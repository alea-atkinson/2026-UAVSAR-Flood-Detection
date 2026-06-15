#!/usr/bin/env python3
"""Generate flood probability and uncertainty maps for selected heldout test tiles.

For each chosen tile:
  - probability map  : sigmoid(logits)                   → .npy and .png
  - uncertainty map  : 1 - |2p - 1|  (max at p=0.5)    → .npy and .png
  - binary prediction: probability > threshold           → .png
  - ground-truth mask                                    → .png
  - land-cover mask  : if available                      → .png

Usage:
    python scripts/generate_probability_uncertainty_maps.py --device cuda
    python scripts/generate_probability_uncertainty_maps.py --device cuda \\
        --num-tiles 10 --fps 1 3 5 --selection mixed
    python scripts/generate_probability_uncertainty_maps.py --device cuda \\
        --num-tiles 5 \\
        --threshold-csv outputs/threshold_sweep_baseline/threshold_sweep_selected_test_results.csv \\
        --outputs-dir outputs/probability_uncertainty_maps_threshold_selected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Bring scripts/ onto sys.path so train_unet_baseline can be imported directly.
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import train_unet_baseline as _tb  # noqa: E402  (runs require_package guards)

_tb.require_package("PIL")

import rasterio  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


CHECKPOINT_TEMPLATE = "filtered_strict_fp{fp}_unet_20epochs_best.pt"
TEST_CSV_TEMPLATE = "heldout_fp{fp}_test.csv"


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

def _flood_pixel_count(flood_path: Path) -> int:
    with rasterio.open(flood_path) as src:
        return int((src.read(1) > 0).sum())


def select_tiles(
    rows: list[dict[str, str]],
    data_root: Path,
    num_tiles: int,
    mode: str,
) -> list[dict[str, str]]:
    """Return up to num_tiles rows from rows according to mode.

    "first"  — rows[:num_tiles], no file reads.
    "mixed"  — count flood pixels in every flood mask, sort ascending,
               then pick num_tiles indices spread evenly across the sorted
               list so the result spans no-flood → heavy-flood tiles.
    """
    if num_tiles <= 0:
        return []
    if num_tiles >= len(rows):
        return list(rows)

    if mode == "first":
        return rows[:num_tiles]

    # mixed: read all flood masks to get counts, then spread picks.
    print("  (scanning flood masks for tile selection ...)")
    counts = [_flood_pixel_count(data_root / r["flood_mask_path"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: counts[i])
    raw_indices = np.linspace(0, len(order) - 1, num_tiles).round().astype(int)
    seen: set[int] = set()
    chosen: list[int] = []
    for idx in raw_indices:
        orig = order[int(idx)]
        if orig not in seen:
            seen.add(orig)
            chosen.append(orig)
    return [rows[i] for i in chosen]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_tile(
    model: torch.nn.Module,
    sar_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Return flood probability map [H, W] float32 for one SAR tile."""
    with rasterio.open(sar_path) as src:
        sar = src.read(out_dtype="float32")
    if sar.shape[0] < 3:
        raise ValueError(f"Expected ≥3 SAR bands, got {sar.shape[0]} in {sar_path}")
    sar = _tb.FloodTileDataset._normalize_per_tile(sar[:3])

    tensor = torch.from_numpy(sar).unsqueeze(0).to(device)  # [1, 3, H, W]
    with torch.inference_mode():
        logits = model(tensor)
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)  # [H, W]


# ---------------------------------------------------------------------------
# PNG helpers
# ---------------------------------------------------------------------------

def _to_uint8_float(arr: np.ndarray) -> np.ndarray:
    """Scale [0,1] float array to uint8 [0,255]."""
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _to_uint8_binary(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold to 0/255 uint8."""
    return ((arr > threshold) * 255).astype(np.uint8)


def save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr, mode="L").save(path)


def save_lc_png(lc: np.ndarray, path: Path) -> None:
    """Save land-cover codes as grayscale, stretching to 0-255 for visibility."""
    max_val = int(lc.max())
    if max_val > 0:
        img = np.clip((lc.astype(np.float32) / max_val) * 255, 0, 255).astype(np.uint8)
    else:
        img = np.zeros_like(lc, dtype=np.uint8)
    save_png(img, path)


# ---------------------------------------------------------------------------
# Threshold map loader
# ---------------------------------------------------------------------------

def load_threshold_map(csv_path: Path) -> dict[str, float]:
    """Read threshold sweep results CSV; return {fp_label: selected_threshold}."""
    import csv as _csv
    result: dict[str, float] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            result[row["heldout_fp"]] = float(row["selected_threshold"])
    if not result:
        raise ValueError(f"No rows found in threshold CSV: {csv_path}")
    return result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_split_dir = Path(
        "csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/strict_no_overlap"
    )
    parser = argparse.ArgumentParser(
        description="Generate flood probability and uncertainty maps for heldout tiles."
    )
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--split-dir", type=Path, default=default_split_dir)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/probability_uncertainty_maps"),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="U-Net base channels fallback if not stored in checkpoint (default: 32)",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Global prediction threshold when --threshold-csv is not provided (default: 0.5)")
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to threshold_sweep_selected_test_results.csv. "
             "When provided, uses the per-fp selected_threshold for prediction PNGs.",
    )
    parser.add_argument("--num-tiles", type=int, default=5,
                        help="Tiles to process per flight path (default: 5)")
    parser.add_argument(
        "--fps",
        type=int,
        nargs="+",
        default=list(range(1, 8)),
        metavar="FP",
        help="Flight paths to process (default: 1 2 3 4 5 6 7)",
    )
    parser.add_argument(
        "--selection",
        choices=["mixed", "first"],
        default="mixed",
        help=(
            "Tile selection strategy: "
            "'mixed' spreads picks across the flood-pixel distribution; "
            "'first' takes the first N rows as-is (default: mixed)"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load per-fp thresholds if a sweep CSV was provided.
    threshold_map: dict[str, float] = {}
    if args.threshold_csv is not None:
        if not args.threshold_csv.exists():
            raise FileNotFoundError(f"--threshold-csv not found: {args.threshold_csv}")
        threshold_map = load_threshold_map(args.threshold_csv)

    print(f"Device      : {device}")
    print(f"Data root   : {args.data_root}")
    print(f"Outputs dir : {args.outputs_dir}")
    print(f"Num tiles   : {args.num_tiles} per fp")
    print(f"FPs         : {args.fps}")
    print(f"Selection   : {args.selection}")
    if threshold_map:
        print(f"Threshold   : per-fp from {args.threshold_csv}")
    else:
        print(f"Threshold   : {args.threshold} (global)")
    print()

    for fp in args.fps:
        fp_label = f"fp{fp}"
        checkpoint_path = args.models_dir / CHECKPOINT_TEMPLATE.format(fp=fp)
        test_csv = args.split_dir / TEST_CSV_TEMPLATE.format(fp=fp)

        if not checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found, skipping: {checkpoint_path}")
            continue
        if not test_csv.exists():
            print(f"[WARN] Test CSV not found, skipping: {test_csv}")
            continue

        threshold = threshold_map.get(fp_label, args.threshold)
        print(f"[{fp_label}] Threshold  : {threshold:.4f}")
        print(f"[{fp_label}] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_args = checkpoint.get("args", {})
        base_channels = saved_args.get("base_channels", args.base_channels)

        model = _tb.UNet(
            in_channels=3, out_channels=1, base_channels=base_channels
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        rows = _tb.FloodTileDataset._read_rows(test_csv)
        print(f"[{fp_label}] {len(rows)} test tiles — selecting {args.num_tiles} ({args.selection})")
        selected = select_tiles(rows, args.data_root, args.num_tiles, args.selection)
        print(f"[{fp_label}] Processing {len(selected)} tiles ...")

        out_dir = args.outputs_dir / fp_label
        out_dir.mkdir(parents=True, exist_ok=True)

        for row in selected:
            sample_id = row.get("sample_id") or Path(row["uavsar_path"]).stem
            sar_path = args.data_root / row["uavsar_path"]
            flood_path = args.data_root / row["flood_mask_path"]

            # --- Inference ---
            prob = infer_tile(model, sar_path, device)            # [H, W] float32 in [0,1]
            uncertainty = (1.0 - np.abs(2.0 * prob - 1.0)).astype(np.float32)
            prediction = (prob > threshold).astype(np.float32)

            # --- Ground truth ---
            with rasterio.open(flood_path) as src:
                gt = (src.read(1) > 0).astype(np.float32)

            # --- Save numpy arrays ---
            np.save(out_dir / f"{sample_id}_probability.npy", prob)
            np.save(out_dir / f"{sample_id}_uncertainty.npy", uncertainty)

            # --- Save PNGs ---
            save_png(_to_uint8_float(prob),        out_dir / f"{sample_id}_probability.png")
            save_png(_to_uint8_float(uncertainty),  out_dir / f"{sample_id}_uncertainty.png")
            save_png(_to_uint8_binary(prediction),  out_dir / f"{sample_id}_prediction.png")
            save_png(_to_uint8_binary(gt),          out_dir / f"{sample_id}_ground_truth.png")

            # --- Optional land-cover PNG ---
            lc_path_str = row.get("land_cover_mask_path", "").strip()
            if lc_path_str:
                lc_path = args.data_root / lc_path_str
                if lc_path.exists():
                    with rasterio.open(lc_path) as src:
                        lc = src.read(1)
                    save_lc_png(lc, out_dir / f"{sample_id}_land_cover.png")

            n_flood_true = int(gt.sum())
            n_flood_pred = int(prediction.sum())
            print(
                f"  {sample_id}: "
                f"true_flood={n_flood_true:,}px  "
                f"pred_flood={n_flood_pred:,}px  "
                f"max_uncertainty={uncertainty.max():.3f}"
            )

        print(f"[{fp_label}] Done → {out_dir}\n")


if __name__ == "__main__":
    main()
