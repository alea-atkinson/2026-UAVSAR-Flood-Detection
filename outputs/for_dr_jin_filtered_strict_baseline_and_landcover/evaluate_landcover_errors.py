#!/usr/bin/env python3
"""Evaluate trained U-Net checkpoints by land-cover class.

Loads each filtered-strict leave-one-fp-out checkpoint, runs inference on the
corresponding heldout test split, and groups pixel-level TP/FP/FN/TN by
land-cover code.

Usage:
    python scripts/evaluate_landcover_errors.py --device cuda
    python scripts/evaluate_landcover_errors.py --device cpu --threshold 0.4
    python scripts/evaluate_landcover_errors.py --device cuda \\
        --threshold-csv outputs/threshold_sweep_baseline/threshold_sweep_selected_test_results.csv \\
        --outputs-dir outputs/landcover_error_analysis_threshold_selected
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Bring scripts/ onto sys.path so we can import train_unet_baseline directly.
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import train_unet_baseline as _tb  # noqa: E402  (runs require_package guards)

import rasterio  # noqa: E402  (already imported by _tb; explicit for clarity)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


FPS = list(range(1, 8))
CHECKPOINT_TEMPLATE = "filtered_strict_fp{fp}_unet_20epochs_best.pt"
TEST_CSV_TEMPLATE = "heldout_fp{fp}_test.csv"

LC_NAMES: dict[int, str] = {
    1: "Water",
    2: "Trees",
    4: "Flooded Vegetation",
    5: "Crops",
    7: "Built Area",
    8: "Bare Ground",
    11: "Rangeland",
}

# Column layout for the three output files.
_METRIC_COLS = [
    "pixels", "tp", "fp", "fn", "tn",
    "true_flood_pixels", "pred_flood_pixels",
    "dice", "iou", "precision", "recall",
    "false_positive_rate", "false_negative_rate",
]
# threshold is included per-fp but omitted from the by-class aggregate because
# that file folds results across multiple fps that may use different thresholds.
COLS_BY_FP_AND_CLASS = ["heldout_fp", "threshold", "land_cover_code", "land_cover_name"] + _METRIC_COLS
COLS_BY_CLASS = ["land_cover_code", "land_cover_name"] + _METRIC_COLS
COLS_BY_FP = ["heldout_fp", "threshold"] + _METRIC_COLS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LandCoverEvalDataset(Dataset):
    """Yields (sar [3,H,W] float32, flood [1,H,W] float32, lc [1,H,W] int32).

    Reuses FloodTileDataset._read_rows and _normalize_per_tile from
    train_unet_baseline so loading and normalization stay identical to training.
    """

    def __init__(self, csv_path: Path, data_root: Path) -> None:
        self.data_root = data_root
        self.rows = _tb.FloodTileDataset._read_rows(csv_path)
        if not self.rows:
            raise ValueError(f"{csv_path} has zero rows.")
        if "land_cover_mask_path" not in self.rows[0]:
            raise ValueError(f"{csv_path} is missing column: land_cover_mask_path")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]

        sar_path = self.data_root / row["uavsar_path"]
        flood_path = self.data_root / row["flood_mask_path"]
        lc_path = self.data_root / row["land_cover_mask_path"]

        with rasterio.open(sar_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            raise ValueError(f"Expected ≥3 SAR bands, got {sar.shape[0]} in {sar_path}")
        sar = sar[:3]

        with rasterio.open(flood_path) as src:
            flood = src.read(1, out_dtype="float32")

        with rasterio.open(lc_path) as src:
            lc = src.read(1).astype(np.int32)

        sar = _tb.FloodTileDataset._normalize_per_tile(sar)
        flood_bin = (flood > 0).astype(np.float32)[None, :, :]  # [1, H, W]
        lc_arr = lc[None, :, :]                                  # [1, H, W]

        return torch.from_numpy(sar), torch.from_numpy(flood_bin), torch.from_numpy(lc_arr)


# ---------------------------------------------------------------------------
# Inference and accumulation
# ---------------------------------------------------------------------------

def accumulate_counts(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    fp_label: str,
    counts: dict[tuple[str, int], dict[str, int]],
) -> None:
    """Accumulate pixel-level TP/FP/FN/TN per (fp_label, lc_code) in-place."""
    model.eval()
    with torch.inference_mode():
        for images, masks, lc_batch in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()   # [B, 1, H, W]
            pred = probs > threshold                       # bool [B, 1, H, W]
            true = masks.numpy() > 0.5                    # bool [B, 1, H, W]
            lc_np = lc_batch.numpy()                      # int32 [B, 1, H, W]

            for b in range(pred.shape[0]):
                p = pred[b, 0]    # [H, W]
                t = true[b, 0]    # [H, W]
                lc = lc_np[b, 0]  # [H, W]

                for code in np.unique(lc):
                    m = lc == code
                    key = (fp_label, int(code))
                    c = counts[key]
                    c["tp"] += int(np.sum(p & t & m))
                    c["fp"] += int(np.sum(p & ~t & m))
                    c["fn"] += int(np.sum(~p & t & m))
                    c["tn"] += int(np.sum(~p & ~t & m))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_NA = ""  # written as a blank cell in CSV output


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    pixels = tp + fp + fn + tn
    true_flood = tp + fn
    pred_flood = tp + fp

    # Dice/IoU: undefined (blank) when there are no flood pixels at all, true or predicted.
    # If pred_flood > 0 but true_flood == 0, the score is legitimately 0 (all FP).
    if true_flood == 0 and pred_flood == 0:
        dice: float | str = _NA
        iou: float | str = _NA
    else:
        dice = round(2 * tp / (2 * tp + fp + fn), 6)
        iou = round(tp / (tp + fp + fn), 6)

    # Conditional metrics: blank when the denominator (the relevant reference set) is empty.
    precision: float | str = round(tp / (tp + fp), 6) if (tp + fp) > 0 else _NA
    recall: float | str = round(tp / (tp + fn), 6) if (tp + fn) > 0 else _NA
    fpr: float | str = round(fp / (fp + tn), 6) if (fp + tn) > 0 else _NA
    fnr: float | str = round(fn / (fn + tp), 6) if (fn + tp) > 0 else _NA

    return {
        "pixels": pixels,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "true_flood_pixels": true_flood,
        "pred_flood_pixels": pred_flood,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


# ---------------------------------------------------------------------------
# Threshold map loader
# ---------------------------------------------------------------------------

def load_threshold_map(csv_path: Path) -> dict[str, float]:
    """Read threshold sweep results CSV; return {fp_label: selected_threshold}."""
    result: dict[str, float] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["heldout_fp"]] = float(row["selected_threshold"])
    if not result:
        raise ValueError(f"No rows found in threshold CSV: {csv_path}")
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_split_dir = Path(
        "csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/strict_no_overlap"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net checkpoints by land-cover class."
    )
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--split-dir", type=Path, default=default_split_dir)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/landcover_error_analysis"),
    )
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Fallback if not stored in checkpoint (default: 32)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Global threshold when --threshold-csv is not provided (default: 0.5)")
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to threshold_sweep_selected_test_results.csv. "
             "When provided, uses the per-fp selected_threshold from that file "
             "instead of --threshold.",
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

    print(f"Device     : {device}")
    print(f"Data root  : {args.data_root}")
    print(f"Split dir  : {args.split_dir}")
    print(f"Models dir : {args.models_dir}")
    print(f"Outputs dir: {args.outputs_dir}")
    if threshold_map:
        print(f"Threshold  : per-fp from {args.threshold_csv}")
    else:
        print(f"Threshold  : {args.threshold} (global)")
    print()

    # counts[(fp_label, lc_code)] = {"tp": N, "fp": N, "fn": N, "tn": N}
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    # Record the threshold actually used for each fp so it can be written to CSVs.
    fp_thresholds: dict[str, float] = {}

    for fp in FPS:
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
        fp_thresholds[fp_label] = threshold

        print(f"[{fp_label}] Checkpoint : {checkpoint_path}")
        print(f"[{fp_label}] Threshold  : {threshold:.4f}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_args = checkpoint.get("args", {})
        base_channels = saved_args.get("base_channels", args.base_channels)

        model = _tb.UNet(
            in_channels=3, out_channels=1, base_channels=base_channels
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        dataset = LandCoverEvalDataset(test_csv, args.data_root)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        print(f"[{fp_label}] Evaluating {len(dataset)} tiles (base_channels={base_channels}) ...")
        accumulate_counts(model, loader, device, threshold, fp_label, counts)
        print(f"[{fp_label}] Done.\n")

    if not counts:
        raise SystemExit("No counts accumulated — check that checkpoints and CSVs exist.")

    # ------------------------------------------------------------------
    # 1. Per (fp, lc_code)
    # ------------------------------------------------------------------
    by_fp_class: list[dict] = []
    for (fp_label, lc_code), c in sorted(counts.items()):
        m = compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"])
        by_fp_class.append({
            "heldout_fp": fp_label,
            "threshold": fp_thresholds.get(fp_label, args.threshold),
            "land_cover_code": lc_code,
            "land_cover_name": LC_NAMES.get(lc_code, f"Unknown ({lc_code})"),
            **m,
        })

    # ------------------------------------------------------------------
    # 2. Overall by lc_code (sum across fps)
    # ------------------------------------------------------------------
    class_totals: dict[int, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    for (_, lc_code), c in counts.items():
        ct = class_totals[lc_code]
        ct["tp"] += c["tp"]
        ct["fp"] += c["fp"]
        ct["fn"] += c["fn"]
        ct["tn"] += c["tn"]

    by_class: list[dict] = []
    for lc_code, c in sorted(class_totals.items()):
        m = compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"])
        by_class.append({
            "land_cover_code": lc_code,
            "land_cover_name": LC_NAMES.get(lc_code, f"Unknown ({lc_code})"),
            **m,
        })

    # ------------------------------------------------------------------
    # 3. Overall by fp (sum across lc codes)
    # ------------------------------------------------------------------
    fp_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    for (fp_label, _), c in counts.items():
        ft = fp_totals[fp_label]
        ft["tp"] += c["tp"]
        ft["fp"] += c["fp"]
        ft["fn"] += c["fn"]
        ft["tn"] += c["tn"]

    by_fp: list[dict] = []
    for fp_label, c in sorted(fp_totals.items()):
        m = compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"])
        by_fp.append({
            "heldout_fp": fp_label,
            "threshold": fp_thresholds.get(fp_label, args.threshold),
            **m,
        })

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    out = args.outputs_dir
    p1 = out / "landcover_error_by_fp_and_class.csv"
    p2 = out / "landcover_error_overall_by_class.csv"
    p3 = out / "landcover_error_by_fp_overall.csv"

    write_csv(p1, COLS_BY_FP_AND_CLASS, by_fp_class)
    write_csv(p2, COLS_BY_CLASS, by_class)
    write_csv(p3, COLS_BY_FP, by_fp)

    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    def _f(v: object, width: int, decimals: int = 4) -> str:
        """Format a metric value, printing N/A for blank entries."""
        if v == _NA:
            return "N/A".rjust(width)
        return f"{v:.{decimals}f}".rjust(width)

    print("\n=== Overall by land-cover class ===")
    print(
        f"{'lc_code':>8}  {'name':<20}  {'pixels':>12}  {'dice':>6}  {'iou':>6}"
        f"  {'precision':>9}  {'recall':>7}  {'FPR':>6}  {'FNR':>6}"
    )
    for row in by_class:
        name = row["land_cover_name"]
        print(
            f"{row['land_cover_code']:>8}  {name:<20}  {row['pixels']:>12,}  "
            f"{_f(row['dice'], 6)}  {_f(row['iou'], 6)}  "
            f"{_f(row['precision'], 9)}  {_f(row['recall'], 7)}  "
            f"{_f(row['false_positive_rate'], 6)}  {_f(row['false_negative_rate'], 6)}"
        )

    print("\n=== Overall by heldout flight path ===")
    print(
        f"{'fp':>5}  {'threshold':>9}  {'pixels':>12}  {'dice':>6}  {'iou':>6}"
        f"  {'precision':>9}  {'recall':>7}  {'FPR':>6}  {'FNR':>6}"
    )
    for row in by_fp:
        print(
            f"{row['heldout_fp']:>5}  {row['threshold']:>9.4f}  {row['pixels']:>12,}  "
            f"{_f(row['dice'], 6)}  {_f(row['iou'], 6)}  "
            f"{_f(row['precision'], 9)}  {_f(row['recall'], 7)}  "
            f"{_f(row['false_positive_rate'], 6)}  {_f(row['false_negative_rate'], 6)}"
        )


if __name__ == "__main__":
    main()
