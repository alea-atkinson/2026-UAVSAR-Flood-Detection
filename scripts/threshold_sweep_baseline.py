#!/usr/bin/env python3
"""Threshold sweep for trained U-Net checkpoints.

For each heldout flight path:
  1. Load the checkpoint.
  2. Run one inference pass on the validation split to collect pixel probabilities.
  3. Sweep thresholds 0.05-0.95 (step 0.05) using global pixel-level Dice.
  4. Select the threshold that maximises validation Dice (ties broken by IoU,
     then by proximity to 0.5).
  5. Evaluate on the test split at the selected threshold.

Usage:
    python scripts/threshold_sweep_baseline.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Bring scripts/ onto sys.path so train_unet_baseline can be imported directly.
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import train_unet_baseline as _tb  # noqa: E402  (runs require_package guards)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


FPS = list(range(1, 8))
CHECKPOINT_TEMPLATE = "filtered_strict_fp{fp}_unet_20epochs_best.pt"
VAL_CSV_TEMPLATE = "heldout_fp{fp}_validation.csv"
TEST_CSV_TEMPLATE = "heldout_fp{fp}_test.csv"

THRESHOLDS: list[float] = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()
DEFAULT_THRESHOLD = 0.5

_NA = ""  # written as a blank cell in CSV

SWEEP_COLS = [
    "heldout_fp", "threshold",
    "val_dice", "val_iou", "val_precision", "val_recall",
    "val_tp", "val_fp", "val_fn", "val_tn",
]
SELECTED_COLS = [
    "heldout_fp", "selected_threshold",
    "val_dice_at_selected_threshold", "val_iou_at_selected_threshold",
    "test_dice", "test_iou", "test_precision", "test_recall",
    "test_tp", "test_fp", "test_fn", "test_tn",
]


# ---------------------------------------------------------------------------
# Inference: one pass, return flat pixel-level arrays
# ---------------------------------------------------------------------------

def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on loader; return (probs [N_pixels] float32, targets [N_pixels] bool).

    All tiles are flattened and concatenated so the threshold sweep operates on
    the full split's pixel population, not on per-tile averages.
    """
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # [B, H, W]
            targets = masks.squeeze(1).numpy() > 0.5               # [B, H, W] bool
            all_probs.append(probs.ravel())
            all_targets.append(targets.ravel())

    return (
        np.concatenate(all_probs).astype(np.float32),
        np.concatenate(all_targets),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def counts_to_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    # Dice and IoU: 0.0 when the denominator is zero (all pixels are TN at extreme
    # thresholds). Using 0.0 rather than blank so tie-breaking comparisons work.
    dice: float = round(2 * tp / denom_dice, 6) if denom_dice > 0 else 0.0
    iou: float = round(tp / denom_iou, 6) if denom_iou > 0 else 0.0
    # Precision and recall: blank when the reference set is empty.
    precision: float | str = round(tp / (tp + fp), 6) if (tp + fp) > 0 else _NA
    recall: float | str = round(tp / (tp + fn), 6) if (tp + fn) > 0 else _NA
    return {
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pixel_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict:
    pred = probs > threshold
    tp = int((pred & targets).sum())
    fp = int((pred & ~targets).sum())
    fn = int((~pred & targets).sum())
    tn = int((~pred & ~targets).sum())
    return counts_to_metrics(tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_best_threshold(sweep_rows: list[dict]) -> dict:
    """Return the sweep row with best val Dice; ties broken by IoU then |t-0.5|."""
    def _key(r: dict) -> tuple:
        return (-r["val_dice"], -r["val_iou"], abs(r["threshold"] - DEFAULT_THRESHOLD))
    return min(sweep_rows, key=_key)


# ---------------------------------------------------------------------------
# CSV writing
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
        description="Threshold sweep for trained U-Net checkpoints."
    )
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--split-dir", type=Path, default=default_split_dir)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/threshold_sweep_baseline"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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

    print(f"Device      : {device}")
    print(f"Data root   : {args.data_root}")
    print(f"Split dir   : {args.split_dir}")
    print(f"Outputs dir : {args.outputs_dir}")
    print(
        f"Thresholds  : {THRESHOLDS[0]:.2f} → {THRESHOLDS[-1]:.2f}  "
        f"(step 0.05, {len(THRESHOLDS)} values)"
    )
    print()

    all_sweep_rows: list[dict] = []
    all_selected_rows: list[dict] = []
    summary: list[dict] = []

    for fp in FPS:
        fp_label = f"fp{fp}"
        checkpoint_path = args.models_dir / CHECKPOINT_TEMPLATE.format(fp=fp)
        val_csv = args.split_dir / VAL_CSV_TEMPLATE.format(fp=fp)
        test_csv = args.split_dir / TEST_CSV_TEMPLATE.format(fp=fp)

        missing = [p for p in (checkpoint_path, val_csv, test_csv) if not p.exists()]
        if missing:
            for p in missing:
                print(f"[WARN] Not found, skipping {fp_label}: {p}")
            continue

        print(f"[{fp_label}] Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_args = checkpoint.get("args", {})
        base_channels = saved_args.get("base_channels", 32)

        model = _tb.UNet(
            in_channels=3, out_channels=1, base_channels=base_channels
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        pin_memory = device.type == "cuda"

        # ------------------------------------------------------------------
        # Validation pass: collect all probabilities in one go
        # ------------------------------------------------------------------
        val_dataset = _tb.FloodTileDataset(val_csv, args.data_root)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        print(f"[{fp_label}] Validation inference ({len(val_dataset)} tiles) ...")
        val_probs, val_targets = collect_probs(model, val_loader, device)

        # ------------------------------------------------------------------
        # Threshold sweep on validation
        # ------------------------------------------------------------------
        fp_sweep_rows: list[dict] = []
        for t in THRESHOLDS:
            m = pixel_metrics(val_probs, val_targets, t)
            fp_sweep_rows.append({
                "heldout_fp": fp_label,
                "threshold": t,
                "val_dice": m["dice"],
                "val_iou": m["iou"],
                "val_precision": m["precision"],
                "val_recall": m["recall"],
                "val_tp": m["tp"],
                "val_fp": m["fp"],
                "val_fn": m["fn"],
                "val_tn": m["tn"],
            })

        all_sweep_rows.extend(fp_sweep_rows)

        # ------------------------------------------------------------------
        # Select best threshold
        # ------------------------------------------------------------------
        best = select_best_threshold(fp_sweep_rows)
        selected_t = best["threshold"]
        default_m = pixel_metrics(val_probs, val_targets, DEFAULT_THRESHOLD)

        # ------------------------------------------------------------------
        # Test pass at selected threshold (no GPU until threshold is decided)
        # ------------------------------------------------------------------
        test_dataset = _tb.FloodTileDataset(test_csv, args.data_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        print(f"[{fp_label}] Test inference ({len(test_dataset)} tiles) ...")
        test_probs, test_targets = collect_probs(model, test_loader, device)
        test_m = pixel_metrics(test_probs, test_targets, selected_t)

        all_selected_rows.append({
            "heldout_fp": fp_label,
            "selected_threshold": selected_t,
            "val_dice_at_selected_threshold": best["val_dice"],
            "val_iou_at_selected_threshold": best["val_iou"],
            "test_dice": test_m["dice"],
            "test_iou": test_m["iou"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_tp": test_m["tp"],
            "test_fp": test_m["fp"],
            "test_fn": test_m["fn"],
            "test_tn": test_m["tn"],
        })

        summary.append({
            "fp": fp_label,
            "val_dice_at_0.5": default_m["dice"],
            "selected_t": selected_t,
            "val_dice_at_best": best["val_dice"],
            "test_dice_at_best": test_m["dice"],
        })

        print(
            f"[{fp_label}] val@0.5={default_m['dice']:.4f}  "
            f"best_t={selected_t:.2f}  "
            f"val@best={best['val_dice']:.4f}  "
            f"test@best={test_m['dice']:.4f}\n"
        )

    if not all_sweep_rows:
        raise SystemExit("No results — check that checkpoints and CSVs exist.")

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    out = args.outputs_dir
    p1 = out / "threshold_sweep_validation_by_fp.csv"
    p2 = out / "threshold_sweep_selected_test_results.csv"

    write_csv(p1, SWEEP_COLS, all_sweep_rows)
    write_csv(p2, SELECTED_COLS, all_selected_rows)

    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print("\n=== Threshold sweep summary ===")
    print(
        f"{'fp':>5}  {'val_dice@0.50':>13}  {'best_t':>6}  "
        f"{'val_dice@best':>13}  {'test_dice@best':>14}"
    )
    for r in summary:
        print(
            f"{r['fp']:>5}  {r['val_dice_at_0.5']:>13.4f}  {r['selected_t']:>6.2f}  "
            f"{r['val_dice_at_best']:>13.4f}  {r['test_dice_at_best']:>14.4f}"
        )


if __name__ == "__main__":
    main()
