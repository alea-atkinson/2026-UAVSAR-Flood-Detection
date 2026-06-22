#!/usr/bin/env python3
"""Threshold sweep for a single trained checkpoint.

Given one checkpoint, one validation CSV, and one test CSV:
  1. Load the checkpoint.
  2. Run one inference pass on the validation split to collect pixel probabilities.
  3. Sweep the given thresholds using global pixel-level Dice.
  4. Select the threshold that maximises validation Dice (ties broken by IoU,
     then by proximity to 0.5).
  5. Evaluate on the test split once, at the selected threshold.

Reuses FloodTileDataset/UNet from train_unet_experiment.py and the pixel-metric
definitions from threshold_sweep_baseline.py, so results are directly
comparable with the baseline sweep.

Usage:
    python scripts/threshold_sweep_single_checkpoint.py \
        --checkpoint models/experiment_fp1_bce_dice_aug_20epochs_best.pt \
        --val-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_validation.csv \
        --test-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_test.csv \
        --output-csv outputs/threshold_sweep_single_checkpoint/experiment_fp1_bce_dice_aug_20epochs.csv \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def print_basic_help() -> None:
    """Show this script's own help, independent of any imported module's argparse."""
    print(
        """usage: threshold_sweep_single_checkpoint.py [options]

Sweep thresholds on a validation split for one checkpoint, select the
threshold that maximizes validation Dice, then evaluate once on the test
split at that threshold.

options:
  --checkpoint PATH    Path to a trained checkpoint (.pt)            [required]
  --val-csv PATH       Validation split CSV                          [required]
  --test-csv PATH      Test split CSV                                [required]
  --output-csv PATH    Where to write the sweep + selected results   [required]
  --data-root PATH     Root folder for tile paths (default: 2025_Tile_Data)
  --device DEVICE      cuda or cpu (default: cuda)
  --thresholds LIST    Comma-separated thresholds to sweep
                        (default: 0.05,0.10,...,0.95 step 0.05)

Output CSV columns:
  threshold,val_dice,val_iou,val_precision,val_recall,
  test_dice,test_iou,test_precision,test_recall,selected

Test columns are blank for every threshold except the selected one.

Tie-breaking when selecting the best threshold:
  1. highest val_dice
  2. highest val_iou
  3. threshold closest to 0.5
"""
    )


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print_basic_help()
    raise SystemExit(0)


# Bring scripts/ onto sys.path so sibling modules can be imported directly.
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import threshold_sweep_baseline as _tsb  # noqa: E402  (reuse pixel-metric definitions)
import train_unet_experiment as _te  # noqa: E402  (reuse FloodTileDataset / UNet)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"
BATCH_SIZE = 8
NUM_WORKERS = 2

OUTPUT_FIELDNAMES = [
    "threshold",
    "val_dice", "val_iou", "val_precision", "val_recall",
    "test_dice", "test_iou", "test_precision", "test_recall",
    "selected",
]


def parse_thresholds(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--thresholds must contain at least one value")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--thresholds contains a non-numeric value: {exc}") from exc


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold sweep for a single trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Val CSV     : {args.val_csv}")
    print(f"Test CSV    : {args.test_csv}")
    print(f"Data root   : {args.data_root}")
    print(
        f"Thresholds  : {args.thresholds[0]:.2f} -> {args.thresholds[-1]:.2f} "
        f"({len(args.thresholds)} values)"
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    base_channels = saved_args.get("base_channels", 32)

    model = _te.UNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pin_memory = device.type == "cuda"

    # ------------------------------------------------------------------
    # Validation pass: collect all probabilities in one go
    # ------------------------------------------------------------------
    val_dataset = _te.FloodTileDataset(args.val_csv, args.data_root, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    print(f"Validation inference ({len(val_dataset)} tiles) ...")
    val_probs, val_targets = _tsb.collect_probs(model, val_loader, device)

    # ------------------------------------------------------------------
    # Threshold sweep on validation
    # ------------------------------------------------------------------
    sweep_rows: list[dict] = []
    for t in args.thresholds:
        m = _tsb.pixel_metrics(val_probs, val_targets, t)
        sweep_rows.append({
            "threshold": t,
            "val_dice": m["dice"],
            "val_iou": m["iou"],
            "val_precision": m["precision"],
            "val_recall": m["recall"],
        })

    best = _tsb.select_best_threshold(sweep_rows)
    selected_t = best["threshold"]
    print(
        f"Selected threshold: {selected_t:.2f} "
        f"(val_dice={best['val_dice']:.4f}, val_iou={best['val_iou']:.4f})"
    )

    # ------------------------------------------------------------------
    # Test pass at selected threshold only
    # ------------------------------------------------------------------
    test_dataset = _te.FloodTileDataset(args.test_csv, args.data_root, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    print(f"Test inference ({len(test_dataset)} tiles) ...")
    test_probs, test_targets = _tsb.collect_probs(model, test_loader, device)
    test_m = _tsb.pixel_metrics(test_probs, test_targets, selected_t)

    # ------------------------------------------------------------------
    # Build output rows: test columns blank except the selected threshold
    # ------------------------------------------------------------------
    output_rows: list[dict] = []
    for row in sweep_rows:
        is_selected = row["threshold"] == selected_t
        out_row = dict(row)
        out_row["selected"] = is_selected
        if is_selected:
            out_row["test_dice"] = test_m["dice"]
            out_row["test_iou"] = test_m["iou"]
            out_row["test_precision"] = test_m["precision"]
            out_row["test_recall"] = test_m["recall"]
        output_rows.append(out_row)

    write_csv(args.output_csv, output_rows)
    print(f"Wrote: {args.output_csv}")
    print(
        f"Test metrics at selected threshold {selected_t:.2f}: "
        f"dice={test_m['dice']:.4f} iou={test_m['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
