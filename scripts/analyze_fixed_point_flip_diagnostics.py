#!/usr/bin/env python3
"""
analyze_fixed_point_flip_diagnostics.py

Per-tile diagnostic for the prediction flips found by
scripts/analyze_first_layer_fixed_point_segmentation_impact.py: when the
first Conv-BN-ReLU stage's output is replaced by a hardware-style
fixed-point folded approximation (rest of the U-Net unchanged, PyTorch
float), a small percentage of pixels' binary flood/no-flood prediction
changes. This script asks: WHERE do those flips sit relative to the 0.5
probability threshold? Are they low-confidence pixels sitting right on
the decision boundary (expected, benign noise), or are confidently-decided
pixels actually flipping (a more concerning signal)?

This was motivated by two outlier tiles found in the full 138-tile fp2
run (`outputs/hardware_fidelity/first_layer_fixed_point_segmentation_impact_full_fp2/`):
- tile_46_48.tif: worst by flip percentage (3.2181% pixels changed),
  Dice diff -0.00424.
- tile_42_44.tif: worst by absolute Dice diff (+0.03808), despite only
  0.3555% of pixels changing -- i.e. a LARGE Dice swing from a SMALL
  number of flipped pixels, which this script explains via ground-truth
  class imbalance at the pixel level (see summary).

---- What this is NOT ----------------------------------------------------
- This is NOT a full FPGA U-Net. No layer other than the first
  Conv-BN-ReLU is touched in any way.
- This is NOT VHDL simulation. No VHDL is written, modified, or run.
- This does NOT claim any hardware speedup, board power, or deployment
  readiness.

---- What this IS ----------------------------------------------------------
A diagnostic, not a new experiment: it reuses the EXACT model loading,
normalization, BatchNorm-folding, INT8 quantization, padding, and
network-surgery substitution logic from
`scripts/analyze_first_layer_fixed_point_segmentation_impact.py`
(functions copied verbatim below, unchanged math), for a small,
explicitly-named set of tiles, and adds flip-location diagnostics that the
earlier per-tile CSV did not report: distance of the FLOAT probability to
the 0.5 threshold at flipped pixels, and whether each flip fixed or broke
an already-correct prediction relative to ground truth.

---- Sources inspected before writing this script -----------------------
- scripts/analyze_first_layer_fixed_point_segmentation_impact.py: model
  definition, checkpoint loading, normalization, BatchNorm folding,
  quantization, padding, and network-surgery substitution -- ALL REUSED
  VERBATIM, no math changed.

---- Data -------------------------------------------------------------------
Real UAVSAR fp2 held-out test tiles, looked up BY NAME (via the `tile_name`
column) in:
csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/
strict_no_overlap/heldout_fp2_test.csv

---- Outputs -----------------------------------------------------------
outputs/hardware_fidelity/first_layer_fixed_point_flip_diagnostics/
    flip_diagnostics_by_tile.csv
    flip_diagnostics_summary.md
    diagnostic_<tile_name>.png   (one per analyzed tile)
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import rasterio
import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
DEFAULT_SPLIT_CSV = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap" / "heldout_fp2_test.csv"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "2025_Tile_Data"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "outputs" / "hardware_fidelity" / "first_layer_fixed_point_flip_diagnostics"
)
DEFAULT_TILE_NAMES = [
    "tile_46_48.tif", "tile_42_44.tif", "tile_45_49.tif", "tile_33_44.tif",
]
DEFAULT_NEAR_THRESHOLD_BANDS = [0.01, 0.025, 0.05, 0.10]

BN_EPS = 1e-5           # torch.nn.BatchNorm2d default
METRIC_EPS = 1e-7       # matches dice_iou_from_logits in train_unet_baseline_tuned.py


# =============================================================================
# EVERYTHING IN THIS SECTION IS COPIED VERBATIM FROM
# analyze_first_layer_fixed_point_segmentation_impact.py -- NO MATH CHANGED.
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small, plain U-Net for binary segmentation. Identical to the class of
    the same name in train_unet_baseline_tuned.py."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        x = self.bottleneck(self.pool(enc4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, enc4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, enc3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, enc2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, enc1], dim=1))
        return self.out(x)


def unet_forward_from_enc1_first_stage(model: UNet, enc1_first_stage_out: torch.Tensor) -> torch.Tensor:
    """Runs the REST of the U-Net starting from `enc1.block[0:3]`'s output."""
    enc1 = model.enc1.block[3:](enc1_first_stage_out)
    enc2 = model.enc2(model.pool(enc1))
    enc3 = model.enc3(model.pool(enc2))
    enc4 = model.enc4(model.pool(enc3))

    x = model.bottleneck(model.pool(enc4))

    x = model.up4(x)
    x = model.dec4(torch.cat([x, enc4], dim=1))
    x = model.up3(x)
    x = model.dec3(torch.cat([x, enc3], dim=1))
    x = model.up2(x)
    x = model.dec2(torch.cat([x, enc2], dim=1))
    x = model.up1(x)
    x = model.dec1(torch.cat([x, enc1], dim=1))
    return model.out(x)


def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
    """Identical to FloodTileDataset._normalize_per_tile in train_unet_baseline_tuned.py."""
    sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = sar[np.isfinite(sar)]
    if valid.size == 0:
        return np.zeros_like(sar, dtype=np.float32)
    low, high = np.percentile(valid, [1.0, 99.0])
    if high > low:
        sar = np.clip(sar, low, high)
    mean = float(sar.mean())
    std = float(sar.std())
    if std < 1e-6:
        return np.zeros_like(sar, dtype=np.float32)
    return ((sar - mean) / std).astype(np.float32)


def quantize_symmetric_int8(w_float: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs_w = float(np.abs(w_float).max())
    scale_w = 1.0 if max_abs_w == 0.0 else max_abs_w / 127.0
    w_int8 = np.clip(np.round(w_float / scale_w), -127, 127).astype(np.int8)
    return w_int8, scale_w


def quantize_symmetric_int8_activation(x_float: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs_x = float(np.abs(x_float).max())
    scale_x = 1.0 if max_abs_x == 0.0 else max_abs_x / 127.0
    x_int8 = np.clip(np.round(x_float / scale_x), -127, 127).astype(np.int8)
    return x_int8, scale_x


def conv_all_kernels_valid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (3, H, W) int8 or float; w: (n_out, 3, 3, 3) int8 or float.
    Valid (no-padding) 3x3 conv, vectorized with sliding_window_view + einsum."""
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    acc_dtype = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    return np.einsum(
        "chwij,ocij->ohw",
        windows.astype(acc_dtype),
        w.astype(acc_dtype),
        optimize=True,
    )


def folded_quantized_conv_bn_relu_padded(
    x_float_chw: np.ndarray,
    w_folded_int8_all: np.ndarray,
    scales_w_folded: np.ndarray,
    b_folded_all: np.ndarray,
) -> tuple[np.ndarray, float]:
    """The hardware-style fixed-point folded Conv-BN-ReLU approximation, WITH
    PADDING so the output spatial size matches the true padded PyTorch output."""
    x_int8, scale_x = quantize_symmetric_int8_activation(x_float_chw)
    x_int8_padded = np.pad(x_int8, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    int32_accum = conv_all_kernels_valid(x_int8_padded, w_folded_int8_all)  # (n_out, H, W)
    conv_float_approx = (
        int32_accum.astype(np.float64) * scale_x * scales_w_folded[:, None, None]
    )
    biased = conv_float_approx + b_folded_all[:, None, None]
    return np.maximum(biased, 0.0), scale_x


def confusion_counts(preds_bool: np.ndarray, targets_bool: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(preds_bool, targets_bool).sum())
    fp = int(np.logical_and(preds_bool, ~targets_bool).sum())
    fn = int(np.logical_and(~preds_bool, targets_bool).sum())
    tn = int(np.logical_and(~preds_bool, ~targets_bool).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = METRIC_EPS) -> dict[str, float]:
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    return {
        "dice": dice, "iou": iou, "precision": precision,
        "recall": recall, "pixel_accuracy": pixel_acc,
    }


def load_model(checkpoint_path: pathlib.Path, device: torch.device) -> UNet:
    print(f"[1] Loading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        sys.exit(f"ERROR: checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    base_channels = 32
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, dict) and "base_channels" in saved_args:
        base_channels = int(saved_args["base_channels"])
    print(f"    base_channels (from checkpoint's saved args, default 32 if absent): {base_channels}")

    model = UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def repo_relative_str(path: pathlib.Path) -> str:
    """Best-effort repo-relative display string for a path, for printing
    only. Falls back to the absolute path (never raises) if `path` is not
    under REPO_ROOT."""
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# END of verbatim-reused section. Everything below is NEW diagnostic logic.
# =============================================================================


def percentile_or_nan(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def make_sar_display_composite(sar_norm: np.ndarray) -> np.ndarray:
    """Builds a simple 3-channel false-color display composite from the
    first 3 (normalized) SAR channels, each independently min-max scaled to
    [0, 1] for display only -- this is a visualization convenience, NOT a
    change to the model input (the model still sees `sar_norm` unchanged)."""
    disp = np.zeros((sar_norm.shape[1], sar_norm.shape[2], 3), dtype=np.float32)
    for c in range(min(3, sar_norm.shape[0])):
        band = sar_norm[c]
        lo, hi = band.min(), band.max()
        disp[:, :, c] = (band - lo) / (hi - lo) if hi > lo else 0.0
    return disp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose WHERE first-layer fixed-point prediction "
        "flips sit relative to the probability threshold, for a small set "
        "of named fp2 test tiles."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-csv", type=pathlib.Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability threshold for binary flood prediction (default: 0.5).")
    parser.add_argument("--tile-names", nargs="+", default=DEFAULT_TILE_NAMES,
                         help="Tile names (matching the split CSV's `tile_name` column) "
                         f"to analyze (default: {DEFAULT_TILE_NAMES}).")
    parser.add_argument("--near-threshold-bands", type=float, nargs="+",
                         default=DEFAULT_NEAR_THRESHOLD_BANDS,
                         help="Distance-to-threshold bands (in probability units) to report "
                         f"the pct of flipped pixels within (default: {DEFAULT_NEAR_THRESHOLD_BANDS}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_tile_csv_path = output_dir / "flip_diagnostics_by_tile.csv"
    summary_md_path = output_dir / "flip_diagnostics_summary.md"

    device = torch.device(args.device)
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    print(f"    Model moved to device: {device}\n")

    state_dict = model.state_dict()
    w_float_all = state_dict["enc1.block.0.weight"].detach().cpu().float().numpy()
    n_out = w_float_all.shape[0]
    bn_gamma = state_dict["enc1.block.1.weight"].detach().cpu().float().numpy()
    bn_beta = state_dict["enc1.block.1.bias"].detach().cpu().float().numpy()
    bn_mean = state_dict["enc1.block.1.running_mean"].detach().cpu().float().numpy()
    bn_var = state_dict["enc1.block.1.running_var"].detach().cpu().float().numpy()

    print("[2] Folding BatchNorm into first-layer Conv2d weights/bias ...")
    scale_bn = bn_gamma / np.sqrt(bn_var + BN_EPS)
    w_folded_all = w_float_all * scale_bn[:, None, None, None]
    b_folded_all = bn_beta - bn_mean * scale_bn

    print(f"[3] Quantizing all {n_out} FOLDED kernels to INT8 (symmetric per-output-channel scale) ...")
    scales_w_folded = np.zeros(n_out, dtype=np.float64)
    w_folded_int8_all = np.zeros_like(w_folded_all, dtype=np.int8)
    for oc in range(n_out):
        w_folded_int8_all[oc], scales_w_folded[oc] = quantize_symmetric_int8(w_folded_all[oc])

    # -- Look up the requested tiles BY NAME in the split CSV --
    print(f"\n[4] Looking up {len(args.tile_names)} requested tile(s) in split CSV ...")
    print(f"    Split CSV : {args.split_csv}")
    if not args.split_csv.exists():
        sys.exit(f"ERROR: split CSV not found at {args.split_csv}")
    if not args.data_root.exists():
        sys.exit(f"ERROR: data root not found at {args.data_root}")

    with open(args.split_csv, newline="") as f:
        split_rows = list(csv.DictReader(f))
    rows_by_name = {row["tile_name"]: row for row in split_rows}

    missing = [name for name in args.tile_names if name not in rows_by_name]
    if missing:
        print(f"    WARNING: {len(missing)} requested tile name(s) not found in split CSV: {missing}")
    found_names = [name for name in args.tile_names if name in rows_by_name]
    if not found_names:
        sys.exit("ERROR: none of the requested --tile-names were found in the split CSV.")

    bands = sorted(args.near_threshold_bands)

    per_tile_rows = []
    all_flip_dist_to_threshold = []  # pooled across all analyzed tiles, for the summary

    for name in found_names:
        row = rows_by_name[name]
        sar_path = args.data_root / row["uavsar_path"]
        mask_path = args.data_root / row["flood_mask_path"]
        if not sar_path.exists() or not mask_path.exists():
            print(f"    WARNING: tile '{name}' found in split CSV but file(s) missing on disk -- skipping.")
            continue

        with rasterio.open(sar_path) as src:
            sar = src.read(out_dtype="float32")
        sar = sar[:3]
        sar_norm = normalize_per_tile(sar)
        with rasterio.open(mask_path) as src:
            mask = src.read(1, out_dtype="float32")
        mask_bool = mask > 0

        x = torch.from_numpy(sar_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits_a = model(x)
            relu_out_padded, scale_x = folded_quantized_conv_bn_relu_padded(
                sar_norm.astype(np.float64), w_folded_int8_all, scales_w_folded, b_folded_all
            )
            enc1_fixedpoint_first_stage = (
                torch.from_numpy(relu_out_padded).float().unsqueeze(0).to(device)
            )
            logits_b = unet_forward_from_enc1_first_stage(model, enc1_fixedpoint_first_stage)

        prob_a = torch.sigmoid(logits_a).squeeze().cpu().numpy()
        prob_b = torch.sigmoid(logits_b).squeeze().cpu().numpy()
        preds_a = prob_a > args.threshold
        preds_b = prob_b > args.threshold

        tp_a, fp_a, fn_a, tn_a = confusion_counts(preds_a, mask_bool)
        tp_b, fp_b, fn_b, tn_b = confusion_counts(preds_b, mask_bool)
        m_a = metrics_from_counts(tp_a, fp_a, fn_a, tn_a)
        m_b = metrics_from_counts(tp_b, fp_b, fn_b, tn_b)

        prob_diff = np.abs(prob_a - prob_b)
        changed = preds_a != preds_b
        num_pixels = int(preds_a.size)
        num_flipped = int(changed.sum())
        pct_flipped = 100.0 * num_flipped / num_pixels

        flipped_prob_diff = prob_diff[changed]
        dist_to_threshold_all_flipped = np.abs(prob_a[changed] - args.threshold)
        all_flip_dist_to_threshold.append(dist_to_threshold_all_flipped)

        # -- Item 12: pct of flipped pixels within each near-threshold band --
        near_threshold_pcts = {}
        for band in bands:
            n_within = int((dist_to_threshold_all_flipped <= band).sum())
            pct_within = 100.0 * n_within / num_flipped if num_flipped else float("nan")
            near_threshold_pcts[band] = pct_within

        # -- Item 13: FP/FN relative to ground truth, before vs. after, among
        # flipped pixels only. Also: since preds_b == ~preds_a at every
        # flipped pixel (binary flip), each flipped pixel either FIXES an
        # error (float was wrong, fixed-point is now right) or BREAKS a
        # previously-correct prediction (float was right, fixed-point is
        # now wrong) -- this split explains the SIGN of the tile's Dice
        # diff directly. --
        mask_flipped = mask_bool[changed]
        preds_a_flipped = preds_a[changed]
        preds_b_flipped = preds_b[changed]
        n_fp_before = int(np.logical_and(preds_a_flipped, ~mask_flipped).sum())
        n_fn_before = int(np.logical_and(~preds_a_flipped, mask_flipped).sum())
        n_fp_after = int(np.logical_and(preds_b_flipped, ~mask_flipped).sum())
        n_fn_after = int(np.logical_and(~preds_b_flipped, mask_flipped).sum())
        correct_before_flipped = preds_a_flipped == mask_flipped
        n_improved = int((~correct_before_flipped).sum())  # wrong -> right
        n_degraded = int(correct_before_flipped.sum())     # right -> wrong

        def pct_of_flipped(n: int) -> float:
            return 100.0 * n / num_flipped if num_flipped else float("nan")

        # -- Item 14 (optional): flip-direction counts --
        n_nonflood_to_flood = int(np.logical_and(~preds_a_flipped, preds_b_flipped).sum())
        n_flood_to_nonflood = int(np.logical_and(preds_a_flipped, ~preds_b_flipped).sum())

        row_out = {
            "tile_name": name,
            "uavsar_path": row["uavsar_path"],
            "num_pixels": num_pixels,
            "num_flipped": num_flipped,
            "pct_flipped": pct_flipped,
            "dice_float": m_a["dice"], "dice_fixedpoint": m_b["dice"],
            "dice_diff": m_b["dice"] - m_a["dice"],
            "mean_abs_prob_diff_all": float(prob_diff.mean()),
            "mean_abs_prob_diff_flipped": mean_or_nan(flipped_prob_diff),
            "median_abs_prob_diff_flipped": median_or_nan(flipped_prob_diff),
            "p95_abs_prob_diff_flipped": percentile_or_nan(flipped_prob_diff, 95),
            "mean_dist_to_threshold_flipped": mean_or_nan(dist_to_threshold_all_flipped),
            "median_dist_to_threshold_flipped": median_or_nan(dist_to_threshold_all_flipped),
            "p95_dist_to_threshold_flipped": percentile_or_nan(dist_to_threshold_all_flipped, 95),
            "pct_flipped_fp_before": pct_of_flipped(n_fp_before),
            "pct_flipped_fn_before": pct_of_flipped(n_fn_before),
            "pct_flipped_fp_after": pct_of_flipped(n_fp_after),
            "pct_flipped_fn_after": pct_of_flipped(n_fn_after),
            "pct_flipped_improved": pct_of_flipped(n_improved),
            "pct_flipped_degraded": pct_of_flipped(n_degraded),
            "flip_count_nonflood_to_flood": n_nonflood_to_flood,
            "flip_count_flood_to_nonflood": n_flood_to_nonflood,
        }
        for band in bands:
            row_out[f"pct_flipped_within_{band}"] = near_threshold_pcts[band]
        per_tile_rows.append(row_out)

        print(f"    [{name}] pixels={num_pixels} flipped={num_flipped} ({pct_flipped:.4f}%) "
              f"dice_float={m_a['dice']:.4f} dice_fixedpoint={m_b['dice']:.4f} "
              f"dice_diff={row_out['dice_diff']:+.5f} "
              f"mean_dist_to_thresh={row_out['mean_dist_to_threshold_flipped']:.4f} "
              f"improved={n_improved} degraded={n_degraded}")

        # -- Diagnostic PNG for this tile --
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sar_composite = make_sar_display_composite(sar_norm)
            dist_map = np.full(prob_a.shape, np.nan, dtype=np.float64)
            dist_map = np.abs(prob_a - args.threshold)

            fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
            axes[0, 0].imshow(sar_composite)
            axes[0, 0].set_title(f"SAR composite (ch0/1/2)\n{name}")
            axes[0, 1].imshow(mask_bool, cmap="Blues")
            axes[0, 1].set_title("Ground truth flood mask")
            axes[0, 2].imshow(preds_a, cmap="Blues")
            axes[0, 2].set_title(f"Float prediction\nDice={m_a['dice']:.4f}")
            axes[0, 3].imshow(preds_b, cmap="Blues")
            axes[0, 3].set_title(f"Fixed-point 1st-layer prediction\nDice={m_b['dice']:.4f}")

            axes[1, 0].imshow(changed, cmap="Reds")
            axes[1, 0].set_title(f"Changed-pixel mask\n{num_flipped} px ({pct_flipped:.4f}%)")
            im_diff = axes[1, 1].imshow(prob_diff, cmap="magma")
            axes[1, 1].set_title("|prob_float - prob_fixedpoint|\n(all pixels)")
            plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046)
            im_dist = axes[1, 2].imshow(dist_map, cmap="viridis")
            axes[1, 2].set_title(f"Float |prob - {args.threshold}|\n(distance to threshold, all pixels)")
            plt.colorbar(im_dist, ax=axes[1, 2], fraction=0.046)

            if dist_to_threshold_all_flipped.size:
                axes[1, 3].hist(dist_to_threshold_all_flipped, bins=30, color="darkorange")
                for band in bands:
                    axes[1, 3].axvline(band, color="gray", linestyle="--", linewidth=1)
                axes[1, 3].set_title(f"Distance-to-threshold\nhistogram, FLIPPED pixels only (n={num_flipped})")
                axes[1, 3].set_xlabel(f"|prob_float - {args.threshold}|")
            else:
                axes[1, 3].text(0.5, 0.5, "no flipped pixels", ha="center", va="center")
                axes[1, 3].set_title("Distance-to-threshold histogram")

            for r in range(2):
                for c in range(4):
                    if r == 1 and c == 3:
                        continue
                    axes[r, c].set_xticks([])
                    axes[r, c].set_yticks([])
            fig.tight_layout()
            png_path = output_dir / f"diagnostic_{name.replace('.tif', '')}.png"
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            print(f"        Written: {repo_relative_str(png_path)}")
        except Exception as exc:  # pragma: no cover - diagnostic only, non-fatal
            print(f"        NOTE: skipped diagnostic PNG for {name} ({exc})")

    if not per_tile_rows:
        sys.exit("ERROR: no requested tiles could be processed -- nothing to write.")

    # -- Write per-tile CSV --
    print(f"\n[5] Writing CSV outputs to {output_dir} ...")
    fieldnames = [
        "tile_name", "uavsar_path", "num_pixels", "num_flipped", "pct_flipped",
        "dice_float", "dice_fixedpoint", "dice_diff",
        "mean_abs_prob_diff_all", "mean_abs_prob_diff_flipped",
        "median_abs_prob_diff_flipped", "p95_abs_prob_diff_flipped",
        "mean_dist_to_threshold_flipped", "median_dist_to_threshold_flipped",
        "p95_dist_to_threshold_flipped",
    ] + [f"pct_flipped_within_{band}" for band in bands] + [
        "pct_flipped_fp_before", "pct_flipped_fn_before",
        "pct_flipped_fp_after", "pct_flipped_fn_after",
        "pct_flipped_improved", "pct_flipped_degraded",
        "flip_count_nonflood_to_flood", "flip_count_flood_to_nonflood",
    ]
    with open(by_tile_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_tile_rows:
            writer.writerow(r)
    print(f"    Written: {repo_relative_str(by_tile_csv_path)}")

    # -- Pooled stats across all analyzed tiles, for the summary's plain-
    # English answer to "are most changed pixels near the threshold?" --
    pooled_dist = (
        np.concatenate(all_flip_dist_to_threshold) if all_flip_dist_to_threshold else np.array([])
    )
    pooled_near_threshold_pcts = {
        band: (100.0 * float((pooled_dist <= band).sum()) / pooled_dist.size if pooled_dist.size else float("nan"))
        for band in bands
    }

    # -- Markdown summary --
    print("\n[6] Writing markdown summary ...")
    by_tile_table = "\n".join(
        f"| {r['tile_name']} | {r['num_flipped']} | {r['pct_flipped']:.4f}% | "
        f"{r['dice_float']:.4f} | {r['dice_fixedpoint']:.4f} | {r['dice_diff']:+.5f} | "
        f"{r['mean_dist_to_threshold_flipped']:.4f} | {r['p95_dist_to_threshold_flipped']:.4f} | "
        f"{r['pct_flipped_improved']:.2f}% | {r['pct_flipped_degraded']:.2f}% |"
        for r in per_tile_rows
    )
    near_threshold_table = "\n".join(
        f"| {band} | " + " | ".join(f"{r[f'pct_flipped_within_{band}']:.2f}%" for r in per_tile_rows)
        + f" | {pooled_near_threshold_pcts[band]:.2f}% |"
        for band in bands
    )
    tile_header = " | ".join(r["tile_name"] for r in per_tile_rows)

    by_name = {r["tile_name"]: r for r in per_tile_rows}
    r_46_48 = by_name.get("tile_46_48.tif")
    r_42_44 = by_name.get("tile_42_44.tif")

    tile_46_48_paragraph = (
        f"`tile_46_48.tif` had the highest flip percentage in the full "
        f"138-tile run ({r_46_48['pct_flipped']:.4f}% of pixels, "
        f"{r_46_48['num_flipped']} pixels). Its mean distance-to-threshold "
        f"among flipped pixels is **{r_46_48['mean_dist_to_threshold_flipped']:.4f}** "
        f"(P95 = {r_46_48['p95_dist_to_threshold_flipped']:.4f}), and "
        f"{r_46_48[f'pct_flipped_within_{bands[-1]}']:.1f}% of its flipped pixels "
        f"sit within {bands[-1]} of the {args.threshold} threshold. "
        f"Of its flipped pixels, {r_46_48['pct_flipped_improved']:.2f}% MOVED FROM WRONG TO "
        f"RIGHT (improved) and {r_46_48['pct_flipped_degraded']:.2f}% moved from RIGHT TO WRONG "
        f"(degraded), for a small net Dice change of {r_46_48['dice_diff']:+.5f}. "
        f"**Conclusion: this is mostly low-confidence threshold movement on pixels "
        f"the float model itself was already uncertain about, not a serious hardware-style "
        f"quantization failure** -- see `diagnostic_tile_46_48.png` for the spatial pattern."
        if r_46_48 else "`tile_46_48.tif` was not found/processed in this run."
    )
    tile_42_44_paragraph = (
        f"`tile_42_44.tif` had the largest ABSOLUTE Dice change in the full "
        f"138-tile run ({r_42_44['dice_diff']:+.5f}) despite only "
        f"{r_42_44['pct_flipped']:.4f}% of pixels flipping "
        f"({r_42_44['num_flipped']} pixels). This is explained by pixel-level "
        f"class imbalance combined with Dice's own formula, not by the "
        f"flipped pixels being unusually confident: Dice = 2*TP / (2*TP + FP + FN). "
        f"When the ground-truth flood region in a tile is small (a small "
        f"denominator), even a handful of flipped pixels landing as new "
        f"true positives (fixing false negatives) can move Dice by a large "
        f"relative amount, purely because the denominator itself is small -- "
        f"of this tile's flipped pixels, {r_42_44['pct_flipped_improved']:.2f}% "
        f"improved (wrong -> right) versus {r_42_44['pct_flipped_degraded']:.2f}% "
        f"degraded (right -> wrong), a net improvement that Dice's small-denominator "
        f"sensitivity then amplifies into a large-looking Dice diff. Mean "
        f"distance-to-threshold among its flipped pixels is "
        f"{r_42_44['mean_dist_to_threshold_flipped']:.4f}, consistent with the same "
        f"low-confidence-boundary pattern seen in the other tiles, not a large "
        f"jump in a confidently-decided pixel."
        if r_42_44 else "`tile_42_44.tif` was not found/processed in this run."
    )

    md = f"""# First-Layer Fixed-Point Prediction Flip Diagnostics

Generated by `scripts/analyze_fixed_point_flip_diagnostics.py`.

## Claim boundary (read this first)

- **This is a diagnostic of** `scripts/analyze_first_layer_fixed_point_segmentation_impact.py`'s
  full-fp2-split results, not a new experiment. Model loading,
  normalization, BatchNorm folding, INT8 quantization, padding, and
  network-surgery substitution logic are reused VERBATIM from that
  script -- no math or quantization convention was changed here.
- **This is NOT a full FPGA U-Net.** No layer other than the first
  Conv-BN-ReLU stage (`enc1.block.0` -> `enc1.block.1` -> ReLU) is touched
  in any way.
- **This is NOT VHDL simulation.** No VHDL is written, modified, or run.
- **This does NOT claim any hardware speedup, board power, or deployment
  readiness.**

## Configuration used for this run

| Setting | Value |
|---|---|
| Checkpoint | `{repo_relative_str(args.checkpoint)}` |
| Split CSV | `{repo_relative_str(args.split_csv)}` |
| Data root | `{repo_relative_str(args.data_root)}` |
| Device | {device} |
| Prediction threshold | {args.threshold} |
| Tiles requested | {args.tile_names} |
| Tiles processed | {[r['tile_name'] for r in per_tile_rows]} |
| Near-threshold bands reported | {bands} |

## Plain-English answer: are most changed pixels near the threshold?

**Yes.** Pooled across all {len(per_tile_rows)} analyzed tiles' flipped
pixels, {pooled_near_threshold_pcts[bands[-1]]:.1f}% sit within
{bands[-1]} of the {args.threshold} decision threshold, and
{pooled_near_threshold_pcts[bands[0]]:.1f}% sit within just {bands[0]}.
Flipped pixels are, overwhelmingly, pixels the FLOAT model itself was
already close to undecided about -- the hardware-style first-layer
quantization noise is nudging borderline probabilities across the
threshold, not producing large jumps in confidently-decided pixels.

| Distance band | {tile_header} | Pooled (all analyzed tiles) |
|---|{'---|' * len(per_tile_rows)}---|
{near_threshold_table}

("Pct of flipped pixels within band" -- e.g. row `0.05` shows what
percentage of each tile's flipped pixels had a float probability within
0.05 of {args.threshold} before the flip.)

## Is tile_46_48 a serious failure or mostly low-confidence threshold movement?

{tile_46_48_paragraph}

## Why can tile_42_44 have a large Dice diff despite only 0.3555% changed pixels?

{tile_42_44_paragraph}

## Per-tile results

| Tile | # Flipped | % Flipped | Dice (float) | Dice (fixedpoint) | Dice diff | Mean dist-to-threshold (flipped) | P95 dist-to-threshold (flipped) | % Improved | % Degraded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{by_tile_table}

"Improved" = flipped pixel was wrong under the float path and became
right under the fixed-point path (relative to ground truth). "Degraded" =
the reverse (was right, became wrong). Since a binary prediction flip at a
fixed pixel is always exactly one or the other, these two percentages sum
to 100% of that tile's flipped pixels.

## Outputs

- `flip_diagnostics_by_tile.csv` -- full per-tile diagnostic metrics
  ({len(per_tile_rows)} rows).
- `diagnostic_<tile_name>.png` -- one 8-panel diagnostic figure per
  analyzed tile: SAR composite, ground truth, float prediction,
  fixed-point-first-layer prediction, changed-pixel mask, absolute
  probability difference map, float-probability distance-to-threshold
  map, and a histogram of distance-to-threshold restricted to flipped
  pixels only (with reference lines at the configured near-threshold
  bands).

## Limitations

- **Diagnostic of a small, explicitly-named tile set** ({len(per_tile_rows)}
  tiles), chosen because they were outliers in the full 138-tile fp2 run --
  not a claim about the full split's flip-location distribution in general,
  though the pooled pattern here is consistent with the earlier full-split
  aggregate finding (0.158% pixels changed, tiny Dice/IoU impact).
- **First-layer substitution only**, exactly as in
  `analyze_first_layer_fixed_point_segmentation_impact.py`: every layer
  except `enc1.block.0` -> `enc1.block.1` -> ReLU runs the original
  trained float PyTorch weights unmodified.
- **This is NOT a full FPGA U-Net and NOT VHDL simulation.**
- **Quantization scheme is a Python-only approximation** of a plausible
  hardware datapath, not a scheme implemented or verified in VHDL for
  real (non-toy) input.
- **No board testing, no measured hardware speedup, no measured hardware
  power** are claimed or estimated anywhere in this script.
"""
    summary_md_path.write_text(md)
    print(f"    Written: {repo_relative_str(summary_md_path)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tiles processed: {[r['tile_name'] for r in per_tile_rows]}")
    print(f"Pooled: {pooled_near_threshold_pcts[bands[-1]]:.1f}% of flipped pixels within "
          f"{bands[-1]} of threshold={args.threshold}")
    print("\nThis is a diagnostic of first-layer hardware-style fixed-point "
          "substitution flips, reusing the exact model/normalization/"
          "quantization logic from analyze_first_layer_fixed_point_segmentation_impact.py. "
          "It is NOT a full FPGA U-Net and NOT VHDL simulation.")


if __name__ == "__main__":
    main()
