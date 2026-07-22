#!/usr/bin/env python3
"""
analyze_first_layer_fixed_point_segmentation_impact.py

Segmentation-level impact study: does replacing ONLY the first
Conv-BN-ReLU stage (`enc1.block.0` Conv2d -> `enc1.block.1` BatchNorm2d ->
ReLU) with a hardware-style fixed-point folded approximation change the
COMPLETE U-Net's final segmentation output, on real UAVSAR fp2 test tiles?

This is a DIRECT FOLLOW-ON to
scripts/analyze_first_layer_real_activation_fidelity.py, which already
showed that the hardware-style fixed-point folded first-layer activation
is numerically very close to the true PyTorch float first-layer
activation (correlation ~0.9999 on real tiles). That earlier study never
propagated the approximation through the rest of the network. THIS script
does exactly that: swap in the approximated first-layer activation, run
the ORIGINAL float PyTorch model for every remaining layer, and measure
whether the final Dice/IoU/precision/recall/pixel-accuracy and predicted
flood mask actually change.

---- What this is NOT ----------------------------------------------------
- This is NOT a full FPGA U-Net. No layer other than the first
  Conv-BN-ReLU is touched in any way.
- This is NOT VHDL simulation. No VHDL is written, modified, or run.
- This does NOT claim any hardware speedup, board power, or deployment
  readiness.
- This does NOT retrain or fine-tune the checkpoint.

---- What this IS ----------------------------------------------------------
An isolation experiment: keep the ENTIRE trained U-Net in PyTorch float,
except substitute the first Conv-BN-ReLU stage's output with a
hardware-style fixed-point folded approximation of that SAME stage, then
measure the end-to-end effect on segmentation metrics and predictions.
The goal is to test whether first-layer hardware-style quantization noise
is large enough to change final segmentation behavior, not to claim any
hardware implementation.

---- Sources inspected before writing this script -----------------------
- scripts/benchmark_full_unet_inference.py /
  scripts/benchmark_full_unet_end_to_end.py: `DoubleConv`/`UNet` class
  definitions (copied verbatim below) and checkpoint-loading convention
  (`torch.load(..., map_location=device, weights_only=False)` then
  `model.load_state_dict(checkpoint["model_state_dict"])`).
- scripts/train_unet_baseline_tuned.py: `FloodTileDataset._normalize_per_tile`
  (reused verbatim) and `dice_iou_from_logits`'s eps=1e-7 convention
  (reused for all confusion-matrix-based metrics here).
- scripts/analyze_first_layer_real_activation_fidelity.py: the BatchNorm-
  folding formula and the symmetric INT8 weight/activation quantization
  convention (reused verbatim below), and its documented padding /
  valid-interior handling, which this script deliberately extends (see
  "Shape and padding handling" below).

---- Shape and padding handling (IMPORTANT -- read before trusting the
     numbers) -----------------------------------------------------------
The real PyTorch `enc1.block.0` Conv2d uses `padding=1`, so its output is
256x256, the same size as the 256x256 input tile. Every prior hardware
fidelity script in this repo computed only a VALID (no-padding) 3x3
convolution, producing a 254x254 interior, and compared that cropped
region against a center-cropped slice of the true padded output -- correct
for a pure activation-fidelity study, but NOT sufficient here: the REST OF
THE U-NET needs a full 256x256 tensor to concatenate against its skip
connections at every decoder stage, so a 254x254 activation cannot simply
be fed into the rest of the network without reshaping every downstream
skip connection -- which would silently change the network being tested.

This script therefore implements PADDING in the hardware-style
approximation itself, per the task's stated preference, rather than
falling back to a crop-based analysis:
  1. The per-tile-normalized SAR input is quantized to INT8 exactly as in
     `analyze_first_layer_real_activation_fidelity.py` (scale computed
     from the UNPADDED tile's own max magnitude).
  2. The resulting INT8 array is THEN zero-padded by 1 pixel on each
     spatial side (INT8 zero maps to exactly float 0.0 after dequantization,
     identical to PyTorch's zero-padding value, so this is equivalent to
     padding before quantizing).
  3. A valid (no-padding) 3x3 convolution over this 258x258 padded INT8
     array, using the folded+quantized weights, produces a 256x256 output
     -- the SAME spatial size as the true padded PyTorch output, so it can
     be substituted directly in place of `enc1.block[0:3]`'s output with
     no further reshaping anywhere else in the network.
This is mathematically equivalent (up to the quantization approximation
itself) to running a padded convolution directly, and it lets the REST OF
THE U-NET run completely unmodified on a full-shape input.

A CORRECTNESS CHECK is performed on every tile before trusting this
substitution: the network is decomposed as
`enc1.block[3:](enc1.block[0:3](x))` -> rest-of-UNet, and this decomposed
path is confirmed to reproduce the model's own direct `model(x)` output
BIT-IDENTICALLY (both are the literal same PyTorch modules on the literal
same tensor, just called in two pieces) before any hardware-style
approximation is introduced. This proves the network-surgery mechanism
itself is correct, isolating all downstream differences to the
hardware-style first-layer approximation, not to a bug in how the model
was split.

---- BatchNorm folding + quantization (identical to
     analyze_first_layer_real_activation_fidelity.py) ---------------------
    scale_bn[oc] = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc] = w[oc] * scale_bn[oc]
    b_folded[oc] = beta[oc] - running_mean[oc] * scale_bn[oc]
    scale_w_folded[oc] = max(|w_folded[oc]|) / 127 ; INT8 round+clip
    scale_x = max(|x_tile|) / 127 ; INT8 round+clip (computed from the
    UNPADDED tile, then the INT8 result is zero-padded -- see above)
    dequant: int32_accum * scale_x * scale_w_folded + b_folded, then ReLU

---- Normalization ----------------------------------------------------------
Reused verbatim from `FloodTileDataset._normalize_per_tile` in
`scripts/train_unet_baseline_tuned.py`.

---- Data -------------------------------------------------------------------
Real UAVSAR fp2 held-out test tiles:
csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/
strict_no_overlap/heldout_fp2_test.csv
Ground-truth flood mask read from each row's `flood_mask_path`
(thresholded `> 0`, identical to `FloodTileDataset.__getitem__`).

---- Outputs -----------------------------------------------------------
outputs/hardware_fidelity/first_layer_fixed_point_segmentation_impact/
    segmentation_impact_overall.csv
    segmentation_impact_by_tile.csv
    segmentation_impact_summary.md
    diagnostic_tile0_comparison.png   (small, optional diagnostic figure)
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
import torch.nn.functional as F

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
    REPO_ROOT / "outputs" / "hardware_fidelity" / "first_layer_fixed_point_segmentation_impact"
)

BN_EPS = 1e-5           # torch.nn.BatchNorm2d default
METRIC_EPS = 1e-7       # matches dice_iou_from_logits in train_unet_baseline_tuned.py


# ---------------------------------------------------------------------------
# Model -- DoubleConv/UNet copied VERBATIM from scripts/benchmark_full_unet_inference.py
# (itself copied verbatim from train_unet_baseline_tuned.py).
# ---------------------------------------------------------------------------
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
    """Runs the REST of the U-Net starting from `enc1.block[0:3]`'s output
    (i.e. the first Conv-BN-ReLU's output, shape (1, base_channels, H, W)).
    This is an EXACT decomposition of UNet.forward -- if
    `enc1_first_stage_out` equals `model.enc1.block[0:3](x)` exactly, this
    function's output equals `model(x)` exactly (see the correctness check
    in main())."""
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


# ---------------------------------------------------------------------------
# Normalization -- reused verbatim from FloodTileDataset._normalize_per_tile.
# ---------------------------------------------------------------------------
def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Quantization -- identical convention to analyze_first_layer_real_activation_fidelity.py
# ---------------------------------------------------------------------------
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
    Valid (no-padding) 3x3 conv, vectorized with sliding_window_view +
    einsum. Returns (n_out, H-2, W-2)."""
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
    """The hardware-style fixed-point folded Conv-BN-ReLU approximation,
    WITH PADDING so the output spatial size matches the true padded
    PyTorch output (see module docstring, "Shape and padding handling").
    x_float_chw: (3, H, W) float, per-tile-normalized (UNPADDED).
    Returns (relu_out (n_out, H, W) float64, scale_x)."""
    x_int8, scale_x = quantize_symmetric_int8_activation(x_float_chw)
    x_int8_padded = np.pad(x_int8, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    int32_accum = conv_all_kernels_valid(x_int8_padded, w_folded_int8_all)  # (n_out, H, W)
    conv_float_approx = (
        int32_accum.astype(np.float64) * scale_x * scales_w_folded[:, None, None]
    )
    biased = conv_float_approx + b_folded_all[:, None, None]
    return np.maximum(biased, 0.0), scale_x


# ---------------------------------------------------------------------------
# Segmentation metrics -- eps convention matches dice_iou_from_logits in
# train_unet_baseline_tuned.py.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmentation-level impact of replacing only the first "
        "Conv-BN-ReLU stage with a hardware-style fixed-point folded "
        "approximation, rest of U-Net unchanged (PyTorch float)."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-csv", type=pathlib.Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tiles", type=int, default=32,
                         help="Maximum number of fp2 test tiles to evaluate (default: 32).")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability threshold for binary flood prediction (default: 0.5).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading -- same convention as benchmark_full_unet_inference.py.
# ---------------------------------------------------------------------------
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
    under REPO_ROOT -- e.g. a relative --output-dir that resolves outside
    the repo, or any path on a different mount entirely."""
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    # Normalize output_dir to an absolute path up front: a relative
    # --output-dir (e.g. "outputs/foo") must not be compared against the
    # absolute REPO_ROOT later via .relative_to(), which raises ValueError
    # if one path is relative and the other absolute.
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_csv_path = output_dir / "segmentation_impact_overall.csv"
    by_tile_csv_path = output_dir / "segmentation_impact_by_tile.csv"
    summary_md_path = output_dir / "segmentation_impact_summary.md"
    diagnostic_png_path = output_dir / "diagnostic_tile0_comparison.png"

    device = torch.device(args.device)
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    print(f"    Model moved to device: {device}\n")

    # -- Extract first-layer Conv2d weight + BatchNorm parameters directly
    # from the loaded model's own state (guarantees byte-for-byte identical
    # weights to whatever load_model actually loaded). --
    state_dict = model.state_dict()
    w_float_all = state_dict["enc1.block.0.weight"].detach().cpu().float().numpy()  # (32, 3, 3, 3)
    n_out = w_float_all.shape[0]
    bn_gamma = state_dict["enc1.block.1.weight"].detach().cpu().float().numpy()
    bn_beta = state_dict["enc1.block.1.bias"].detach().cpu().float().numpy()
    bn_mean = state_dict["enc1.block.1.running_mean"].detach().cpu().float().numpy()
    bn_var = state_dict["enc1.block.1.running_var"].detach().cpu().float().numpy()

    print("[2] Folding BatchNorm into first-layer Conv2d weights/bias ...")
    scale_bn = bn_gamma / np.sqrt(bn_var + BN_EPS)
    w_folded_all = w_float_all * scale_bn[:, None, None, None]
    b_folded_all = bn_beta - bn_mean * scale_bn
    print(f"    scale_bn range : [{scale_bn.min():.6f}, {scale_bn.max():.6f}]")
    print(f"    b_folded range : [{b_folded_all.min():.6f}, {b_folded_all.max():.6f}]")

    print(f"\n[3] Quantizing all {n_out} FOLDED kernels to INT8 (symmetric per-output-channel scale) ...")
    scales_w_folded = np.zeros(n_out, dtype=np.float64)
    w_folded_int8_all = np.zeros_like(w_folded_all, dtype=np.int8)
    for oc in range(n_out):
        w_folded_int8_all[oc], scales_w_folded[oc] = quantize_symmetric_int8(w_folded_all[oc])
    print(f"    scale_w_folded range: [{scales_w_folded.min():.8f}, {scales_w_folded.max():.8f}]")

    # -- Load real test tiles + ground-truth masks --
    print(f"\n[4] Loading up to {args.max_tiles} real test tiles ...")
    print(f"    Split CSV : {args.split_csv}")
    if not args.split_csv.exists():
        sys.exit(f"ERROR: split CSV not found at {args.split_csv}")
    if not args.data_root.exists():
        sys.exit(f"ERROR: data root not found at {args.data_root}")

    with open(args.split_csv, newline="") as f:
        split_rows = list(csv.DictReader(f))

    tiles = []
    tried = 0
    for row in split_rows:
        if len(tiles) >= args.max_tiles:
            break
        tried += 1
        sar_path = args.data_root / row["uavsar_path"]
        mask_path = args.data_root / row["flood_mask_path"]
        if not sar_path.exists() or not mask_path.exists():
            continue
        with rasterio.open(sar_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            continue
        sar = sar[:3]
        sar_norm = normalize_per_tile(sar)
        with rasterio.open(mask_path) as src:
            mask = src.read(1, out_dtype="float32")
        mask_bool = mask > 0
        tiles.append({
            "tile_name": row["tile_name"],
            "uavsar_path": row["uavsar_path"],
            "sar_norm": sar_norm,
            "mask_bool": mask_bool,
        })
    print(f"    Tried {tried} split rows, loaded {len(tiles)} usable tiles "
          f"(requested up to {args.max_tiles}).")
    if not tiles:
        sys.exit("ERROR: no usable tiles were loaded -- cannot run this study.")

    # -- Per-tile: path A (float reference), correctness check, path B
    # (hardware-style fixed-point first layer), metrics --
    print(f"\n[5] Running float reference + fixed-point-first-layer paths, "
          f"threshold={args.threshold} ...")

    per_tile_rows = []
    surgery_max_diffs = []
    all_prob_diffs = []  # pooled abs(prob_A - prob_B), for global MAE/max/P99
    global_tp_a = global_fp_a = global_fn_a = global_tn_a = 0
    global_tp_b = global_fp_b = global_fn_b = global_tn_b = 0
    global_changed_pixels = 0
    global_total_pixels = 0

    for idx, t in enumerate(tiles):
        sar_norm = t["sar_norm"]
        mask_bool = t["mask_bool"]
        x = torch.from_numpy(sar_norm).float().unsqueeze(0).to(device)  # (1, 3, 256, 256)

        with torch.no_grad():
            # -- Path A: direct float forward pass (production path). --
            logits_a = model(x)

            # -- Correctness check: decompose the SAME float forward pass
            # into enc1.block[0:3](x) -> unet_forward_from_enc1_first_stage,
            # and confirm it reproduces logits_a exactly. This validates the
            # network-surgery mechanism BEFORE any approximation is used. --
            enc1_true_first_stage = model.enc1.block[0:3](x)
            logits_a_decomposed = unet_forward_from_enc1_first_stage(model, enc1_true_first_stage)
            surgery_diff = float(torch.abs(logits_a_decomposed - logits_a).max().item())
            surgery_max_diffs.append(surgery_diff)

            # -- Path B: hardware-style fixed-point folded first-layer
            # approximation (padded, full 256x256), substituted in place of
            # enc1.block[0:3](x), rest of the network unchanged float. --
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
        n_pixels = preds_a.size
        pct_changed = 100.0 * float(changed.sum()) / n_pixels

        global_tp_a += tp_a; global_fp_a += fp_a; global_fn_a += fn_a; global_tn_a += tn_a
        global_tp_b += tp_b; global_fp_b += fp_b; global_fn_b += fn_b; global_tn_b += tn_b
        global_changed_pixels += int(changed.sum())
        global_total_pixels += n_pixels
        all_prob_diffs.append(prob_diff.ravel())

        per_tile_rows.append({
            "tile_name": t["tile_name"],
            "uavsar_path": t["uavsar_path"],
            "num_pixels": n_pixels,
            "surgery_correctness_max_diff": surgery_diff,
            "scale_x": scale_x,
            "dice_float": m_a["dice"], "iou_float": m_a["iou"],
            "precision_float": m_a["precision"], "recall_float": m_a["recall"],
            "pixel_accuracy_float": m_a["pixel_accuracy"],
            "dice_fixedpoint": m_b["dice"], "iou_fixedpoint": m_b["iou"],
            "precision_fixedpoint": m_b["precision"], "recall_fixedpoint": m_b["recall"],
            "pixel_accuracy_fixedpoint": m_b["pixel_accuracy"],
            "dice_diff": m_b["dice"] - m_a["dice"],
            "iou_diff": m_b["iou"] - m_a["iou"],
            "precision_diff": m_b["precision"] - m_a["precision"],
            "recall_diff": m_b["recall"] - m_a["recall"],
            "pixel_accuracy_diff": m_b["pixel_accuracy"] - m_a["pixel_accuracy"],
            "mean_abs_prob_diff": float(prob_diff.mean()),
            "max_abs_prob_diff": float(prob_diff.max()),
            "p99_abs_prob_diff": float(np.percentile(prob_diff, 99)),
            "pct_pixels_pred_changed": pct_changed,
        })
        print(f"    [{idx + 1}/{len(tiles)}] {t['tile_name']:20s} "
              f"dice_float={m_a['dice']:.4f} dice_fixedpoint={m_b['dice']:.4f} "
              f"pct_changed={pct_changed:.4f}%  surgery_max_diff={surgery_diff:.3e}")

        if idx == 0:
            first_tile_diag = {
                "tile_name": t["tile_name"], "sar_norm": sar_norm, "mask_bool": mask_bool,
                "prob_a": prob_a, "prob_b": prob_b, "preds_a": preds_a, "preds_b": preds_b,
            }

    num_tiles = len(tiles)
    surgery_max_diff_overall = max(surgery_max_diffs)
    surgery_check_passed = surgery_max_diff_overall < 1e-4
    print(f"\n    Network-surgery correctness check across all {num_tiles} tiles: "
          f"max diff = {surgery_max_diff_overall:.3e} "
          f"({'PASSED' if surgery_check_passed else 'FAILED -- investigate'})")

    global_m_a = metrics_from_counts(global_tp_a, global_fp_a, global_fn_a, global_tn_a)
    global_m_b = metrics_from_counts(global_tp_b, global_fp_b, global_fn_b, global_tn_b)
    all_prob_diffs_concat = np.concatenate(all_prob_diffs)
    global_mean_abs_prob_diff = float(all_prob_diffs_concat.mean())
    global_max_abs_prob_diff = float(all_prob_diffs_concat.max())
    global_p99_abs_prob_diff = float(np.percentile(all_prob_diffs_concat, 99))
    global_pct_changed = 100.0 * global_changed_pixels / global_total_pixels

    per_tile_dice_diffs = np.array([r["dice_diff"] for r in per_tile_rows])
    per_tile_iou_diffs = np.array([r["iou_diff"] for r in per_tile_rows])

    print("\n" + "=" * 70)
    print("GLOBAL (all tiles pooled) SUMMARY")
    print("=" * 70)
    print(f"  Float reference     : dice={global_m_a['dice']:.6f} iou={global_m_a['iou']:.6f} "
          f"precision={global_m_a['precision']:.6f} recall={global_m_a['recall']:.6f} "
          f"pixel_acc={global_m_a['pixel_accuracy']:.6f}")
    print(f"  Fixed-point 1st-layer: dice={global_m_b['dice']:.6f} iou={global_m_b['iou']:.6f} "
          f"precision={global_m_b['precision']:.6f} recall={global_m_b['recall']:.6f} "
          f"pixel_acc={global_m_b['pixel_accuracy']:.6f}")
    print(f"  Mean abs prob diff  : {global_mean_abs_prob_diff:.6f}")
    print(f"  Max abs prob diff   : {global_max_abs_prob_diff:.6f}")
    print(f"  P99 abs prob diff   : {global_p99_abs_prob_diff:.6f}")
    print(f"  Pct pixels pred changed @ threshold={args.threshold}: {global_pct_changed:.4f}%")

    # -- Write per-tile CSV --
    print(f"\n[6] Writing CSV outputs to {output_dir} ...")
    by_tile_fieldnames = [
        "tile_name", "uavsar_path", "num_pixels", "surgery_correctness_max_diff", "scale_x",
        "dice_float", "iou_float", "precision_float", "recall_float", "pixel_accuracy_float",
        "dice_fixedpoint", "iou_fixedpoint", "precision_fixedpoint", "recall_fixedpoint",
        "pixel_accuracy_fixedpoint",
        "dice_diff", "iou_diff", "precision_diff", "recall_diff", "pixel_accuracy_diff",
        "mean_abs_prob_diff", "max_abs_prob_diff", "p99_abs_prob_diff", "pct_pixels_pred_changed",
    ]
    with open(by_tile_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=by_tile_fieldnames)
        writer.writeheader()
        for row in per_tile_rows:
            writer.writerow(row)
    print(f"    Written: {repo_relative_str(by_tile_csv_path)}")

    overall_fieldnames = [
        "num_tiles", "threshold", "surgery_correctness_max_diff_overall", "surgery_check_passed",
        "dice_float_global", "iou_float_global", "precision_float_global",
        "recall_float_global", "pixel_accuracy_float_global",
        "dice_fixedpoint_global", "iou_fixedpoint_global", "precision_fixedpoint_global",
        "recall_fixedpoint_global", "pixel_accuracy_fixedpoint_global",
        "dice_diff_global", "iou_diff_global", "precision_diff_global",
        "recall_diff_global", "pixel_accuracy_diff_global",
        "mean_abs_prob_diff_global", "max_abs_prob_diff_global", "p99_abs_prob_diff_global",
        "pct_pixels_pred_changed_global",
        "mean_per_tile_dice_diff", "std_per_tile_dice_diff",
        "mean_per_tile_iou_diff", "std_per_tile_iou_diff",
    ]
    with open(overall_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        writer.writerow({
            "num_tiles": num_tiles,
            "threshold": args.threshold,
            "surgery_correctness_max_diff_overall": surgery_max_diff_overall,
            "surgery_check_passed": surgery_check_passed,
            "dice_float_global": global_m_a["dice"], "iou_float_global": global_m_a["iou"],
            "precision_float_global": global_m_a["precision"],
            "recall_float_global": global_m_a["recall"],
            "pixel_accuracy_float_global": global_m_a["pixel_accuracy"],
            "dice_fixedpoint_global": global_m_b["dice"], "iou_fixedpoint_global": global_m_b["iou"],
            "precision_fixedpoint_global": global_m_b["precision"],
            "recall_fixedpoint_global": global_m_b["recall"],
            "pixel_accuracy_fixedpoint_global": global_m_b["pixel_accuracy"],
            "dice_diff_global": global_m_b["dice"] - global_m_a["dice"],
            "iou_diff_global": global_m_b["iou"] - global_m_a["iou"],
            "precision_diff_global": global_m_b["precision"] - global_m_a["precision"],
            "recall_diff_global": global_m_b["recall"] - global_m_a["recall"],
            "pixel_accuracy_diff_global": global_m_b["pixel_accuracy"] - global_m_a["pixel_accuracy"],
            "mean_abs_prob_diff_global": global_mean_abs_prob_diff,
            "max_abs_prob_diff_global": global_max_abs_prob_diff,
            "p99_abs_prob_diff_global": global_p99_abs_prob_diff,
            "pct_pixels_pred_changed_global": global_pct_changed,
            "mean_per_tile_dice_diff": float(per_tile_dice_diffs.mean()),
            "std_per_tile_dice_diff": float(per_tile_dice_diffs.std()),
            "mean_per_tile_iou_diff": float(per_tile_iou_diffs.mean()),
            "std_per_tile_iou_diff": float(per_tile_iou_diffs.std()),
        })
    print(f"    Written: {repo_relative_str(overall_csv_path)}")

    # -- Optional small diagnostic PNG (first tile only) --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = first_tile_diag
        sar_display = d["sar_norm"][0]  # first channel, normalized
        diff_map = np.abs(d["prob_a"] - d["prob_b"])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
        axes[0].imshow(sar_display, cmap="gray")
        axes[0].set_title(f"SAR input (ch0)\n{d['tile_name']}")
        axes[1].imshow(d["mask_bool"], cmap="Blues")
        axes[1].set_title("Ground truth flood mask")
        axes[2].imshow(d["preds_a"], cmap="Blues")
        axes[2].set_title("Float prediction")
        axes[3].imshow(d["preds_b"], cmap="Blues")
        axes[3].set_title("Fixed-point 1st-layer\nprediction")
        im4 = axes[4].imshow(diff_map, cmap="magma")
        axes[4].set_title("|prob_float - prob_fixedpoint|")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(diagnostic_png_path, dpi=110)
        plt.close(fig)
        print(f"    Written: {repo_relative_str(diagnostic_png_path)}")
    except Exception as exc:  # pragma: no cover - diagnostic only, non-fatal
        print(f"    NOTE: skipped diagnostic PNG ({exc})")

    # -- Markdown summary --
    print("\n[7] Writing markdown summary ...")
    by_tile_table = "\n".join(
        f"| {r['tile_name']} | {r['dice_float']:.4f} | {r['dice_fixedpoint']:.4f} | "
        f"{r['dice_diff']:+.4f} | {r['iou_diff']:+.4f} | {r['pct_pixels_pred_changed']:.4f}% | "
        f"{r['mean_abs_prob_diff']:.6f} | {r['max_abs_prob_diff']:.6f} |"
        for r in per_tile_rows
    )

    md = f"""# First-Layer Hardware-Style Fixed-Point Segmentation Impact Study

Generated by `scripts/analyze_first_layer_fixed_point_segmentation_impact.py`.

## Claim boundary (read this first)

- **This is NOT a full FPGA U-Net.** No layer other than the first
  Conv-BN-ReLU stage (`enc1.block.0` -> `enc1.block.1` -> ReLU) is touched
  in any way; every other layer runs the original trained PyTorch float
  weights unmodified.
- **This is NOT VHDL simulation.** No VHDL is written, modified, run, or
  claimed to implement this experiment's fixed-point scheme.
- **This isolates the effect of replacing only the first Conv-BN-ReLU
  stage** with a hardware-style fixed-point folded approximation
  (BatchNorm folded into Conv2d weights/bias, then symmetric INT8
  weight+activation quantization, identical convention to
  `scripts/analyze_first_layer_real_activation_fidelity.py`), while every
  downstream layer (the rest of `enc1.block`, `enc2`-`enc4`, bottleneck,
  all decoder stages, final 1x1 conv) remains PyTorch float, unchanged.
- **The goal is to test whether first-layer hardware-style quantization
  noise is large enough to change final segmentation behavior** -- not to
  claim any hardware implementation, board speedup, or deployment
  readiness.
- No retraining, fine-tuning, or checkpoint modification occurred.

## Shape / padding handling

The real `enc1.block.0` Conv2d uses `padding=1`, so its output is the same
256x256 size as the input tile -- but every prior hardware-style
approximation script in this repo computed only a VALID (no-padding)
254x254 interior. That is fine for a pure activation-fidelity study, but
NOT sufficient here, because the rest of the U-Net needs a full 256x256
tensor to concatenate against its skip connections at every decoder stage.

Per the task's stated preference, **padding is implemented directly in the
hardware-style approximation**, rather than falling back to a crop-based
analysis:
1. The per-tile-normalized SAR input is quantized to INT8 (scale computed
   from the UNPADDED tile's own max magnitude, matching the existing
   convention).
2. The resulting INT8 array is zero-padded by 1 pixel on each spatial side
   (INT8 zero dequantizes to exactly float 0.0, identical to PyTorch's
   zero-padding value).
3. A valid 3x3 convolution over the resulting 258x258 padded INT8 array,
   using the folded+quantized weights, produces a 256x256 output -- the
   same spatial size as the true padded PyTorch output -- so it can be
   substituted directly for `enc1.block[0:3]`'s output with no reshaping
   anywhere else in the network.

**Network-surgery correctness check**: before trusting this substitution,
every tile also runs a decomposed-but-unmodified path -- computing
`enc1.block[0:3](x)` via the model's own true float modules (no
approximation at all) and feeding that into the same "rest of the
network" code path used for the fixed-point experiment. This decomposed
path is confirmed to reproduce the model's own direct `model(x)` output,
to a max absolute difference of **{surgery_max_diff_overall:.3e}** across
all {num_tiles} tiles ({"PASSED, well below the 1e-4 threshold used here" if surgery_check_passed else "FAILED -- investigate before trusting the fixed-point results below"}).
This proves the network-splitting mechanism itself is correct, so any
difference measured between the float and fixed-point paths below is
attributable to the hardware-style first-layer approximation, not to a
bug in how the model was split.

## Normalization

Reused verbatim from `FloodTileDataset._normalize_per_tile` in
`scripts/train_unet_baseline_tuned.py`.

## Configuration used for this run

| Setting | Value |
|---|---|
| Checkpoint | `{repo_relative_str(args.checkpoint)}` |
| Split CSV | `{repo_relative_str(args.split_csv)}` |
| Data root | `{repo_relative_str(args.data_root)}` |
| Device | {device} |
| Max tiles requested | {args.max_tiles} |
| Tiles actually loaded | {num_tiles} |
| Split rows tried | {tried} |
| Prediction threshold | {args.threshold} |

## Global metrics (all {num_tiles} tiles' pixels pooled into one confusion matrix)

| Metric | Float reference | Fixed-point 1st-layer | Diff (fixedpoint - float) |
|---|---:|---:|---:|
| Dice | {global_m_a['dice']:.6f} | {global_m_b['dice']:.6f} | {global_m_b['dice'] - global_m_a['dice']:+.6f} |
| IoU | {global_m_a['iou']:.6f} | {global_m_b['iou']:.6f} | {global_m_b['iou'] - global_m_a['iou']:+.6f} |
| Precision | {global_m_a['precision']:.6f} | {global_m_b['precision']:.6f} | {global_m_b['precision'] - global_m_a['precision']:+.6f} |
| Recall | {global_m_a['recall']:.6f} | {global_m_b['recall']:.6f} | {global_m_b['recall'] - global_m_a['recall']:+.6f} |
| Pixel accuracy | {global_m_a['pixel_accuracy']:.6f} | {global_m_b['pixel_accuracy']:.6f} | {global_m_b['pixel_accuracy'] - global_m_a['pixel_accuracy']:+.6f} |

## Probability-level difference (all {num_tiles} tiles' pixels pooled)

| Metric | Value |
|---|---:|
| Mean absolute probability difference | {global_mean_abs_prob_diff:.6f} |
| Max absolute probability difference | {global_max_abs_prob_diff:.6f} |
| P99 absolute probability difference | {global_p99_abs_prob_diff:.6f} |
| Pct pixels whose binary prediction changed @ threshold={args.threshold} | {global_pct_changed:.4f}% |

## Per-tile summary (mean +/- std of per-tile diffs)

| Metric | Mean | Std |
|---|---:|---:|
| Dice diff (fixedpoint - float) | {per_tile_dice_diffs.mean():+.6f} | {per_tile_dice_diffs.std():.6f} |
| IoU diff (fixedpoint - float) | {per_tile_iou_diffs.mean():+.6f} | {per_tile_iou_diffs.std():.6f} |

## Per-tile results

| Tile | Dice (float) | Dice (fixedpoint) | Dice diff | IoU diff | % pixels changed | Mean abs prob diff | Max abs prob diff |
|---|---:|---:|---:|---:|---:|---:|---:|
{by_tile_table}

## Outputs

- `segmentation_impact_overall.csv` -- global/aggregate metrics (one row).
- `segmentation_impact_by_tile.csv` -- per-tile metrics ({num_tiles} rows).
- `diagnostic_tile0_comparison.png` -- optional diagnostic figure (first
  tile: SAR input, ground truth, float prediction, fixed-point-first-layer
  prediction, absolute probability difference map). Diagnostic only, not
  a metric.

## Limitations

- **First-layer substitution only.** Every layer except `enc1.block.0` ->
  `enc1.block.1` -> ReLU runs the original trained float PyTorch weights
  unmodified.
- **This is NOT a full FPGA U-Net and NOT VHDL simulation.** No VHDL is
  written, modified, or claimed to implement this experiment.
- **Quantization scheme is a Python-only approximation** of a plausible
  hardware datapath (BatchNorm-folded, symmetric per-output-channel INT8
  weights, per-tile symmetric INT8 activations), not a scheme implemented
  or verified in VHDL for real (non-toy) input.
- **Padding is implemented only in this fixed-point approximation's first
  layer**, specifically to allow full-shape substitution into the rest of
  the network; no VHDL prototype in this repo implements padding.
- **Sample size is {num_tiles} tile(s)** from the fp2 held-out test split
  -- a larger `--max-tiles` gives a more representative estimate but this
  remains a held-out-split sample, not the full split.
- **No board testing, no measured hardware speedup, no measured hardware
  power** are claimed or estimated anywhere in this script.
- **Does not retrain, fine-tune, or otherwise modify the checkpoint.**
"""
    summary_md_path.write_text(md)
    print(f"    Written: {repo_relative_str(summary_md_path)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tiles evaluated              : {num_tiles}")
    print(f"Network-surgery check        : max diff = {surgery_max_diff_overall:.3e} "
          f"({'PASSED' if surgery_check_passed else 'FAILED'})")
    print(f"Dice   (float -> fixedpoint) : {global_m_a['dice']:.6f} -> {global_m_b['dice']:.6f}")
    print(f"IoU    (float -> fixedpoint) : {global_m_a['iou']:.6f} -> {global_m_b['iou']:.6f}")
    print(f"Pct pixels pred changed @ {args.threshold}: {global_pct_changed:.4f}%")
    print("\nThis isolates the effect of replacing ONLY the first Conv-BN-ReLU "
          "stage with a hardware-style fixed-point approximation, keeping the "
          "rest of the U-Net in PyTorch float. It is NOT a full FPGA U-Net and "
          "NOT VHDL simulation.")


if __name__ == "__main__":
    main()
