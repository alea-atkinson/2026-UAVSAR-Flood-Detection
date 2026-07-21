"""
analyze_first_layer_real_activation_fidelity.py

Real-tile numerical fidelity study for the first U-Net Conv-BN-ReLU stage
(enc1.block.0 -> enc1.block.1 -> ReLU), comparing PyTorch floating-point
activations against a hardware-style fixed-point folded computation.

This EXTENDS the earlier, smaller-scale studies in
scripts/analyze_first_layer_quantization_fidelity.py (raw Conv2d only,
2 real tiles, 16x16 center crops) and
scripts/analyze_first_conv_bn_relu_fidelity.py (folded Conv-BN-ReLU, 2 real
tiles, 16x16 center crops, plus the synthetic 5x5x3 toy patch). Those
scripts were useful for correctness/consistency checks against the
VHDL-verified toy path. This script runs the SAME BatchNorm-folding and
INT8 quantization convention over a configurable NUMBER of FULL real
UAVSAR tiles (default up to 16, no cropping), to get a fidelity estimate
that is representative of real held-out data rather than a couple of
small crops.

This is ANALYSIS ONLY -- it does not retrain, modify the model, or touch
any dataset splits. It does NOT write or modify any VHDL. Existing VHDL prototypes in hardware/vhdl_conv3x3/ include raw Conv2d
datapaths and toy-input folded Conv-BN-ReLU datapaths. This script does
not add a new real-tile VHDL input pipeline, does not simulate VHDL on
full real UAVSAR tiles, and does not claim that the real-tile
per-activation quantization path is implemented in VHDL.

---- Scope ------------------------------------------------------------------
- First Conv-BN-ReLU stage ONLY: enc1.block.0 (Conv2d) -> enc1.block.1
  (BatchNorm2d) -> ReLU. Does NOT analyze the second Conv2d in
  enc1.block, the rest of DoubleConv, or any other part of the U-Net.
- Does NOT claim full DoubleConv, full U-Net, full quantized model
  accuracy, Dice, recall, or end-to-end segmentation impact.
- Does NOT claim the full real-tile input/quantization/fidelity path is
  implemented in VHDL -- this is Python numerical fidelity only. Existing
  toy-input VHDL prototypes for folded Conv-BN-ReLU remain separate
  correctness checks.

---- Padding / valid-interior-region handling (read this before trusting
     the numbers) ----------------------------------------------------------
The real PyTorch Conv2d uses padding=1 (`enc1.block.0` is
`nn.Conv2d(3, 32, 3, padding=1, bias=False)`), so its output has the same
spatial size as its input. Every VHDL prototype in this repo computes
VALID (no padding) 3x3 windows only, and padding is not implemented in
VHDL. To keep this an apples-to-apples comparison, this script:
  1. Runs the REAL PyTorch Conv2d(padding=1) -> BatchNorm(eval) -> ReLU on
     the FULL tile, exactly as the real model would.
  2. Extracts ONLY the valid interior region from that padded output
     (crops away the outermost 1-pixel border on all four sides) -- this
     is exactly the region whose 3x3 receptive field never touches the
     zero-padding, so it is mathematically identical to a true valid
     (no-pad) convolution over the same tile.
  3. Compares that cropped region against the valid (no-pad) folded/
     quantized hardware-style computation, run over the SAME full tile.
This script does NOT evaluate padded-border behavior and does NOT imply
that VHDL implements padding.

---- BatchNorm folding ----------------------------------------------------
    scale_bn[oc] = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc] = w[oc] * scale_bn[oc]
    b_folded[oc] = beta[oc] - running_mean[oc] * scale_bn[oc]
eps = 1e-5 (torch.nn.BatchNorm2d default; train_unet_baseline_tuned.py's
DoubleConv instantiates nn.BatchNorm2d(out_channels) with no eps override).

---- Quantization scheme (same convention as the existing hardware
     fidelity scripts; NOT a new "official" hardware convention -- no
     VHDL in this repo quantizes real-image input yet) -----------------
Weights (folded): symmetric per-output-channel INT8, identical formula to
analyze_first_conv_bn_relu_fidelity.py:
    scale_w_folded[oc] = max(|w_folded[oc]|) / 127
    w_folded_int8[oc]  = round(w_folded[oc] / scale_w_folded[oc]), clipped
                         to [-127, 127]
Activations: per-tile symmetric INT8, identical formula:
    scale_x = max(|x_tile|) / 127
    x_int8  = round(x_tile / scale_x), clipped to [-127, 127]
Reconstruction:
    conv_float_approx[oc] = int32_accum[oc] * scale_x * scale_w_folded[oc]
    + b_folded[oc]   (added in float, NOT quantized)
    -> ReLU

---- Normalization ----------------------------------------------------------
Reused VERBATIM from FloodTileDataset._normalize_per_tile in
scripts/train_unet_baseline_tuned.py (percentile-clip 1-99 + zero-mean/
unit-variance per tile) -- confirmed by direct inspection of that file, not
invented here. No `_normalize_per_band` method exists anywhere in this
repo.

---- Data -------------------------------------------------------------------
Real UAVSAR tiles from
csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/
strict_no_overlap/heldout_<fp>_<test|validation>.csv (default fp2, the
held-out flight-path split that matches this checkpoint's
"filtered_strict_fp2" training config). Tiles are read with rasterio,
the first 3 SAR bands are used (matching the model's 3-channel input),
and each FULL tile (no cropping) is normalized and compared.

---- Outputs -----------------------------------------------------------
outputs/hardware_fidelity/first_layer_real_tile_activation_fidelity/
    activation_fidelity_summary.md
    activation_fidelity_overall.csv
    activation_fidelity_by_channel.csv
    diagnostic_tile0_kernel0.png   (small, optional diagnostic figure)
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CONV_KEY_EXPECTED = "enc1.block.0.weight"
BN_PREFIX_EXPECTED = "enc1.block.1"
BN_EPS = 1e-5  # torch.nn.BatchNorm2d default; verified no override in train_unet_baseline_tuned.py

SPLIT_BASE_DIR = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-tile fidelity check: PyTorch float Conv-BN-ReLU "
        "vs. hardware-style folded/quantized fixed-point Conv-BN-ReLU, "
        "first layer only."
    )
    parser.add_argument("--max-tiles", type=int, default=16,
                         help="Maximum number of real tiles to sample (default: 16).")
    parser.add_argument("--split", choices=["auto", "test", "val"], default="auto",
                         help="Which split to use. 'auto' (default) prefers the "
                         "test split and falls back to validation if the test "
                         "split CSV is missing or empty.")
    parser.add_argument("--heldout-fp", default="fp2",
                         help="Held-out flight path (default: fp2, matching this "
                         "checkpoint's filtered_strict_fp2 training config).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Device for the PyTorch float reference computation "
                         "(default: cuda if available, else cpu).")
    parser.add_argument("--output-dir", type=pathlib.Path,
                         default=REPO_ROOT / "outputs" / "hardware_fidelity"
                         / "first_layer_real_tile_activation_fidelity",
                         help="Directory to write output CSV/MD/diagnostic files.")
    parser.add_argument("--data-root", type=pathlib.Path,
                         default=REPO_ROOT / "2025_Tile_Data",
                         help="Root directory containing the UAVSAR tile folders.")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                         default=REPO_ROOT / "models"
                         / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt",
                         help="Path to the trained checkpoint to load "
                         "enc1.block.0/enc1.block.1 from.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Normalization -- reused verbatim from train_unet_baseline_tuned.py
# ---------------------------------------------------------------------------
def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
    """Identical to FloodTileDataset._normalize_per_tile in
    train_unet_baseline_tuned.py -- reproduces the real training/eval-time
    input distribution. Do not modify without re-verifying against that
    file."""
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
# Quantization -- identical convention to analyze_first_conv_bn_relu_fidelity.py
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


def conv_all_kernels(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (3, H, W) int8 or float; w: (n_out, 3, 3, 3) int8 or float.
    Valid (no-padding) 3x3 conv, vectorized with sliding_window_view +
    einsum. Returns (n_out, H-2, W-2) int64/float64 accumulation (no bias
    added here -- bias is added separately after rescaling)."""
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    acc_dtype = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    return np.einsum(
        "chwij,ocij->ohw",
        windows.astype(acc_dtype),
        w.astype(acc_dtype),
        optimize=True,
    )


# ---------------------------------------------------------------------------
# Metrics helpers -- identical convention to the existing hardware fidelity
# scripts (per-array-own-threshold relative error, to avoid one tile's scale
# silently excluding another tile's samples from the "safe" subset).
# ---------------------------------------------------------------------------
def safe_relative_error_values(float_flat: np.ndarray, rescaled_flat: np.ndarray,
                                eps_frac: float = 0.05) -> tuple[np.ndarray, int]:
    max_mag = np.abs(float_flat).max() if float_flat.size else 0.0
    threshold = eps_frac * max_mag if max_mag > 0 else 0.0
    mask = np.abs(float_flat) > max(threshold, 1e-8)
    if not np.any(mask):
        return np.array([], dtype=np.float64), 0
    rel_err = np.abs(rescaled_flat[mask] - float_flat[mask]) / np.abs(float_flat[mask])
    return rel_err, int(mask.sum())


def mean_of_values(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with float64 accumulation.

    The activation vectors can contain tens of millions of float32 values.
    Computing the dot product/norms in float32 can produce tiny numerical
    overshoots above 1.0, which is mathematically invalid for cosine
    similarity. Cast to float64 before accumulation and clip only to remove
    residual floating-point roundoff.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    cos = float(np.dot(a64, b64) / (na * nb))
    return float(np.clip(cos, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Split resolution
# ---------------------------------------------------------------------------
def resolve_split_csv(heldout_fp: str, split_choice: str) -> tuple[pathlib.Path, str]:
    """Returns (csv_path, split_name_used)."""
    def path_for(name: str) -> pathlib.Path:
        return SPLIT_BASE_DIR / f"heldout_{heldout_fp}_{name}.csv"

    def has_rows(p: pathlib.Path) -> bool:
        if not p.exists():
            return False
        with open(p, newline="") as f:
            return len(list(csv.DictReader(f))) > 0

    if split_choice == "test":
        p = path_for("test")
        if not has_rows(p):
            sys.exit(f"ERROR: requested --split test but {p} is missing or empty.")
        return p, "test"
    if split_choice == "val":
        p = path_for("validation")
        if not has_rows(p):
            sys.exit(f"ERROR: requested --split val but {p} is missing or empty.")
        return p, "validation"

    # auto: prefer test, fall back to validation
    test_p = path_for("test")
    if has_rows(test_p):
        return test_p, "test"
    val_p = path_for("validation")
    if has_rows(val_p):
        print(f"    NOTE: test split {test_p.name} unavailable/empty; "
              f"falling back to validation split.")
        return val_p, "validation"
    sys.exit(f"ERROR: neither {test_p} nor {val_p} exist / have rows.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "activation_fidelity_overall.csv"
    per_channel_csv_path = output_dir / "activation_fidelity_by_channel.csv"
    summary_md_path = output_dir / "activation_fidelity_summary.md"
    diagnostic_png_path = output_dir / "diagnostic_tile0_kernel0.png"

    # -- 1. Load checkpoint, extract Conv2d weight + BatchNorm parameters --
    print(f"[1] Loading checkpoint: {args.checkpoint}")
    if not args.checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found at {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    if CONV_KEY_EXPECTED not in state_dict:
        sys.exit(f"ERROR: expected tensor key '{CONV_KEY_EXPECTED}' not found in checkpoint.")
    w_float_all = state_dict[CONV_KEY_EXPECTED].float().numpy()  # (32, 3, 3, 3)
    n_out = w_float_all.shape[0]
    print(f"    Tensor key   : {CONV_KEY_EXPECTED}  shape {list(w_float_all.shape)}")

    bn_keys = {
        "weight": f"{BN_PREFIX_EXPECTED}.weight",
        "bias": f"{BN_PREFIX_EXPECTED}.bias",
        "running_mean": f"{BN_PREFIX_EXPECTED}.running_mean",
        "running_var": f"{BN_PREFIX_EXPECTED}.running_var",
    }
    if not all(k in state_dict for k in bn_keys.values()):
        sys.exit(f"ERROR: expected BatchNorm keys under '{BN_PREFIX_EXPECTED}' not found.")
    bn_gamma = state_dict[bn_keys["weight"]].float().numpy()
    bn_beta = state_dict[bn_keys["bias"]].float().numpy()
    bn_mean = state_dict[bn_keys["running_mean"]].float().numpy()
    bn_var = state_dict[bn_keys["running_var"]].float().numpy()
    print(f"    Found {BN_PREFIX_EXPECTED} (BatchNorm2d, affine=True)")

    # -- 2. Fold BatchNorm into Conv2d weights/bias --
    print("\n[2] Folding BatchNorm into Conv2d weights/bias ...")
    scale_bn = bn_gamma / np.sqrt(bn_var + BN_EPS)
    w_folded_all = w_float_all * scale_bn[:, None, None, None]
    b_folded_all = bn_beta - bn_mean * scale_bn
    print(f"    scale_bn range : [{scale_bn.min():.6f}, {scale_bn.max():.6f}]")
    print(f"    b_folded range : [{b_folded_all.min():.6f}, {b_folded_all.max():.6f}]")

    # -- 3. Quantize folded weights (per-output-channel symmetric INT8) --
    print(f"\n[3] Quantizing all {n_out} FOLDED kernels to INT8 "
          f"(symmetric per-output-channel scale) ...")
    scales_w_folded = np.zeros(n_out, dtype=np.float64)
    w_folded_int8_all = np.zeros_like(w_folded_all, dtype=np.int8)
    for oc in range(n_out):
        w_folded_int8_all[oc], scales_w_folded[oc] = quantize_symmetric_int8(w_folded_all[oc])
    print(f"    scale_w_folded range: [{scales_w_folded.min():.8f}, {scales_w_folded.max():.8f}]")

    # -- 4. PyTorch reference (float) Conv2d(padding=1) -> BatchNorm -> ReLU --
    device = torch.device(args.device)
    print(f"\n[4] PyTorch float reference will run on device: {device}")
    w_tensor_f32 = torch.from_numpy(w_float_all).float().to(device)
    bn_gamma_t = torch.from_numpy(bn_gamma).float().to(device)
    bn_beta_t = torch.from_numpy(bn_beta).float().to(device)
    bn_mean_t = torch.from_numpy(bn_mean).float().to(device)
    bn_var_t = torch.from_numpy(bn_var).float().to(device)

    def pytorch_conv_bn_relu_valid_region(x_float_chw: np.ndarray) -> np.ndarray:
        """Runs the REAL Conv2d(padding=1) -> BatchNorm(eval) -> ReLU on the
        full tile, then crops to the valid interior (H-2, W-2) region."""
        x = torch.from_numpy(x_float_chw).float().unsqueeze(0).to(device)
        conv_out = F.conv2d(x, w_tensor_f32, bias=None, padding=1)
        bn_out = F.batch_norm(
            conv_out, bn_mean_t, bn_var_t, weight=bn_gamma_t, bias=bn_beta_t,
            training=False, eps=BN_EPS,
        )
        relu_out = F.relu(bn_out)
        relu_out_np = relu_out.squeeze(0).cpu().numpy()
        return relu_out_np[:, 1:-1, 1:-1]

    def folded_float_conv_bn_relu_valid(x_float_chw: np.ndarray) -> np.ndarray:
        """Float folded valid conv + folded bias + ReLU (no quantization) --
        used only for the folding sanity check."""
        conv_out = conv_all_kernels(x_float_chw.astype(np.float64), w_folded_all)
        biased = conv_out + b_folded_all[:, None, None]
        return np.maximum(biased, 0.0)

    def folded_quantized_conv_bn_relu_valid(x_float_chw: np.ndarray) -> tuple[np.ndarray, float]:
        """INT8-quantized folded valid conv + folded bias + ReLU (the
        hardware-style fixed-point approximation under test)."""
        x_int8, scale_x = quantize_symmetric_int8_activation(x_float_chw)
        int32_accum = conv_all_kernels(x_int8, w_folded_int8_all)
        conv_float_approx = (
            int32_accum.astype(np.float64) * scale_x * scales_w_folded[:, None, None]
        )
        biased = conv_float_approx + b_folded_all[:, None, None]
        return np.maximum(biased, 0.0), scale_x

    # -- 5. Resolve split and load real tiles --
    print(f"\n[5] Resolving split (heldout_fp={args.heldout_fp}, split={args.split}) ...")
    split_csv_path, split_used = resolve_split_csv(args.heldout_fp, args.split)
    print(f"    Using split CSV: {split_csv_path.relative_to(REPO_ROOT)} (split={split_used})")

    if not args.data_root.exists():
        sys.exit(f"ERROR: data root not found at {args.data_root}")

    with open(split_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    tiles_used = []
    fold_diffs = []
    per_tile_kernel_errors = []
    tried = 0
    for row in rows:
        if len(tiles_used) >= args.max_tiles:
            break
        tried += 1
        tile_path = args.data_root / row["uavsar_path"]
        if not tile_path.exists():
            continue
        with rasterio.open(tile_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            continue
        sar = sar[:3]
        sar_norm = normalize_per_tile(sar).astype(np.float64)

        pytorch_ref = pytorch_conv_bn_relu_valid_region(sar_norm)
        folded_float = folded_float_conv_bn_relu_valid(sar_norm)
        fold_diff = np.abs(pytorch_ref - folded_float)
        fold_diffs.append((row["tile_name"], float(fold_diff.max()), float(fold_diff.mean())))

        quant_out, scale_x = folded_quantized_conv_bn_relu_valid(sar_norm)

        tiles_used.append({
            "tile_name": row["tile_name"],
            "uavsar_path": row["uavsar_path"],
            "shape": sar_norm.shape,
        })
        per_tile_kernel_errors.append({
            "label": row["tile_name"],
            "float_out": pytorch_ref,
            "rescaled_out": quant_out,
            "scale_x": scale_x,
            "sar_norm": sar_norm,
        })
    print(f"    Tried {tried} split rows, loaded {len(tiles_used)} usable full tiles "
          f"(requested up to {args.max_tiles}).")
    if not tiles_used:
        sys.exit("ERROR: no usable real tiles were loaded -- cannot compute fidelity metrics.")

    overall_fold_max_diff = max(d[1] for d in fold_diffs)
    overall_fold_mean_diff = float(np.mean([d[2] for d in fold_diffs]))
    print(f"\n    Folding sanity check (float, no quantization, ALL {len(tiles_used)} tiles): "
          f"max diff = {overall_fold_max_diff:.3e}, "
          f"mean of per-tile mean diffs = {overall_fold_mean_diff:.3e}")
    fold_check_passed = overall_fold_max_diff <= 1e-3
    print("    Folding sanity check "
          + ("PASSED" if fold_check_passed else "FAILED -- investigate")
          + ": float folded valid-conv + folded bias + ReLU vs. true PyTorch "
            "Conv2d(padding=1) -> BatchNorm -> ReLU (cropped to valid interior).")

    # -- 6. Error metrics: overall + per-channel --
    print("\n[6] Computing error metrics (MAE, max abs error, P99 abs error, "
          "mean relative error, Pearson correlation, cosine similarity) ...")

    per_tile_rel_err_values = []
    for r in per_tile_kernel_errors:
        f = r["float_out"].ravel()
        q = r["rescaled_out"].ravel()
        rel_vals, _ = safe_relative_error_values(f, q)
        per_tile_rel_err_values.append(rel_vals)

    all_float = np.concatenate([r["float_out"].ravel() for r in per_tile_kernel_errors])
    all_rescaled = np.concatenate([r["rescaled_out"].ravel() for r in per_tile_kernel_errors])
    abs_err_all = np.abs(all_rescaled - all_float)

    num_tiles = len(tiles_used)
    num_activation_values = int(all_float.size)
    mae_all = float(np.mean(abs_err_all))
    max_err_all = float(np.max(abs_err_all))
    p99_all = float(np.percentile(abs_err_all, 99))

    all_safe_rel_err_values = (
        np.concatenate(per_tile_rel_err_values) if per_tile_rel_err_values else np.array([])
    )
    mean_rel_err_all = mean_of_values(all_safe_rel_err_values)
    n_rel_samples_all = int(all_safe_rel_err_values.size)

    corr_all = float(np.corrcoef(all_float, all_rescaled)[0, 1]) if all_float.size > 1 else float("nan")
    cos_all = cosine_similarity(all_float, all_rescaled)

    print(f"    OVERALL (all {num_tiles} tiles, all {n_out} kernels, "
          f"{num_activation_values} total activation values):")
    print(f"      MAE                 = {mae_all:.6f}")
    print(f"      Max abs error       = {max_err_all:.6f}")
    print(f"      P99 abs error       = {p99_all:.6f}")
    print(f"      Mean relative error = {mean_rel_err_all:.6f} (n={n_rel_samples_all} safe samples)")
    print(f"      Pearson correlation = {corr_all:.6f}")
    print(f"      Cosine similarity   = {cos_all:.6f}")

    per_channel_rows = []
    for oc in range(n_out):
        f_oc = np.concatenate([r["float_out"][oc].ravel() for r in per_tile_kernel_errors])
        q_oc = np.concatenate([r["rescaled_out"][oc].ravel() for r in per_tile_kernel_errors])
        abs_err_oc = np.abs(q_oc - f_oc)

        per_tile_rel_vals_oc = []
        for r in per_tile_kernel_errors:
            f_oc_t = r["float_out"][oc].ravel()
            q_oc_t = r["rescaled_out"][oc].ravel()
            rel_vals_t, _ = safe_relative_error_values(f_oc_t, q_oc_t)
            per_tile_rel_vals_oc.append(rel_vals_t)
        rel_vals_oc = (
            np.concatenate(per_tile_rel_vals_oc) if per_tile_rel_vals_oc else np.array([])
        )

        corr_oc = float(np.corrcoef(f_oc, q_oc)[0, 1]) if f_oc.size > 1 else float("nan")
        cos_oc = cosine_similarity(f_oc, q_oc)
        per_channel_rows.append({
            "kernel": oc,
            "num_samples": int(f_oc.size),
            "mae": float(np.mean(abs_err_oc)),
            "max_abs_error": float(np.max(abs_err_oc)),
            "p99_abs_error": float(np.percentile(abs_err_oc, 99)),
            "mean_relative_error_safe_subset": mean_of_values(rel_vals_oc),
            "n_relative_error_samples": int(rel_vals_oc.size),
            "pearson_correlation": corr_oc,
            "cosine_similarity": cos_oc,
        })

    worst_channel_by_mae = max(per_channel_rows, key=lambda r: r["mae"])
    best_channel_by_mae = min(per_channel_rows, key=lambda r: r["mae"])

    # -- 7. Write CSV outputs --
    print(f"\n[7] Writing CSV outputs to {output_dir} ...")
    with open(summary_csv_path, "w", newline="") as f:
        fieldnames = [
            "num_tiles", "num_kernels", "num_activation_values",
            "mae", "max_abs_error", "p99_abs_error",
            "mean_relative_error_safe_subset", "n_relative_error_samples",
            "pearson_correlation", "cosine_similarity",
            "fold_sanity_max_diff", "fold_sanity_mean_diff",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "num_tiles": num_tiles,
            "num_kernels": n_out,
            "num_activation_values": num_activation_values,
            "mae": mae_all,
            "max_abs_error": max_err_all,
            "p99_abs_error": p99_all,
            "mean_relative_error_safe_subset": mean_rel_err_all,
            "n_relative_error_samples": n_rel_samples_all,
            "pearson_correlation": corr_all,
            "cosine_similarity": cos_all,
            "fold_sanity_max_diff": overall_fold_max_diff,
            "fold_sanity_mean_diff": overall_fold_mean_diff,
        })
    print(f"    Written: {summary_csv_path.relative_to(REPO_ROOT)}")

    with open(per_channel_csv_path, "w", newline="") as f:
        fieldnames = [
            "kernel", "num_samples", "mae", "max_abs_error", "p99_abs_error",
            "mean_relative_error_safe_subset", "n_relative_error_samples",
            "pearson_correlation", "cosine_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_channel_rows:
            writer.writerow(row)
    print(f"    Written: {per_channel_csv_path.relative_to(REPO_ROOT)}")

    # -- 8. Small diagnostic PNG (first tile, kernel 0 only -- kept small) --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        first = per_tile_kernel_errors[0]
        f0 = first["float_out"][0]
        q0 = first["rescaled_out"][0]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(f0, cmap="viridis")
        axes[0].set_title(f"PyTorch float\n{first['label']}, kernel 0")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(q0, cmap="viridis")
        axes[1].set_title("Hardware-style fixed-point\n(folded + INT8 + dequant)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        f0_flat, q0_flat = f0.ravel(), q0.ravel()
        axes[2].scatter(f0_flat, q0_flat, s=4, alpha=0.4)
        lims = [min(f0_flat.min(), q0_flat.min()), max(f0_flat.max(), q0_flat.max())]
        axes[2].plot(lims, lims, "r--", linewidth=1)
        axes[2].set_xlabel("PyTorch float")
        axes[2].set_ylabel("Fixed-point (dequantized)")
        axes[2].set_title("Value scatter (kernel 0)")
        fig.tight_layout()
        fig.savefig(diagnostic_png_path, dpi=110)
        plt.close(fig)
        print(f"    Written: {diagnostic_png_path.relative_to(REPO_ROOT)}")
    except Exception as exc:  # pragma: no cover - diagnostic only, non-fatal
        print(f"    NOTE: skipped diagnostic PNG ({exc})")

    # -- 9. Markdown summary --
    print(f"\n[8] Writing markdown summary ...")
    tile_list_lines = "\n".join(
        f"| {t['tile_name']} | {t['uavsar_path']} | {t['shape'][1]}x{t['shape'][2]} |"
        for t in tiles_used
    )
    per_channel_table_lines = "\n".join(
        f"| {r['kernel']} | {r['num_samples']} | {r['mae']:.6f} | {r['max_abs_error']:.6f} | "
        f"{r['p99_abs_error']:.6f} | {r['pearson_correlation']:.6f} | {r['cosine_similarity']:.6f} |"
        for r in per_channel_rows
    )

    md = f"""# First Conv-BN-ReLU Stage: Real-Tile Activation Fidelity (PyTorch Float vs. Hardware-Style Fixed-Point)

Generated by `scripts/analyze_first_layer_real_activation_fidelity.py`.

## Claim boundary (read this first)

- This is a Python numerical fidelity study only. **No VHDL was written or
  modified for this task.** Existing VHDL prototypes in
  `hardware/vhdl_conv3x3/` include toy-input folded Conv-BN-ReLU datapaths,
  but this script does not implement or simulate a full real-tile VHDL input
  pipeline.
- Scope: the FIRST Conv-BN-ReLU stage only (`enc1.block.0` Conv2d +
  `enc1.block.1` BatchNorm2d + ReLU). This is **not** a full DoubleConv or
  full U-Net comparison, and makes no Dice/recall/end-to-end accuracy claim.
- "Hardware-style fixed-point" here means: BatchNorm folded into the Conv2d
  weights/bias (float, exact), then symmetric per-output-channel INT8
  weight quantization and per-tile symmetric INT8 activation quantization,
  matching the convention already used in
  `scripts/analyze_first_conv_bn_relu_fidelity.py`. It is **not** a new
  "official" hardware convention. Existing VHDL prototypes verify toy-input
  folded Conv-BN-ReLU datapaths, but this script does not simulate or
  synthesize a full real-tile input pipeline in VHDL.

## Padding / valid-interior-region handling

The real PyTorch `enc1.block.0` Conv2d uses `padding=1`. Every VHDL
prototype in this repo computes VALID (no-padding) 3x3 windows only.
To keep this comparison apples-to-apples:
1. The real PyTorch `Conv2d(padding=1) -> BatchNorm(eval) -> ReLU` is run
   on the FULL tile, exactly as the real model would run it.
2. Only the valid interior region (the outermost 1-pixel border cropped
   away on all sides) is kept from that output -- this is exactly the
   region whose 3x3 receptive field never touches the zero-padding, so it
   is mathematically identical to a true valid (no-pad) convolution over
   the same tile.
3. The hardware-style folded/quantized computation is run as a true valid
   (no-pad) convolution over the SAME full tile, producing the same-shape
   output, and compared directly against the cropped PyTorch reference.

Padded-border behavior is **not** evaluated here, and VHDL is **not**
claimed to implement padding.

## Normalization

Reused verbatim from `FloodTileDataset._normalize_per_tile` in
`scripts/train_unet_baseline_tuned.py` (percentile-clip 1-99 +
zero-mean/unit-variance per tile), confirmed by direct inspection of that
file before implementation -- no different normalization was invented for
this script.

## Configuration used for this run

| Setting | Value |
|---|---|
| Checkpoint | `{args.checkpoint.relative_to(REPO_ROOT) if args.checkpoint.is_relative_to(REPO_ROOT) else args.checkpoint}` |
| Conv2d tensor | `{CONV_KEY_EXPECTED}` (shape {list(w_float_all.shape)}) |
| BatchNorm prefix | `{BN_PREFIX_EXPECTED}` |
| BN eps | {BN_EPS} |
| Held-out flight path | {args.heldout_fp} |
| Split requested / used | {args.split} / {split_used} |
| Split CSV | `{split_csv_path.relative_to(REPO_ROOT)}` |
| Data root | `{args.data_root.relative_to(REPO_ROOT) if args.data_root.is_relative_to(REPO_ROOT) else args.data_root}` |
| Max tiles requested | {args.max_tiles} |
| Tiles actually loaded | {num_tiles} |
| Split rows tried | {tried} |
| PyTorch reference device | {device} |

## Folding sanity check (float only, no quantization)

Confirms `valid_conv(x, w_folded) + b_folded`, ReLU'd, equals the
center-cropped `Conv2d(x, w, padding=1) -> BatchNorm -> ReLU`, before any
quantization is introduced.

| Metric | Value |
|---|---:|
| Max diff across all {num_tiles} tiles | {overall_fold_max_diff:.3e} |
| Mean of per-tile mean diffs | {overall_fold_mean_diff:.3e} |
| Result | {"PASSED" if fold_check_passed else "FAILED -- investigate"} |

## Overall metrics (all {num_tiles} tiles, all {n_out} kernels combined)

| Metric | Value |
|---|---:|
| Number of tiles | {num_tiles} |
| Number of activation values compared | {num_activation_values} |
| MAE | {mae_all:.6f} |
| Max absolute error | {max_err_all:.6f} |
| P99 absolute error | {p99_all:.6f} |
| Mean relative error (safe subset, n={n_rel_samples_all}) | {mean_rel_err_all:.6f} |
| Pearson correlation | {corr_all:.6f} |
| Cosine similarity | {cos_all:.6f} |

"Safe subset" relative error excludes, per tile, activation values whose
magnitude is below 5% of that tile's own max magnitude (or 1e-8, whichever
is larger) -- this avoids division blowing up near ReLU's zero-clamp
region and near zero-crossings, and each tile's threshold is computed from
that tile's own values only (never a different tile's scale).

## Per-output-channel metrics

| Kernel | N samples | MAE | Max abs error | P99 abs error | Pearson corr | Cosine sim |
|---:|---:|---:|---:|---:|---:|---:|
{per_channel_table_lines}

Worst channel by MAE: kernel {worst_channel_by_mae['kernel']} (MAE = {worst_channel_by_mae['mae']:.6f}).
Best channel by MAE: kernel {best_channel_by_mae['kernel']} (MAE = {best_channel_by_mae['mae']:.6f}).

## Tiles used

| Tile name | uavsar_path | Tile size (HxW) |
|---|---|---|
{tile_list_lines}

## Outputs

- `activation_fidelity_overall.csv` -- overall metrics (one row).
- `activation_fidelity_by_channel.csv` -- per-output-channel metrics ({n_out} rows).
- `diagnostic_tile0_kernel0.png` -- optional small diagnostic figure (first
  tile, kernel 0 float vs. fixed-point maps + value scatter). Diagnostic
  only, not a metric.

## Limitations

- **First-layer convolution only.** Does not analyze the second Conv2d in
  `enc1.block`, the rest of `DoubleConv`, or any other U-Net stage.
- **Does not claim the full real-tile input/quantization/fidelity path is
  implemented in VHDL.** No VHDL in this repo was written or modified for
  this task. Existing toy-input VHDL folded Conv-BN-ReLU prototypes are
  separate correctness checks.
- **Does not measure quantized end-to-end model accuracy.** No Dice,
  recall, IoU, or other segmentation metric is computed here.
- **Does not evaluate padded-border pixels.** Only the valid interior
  region (receptive field never touching zero-padding) is compared; the
  outermost 1-pixel border of the real (padded) PyTorch output is
  discarded from this comparison.
- **Quantization scheme is a Python-only approximation of a plausible
  hardware datapath**, not a scheme that has been implemented or verified
  in VHDL for real (non-toy) input.
- **Sample size is {num_tiles} tile(s)** from the `{args.heldout_fp}` `{split_used}`
  split -- a larger `--max-tiles` value gives a more representative
  estimate but this remains a held-out-split sample, not the full split.
- All metrics are computed in Python/NumPy; no board testing, no VHDL
  simulation, and no synthesis are involved in this script.
"""
    summary_md_path.write_text(md)
    print(f"    Written: {summary_md_path.relative_to(REPO_ROOT)}")

    # -- Final stdout summary --
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Stage analyzed    : {CONV_KEY_EXPECTED.rsplit('.', 1)[0]} -> "
          f"{BN_PREFIX_EXPECTED} -> ReLU  ({n_out} kernels)")
    print(f"Comparison scope  : VALID INTERIOR positions only (no padding evaluated)")
    print(f"Split used        : {args.heldout_fp} / {split_used} ({num_tiles} tiles loaded)")
    print(f"Folding sanity check (float, pre-quantization): max diff = "
          f"{overall_fold_max_diff:.3e}, {'PASSED' if fold_check_passed else 'FAILED'}")
    print(f"Overall MAE                 : {mae_all:.6f}")
    print(f"Overall max abs error       : {max_err_all:.6f}")
    print(f"Overall P99 abs error       : {p99_all:.6f}")
    print(f"Overall mean relative error : {mean_rel_err_all:.6f} (n={n_rel_samples_all})")
    print(f"Overall Pearson correlation : {corr_all:.6f}")
    print(f"Overall cosine similarity   : {cos_all:.6f}")
    print("\nThis compares the FIRST Conv-BN-ReLU stage only, on real UAVSAR "
          "tiles, restricted to valid interior positions. It does NOT analyze "
          "the second Conv2d, does NOT claim full DoubleConv/U-Net accuracy, "
          "does NOT claim the full real-tile input/quantization path is implemented in VHDL, and does NOT "
          "measure Dice/recall/end-to-end impact.")


if __name__ == "__main__":
    main()
