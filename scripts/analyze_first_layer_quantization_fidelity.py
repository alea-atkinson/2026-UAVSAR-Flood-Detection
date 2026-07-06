"""
analyze_first_layer_quantization_fidelity.py

Quantization fidelity study for the first U-Net convolution layer
(enc1.block.0.weight, shape [32, 3, 3, 3]).

This is ANALYSIS ONLY -- it does not retrain, modify the model, or touch
any dataset splits. It reuses the exact symmetric per-tensor INT8 weight
quantization convention already established in
scripts/export_first_layer_multi_kernel_vhdl_vectors.py, and answers a
question that script does not: how close is the INT8/fixed-point-style
first-layer computation to the ORIGINAL PYTORCH FLOATING-POINT first-layer
computation (not just "does the VHDL match the Python INT32 golden
values" -- that was already established bit-for-bit by the GHDL
testbenches in this repo)?

---- Two different claims, kept clearly separate ---------------------------
1. "VHDL matches Python fixed-point": ALREADY ESTABLISHED (GHDL testbenches,
   72/72 etc.) -- the hardware computes exactly the INT32 values this
   family of export scripts predicts. This script does NOT re-verify that.
2. "Fixed-point approximates PyTorch float": THIS is what this script
   measures -- how much the INT8-weight / INT8-activation approximation
   deviates from the original float32 Conv2d output, before any BatchNorm,
   activation, or further layers are applied.

These are independent questions. A perfect answer to (1) says nothing
about (2), and this script's answer to (2) says nothing about downstream
model accuracy (Dice, recall, etc.) -- that would require a full
end-to-end quantized forward pass and evaluation, which is NOT done here.

---- Scope ------------------------------------------------------------------
- First convolution layer ONLY (enc1.block.0, a Conv2d(3, 32, 3, padding=1,
  bias=False) immediately followed by BatchNorm2d(32) and ReLU in the real
  model -- see Section on BatchNorm below).
- No BatchNorm, no activation, no padding, no downstream layers are applied
  here -- this compares the RAW Conv2d output only, matching the scope of
  every VHDL prototype in hardware/vhdl_conv3x3/.
- Does NOT claim full quantized U-Net accuracy and does NOT claim any
  Dice/recall/end-to-end model impact -- no such evaluation is performed.

---- Inputs used --------------------------------------------------------
A. The canonical 5x5x3 TOY patch used throughout hardware/vhdl_conv3x3/'s
   VHDL testbenches (channel 0 = 1..25 row-major, channel 1 = 2x channel 0,
   channel 2 = -1x channel 0). This is SYNTHETIC/TOY data, included here as
   a sanity cross-check: running kernel 0 through this script's own
   quantization path must reproduce the SAME INT32 value (11287 at the
   top-left position) already verified in VHDL, confirming this script's
   quantization arithmetic is consistent with the hardware-verified path.
B. REAL UAVSAR tiles from the IEEE PNG-filtered strict held-out fp2 test
   split (csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/
   strict_no_overlap/heldout_fp2_test.csv). This is the split that actually
   matches the checkpoint under test
   (models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
   was trained on the "filtered_strict" / IEEE PNG-filtered dataset variant,
   not the broader "standard_strict" split -- using the broader split would
   describe the wrong held-out fold for this specific checkpoint). Tiles are
   loaded with rasterio and normalized with the SAME per-tile
   percentile-clip + zero-mean/unit-variance normalization used by
   FloodTileDataset in train_unet_baseline_tuned.py, so the input
   distribution matches what the model actually saw during training/
   evaluation. A small center crop is taken from each tile (not the full
   256x256 tile) to keep this analysis fast and the output small.

---- Quantization scheme --------------------------------------------------
Weights: symmetric per-tensor INT8, SAME formula as
export_first_layer_multi_kernel_vhdl_vectors.py:
    scale_w[oc] = max(|w_float[oc]|) / 127
    w_int8[oc]  = round(w_float[oc] / scale_w[oc]), clipped to [-127, 127]
(scale computed independently per output channel/kernel, matching the
existing VHDL package generation convention.)

Activations: the existing hardware scripts only ever used an INTEGER toy
input with scale_x = 1.0 (no quantization was needed because the toy input
was already integer-valued). Real normalized UAVSAR patches are
floating-point, so THIS script introduces one new choice, clearly labeled
as such: a simple per-patch symmetric quantization,
    scale_x = max(|x_float_patch|) / 127
    x_int8  = round(x_float_patch / scale_x), clipped to [-127, 127]
applied independently to each analyzed crop. This is a natural extension of
the same symmetric scheme already used for weights, not a new "official"
hardware convention -- no VHDL in this repo implements real-image input
quantization yet.

---- Outputs -----------------------------------------------------------
hardware/vhdl_conv3x3/test_vectors/quantization_fidelity/first_layer_quantization_error_summary.csv
hardware/vhdl_conv3x3/test_vectors/quantization_fidelity/first_layer_quantization_per_channel_metrics.csv
(printed to stdout: full metrics, BatchNorm inspection, and a toy-patch
 cross-check against the existing VHDL-verified INT32 golden value)
"""

import csv
import pathlib
import sys

import numpy as np
import rasterio
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
CONV_KEY_EXPECTED = "enc1.block.0.weight"
BN_PREFIX_EXPECTED = "enc1.block.1"

FP2_TEST_SPLIT_CSV = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap" / "heldout_fp2_test.csv"
)
DATA_ROOT = REPO_ROOT / "2025_Tile_Data"

OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors" / "quantization_fidelity"
SUMMARY_CSV_PATH     = OUT_DIR / "first_layer_quantization_error_summary.csv"
PER_CHANNEL_CSV_PATH = OUT_DIR / "first_layer_quantization_per_channel_metrics.csv"

NUM_REAL_TILES = 2          # small, meaningful sample -- not the full split
CROP_SIZE      = 16         # 16x16 center crop -> 14x14 = 196 valid 3x3 windows per tile

# ---------------------------------------------------------------------------
# 1. Load checkpoint and extract first-layer Conv2d weight + BatchNorm
# ---------------------------------------------------------------------------
print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
if not CHECKPOINT_PATH.exists():
    sys.exit(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")

ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]

if CONV_KEY_EXPECTED not in state_dict:
    sys.exit(f"ERROR: expected tensor key '{CONV_KEY_EXPECTED}' not found in checkpoint.")

w_tensor = state_dict[CONV_KEY_EXPECTED]           # [32, 3, 3, 3]
w_float_all = w_tensor.float().numpy()             # (32, 3, 3, 3)
n_out = w_float_all.shape[0]
print(f"    Tensor key   : {CONV_KEY_EXPECTED}")
print(f"    Tensor shape : {list(w_tensor.shape)}")
print(f"    Output channels available: {n_out}")

bias_key = CONV_KEY_EXPECTED.replace(".weight", ".bias")
has_direct_bias = bias_key in state_dict
print(f"    Direct Conv bias present: {has_direct_bias} "
      f"(expected False -- this Conv2d uses BatchNorm instead)")

# ---------------------------------------------------------------------------
# 2. BatchNorm inspection (Task item 3) -- inspect only, do NOT fold here.
# ---------------------------------------------------------------------------
print(f"\n[2] Inspecting BatchNorm immediately following {CONV_KEY_EXPECTED} ...")

bn_keys = {
    "weight": f"{BN_PREFIX_EXPECTED}.weight",
    "bias": f"{BN_PREFIX_EXPECTED}.bias",
    "running_mean": f"{BN_PREFIX_EXPECTED}.running_mean",
    "running_var": f"{BN_PREFIX_EXPECTED}.running_var",
}
bn_present = all(k in state_dict for k in bn_keys.values())

if bn_present:
    bn_gamma = state_dict[bn_keys["weight"]].float().numpy()
    bn_beta = state_dict[bn_keys["bias"]].float().numpy()
    bn_mean = state_dict[bn_keys["running_mean"]].float().numpy()
    bn_var = state_dict[bn_keys["running_var"]].float().numpy()
    bn_eps = 1e-5  # torch.nn.BatchNorm2d default; not stored in state_dict

    print(f"    Found {BN_PREFIX_EXPECTED} (BatchNorm2d, affine=True): "
          f"gamma{bn_gamma.shape}, beta{bn_beta.shape}, "
          f"running_mean{bn_mean.shape}, running_var{bn_var.shape}")
    print(f"    gamma  range: [{bn_gamma.min():.4f}, {bn_gamma.max():.4f}]")
    print(f"    beta   range: [{bn_beta.min():.4f}, {bn_beta.max():.4f}]")
    print(f"    run.mean range: [{bn_mean.min():.4f}, {bn_mean.max():.4f}]")
    print(f"    run.var  range: [{bn_var.min():.6f}, {bn_var.max():.6f}]")
    print(
        "    BatchNorm DOES immediately follow the first Conv2d in this model "
        "(enc1.block = Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU)."
    )
    print(
        "    Folding BatchNorm into the Conv2d weights/bias is mathematically "
        "straightforward (a per-output-channel affine rescale: "
        "w_folded = w * gamma / sqrt(var + eps), "
        "b_folded = beta - gamma * running_mean / sqrt(var + eps)), "
        "but is NOT implemented in this script -- this analysis compares the "
        "RAW Conv2d output only (no BN, no activation), matching the scope of "
        "every VHDL prototype in hardware/vhdl_conv3x3/. BN folding remains a "
        "recommended next step for a more realistic hardware approximation."
    )
else:
    print(f"    WARNING: expected BatchNorm keys under '{BN_PREFIX_EXPECTED}' not found.")

# ---------------------------------------------------------------------------
# 3. Symmetric per-tensor INT8 weight quantization
#    (identical formula to export_first_layer_multi_kernel_vhdl_vectors.py)
# ---------------------------------------------------------------------------
def quantize_symmetric_int8(w_float: np.ndarray):
    max_abs_w = float(np.abs(w_float).max())
    scale_w = 1.0 if max_abs_w == 0.0 else max_abs_w / 127.0
    w_int8 = np.clip(np.round(w_float / scale_w), -127, 127).astype(np.int8)
    return w_int8, scale_w


print(f"\n[3] Quantizing all {n_out} output-channel kernels to INT8 "
      f"(symmetric per-tensor, per-kernel scale) ...")
scales_w = np.zeros(n_out, dtype=np.float64)
w_int8_all = np.zeros_like(w_float_all, dtype=np.int8)
for oc in range(n_out):
    w_int8_all[oc], scales_w[oc] = quantize_symmetric_int8(w_float_all[oc])
print(f"    scale_w range across {n_out} kernels: "
      f"[{scales_w.min():.8f}, {scales_w.max():.8f}]")


def quantize_symmetric_int8_activation(x_float: np.ndarray):
    """Same symmetric scheme applied to one activation patch (new for this
    script -- no VHDL in this repo yet quantizes real-image input scale)."""
    max_abs_x = float(np.abs(x_float).max())
    scale_x = 1.0 if max_abs_x == 0.0 else max_abs_x / 127.0
    x_int8 = np.clip(np.round(x_float / scale_x), -127, 127).astype(np.int8)
    return x_int8, scale_x


# ---------------------------------------------------------------------------
# 4. Valid (no-padding) 3x3 conv over all windows of a patch, for all kernels,
#    vectorized with sliding_window_view + einsum. Returns (n_out, H', W').
# ---------------------------------------------------------------------------
def conv_all_kernels(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (3, H, W) int8 or float; w: (n_out, 3, 3, 3) int8 or float.
    Returns (n_out, H-2, W-2) INT64/float64 valid-convolution accumulation
    (no bias -- matches every VHDL prototype's bias_int32 = 0 convention)."""
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    # windows shape: (3, H-2, W-2, 3, 3)  -- (channel, row, col, ky, kx)
    acc_dtype = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    return np.einsum(
        "chwij,ocij->ohw",
        windows.astype(acc_dtype),
        w.astype(acc_dtype),
        optimize=True,
    )


# ---------------------------------------------------------------------------
# 5. Toy patch (synthetic/toy) -- sanity cross-check against VHDL golden value
# ---------------------------------------------------------------------------
print("\n[4] Building canonical 5x5x3 TOY patch (SYNTHETIC, matches VHDL testbenches) ...")
base = np.arange(1, 26, dtype=np.int64).reshape(5, 5)
toy_x_int8 = np.stack(
    [base, (base * 2), (-base)], axis=0
).astype(np.int8)   # (3, 5, 5); scale_x = 1.0, matching every VHDL testbench
toy_scale_x = 1.0

toy_float_out = conv_all_kernels(toy_x_int8.astype(np.float64), w_float_all)      # (32, 3, 3)
toy_int_out = conv_all_kernels(toy_x_int8, w_int8_all)                            # (32, 3, 3)
toy_rescaled_out = toy_int_out.astype(np.float64) * scales_w[:, None, None] * toy_scale_x

# Cross-check: kernel 0, top-left position must equal the VHDL-verified 11287
kernel0_topleft_int32 = int(toy_int_out[0, 0, 0])
print(f"    Cross-check: kernel 0, top-left INT32 output = {kernel0_topleft_int32} "
      f"(expected 11287, matching every VHDL-verified prototype in this repo)")
if kernel0_topleft_int32 != 11287:
    print("    WARNING: cross-check MISMATCH -- this script's quantization path "
          "diverges from the VHDL-verified golden value. Investigate before "
          "trusting the results below.")
else:
    print("    Cross-check PASSED: this script's INT8 quantization arithmetic "
          "reproduces the same INT32 value already verified bit-for-bit in "
          "GHDL for kernel 0 (see hardware/vhdl_conv3x3/tb_stream_conv3x3_3chan_*).")

# ---------------------------------------------------------------------------
# 6. Real UAVSAR patches from the IEEE PNG-filtered strict held-out fp2 test split
# ---------------------------------------------------------------------------
def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
    """Identical to FloodTileDataset._normalize_per_tile in
    train_unet_baseline_tuned.py -- reproduces the real training/eval-time
    input distribution."""
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


print(f"\n[5] Loading {NUM_REAL_TILES} REAL UAVSAR tiles from the IEEE PNG-filtered "
      f"strict held-out fp2 test split (matches the checkpoint's training config) ...")
print(f"    Split CSV: {FP2_TEST_SPLIT_CSV.relative_to(REPO_ROOT)}")

real_patches_used = []
using_real_data = FP2_TEST_SPLIT_CSV.exists() and DATA_ROOT.exists()

if using_real_data:
    import csv as csv_mod
    with open(FP2_TEST_SPLIT_CSV, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    tried = 0
    for row in rows:
        if len(real_patches_used) >= NUM_REAL_TILES:
            break
        tried += 1
        tile_path = DATA_ROOT / row["uavsar_path"]
        if not tile_path.exists():
            continue
        with rasterio.open(tile_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            continue
        sar = sar[:3]
        sar_norm = normalize_per_tile(sar)

        h, w = sar_norm.shape[1], sar_norm.shape[2]
        cy, cx = h // 2, w // 2
        half = CROP_SIZE // 2
        crop = sar_norm[:, cy - half:cy + half, cx - half:cx + half]  # (3, CROP_SIZE, CROP_SIZE)
        if crop.shape[1] != CROP_SIZE or crop.shape[2] != CROP_SIZE:
            continue

        real_patches_used.append({
            "tile_name": row["tile_name"],
            "uavsar_path": row["uavsar_path"],
            "crop_float": crop,
        })
    if not real_patches_used:
        using_real_data = False
        print("    WARNING: could not load any real tiles from the split "
              "(tiles missing on disk?) -- falling back to synthetic-only analysis.")
else:
    print("    Real UAVSAR split CSV or data root not found -- "
          "falling back to synthetic-only analysis.")

if using_real_data:
    for p in real_patches_used:
        print(f"    Loaded real tile: {p['tile_name']} "
              f"({p['uavsar_path']}) -- {CROP_SIZE}x{CROP_SIZE} center crop, "
              f"per-tile normalized (same as training)")
else:
    print("    NOTE: this run used SYNTHETIC/TOY data only "
          "(see 'input source' in the output CSVs and summary).")

# ---------------------------------------------------------------------------
# 7. Compute float vs. INT8-rescaled outputs for every patch, all kernels
# ---------------------------------------------------------------------------
print("\n[6] Computing float vs. INT8/fixed-point-rescaled outputs for all kernels ...")

# error_records: list of dicts, one per (patch_source, kernel) pair, with all
# per-position errors flattened for that (patch, kernel) combination.
per_patch_kernel_errors = []   # for overall + per-channel aggregation

def add_patch_result(label: str, is_synthetic: bool, x_float: np.ndarray):
    x_int8, scale_x = quantize_symmetric_int8_activation(x_float)
    float_out = conv_all_kernels(x_float.astype(np.float64), w_float_all)          # (32, H', W')
    int_out = conv_all_kernels(x_int8, w_int8_all)                                  # (32, H', W')
    rescaled_out = int_out.astype(np.float64) * scales_w[:, None, None] * scale_x

    per_patch_kernel_errors.append({
        "label": label,
        "is_synthetic": is_synthetic,
        "float_out": float_out,
        "rescaled_out": rescaled_out,
        "scale_x": scale_x,
    })


# NOTE: this re-quantizes the toy patch with the SAME per-patch symmetric
# activation scheme used for the real tiles (scale_x = max(|x|)/127), for a
# consistent, apples-to-apples fidelity comparison across all patches. This
# is DIFFERENT from the cross-check above, which used the original hardware
# convention (scale_x = 1.0, no requantization, since the toy patch is
# already integer-valued and small enough to use as-is). Both are valid but
# answer different questions: the cross-check confirms this script's
# arithmetic matches the VHDL-verified path; this call measures genuine
# activation-quantization error the same way it would apply to real,
# non-integer image data. See first_layer_quantization_fidelity_summary.md.
add_patch_result("toy_5x5_synthetic", True, toy_x_int8.astype(np.float64))

if using_real_data:
    for p in real_patches_used:
        add_patch_result(f"real_{p['tile_name']}", False, p["crop_float"].astype(np.float64))

# ---------------------------------------------------------------------------
# 8. Error metrics: overall + per-channel
# ---------------------------------------------------------------------------
print("\n[7] Computing error metrics (MAE, max abs error, relative error, "
      "correlation, cosine similarity) ...")


def safe_relative_error_values(float_flat: np.ndarray, rescaled_flat: np.ndarray, eps_frac: float = 0.05):
    """Relative-error VALUES (not just their mean) computed only where
    |float_out| exceeds a small fraction of THIS ARRAY'S OWN max magnitude,
    to avoid division blowing up near zero-crossings.

    IMPORTANT: the "safe" threshold is always derived from the array passed
    in, never from a different array's magnitude. Callers must therefore
    call this ONCE PER PATCH (and, for per-channel stats, once per
    (patch, kernel) pair) -- never on data concatenated across patches of
    very different scale (e.g. the large-magnitude toy patch vs. the
    small-magnitude normalized real UAVSAR patches), or the toy patch's
    much larger threshold would silently exclude nearly all real-patch
    samples from the "safe" subset. Returns (values_array, n_safe).
    """
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
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ---- Per-patch summary (each patch's relative error uses ITS OWN safe
# threshold -- this part was already correct and is unchanged in method) ----
per_patch_summary_rows = []
per_patch_rel_err_values = []   # kept for the overall aggregation below
for r in per_patch_kernel_errors:
    f = r["float_out"].ravel()
    q = r["rescaled_out"].ravel()
    abs_err = np.abs(q - f)
    rel_vals, n_rel = safe_relative_error_values(f, q)
    per_patch_rel_err_values.append(rel_vals)
    mre = mean_of_values(rel_vals)
    corr = float(np.corrcoef(f, q)[0, 1]) if f.size > 1 else float("nan")
    cos = cosine_similarity(f, q)
    per_patch_summary_rows.append({
        "patch_label": r["label"],
        "input_source": "synthetic_toy" if r["is_synthetic"] else "real_uavsar_fp2_test",
        "num_kernels": n_out,
        "num_output_positions_per_kernel": r["float_out"].shape[1] * r["float_out"].shape[2],
        "scale_x": r["scale_x"],
        "mae": float(np.mean(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
        "p50_abs_error": float(np.percentile(abs_err, 50)),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "mean_relative_error_safe_subset": mre,
        "n_relative_error_samples": n_rel,
        "pearson_correlation": corr,
        "cosine_similarity": cos,
    })

# ---- Overall summary (absolute-error metrics across ALL outputs; relative
# error is a SEPARATE aggregation -- see below -- not filtered by one
# global threshold) ----
all_float = np.concatenate([r["float_out"].ravel() for r in per_patch_kernel_errors])
all_rescaled = np.concatenate([r["rescaled_out"].ravel() for r in per_patch_kernel_errors])
abs_err_all = np.abs(all_rescaled - all_float)

mae_all = float(np.mean(abs_err_all))
max_err_all = float(np.max(abs_err_all))
p50_all = float(np.percentile(abs_err_all, 50))
p90_all = float(np.percentile(abs_err_all, 90))
p99_all = float(np.percentile(abs_err_all, 99))

# Relative error, aggregated correctly: collect EACH PATCH's own safe
# relative-error values (already computed above using that patch's own
# threshold), concatenate them, then average -- this is the fix for the
# bug where a single global threshold (dominated by the large-magnitude
# toy patch) silently excluded nearly all real-tile samples from the
# "safe" subset.
all_safe_rel_err_values = (
    np.concatenate(per_patch_rel_err_values) if per_patch_rel_err_values else np.array([])
)
mean_rel_err_all = mean_of_values(all_safe_rel_err_values)
n_rel_samples_all = int(all_safe_rel_err_values.size)

corr_all = float(np.corrcoef(all_float, all_rescaled)[0, 1]) if all_float.size > 1 else float("nan")
cos_all = cosine_similarity(all_float, all_rescaled)

print(f"    Overall ABSOLUTE-error metrics (all patches, all {n_out} kernels, "
      f"{all_float.size} total output values):")
print(f"      MAE                 = {mae_all:.6f}")
print(f"      Max abs error       = {max_err_all:.6f}")
print(f"      P50 / P90 / P99 abs error = {p50_all:.6f} / {p90_all:.6f} / {p99_all:.6f}")
print(f"      Pearson correlation = {corr_all:.6f}")
print(f"      Cosine similarity   = {cos_all:.6f}")
print(f"    Overall RELATIVE-error metric (aggregated from each patch's OWN "
      f"safe subset, then averaged -- NOT one global threshold):")
print(f"      Mean relative error (n={n_rel_samples_all} safe samples "
      f"pooled from {len(per_patch_rel_err_values)} patches) = {mean_rel_err_all:.6f}")
for row in per_patch_summary_rows:
    print(f"        {row['patch_label']:<24} n_safe={row['n_relative_error_samples']:>5}  "
          f"mean_rel_err={row['mean_relative_error_safe_subset']:.6f}")

# ---- Per-channel summary (aggregated across all patches, per kernel oc).
# Relative error uses the SAME per-patch-threshold fix: for each kernel,
# each patch's own safe subset (using that patch's own threshold for that
# kernel's data) is computed first, then pooled across patches. ----
per_channel_rows = []
for oc in range(n_out):
    f_oc = np.concatenate([r["float_out"][oc].ravel() for r in per_patch_kernel_errors])
    q_oc = np.concatenate([r["rescaled_out"][oc].ravel() for r in per_patch_kernel_errors])
    abs_err_oc = np.abs(q_oc - f_oc)

    per_patch_rel_vals_oc = []
    for r in per_patch_kernel_errors:
        f_oc_p = r["float_out"][oc].ravel()
        q_oc_p = r["rescaled_out"][oc].ravel()
        rel_vals_p, _ = safe_relative_error_values(f_oc_p, q_oc_p)
        per_patch_rel_vals_oc.append(rel_vals_p)
    rel_vals_oc = (
        np.concatenate(per_patch_rel_vals_oc) if per_patch_rel_vals_oc else np.array([])
    )
    mre_oc = mean_of_values(rel_vals_oc)
    n_rel_oc = int(rel_vals_oc.size)

    corr_oc = float(np.corrcoef(f_oc, q_oc)[0, 1]) if f_oc.size > 1 else float("nan")
    cos_oc = cosine_similarity(f_oc, q_oc)
    per_channel_rows.append({
        "kernel": oc,
        "scale_w": scales_w[oc],
        "num_samples": f_oc.size,
        "mae": float(np.mean(abs_err_oc)),
        "max_abs_error": float(np.max(abs_err_oc)),
        "mean_relative_error_safe_subset": mre_oc,
        "n_relative_error_samples": n_rel_oc,
        "pearson_correlation": corr_oc,
        "cosine_similarity": cos_oc,
    })

print("\n    Per-channel summary (aggregated across all patches):")
print(f"    {'kernel':>6} {'scale_w':>10} {'MAE':>10} {'max_err':>10} {'corr':>8} {'cos_sim':>8}")
for row in per_channel_rows:
    print(f"    {row['kernel']:>6} {row['scale_w']:>10.6f} {row['mae']:>10.4f} "
          f"{row['max_abs_error']:>10.4f} {row['pearson_correlation']:>8.4f} "
          f"{row['cosine_similarity']:>8.4f}")

# ---------------------------------------------------------------------------
# 9. Save small CSV outputs
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[8] Writing summary CSVs to {OUT_DIR} ...")

with open(SUMMARY_CSV_PATH, "w", newline="") as f:
    fieldnames = [
        "patch_label", "input_source", "num_kernels",
        "num_output_positions_per_kernel", "scale_x", "mae", "max_abs_error",
        "p50_abs_error", "p90_abs_error", "p99_abs_error",
        "mean_relative_error_safe_subset", "n_relative_error_samples",
        "pearson_correlation", "cosine_similarity",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in per_patch_summary_rows:
        writer.writerow(row)
    # Final row: overall aggregate across all patches
    writer.writerow({
        "patch_label": "OVERALL_ALL_PATCHES",
        "input_source": "synthetic_toy+real_uavsar_fp2_test" if using_real_data else "synthetic_toy_only",
        "num_kernels": n_out,
        "num_output_positions_per_kernel": "",
        "scale_x": "",
        "mae": mae_all,
        "max_abs_error": max_err_all,
        "p50_abs_error": p50_all,
        "p90_abs_error": p90_all,
        "p99_abs_error": p99_all,
        "mean_relative_error_safe_subset": mean_rel_err_all,
        "n_relative_error_samples": n_rel_samples_all,
        "pearson_correlation": corr_all,
        "cosine_similarity": cos_all,
    })
print(f"    Written: {SUMMARY_CSV_PATH.relative_to(REPO_ROOT)}")

with open(PER_CHANNEL_CSV_PATH, "w", newline="") as f:
    fieldnames = [
        "kernel", "scale_w", "num_samples", "mae", "max_abs_error",
        "mean_relative_error_safe_subset", "n_relative_error_samples",
        "pearson_correlation", "cosine_similarity",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in per_channel_rows:
        writer.writerow(row)
print(f"    Written: {PER_CHANNEL_CSV_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Checkpoint        : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}")
print(f"Tensor            : {CONV_KEY_EXPECTED}  shape {list(w_tensor.shape)}")
print(f"Kernels analyzed  : 0-{n_out - 1} ({n_out} total)")
print(f"Input source      : {'synthetic toy patch + ' + str(len(real_patches_used)) + ' real UAVSAR fp2-test tiles' if using_real_data else 'synthetic toy patch ONLY (real data unavailable)'}")
print(f"Toy-patch cross-check vs VHDL golden (kernel 0, top-left) : "
      f"{'PASSED' if kernel0_topleft_int32 == 11287 else 'FAILED'}")
print(f"Overall MAE (float units)      : {mae_all:.6f}")
print(f"Overall max abs error          : {max_err_all:.6f}")
print(f"Overall Pearson correlation    : {corr_all:.6f}")
print(f"Overall cosine similarity      : {cos_all:.6f}")
print("\nThis compares RAW Conv2d output only (no BatchNorm, no activation, "
      "no downstream layers). It does NOT measure full quantized U-Net "
      "accuracy and does NOT measure any Dice/recall/end-to-end impact.")
