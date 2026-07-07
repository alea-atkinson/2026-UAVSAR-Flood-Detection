"""
analyze_first_conv_bn_relu_fidelity.py

BatchNorm-folding fidelity study for the first U-Net Conv-BN-ReLU stage
(enc1.block.0 -> enc1.block.1 -> ReLU).

This EXTENDS scripts/analyze_first_layer_quantization_fidelity.py, which
compared raw PyTorch float Conv2d output against an INT8/fixed-point-style
raw Conv2d approximation (no BatchNorm, no activation). That earlier study's
own "Recommended next step" was exactly this: fold BatchNorm into the
Conv2d weights/bias and measure the fidelity of the folded, quantized
Conv-BN-ReLU stage against the true PyTorch Conv2d -> BatchNorm -> ReLU
computation. This script performs that comparison.

This is ANALYSIS ONLY -- it does not retrain, modify the model, or touch
any dataset splits. It does NOT write any new VHDL, and it does NOT claim
that BatchNorm folding, the bias, or ReLU are implemented in any VHDL
module in this repo -- every VHDL prototype in hardware/vhdl_conv3x3/
still only computes bias_int32 = 0, no BatchNorm, no activation.

---- Two different claims, kept clearly separate (same principle as the
     earlier study) --------------------------------------------------------
1. "VHDL matches Python fixed-point": NOT applicable here -- no VHDL
   implements BatchNorm/bias/ReLU. Only the raw Conv2d datapath (bias=0,
   no BN, no activation) has ever been verified in VHDL.
2. "Fixed-point (folded) approximates PyTorch float (Conv-BN-ReLU)": THIS
   is what this script measures.

---- Scope ------------------------------------------------------------------
- First Conv-BN-ReLU stage ONLY: enc1.block.0 (Conv2d) -> enc1.block.1
  (BatchNorm2d) -> ReLU. Does NOT analyze the second Conv2d in
  enc1.block (enc1.block.3 / enc1.block.4), the rest of DoubleConv, or any
  other part of the U-Net.
- Does NOT claim full DoubleConv, full U-Net, full quantized model
  accuracy, Dice, recall, or end-to-end segmentation impact.
- Does NOT claim VHDL implements BatchNorm folding, bias, or ReLU -- this
  is Python numerical fidelity only.

---- Important spatial-scope clarification (valid-interior-only) -----------
The real PyTorch Conv2d uses padding=1, so its output has the same spatial
size as its input. Every VHDL prototype in this repo computes VALID (no
padding) 3x3 windows only -- e.g. a 5x5 input produces a 3x3 valid output,
a 16x16 input produces a 14x14 valid output. Padding is NOT implemented in
VHDL. To make an apples-to-apples comparison, this script:
  1. Runs the real PyTorch Conv2d(padding=1) -> BatchNorm -> ReLU on the
     FULL patch (5x5 or 16x16), exactly as the real model would.
  2. Extracts ONLY the valid interior region from that padded output
     (the center 3x3 of a 5x5-input output, or the center 14x14 of a
     16x16-input output) -- i.e. crops away the outermost 1-pixel border,
     which is exactly the region whose receptive field never touches the
     padding, so it is mathematically identical to a true valid (no-pad)
     convolution over the same patch.
  3. Compares that cropped region against the valid (no-pad) folded/
     quantized computation.
This script does NOT evaluate padded border behavior, and does NOT imply
that VHDL implements padding.

---- BatchNorm folding ----------------------------------------------------
    scale_bn[oc] = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc] = w[oc] * scale_bn[oc]
    b_folded[oc] = beta[oc] - running_mean[oc] * scale_bn[oc]
eps = 1e-5 (torch.nn.BatchNorm2d default; verified that
train_unet_baseline_tuned.py's DoubleConv instantiates
nn.BatchNorm2d(out_channels) with no eps override, so the default applies).

A folding SANITY CHECK (float, no quantization) confirms:
    valid_conv(x, w_folded) + b_folded, ReLU'd
      ==  center-cropped [ Conv2d(x, w, padding=1) -> BatchNorm -> ReLU ]
before any quantization is introduced (see "Folding sanity check" printout
and the markdown summary).

---- Quantization scheme (folded weights + same activation convention) ----
Weights (folded): symmetric per-output-channel INT8, SAME formula as the
raw-Conv2d study but applied to the FOLDED weights:
    scale_w_folded[oc] = max(|w_folded[oc]|) / 127
    w_folded_int8[oc]  = round(w_folded[oc] / scale_w_folded[oc]), clipped
                         to [-127, 127]

Activations: identical per-patch symmetric convention already established
in analyze_first_layer_quantization_fidelity.py:
    scale_x = max(|x_patch|) / 127
    x_int8  = round(x_patch / scale_x), clipped to [-127, 127]

Reconstruction:
    conv_float_approx[oc] = int32_accum[oc] * scale_x * scale_w_folded[oc]
    + b_folded[oc]   (added in float, NOT quantized)
    -> ReLU

---- Data -------------------------------------------------------------------
Same corrected source as the earlier study:
    csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/
    strict_no_overlap/heldout_fp2_test.csv
(the IEEE PNG-filtered strict held-out fp2 split that actually matches this
checkpoint's "filtered_strict_fp2" training config).

Normalization: VERIFIED (not assumed) by inspecting
train_unet_baseline_tuned.py directly -- the dataset class is
FloodTileDataset, and its normalization method is `_normalize_per_tile`
(percentile-clip + zero-mean/unit-variance per tile). No
`_normalize_per_band` method exists anywhere in this repo. This script
reuses that exact `_normalize_per_tile` logic.

Patches: the same canonical 5x5x3 toy patch used throughout
hardware/vhdl_conv3x3/'s VHDL testbenches, plus the first 2 loadable real
UAVSAR tiles from the corrected split, with the same 16x16 center crop used
by the earlier study.

---- Outputs -----------------------------------------------------------
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_fidelity/first_conv_bn_relu_error_summary.csv
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_fidelity/first_conv_bn_relu_per_channel_metrics.csv
(printed to stdout: BatchNorm folding parameters, the folding sanity check,
 and full error metrics)
"""

import csv
import pathlib
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

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
BN_EPS = 1e-5   # torch.nn.BatchNorm2d default; verified no override in train_unet_baseline_tuned.py

FP2_TEST_SPLIT_CSV = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap" / "heldout_fp2_test.csv"
)
DATA_ROOT = REPO_ROOT / "2025_Tile_Data"

OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors" / "conv_bn_relu_fidelity"
SUMMARY_CSV_PATH     = OUT_DIR / "first_conv_bn_relu_error_summary.csv"
PER_CHANNEL_CSV_PATH = OUT_DIR / "first_conv_bn_relu_per_channel_metrics.csv"

NUM_REAL_TILES = 2          # small, meaningful sample -- not the full split
CROP_SIZE      = 16         # 16x16 center crop -> 14x14 = 196 valid 3x3 windows per tile

# ---------------------------------------------------------------------------
# 1. Load checkpoint and extract Conv2d weight + BatchNorm parameters
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

print(f"    Found {BN_PREFIX_EXPECTED} (BatchNorm2d, affine=True): "
      f"gamma{bn_gamma.shape}, beta{bn_beta.shape}, "
      f"running_mean{bn_mean.shape}, running_var{bn_var.shape}")
print(f"    gamma  range: [{bn_gamma.min():.4f}, {bn_gamma.max():.4f}]")
print(f"    beta   range: [{bn_beta.min():.4f}, {bn_beta.max():.4f}]")
print(f"    run.mean range: [{bn_mean.min():.4f}, {bn_mean.max():.4f}]")
print(f"    run.var  range: [{bn_var.min():.6f}, {bn_var.max():.6f}]")
print(f"    BN eps used: {BN_EPS} (torch.nn.BatchNorm2d default; verified no override "
      f"in train_unet_baseline_tuned.py's DoubleConv)")

# ---------------------------------------------------------------------------
# 2. Fold BatchNorm into the Conv2d weights and bias
# ---------------------------------------------------------------------------
print("\n[2] Folding BatchNorm into Conv2d weights/bias ...")
scale_bn = bn_gamma / np.sqrt(bn_var + BN_EPS)                       # (32,)
w_folded_all = w_float_all * scale_bn[:, None, None, None]           # (32, 3, 3, 3)
b_folded_all = bn_beta - bn_mean * scale_bn                          # (32,)
print(f"    scale_bn range : [{scale_bn.min():.6f}, {scale_bn.max():.6f}]")
print(f"    b_folded range : [{b_folded_all.min():.6f}, {b_folded_all.max():.6f}]")

# ---------------------------------------------------------------------------
# 3. Symmetric per-output-channel INT8 quantization of the FOLDED weights
# ---------------------------------------------------------------------------
def quantize_symmetric_int8(w_float: np.ndarray):
    max_abs_w = float(np.abs(w_float).max())
    scale_w = 1.0 if max_abs_w == 0.0 else max_abs_w / 127.0
    w_int8 = np.clip(np.round(w_float / scale_w), -127, 127).astype(np.int8)
    return w_int8, scale_w


print(f"\n[3] Quantizing all {n_out} FOLDED kernels to INT8 "
      f"(symmetric per-output-channel scale) ...")
scales_w_folded = np.zeros(n_out, dtype=np.float64)
w_folded_int8_all = np.zeros_like(w_folded_all, dtype=np.int8)
for oc in range(n_out):
    w_folded_int8_all[oc], scales_w_folded[oc] = quantize_symmetric_int8(w_folded_all[oc])
print(f"    scale_w_folded range across {n_out} kernels: "
      f"[{scales_w_folded.min():.8f}, {scales_w_folded.max():.8f}]")


def quantize_symmetric_int8_activation(x_float: np.ndarray):
    """Identical convention to analyze_first_layer_quantization_fidelity.py."""
    max_abs_x = float(np.abs(x_float).max())
    scale_x = 1.0 if max_abs_x == 0.0 else max_abs_x / 127.0
    x_int8 = np.clip(np.round(x_float / scale_x), -127, 127).astype(np.int8)
    return x_int8, scale_x


# ---------------------------------------------------------------------------
# 4. Valid (no-padding) 3x3 conv over all windows of a patch, for all
#    kernels, vectorized with sliding_window_view + einsum.
#    Returns (n_out, H-2, W-2).
# ---------------------------------------------------------------------------
def conv_all_kernels(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (3, H, W) int8 or float; w: (n_out, 3, 3, 3) int8 or float.
    Returns (n_out, H-2, W-2) INT64/float64 valid-convolution accumulation
    (no bias added here -- bias is added separately after rescaling)."""
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    acc_dtype = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    return np.einsum(
        "chwij,ocij->ohw",
        windows.astype(acc_dtype),
        w.astype(acc_dtype),
        optimize=True,
    )


# ---------------------------------------------------------------------------
# 5. Reference computation: TRUE PyTorch Conv2d(padding=1) -> BatchNorm ->
#    ReLU, run via torch.nn.functional so it is bit-for-bit the same
#    operation the real model performs, then crop to the valid interior.
# ---------------------------------------------------------------------------
w_tensor_f32 = torch.from_numpy(w_float_all).float()
bn_gamma_t = torch.from_numpy(bn_gamma).float()
bn_beta_t = torch.from_numpy(bn_beta).float()
bn_mean_t = torch.from_numpy(bn_mean).float()
bn_var_t = torch.from_numpy(bn_var).float()


def pytorch_conv_bn_relu_valid_region(x_float_chw: np.ndarray) -> np.ndarray:
    """x_float_chw: (3, H, W) float. Runs the REAL Conv2d(padding=1) ->
    BatchNorm(eval, running stats) -> ReLU, then crops to the valid
    interior (H-2, W-2) region -- the region whose receptive field never
    touches the padding, hence mathematically identical to a true valid
    (no-pad) convolution over the same patch. Returns (n_out, H-2, W-2)."""
    x = torch.from_numpy(x_float_chw).float().unsqueeze(0)   # (1, 3, H, W)
    conv_out = F.conv2d(x, w_tensor_f32, bias=None, padding=1)  # (1, 32, H, W)
    bn_out = F.batch_norm(
        conv_out, bn_mean_t, bn_var_t, weight=bn_gamma_t, bias=bn_beta_t,
        training=False, eps=BN_EPS,
    )
    relu_out = F.relu(bn_out)
    relu_out_np = relu_out.squeeze(0).numpy()   # (32, H, W)
    return relu_out_np[:, 1:-1, 1:-1]            # crop 1-pixel border -> valid interior


def folded_float_conv_bn_relu_valid(x_float_chw: np.ndarray) -> np.ndarray:
    """Float folded valid (no-pad) conv + folded bias + ReLU -- the
    float-only half of the folding sanity check (no quantization)."""
    conv_out = conv_all_kernels(x_float_chw.astype(np.float64), w_folded_all)  # (32, H-2, W-2)
    biased = conv_out + b_folded_all[:, None, None]
    return np.maximum(biased, 0.0)


def folded_quantized_conv_bn_relu_valid(x_float_chw: np.ndarray):
    """INT8-quantized folded valid conv + folded bias + ReLU (the
    fixed-point-style approximation under test). Returns (result, scale_x)."""
    x_int8, scale_x = quantize_symmetric_int8_activation(x_float_chw)
    int32_accum = conv_all_kernels(x_int8, w_folded_int8_all)                  # (32, H-2, W-2)
    conv_float_approx = int32_accum.astype(np.float64) * scale_x * scales_w_folded[:, None, None]
    biased = conv_float_approx + b_folded_all[:, None, None]
    return np.maximum(biased, 0.0), scale_x


# ---------------------------------------------------------------------------
# 6. Toy patch (synthetic/toy) -- folding sanity check + fidelity check
# ---------------------------------------------------------------------------
print("\n[4] Building canonical 5x5x3 TOY patch (SYNTHETIC, matches VHDL testbenches) ...")
base = np.arange(1, 26, dtype=np.int64).reshape(5, 5)
toy_x_float = np.stack([base, base * 2, -base], axis=0).astype(np.float64)   # (3, 5, 5)

print("\n[5] Folding sanity check (FLOAT only, no quantization) ...")
print("    Confirms: valid_conv(x, w_folded) + b_folded, ReLU'd")
print("              == center-cropped [Conv2d(x, w, padding=1) -> BatchNorm -> ReLU]")


def run_folding_sanity_check(label: str, x_float_chw: np.ndarray):
    pytorch_ref = pytorch_conv_bn_relu_valid_region(x_float_chw)
    folded_float = folded_float_conv_bn_relu_valid(x_float_chw)
    diff = np.abs(pytorch_ref - folded_float)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"    [{label}] max diff = {max_diff:.3e}   mean diff = {mean_diff:.3e}   "
          f"shape = {pytorch_ref.shape}")
    return pytorch_ref, max_diff, mean_diff


toy_pytorch_ref, toy_fold_max_diff, toy_fold_mean_diff = run_folding_sanity_check(
    "toy_5x5_synthetic", toy_x_float
)

# ---------------------------------------------------------------------------
# 7. Real UAVSAR patches from the IEEE PNG-filtered strict held-out fp2 test split
# ---------------------------------------------------------------------------
def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
    """VERIFIED (not assumed) to be the exact normalization used by the
    checkpoint's training/eval workflow: FloodTileDataset._normalize_per_tile
    in train_unet_baseline_tuned.py (percentile-clip 1-99 + zero-mean/
    unit-variance per tile). No _normalize_per_band method exists anywhere
    in this repo -- confirmed by direct inspection before writing this
    function."""
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


print(f"\n[6] Loading {NUM_REAL_TILES} REAL UAVSAR tiles from the IEEE PNG-filtered "
      f"strict held-out fp2 test split (matches the checkpoint's training config) ...")
print(f"    Split CSV: {FP2_TEST_SPLIT_CSV.relative_to(REPO_ROOT)}")

real_patches_used = []
using_real_data = FP2_TEST_SPLIT_CSV.exists() and DATA_ROOT.exists()

if using_real_data:
    import csv as csv_mod
    with open(FP2_TEST_SPLIT_CSV, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    for row in rows:
        if len(real_patches_used) >= NUM_REAL_TILES:
            break
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
        crop = sar_norm[:, cy - half:cy + half, cx - half:cx + half]
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
              f"per-tile normalized (_normalize_per_tile, same as training)")
else:
    print("    NOTE: this run used SYNTHETIC/TOY data only.")

# ---------------------------------------------------------------------------
# 8. Compute PyTorch reference vs. folded+quantized output for every patch
# ---------------------------------------------------------------------------
print("\n[7] Computing PyTorch Conv-BN-ReLU reference vs. folded/quantized "
      "approximation for all kernels, all patches ...")

per_patch_kernel_errors = []


def add_patch_result(label: str, is_synthetic: bool, x_float: np.ndarray, pytorch_ref=None):
    if pytorch_ref is None:
        pytorch_ref = pytorch_conv_bn_relu_valid_region(x_float)
    folded_quant_out, scale_x = folded_quantized_conv_bn_relu_valid(x_float)
    per_patch_kernel_errors.append({
        "label": label,
        "is_synthetic": is_synthetic,
        "float_out": pytorch_ref,          # PyTorch Conv-BN-ReLU, valid interior
        "rescaled_out": folded_quant_out,  # folded + quantized + bias + ReLU
        "scale_x": scale_x,
    })


# Reuse the already-computed PyTorch reference for the toy patch (avoids
# recomputation; also keeps the sanity check and fidelity check consistent).
add_patch_result("toy_5x5_synthetic", True, toy_x_float, pytorch_ref=toy_pytorch_ref)

real_fold_diffs = []
if using_real_data:
    for p in real_patches_used:
        crop = p["crop_float"].astype(np.float64)
        pytorch_ref, max_diff, mean_diff = run_folding_sanity_check(
            f"real_{p['tile_name']}", crop
        )
        real_fold_diffs.append((p["tile_name"], max_diff, mean_diff))
        add_patch_result(f"real_{p['tile_name']}", False, crop, pytorch_ref=pytorch_ref)

overall_fold_max_diff = max([toy_fold_max_diff] + [d[1] for d in real_fold_diffs])
overall_fold_mean_diff = float(np.mean([toy_fold_mean_diff] + [d[2] for d in real_fold_diffs]))
print(f"\n    Folding sanity check (ALL patches): max diff = {overall_fold_max_diff:.3e}, "
      f"mean of per-patch mean diffs = {overall_fold_mean_diff:.3e}")
if overall_fold_max_diff > 1e-3:
    print("    WARNING: folding sanity check shows a larger-than-expected float "
          "discrepancy -- investigate before trusting the quantized results below.")
else:
    print("    Folding sanity check PASSED: float folded valid-conv + folded bias "
          "+ ReLU matches the true PyTorch Conv2d(padding=1) -> BatchNorm -> ReLU "
          "(cropped to the valid interior) to within floating-point precision.")

# ---------------------------------------------------------------------------
# 9. Error metrics: overall + per-channel (same corrected aggregation
#    method as analyze_first_layer_quantization_fidelity.py)
# ---------------------------------------------------------------------------
print("\n[8] Computing error metrics (MAE, max abs error, relative error, "
      "correlation, cosine similarity) ...")


def safe_relative_error_values(float_flat: np.ndarray, rescaled_flat: np.ndarray, eps_frac: float = 0.05):
    """Relative-error VALUES computed using THIS ARRAY'S OWN max magnitude
    as the safe-subset threshold. Must be called once per patch (and once
    per (patch, kernel) pair for per-channel stats) -- never on data
    concatenated across patches of different scale. Returns (values, n_safe)."""
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


# ---- Per-patch summary (each patch uses ITS OWN safe threshold) ----
per_patch_summary_rows = []
per_patch_rel_err_values = []
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

# ---- Overall absolute-error metrics (across ALL outputs) ----
all_float = np.concatenate([r["float_out"].ravel() for r in per_patch_kernel_errors])
all_rescaled = np.concatenate([r["rescaled_out"].ravel() for r in per_patch_kernel_errors])
abs_err_all = np.abs(all_rescaled - all_float)

mae_all = float(np.mean(abs_err_all))
max_err_all = float(np.max(abs_err_all))
p50_all = float(np.percentile(abs_err_all, 50))
p90_all = float(np.percentile(abs_err_all, 90))
p99_all = float(np.percentile(abs_err_all, 99))

# ---- Overall relative error: pool each patch's OWN safe values, then
# average -- NOT a single global threshold (corrected aggregation method) ----
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

# ---- Per-channel summary (same corrected per-patch-threshold-pooling method) ----
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
        "scale_w_folded": scales_w_folded[oc],
        "b_folded": b_folded_all[oc],
        "num_samples": f_oc.size,
        "mae": float(np.mean(abs_err_oc)),
        "max_abs_error": float(np.max(abs_err_oc)),
        "mean_relative_error_safe_subset": mre_oc,
        "n_relative_error_samples": n_rel_oc,
        "pearson_correlation": corr_oc,
        "cosine_similarity": cos_oc,
    })

print("\n    Per-channel summary (aggregated across all patches):")
print(f"    {'kernel':>6} {'scale_wf':>10} {'b_folded':>10} {'MAE':>10} {'max_err':>10} {'corr':>8} {'cos_sim':>8}")
for row in per_channel_rows:
    print(f"    {row['kernel']:>6} {row['scale_w_folded']:>10.6f} {row['b_folded']:>10.4f} "
          f"{row['mae']:>10.4f} {row['max_abs_error']:>10.4f} "
          f"{row['pearson_correlation']:>8.4f} {row['cosine_similarity']:>8.4f}")

# ---------------------------------------------------------------------------
# 10. Save small CSV outputs
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[9] Writing summary CSVs to {OUT_DIR} ...")

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
        "kernel", "scale_w_folded", "b_folded", "num_samples", "mae", "max_abs_error",
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
print(f"Stage analyzed    : {CONV_KEY_EXPECTED.rsplit('.', 1)[0]} -> "
      f"{BN_PREFIX_EXPECTED} -> ReLU  (32 kernels)")
print(f"Comparison scope  : VALID INTERIOR positions only (no padding evaluated)")
print(f"Input source      : {'synthetic toy patch + ' + str(len(real_patches_used)) + ' real UAVSAR fp2-test tiles' if using_real_data else 'synthetic toy patch ONLY (real data unavailable)'}")
print(f"Folding sanity check (float, pre-quantization): max diff = {overall_fold_max_diff:.3e}, "
      f"{'PASSED' if overall_fold_max_diff <= 1e-3 else 'FAILED -- investigate'}")
print(f"Overall MAE (float units)      : {mae_all:.6f}")
print(f"Overall max abs error          : {max_err_all:.6f}")
print(f"Overall mean relative error    : {mean_rel_err_all:.6f} (n={n_rel_samples_all})")
print(f"Overall Pearson correlation    : {corr_all:.6f}")
print(f"Overall cosine similarity      : {cos_all:.6f}")
print("\nThis compares the FIRST Conv-BN-ReLU stage only, restricted to valid "
      "interior positions. It does NOT analyze the second Conv2d, does NOT "
      "claim full DoubleConv/U-Net accuracy, does NOT claim VHDL implements "
      "BatchNorm/bias/ReLU, and does NOT measure Dice/recall/end-to-end impact.")
