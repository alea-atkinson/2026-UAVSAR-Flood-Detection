"""
export_first_layer_conv3x3_vhdl_vectors.py

Extract the first-layer 3×3 convolution kernel (output channel 0) from the
Alea-tuned U-Net checkpoint, quantize it to INT8, apply it to the canonical
toy 5×5×3 input patch that matches the VHDL testbench, and save test-vector
files for future VHDL golden-value comparison.

This script is ANALYSIS ONLY — it does not retrain, modify the model, or touch
any dataset splits.

Output files
------------
hardware/vhdl_conv3x3/test_vectors/first_layer_kernel0_int8.json
hardware/vhdl_conv3x3/test_vectors/first_layer_kernel0_expected_outputs.csv
hardware/vhdl_conv3x3/test_vectors/first_layer_kernel0_summary.md
"""

import json
import csv
import math
import os
import pathlib
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors"

JSON_PATH   = OUT_DIR / "first_layer_kernel0_int8.json"
CSV_PATH    = OUT_DIR / "first_layer_kernel0_expected_outputs.csv"
MD_PATH     = OUT_DIR / "first_layer_kernel0_summary.md"

# ---------------------------------------------------------------------------
# 1. Locate checkpoint
# ---------------------------------------------------------------------------
print(f"[1] Looking for checkpoint: {CHECKPOINT_PATH}")
if not CHECKPOINT_PATH.exists():
    # Fallback: search models/ for any alea_tuned fp2 checkpoint
    candidates = sorted(REPO_ROOT.glob("models/*fp2*.pt"))
    if not candidates:
        sys.exit("ERROR: No fp2 checkpoint found in models/")
    CHECKPOINT_PATH = candidates[0]
    print(f"    Fallback to: {CHECKPOINT_PATH}")
else:
    print("    Found.")

# ---------------------------------------------------------------------------
# 2. Load checkpoint, inspect keys
# ---------------------------------------------------------------------------
print(f"\n[2] Loading checkpoint ...")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)

state_dict = ckpt["model_state_dict"]
args       = ckpt.get("args", {})
base_ch    = getattr(args, "base_channels", None) or args.get("base_channels", "unknown")
print(f"    base_channels from checkpoint args: {base_ch}")
print(f"    Total state-dict keys: {len(state_dict)}")

# Find first Conv2d weight tensor: 4-D, input-channels == 3, 3×3 kernel
conv_key = None
for key, val in state_dict.items():
    if val.ndim == 4 and val.shape[1] == 3 and val.shape[2] == 3 and val.shape[3] == 3:
        conv_key = key
        break

if conv_key is None:
    sys.exit(
        "ERROR: Could not find a 4-D tensor with shape (*, 3, 3, 3) in state dict."
    )

w_tensor = state_dict[conv_key]  # shape: [out_ch, 3, 3, 3]
print(f"    Selected tensor key : {conv_key}")
print(f"    Tensor shape        : {list(w_tensor.shape)}")

# ---------------------------------------------------------------------------
# 3. Extract output channel 0 (3 × 3 × 3 kernel)
# ---------------------------------------------------------------------------
OUT_CHAN = 0
w_float = w_tensor[OUT_CHAN].float().numpy()   # shape (3, 3, 3) — (in_ch, ky, kx)
print(f"\n[3] Extracted output channel {OUT_CHAN}, shape {w_float.shape}")

# Check for direct bias (Conv2d may or may not have a bias term)
bias_key = conv_key.replace(".weight", ".bias")
has_direct_bias = bias_key in state_dict
if has_direct_bias:
    bias_float = float(state_dict[bias_key][OUT_CHAN])
    print(f"    Direct Conv bias found: {bias_float:.6f}")
else:
    bias_float = 0.0
    print(
        "    No direct Conv bias (layer uses BatchNorm; bias absorbed into BN). "
        "Using bias = 0.0 for this integer-datapath test vector."
    )

# ---------------------------------------------------------------------------
# 4. Quantize weights to signed INT8 (symmetric per-tensor)
# ---------------------------------------------------------------------------
print("\n[4] Quantizing kernel weights to INT8 ...")

max_abs_w = float(np.abs(w_float).max())
if max_abs_w == 0.0:
    scale_w = 1.0   # degenerate all-zero kernel
else:
    scale_w = max_abs_w / 127.0

w_float_scaled = w_float / scale_w
w_int8 = np.clip(np.round(w_float_scaled), -127, 127).astype(np.int8)

print(f"    max |w_float|  = {max_abs_w:.8f}")
print(f"    scale_w        = {scale_w:.8f}  (= max_abs / 127)")
print(f"    INT8 kernel (channel 0):\n{w_int8[0]}")
print(f"    INT8 kernel (channel 1):\n{w_int8[1]}")
print(f"    INT8 kernel (channel 2):\n{w_int8[2]}")

# ---------------------------------------------------------------------------
# 5. Build the canonical 5×5×3 toy input patch
#    Channel 0: 1..25 row-major
#    Channel 1: 2 × channel 0
#    Channel 2: -1 × channel 0
#    This matches tb_stream_conv3x3_3chan_cell.vhd exactly.
# ---------------------------------------------------------------------------
print("\n[5] Building 5×5×3 toy input patch ...")
base = np.arange(1, 26, dtype=np.int8).reshape(5, 5)   # values 1..25
x_int8 = np.stack([base, (base * 2).astype(np.int8), (-base).astype(np.int8)], axis=0)
# x_int8 shape: (3, 5, 5); channel 1 max = 50 (fits in signed INT8)

scale_x = 1.0
print(f"    Channel 0 range: {int(x_int8[0].min())} .. {int(x_int8[0].max())}")
print(f"    Channel 1 range: {int(x_int8[1].min())} .. {int(x_int8[1].max())}  (2× ch0)")
print(f"    Channel 2 range: {int(x_int8[2].min())} .. {int(x_int8[2].max())}  (-1× ch0)")

# ---------------------------------------------------------------------------
# 6. Quantize bias for integer datapath
#    Conv2d has no direct bias here (BN instead).
#    For the VHDL integer-datapath we use bias_int32 = 0.
#    In a full quantized inference pipeline, the BN parameters would be
#    folded into the weights and bias before deployment — not done here.
# ---------------------------------------------------------------------------
bias_int32 = 0
print(f"\n[6] Bias: float = {bias_float}  →  int32 used in VHDL datapath = {bias_int32}")

# ---------------------------------------------------------------------------
# 7. Compute INT32 valid convolution (5×5 → 3×3, no padding)
# ---------------------------------------------------------------------------
print("\n[7] Computing INT32 valid convolution output ...")

y_int32 = np.zeros((3, 3), dtype=np.int64)   # use int64 for accumulation safety
for r in range(3):
    for c in range(3):
        acc = bias_int32
        for ch in range(3):
            for ky in range(3):
                for kx in range(3):
                    acc += int(x_int8[ch, r + ky, c + kx]) * int(w_int8[ch, ky, kx])
        y_int32[r, c] = acc

y_int32 = y_int32.astype(np.int32)
print("    INT32 output (3×3):")
print(f"    {y_int32}")

# ---------------------------------------------------------------------------
# 8. Float reference: use original float weights, toy-integer input as floats
# ---------------------------------------------------------------------------
print("\n[8] Computing float reference output ...")

x_float = x_int8.astype(np.float32)   # same numeric values, float precision
y_float = np.zeros((3, 3), dtype=np.float64)
for r in range(3):
    for c in range(3):
        for ch in range(3):
            for ky in range(3):
                for kx in range(3):
                    y_float[r, c] += float(x_float[ch, r + ky, c + kx]) * float(w_float[ch, ky, kx])
# Note: no bias term in float reference either (Conv2d has no bias)

y_float_f32 = y_float.astype(np.float32)
print("    Float reference output (3×3):")
print(f"    {y_float_f32}")

# Scale to report what the integer output approximates in float space
y_int32_rescaled = y_int32.astype(np.float32) * (scale_w * scale_x)
print("    INT32 rescaled back to float units (y_int32 × scale_w × scale_x):")
print(f"    {y_int32_rescaled}")
max_rescale_err = float(np.abs(y_float_f32 - y_int32_rescaled).max())
print(f"    Max rescale error vs float reference: {max_rescale_err:.6f}")

# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[9] Writing output files to {OUT_DIR} ...")

# ---- JSON ----
data = {
    "checkpoint_path": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
    "selected_tensor_key": conv_key,
    "selected_output_channel": OUT_CHAN,
    "tensor_shape": list(w_tensor.shape),
    "base_channels_from_args": str(base_ch),
    "quantization": {
        "scheme": "symmetric per-tensor INT8 for weights",
        "weight_max_abs": max_abs_w,
        "scale_w": scale_w,
        "input_scale": scale_x,
        "bias_note": (
            "Conv2d has no direct bias (layer uses BatchNorm). "
            "bias_float=0.0, bias_int32=0 used for integer-datapath test vector. "
            "Full quantized inference would require folding BN params into weights/bias."
        ),
    },
    "float_weights": {
        f"channel_{c}": w_float[c].tolist()
        for c in range(3)
    },
    "int8_weights": {
        f"channel_{c}": w_int8[c].tolist()
        for c in range(3)
    },
    "input_patch_int8": {
        f"channel_{c}": x_int8[c].tolist()
        for c in range(3)
    },
    "input_patch_description": (
        "5×5 toy patch: ch0=1..25 row-major, ch1=2×ch0, ch2=-1×ch0. "
        "Matches tb_stream_conv3x3_3chan_cell.vhd stimulus."
    ),
    "bias_float": bias_float,
    "bias_int32": bias_int32,
    "int32_expected_outputs_3x3": y_int32.tolist(),
    "float_reference_outputs_3x3": y_float_f32.tolist(),
    "int32_rescaled_to_float_3x3": y_int32_rescaled.tolist(),
    "max_rescale_error_vs_float_reference": max_rescale_err,
    "notes": [
        "This validates integer datapath arithmetic for the VHDL stream_conv3x3_3chan_cell.",
        "It is NOT a full quantized U-Net inference pipeline.",
        "Conv2d bias is zero here because the model uses Conv → BatchNorm (no bias).",
        "The BN running_mean/var and gamma/beta are not folded into these weights.",
        "For production deployment, BN folding would be required before VHDL synthesis.",
        f"Checkpoint base_channels={base_ch}; the full first layer would need "
        f"{base_ch} instances of stream_conv3x3_3chan_cell for all output channels.",
        "Symmetric per-tensor INT8 quantization of weights only; input uses scale=1.0.",
    ],
}
with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"    Written: {JSON_PATH.relative_to(REPO_ROOT)}")

# ---- CSV ----
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row", "col", "output_index", "y_int32", "y_float_reference",
                     "y_int32_rescaled_to_float"])
    idx = 0
    for r in range(3):
        for c in range(3):
            writer.writerow([
                r, c, idx,
                int(y_int32[r, c]),
                f"{float(y_float_f32[r, c]):.6f}",
                f"{float(y_int32_rescaled[r, c]):.6f}",
            ])
            idx += 1
print(f"    Written: {CSV_PATH.relative_to(REPO_ROOT)}")

# ---- Markdown summary ----
w_int8_ch_str = "\n".join(
    f"**Channel {c}:**\n```\n{w_int8[c]}\n```" for c in range(3)
)

md_lines = f"""\
# First-Layer Kernel 0 — INT8 Test Vectors

Generated by `scripts/export_first_layer_conv3x3_vhdl_vectors.py`.

---

## Checkpoint

| Field | Value |
|---|---|
| File | `{CHECKPOINT_PATH.relative_to(REPO_ROOT)}` |
| Epoch saved at | `{ckpt.get("epoch", "unknown")}` |
| Best val loss | `{ckpt.get("best_val_loss", "unknown")}` |
| `base_channels` | `{base_ch}` |

---

## Selected First-Layer Tensor

| Field | Value |
|---|---|
| State-dict key | `{conv_key}` |
| Full shape | `{list(w_tensor.shape)}` — `[out_ch, in_ch, ky, kx]` |
| Output channel extracted | `{OUT_CHAN}` |
| Direct Conv bias | `{"Yes: " + str(bias_float) if has_direct_bias else "No (Conv2d uses BatchNorm; bias_int32 = 0)"}` |

The first Conv2d in this U-Net is `Conv2d(3, {base_ch}, 3, padding=1)` with no direct
bias term (the bias is absorbed into the subsequent BatchNorm layer).  For this
integer-datapath test vector, `bias_int32 = 0`.

---

## Quantization

Symmetric per-tensor INT8 quantization for weights only:

```
scale_w = max(|w_float|) / 127 = {max_abs_w:.8f} / 127 = {scale_w:.8f}
w_int8  = round(w_float / scale_w), clipped to [-127, 127]
scale_x = 1.0  (toy input already integer-valued, fits in INT8)
```

### INT8 Kernel (output channel 0)

{w_int8_ch_str}

---

## Input Patch

5×5×3 toy patch, matching `tb_stream_conv3x3_3chan_cell.vhd`:

| Channel | Values |
|---|---|
| 0 | 1..25 row-major |
| 1 | 2 × channel 0 (2..50) |
| 2 | −1 × channel 0 (−1..−25) |

---

## Expected Outputs (3×3 valid convolution)

Using INT8 weights and INT8 input → INT32 accumulator:

```
y_int32[r,c] = bias_int32 + Σ_c Σ_ky Σ_kx  x_int8[c, r+ky, c+kx] * w_int8[c, ky, kx]
```

| row | col | output_index | y_int32 | y_float_reference | y_int32 rescaled (× scale_w) |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r} | {c} | {r*3+c} | {int(y_int32[r,c])} | {float(y_float_f32[r,c]):.4f} | {float(y_int32_rescaled[r,c]):.4f} |"
    for r in range(3) for c in range(3)
) + f"""

Max rescaling error vs float reference: **{max_rescale_err:.6f}**

---

## Connection to `stream_conv3x3_3chan_cell.vhd`

The VHDL cell implements exactly this computation for one output channel:

```
y = bias + dot(x_ch0_window, w_ch0) + dot(x_ch1_window, w_ch1) + dot(x_ch2_window, w_ch2)
```

To use these test vectors with the VHDL cell:

1. Load `w_int8` channel 0, 1, 2 values from the JSON as the VHDL `c0_w*`, `c1_w*`, `c2_w*` ports.
2. Drive the pixel stream with the toy patch (or adapt for real image data).
3. Drive `bias` with `bias_int32` (= 0 here).
4. After the 4-clock latency (3-clock dot product + 1-clock final summation), compare
   `valid_out` / `y` against the 9 `y_int32` values in this file.

---

## Limitations

- **Not full quantized inference.** BatchNorm parameters (`enc1.block.1.*`) are
  not folded into the weights or bias.  A production deployment would require
  BN folding before synthesis.
- **Single output channel only.** The full first layer produces `{base_ch}` output
  channels; this file covers output channel 0 only.
- **Symmetric per-tensor quantization** of weights only.  A calibrated per-channel
  or per-tensor scheme using representative input statistics would give lower error.
- **Toy input patch.** Real UAVSAR tiles have different dynamic range.  For
  production, the input would also need its own calibrated quantization scale.
- The INT32 output must be rescaled by `scale_w × scale_x` to convert back to
  approximate float units.
"""

with open(MD_PATH, "w") as f:
    f.write(md_lines)
print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Checkpoint       : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}")
print(f"Tensor key       : {conv_key}")
print(f"Tensor shape     : {list(w_tensor.shape)}")
print(f"Output channel   : {OUT_CHAN}")
print(f"scale_w          : {scale_w:.8f}")
print(f"bias_int32       : {bias_int32}")
print(f"INT32 outputs:")
for r in range(3):
    print(f"  row {r}: {y_int32[r].tolist()}")
print(f"\nOutput files:")
print(f"  {JSON_PATH.relative_to(REPO_ROOT)}")
print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
