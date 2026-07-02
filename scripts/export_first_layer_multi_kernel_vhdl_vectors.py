"""
export_first_layer_multi_kernel_vhdl_vectors.py

Generalization of export_first_layer_conv3x3_vhdl_vectors.py to multiple
output channels ("kernels") of the same first-layer Conv2d tensor.

Extracts output channels 0-3 (by default) of the first-layer 3x3 convolution
from the Alea-tuned U-Net checkpoint, quantizes each to INT8 using the same
symmetric per-tensor scheme as the original single-kernel script, applies
each kernel to the same canonical toy 5x5x3 input patch used by the existing
VHDL testbenches, and emits:

  - a JSON file with float/INT8 weights and INT32 golden outputs per kernel
  - a CSV file with per-kernel/per-output golden INT32 values
  - a Markdown summary
  - a VHDL-2008 package of constants for kernels 0-3 (weights + expected
    outputs + bias), for use by a self-checking testbench

This script is ANALYSIS ONLY -- it does not retrain, modify the model, or
touch any dataset splits. It reuses the exact quantization method and toy
input patch from export_first_layer_conv3x3_vhdl_vectors.py so the kernel-0
results here match that script's output bit-for-bit.

IMPORTANT LIMITATIONS (carried over from the single-kernel script):
  - This is NOT full quantized U-Net inference.
  - BatchNorm parameters are NOT folded into the weights or bias
    (enc1.block.0 is a Conv2d with no direct bias; BN follows separately).
  - Only a handful of the model's base_channels output channels are covered.
  - Weights use symmetric per-tensor INT8 quantization; the toy input is
    already integer-valued (scale_x = 1.0).

Output files
------------
hardware/vhdl_conv3x3/test_vectors/multi_kernel_first_layer/first_layer_kernels0_to3_int8.json
hardware/vhdl_conv3x3/test_vectors/multi_kernel_first_layer/first_layer_kernels0_to3_expected_outputs.csv
hardware/vhdl_conv3x3/test_vectors/multi_kernel_first_layer/first_layer_kernels0_to3_summary.md
hardware/vhdl_conv3x3/first_layer_kernels0_to3_pkg.vhd
"""

import json
import csv
import pathlib
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_CHANNELS = [0, 1, 2, 3]   # kernels to export by default

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
CONV_KEY_EXPECTED = "enc1.block.0.weight"

OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors" / "multi_kernel_first_layer"
PKG_PATH = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "first_layer_kernels0_to3_pkg.vhd"

KSTR = f"{OUT_CHANNELS[0]}_to{OUT_CHANNELS[-1]}"
JSON_PATH = OUT_DIR / f"first_layer_kernels{KSTR}_int8.json"
CSV_PATH  = OUT_DIR / f"first_layer_kernels{KSTR}_expected_outputs.csv"
MD_PATH   = OUT_DIR / f"first_layer_kernels{KSTR}_summary.md"

# ---------------------------------------------------------------------------
# 1. Locate checkpoint
# ---------------------------------------------------------------------------
print(f"[1] Looking for checkpoint: {CHECKPOINT_PATH}")
if not CHECKPOINT_PATH.exists():
    candidates = sorted(REPO_ROOT.glob("models/*fp2*.pt"))
    if not candidates:
        sys.exit("ERROR: No fp2 checkpoint found in models/")
    CHECKPOINT_PATH = candidates[0]
    print(f"    Fallback to: {CHECKPOINT_PATH}")
else:
    print("    Found.")

# ---------------------------------------------------------------------------
# 2. Load checkpoint, find first Conv2d weight tensor (same rule as before)
# ---------------------------------------------------------------------------
print("\n[2] Loading checkpoint ...")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)

state_dict = ckpt["model_state_dict"]
args       = ckpt.get("args", {})
base_ch    = getattr(args, "base_channels", None) or args.get("base_channels", "unknown")
print(f"    base_channels from checkpoint args: {base_ch}")
print(f"    Total state-dict keys: {len(state_dict)}")

conv_key = None
for key, val in state_dict.items():
    if val.ndim == 4 and val.shape[1] == 3 and val.shape[2] == 3 and val.shape[3] == 3:
        conv_key = key
        break

if conv_key is None:
    sys.exit("ERROR: Could not find a 4-D tensor with shape (*, 3, 3, 3) in state dict.")

if conv_key != CONV_KEY_EXPECTED:
    print(f"    WARNING: selected tensor key '{conv_key}' != expected '{CONV_KEY_EXPECTED}'")

w_tensor = state_dict[conv_key]  # shape: [out_ch, 3, 3, 3]
print(f"    Selected tensor key : {conv_key}")
print(f"    Tensor shape        : {list(w_tensor.shape)}")

n_out = w_tensor.shape[0]
for oc in OUT_CHANNELS:
    if oc >= n_out:
        sys.exit(f"ERROR: requested output channel {oc} but tensor only has {n_out} output channels")

bias_key = conv_key.replace(".weight", ".bias")
has_direct_bias = bias_key in state_dict
if has_direct_bias:
    print(f"    Direct Conv bias tensor found: {bias_key}")
else:
    print("    No direct Conv bias (layer uses BatchNorm; bias absorbed into BN). "
          "Using bias_int32 = 0 for all kernels' integer-datapath test vectors.")

# ---------------------------------------------------------------------------
# 3. Canonical 5x5x3 toy input patch (identical to single-kernel script)
# ---------------------------------------------------------------------------
print("\n[3] Building 5x5x3 toy input patch ...")
base = np.arange(1, 26, dtype=np.int8).reshape(5, 5)   # values 1..25
x_int8 = np.stack([base, (base * 2).astype(np.int8), (-base).astype(np.int8)], axis=0)
scale_x = 1.0
print(f"    Channel 0 range: {int(x_int8[0].min())} .. {int(x_int8[0].max())}")
print(f"    Channel 1 range: {int(x_int8[1].min())} .. {int(x_int8[1].max())}  (2x ch0)")
print(f"    Channel 2 range: {int(x_int8[2].min())} .. {int(x_int8[2].max())}  (-1x ch0)")


def quantize_symmetric_int8(w_float: np.ndarray):
    """Symmetric per-tensor INT8 quantization, same rule as the single-kernel script."""
    max_abs_w = float(np.abs(w_float).max())
    scale_w = 1.0 if max_abs_w == 0.0 else max_abs_w / 127.0
    w_int8 = np.clip(np.round(w_float / scale_w), -127, 127).astype(np.int8)
    return w_int8, scale_w, max_abs_w


def valid_conv_int32(x_i8: np.ndarray, w_i8: np.ndarray, bias_i32: int) -> np.ndarray:
    """5x5x3 -> 3x3 valid convolution, INT32 accumulation (matches VHDL cell)."""
    y = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            acc = bias_i32
            for ch in range(3):
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x_i8[ch, r + ky, c + kx]) * int(w_i8[ch, ky, kx])
            y[r, c] = acc
    return y.astype(np.int32)


def float_ref_conv(x_i8: np.ndarray, w_float: np.ndarray) -> np.ndarray:
    x_float = x_i8.astype(np.float32)
    y = np.zeros((3, 3), dtype=np.float64)
    for r in range(3):
        for c in range(3):
            for ch in range(3):
                for ky in range(3):
                    for kx in range(3):
                        y[r, c] += float(x_float[ch, r + ky, c + kx]) * float(w_float[ch, ky, kx])
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Process each requested kernel (output channel)
# ---------------------------------------------------------------------------
print(f"\n[4] Processing kernels {OUT_CHANNELS} ...")

kernels = {}   # oc -> dict of results
for oc in OUT_CHANNELS:
    w_float = w_tensor[oc].float().numpy()   # (3, 3, 3) = (in_ch, ky, kx)
    bias_float = float(state_dict[bias_key][oc]) if has_direct_bias else 0.0
    bias_int32 = 0  # BN not folded; direct Conv bias (if any) not used in integer datapath

    w_int8, scale_w, max_abs_w = quantize_symmetric_int8(w_float)
    y_int32 = valid_conv_int32(x_int8, w_int8, bias_int32)
    y_float = float_ref_conv(x_int8, w_float)
    y_int32_rescaled = y_int32.astype(np.float32) * (scale_w * scale_x)
    max_rescale_err = float(np.abs(y_float - y_int32_rescaled).max())

    kernels[oc] = dict(
        w_float=w_float, w_int8=w_int8, scale_w=scale_w, max_abs_w=max_abs_w,
        bias_float=bias_float, bias_int32=bias_int32,
        y_int32=y_int32, y_float=y_float, y_int32_rescaled=y_int32_rescaled,
        max_rescale_err=max_rescale_err,
    )

    print(f"\n    --- Kernel (output channel) {oc} ---")
    print(f"    scale_w = {scale_w:.8f}")
    print(f"    INT8 weights ch0:\n{w_int8[0]}")
    print(f"    INT8 weights ch1:\n{w_int8[1]}")
    print(f"    INT8 weights ch2:\n{w_int8[2]}")
    print(f"    INT32 outputs (3x3):\n{y_int32}")
    print(f"    Max rescale error vs float reference: {max_rescale_err:.6f}")

# ---------------------------------------------------------------------------
# 5. Write JSON
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[5] Writing output files to {OUT_DIR} ...")

data = {
    "checkpoint_path": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
    "selected_tensor_key": conv_key,
    "selected_output_channels": OUT_CHANNELS,
    "tensor_shape": list(w_tensor.shape),
    "base_channels_from_args": str(base_ch),
    "quantization": {
        "scheme": "symmetric per-tensor INT8 for weights, per-kernel scale",
        "input_scale": scale_x,
        "bias_note": (
            "Conv2d has no direct bias (layer uses BatchNorm). "
            "bias_int32=0 used for every kernel's integer-datapath test vector. "
            "Full quantized inference would require folding BN params into weights/bias."
        ),
    },
    "input_patch_int8": {f"channel_{c}": x_int8[c].tolist() for c in range(3)},
    "input_patch_description": (
        "5x5 toy patch: ch0=1..25 row-major, ch1=2xch0, ch2=-1xch0. "
        "Matches tb_stream_conv3x3_3chan_cell.vhd stimulus."
    ),
    "kernels": {},
    "notes": [
        "This validates integer datapath arithmetic for the VHDL stream_conv3x3_3chan_cell "
        "across multiple output channels of the first Conv2d layer.",
        "It is NOT a full quantized U-Net inference pipeline.",
        "Conv2d bias is zero in the integer datapath because the model uses Conv -> BatchNorm "
        "(no bias) for this layer; BN running_mean/var and gamma/beta are not folded in.",
        "For production deployment, BN folding would be required before VHDL synthesis.",
        f"Checkpoint base_channels={base_ch}; the full first layer would need "
        f"{base_ch} instances of stream_conv3x3_3chan_cell for all output channels. "
        f"This export covers only channels {OUT_CHANNELS}.",
        "Symmetric per-tensor INT8 quantization of weights only (scale computed per kernel); "
        "input uses scale=1.0.",
    ],
}

for oc, k in kernels.items():
    data["kernels"][str(oc)] = {
        "output_channel": oc,
        "weight_max_abs": k["max_abs_w"],
        "scale_w": k["scale_w"],
        "float_weights": {f"channel_{c}": k["w_float"][c].tolist() for c in range(3)},
        "int8_weights": {f"channel_{c}": k["w_int8"][c].tolist() for c in range(3)},
        "bias_float": k["bias_float"],
        "bias_int32": k["bias_int32"],
        "int32_expected_outputs_3x3": k["y_int32"].tolist(),
        "float_reference_outputs_3x3": k["y_float"].tolist(),
        "int32_rescaled_to_float_3x3": k["y_int32_rescaled"].tolist(),
        "max_rescale_error_vs_float_reference": k["max_rescale_err"],
    }

with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"    Written: {JSON_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 6. Write CSV
# ---------------------------------------------------------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel", "row", "col", "output_index", "y_int32",
                      "y_float_reference", "y_int32_rescaled_to_float"])
    for oc, k in kernels.items():
        idx = 0
        for r in range(3):
            for c in range(3):
                writer.writerow([
                    oc, r, c, idx,
                    int(k["y_int32"][r, c]),
                    f"{float(k['y_float'][r, c]):.6f}",
                    f"{float(k['y_int32_rescaled'][r, c]):.6f}",
                ])
                idx += 1
print(f"    Written: {CSV_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 7. Write Markdown summary
# ---------------------------------------------------------------------------
kernel_sections = []
for oc, k in kernels.items():
    w_int8_ch_str = "\n".join(
        f"**Channel {c}:**\n```\n{k['w_int8'][c]}\n```" for c in range(3)
    )
    out_rows = "\n".join(
        f"| {r} | {c} | {r*3+c} | {int(k['y_int32'][r,c])} | "
        f"{float(k['y_float'][r,c]):.4f} | {float(k['y_int32_rescaled'][r,c]):.4f} |"
        for r in range(3) for c in range(3)
    )
    kernel_sections.append(f"""\
## Kernel (output channel) {oc}

`scale_w = max(|w_float|) / 127 = {k['max_abs_w']:.8f} / 127 = {k['scale_w']:.8f}`

{w_int8_ch_str}

### Expected Outputs (3x3 valid convolution)

| row | col | output_index | y_int32 | y_float_reference | y_int32 rescaled (x scale_w) |
|---|---|---|---|---|---|
{out_rows}

Max rescaling error vs float reference: **{k['max_rescale_err']:.6f}**
""")

md_lines = f"""\
# First-Layer Kernels {OUT_CHANNELS[0]}-{OUT_CHANNELS[-1]} -- INT8 Test Vectors

Generated by `scripts/export_first_layer_multi_kernel_vhdl_vectors.py`.

This is the generalization of `first_layer_kernel0_summary.md` to multiple
output channels of the same first-layer Conv2d tensor. Kernel 0 here is
numerically identical to the original single-kernel export.

---

## Checkpoint

| Field | Value |
|---|---|
| File | `{CHECKPOINT_PATH.relative_to(REPO_ROOT)}` |
| Epoch saved at | `{ckpt.get("epoch", "unknown")}` |
| Best val loss | `{ckpt.get("best_val_loss", "unknown")}` |
| `base_channels` | `{base_ch}` |

## Selected First-Layer Tensor

| Field | Value |
|---|---|
| State-dict key | `{conv_key}` |
| Full shape | `{list(w_tensor.shape)}` -- `[out_ch, in_ch, ky, kx]` |
| Output channels extracted | `{OUT_CHANNELS}` |
| Direct Conv bias | `{"Yes (not used in integer datapath)" if has_direct_bias else "No (Conv2d uses BatchNorm; bias_int32 = 0)"}` |

The first Conv2d in this U-Net is `Conv2d(3, {base_ch}, 3, padding=1)` with no direct
bias term (the bias is absorbed into the subsequent BatchNorm layer). For every
kernel's integer-datapath test vector, `bias_int32 = 0`.

## Input Patch

5x5x3 toy patch, matching `tb_stream_conv3x3_3chan_cell.vhd`:

| Channel | Values |
|---|---|
| 0 | 1..25 row-major |
| 1 | 2 x channel 0 (2..50) |
| 2 | -1 x channel 0 (-1..-25) |

---

{"".join(kernel_sections)}
---

## Connection to `stream_conv3x3_3chan_cell.vhd`

Each kernel above drives exactly one instantiation (or one time-multiplexed pass)
of `stream_conv3x3_3chan_cell`, which computes one output channel of the first
Conv2d layer:

```
y = bias + dot(x_ch0_window, w_ch0) + dot(x_ch1_window, w_ch1) + dot(x_ch2_window, w_ch2)
```

See `hardware/vhdl_conv3x3/first_layer_kernels0_to3_pkg.vhd` for VHDL constants
and `hardware/vhdl_conv3x3/tb_stream_conv3x3_3chan_kernels0_to3.vhd` for the
self-checking testbench that runs all {len(OUT_CHANNELS)} kernels sequentially
through the same DUT instance.

## Limitations

- **Not full quantized inference.** BatchNorm parameters (`enc1.block.1.*`) are
  not folded into the weights or bias. A production deployment would require
  BN folding before synthesis.
- **Only {len(OUT_CHANNELS)} of {base_ch} output channels covered.** The full first layer
  produces `{base_ch}` output channels; this file covers channels {OUT_CHANNELS} only.
- **Symmetric per-tensor quantization** of weights only (scale computed per kernel).
  A calibrated per-channel or per-tensor scheme using representative input statistics
  would give lower error.
- **Toy input patch.** Real UAVSAR tiles have different dynamic range. For
  production, the input would also need its own calibrated quantization scale.
- The INT32 output must be rescaled by `scale_w x scale_x` (per kernel) to convert
  back to approximate float units.
"""

with open(MD_PATH, "w") as f:
    f.write(md_lines)
print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 8. Write VHDL-2008 package of constants
# ---------------------------------------------------------------------------
print(f"\n[6] Writing VHDL package to {PKG_PATH} ...")


def vhdl_int8_array(vals) -> str:
    """9 flattened row-major INT8 weight values -> VHDL aggregate for int8_kernel_t."""
    return "(" + ", ".join(str(int(v)) for v in vals) + ")"


pkg_lines = []
pkg_lines.append(f"""\
-- first_layer_kernels0_to3_pkg.vhd
-- VHDL-2008 package of INT8 weight and INT32 golden-output constants for
-- first-layer U-Net convolution kernels {OUT_CHANNELS[0]}-{OUT_CHANNELS[-1]}.
--
-- Generated by: scripts/export_first_layer_multi_kernel_vhdl_vectors.py
-- Source data : {JSON_PATH.relative_to(REPO_ROOT)}
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}
-- Tensor key : {conv_key}  (shape {list(w_tensor.shape)})
-- Output chs : {OUT_CHANNELS}
--
-- ---- Quantization -------------------------------------------------------
-- Scheme : symmetric per-tensor INT8 for weights, scale computed per kernel
--          scale_w = max(|w_float|) / 127
-- Input  : canonical 5x5x3 toy patch (ch0=1..25, ch1=2xch0, ch2=-1xch0),
--          scale_x = 1.0 (already integer-valued)
--
-- ---- Bias -----------------------------------------------------------------
-- The first Conv2d in this U-Net has NO direct bias (Conv2d -> BatchNorm;
-- BN params are NOT folded here). Therefore bias_int32 = 0 for every kernel.
--
-- ---- What this validates --------------------------------------------------
-- This package supplies weight and golden-output constants so a testbench can
-- confirm the VHDL stream_conv3x3_3chan_cell integer datapath produces the
-- SAME INT32 values as the Python golden-vector script when driven with real
-- trained-model INT8 weights, across kernels {OUT_CHANNELS[0]}-{OUT_CHANNELS[-1]}.
-- It does NOT validate full quantized U-Net inference (BatchNorm is not applied),
-- and it does NOT cover the full set of {base_ch} first-layer output channels.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package first_layer_kernels0_to3_pkg is

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;

    -- 9 INT32 expected outputs per kernel, row-major (output_index 0..8)
    type int32_outputs_t is array (0 to 8) of integer;

    constant NUM_KERNELS : integer := {len(OUT_CHANNELS)};
    type kernel_id_t is array (0 to NUM_KERNELS - 1) of integer;
    constant KERNEL_IDS : kernel_id_t := ({", ".join(str(oc) for oc in OUT_CHANNELS)});

""")

for oc, k in kernels.items():
    w = k["w_int8"]
    y = k["y_int32"]
    pkg_lines.append(f"""\
    -- ==================================================================
    -- Kernel (output channel) {oc}
    --   scale_w = {k['scale_w']:.8f}
    -- ==================================================================
    constant KERNEL{oc}_CH0_W : int8_kernel_t := {vhdl_int8_array(w[0].flatten())};
    constant KERNEL{oc}_CH1_W : int8_kernel_t := {vhdl_int8_array(w[1].flatten())};
    constant KERNEL{oc}_CH2_W : int8_kernel_t := {vhdl_int8_array(w[2].flatten())};
    constant KERNEL{oc}_BIAS  : integer := {k['bias_int32']};
    constant KERNEL{oc}_EXPECTED : int32_outputs_t := (
        {y[0,0]}, {y[0,1]}, {y[0,2]},
        {y[1,0]}, {y[1,1]}, {y[1,2]},
        {y[2,0]}, {y[2,1]}, {y[2,2]}
    );

""")

pkg_lines.append("end package first_layer_kernels0_to3_pkg;\n")

with open(PKG_PATH, "w") as f:
    f.write("".join(pkg_lines))
print(f"    Written: {PKG_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Checkpoint       : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}")
print(f"Tensor key       : {conv_key}")
print(f"Tensor shape     : {list(w_tensor.shape)}")
print(f"Output channels  : {OUT_CHANNELS}")
for oc, k in kernels.items():
    print(f"\nKernel {oc}: scale_w={k['scale_w']:.8f}  bias_int32={k['bias_int32']}")
    for r in range(3):
        print(f"  row {r}: {k['y_int32'][r].tolist()}")
print("\nOutput files:")
print(f"  {JSON_PATH.relative_to(REPO_ROOT)}")
print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
print(f"  {PKG_PATH.relative_to(REPO_ROOT)}")
