"""
generate_kernel0_bn_relu_fixed_point_vectors.py

Generates a GENUINE fixed-point (integer-only) golden reference for a
VHDL Conv -> folded-BatchNorm-bias/scale -> ReLU proof-of-concept, for
kernel (output channel) 0 of the first U-Net Conv2d layer only.

This is a NEW, separate script from
scripts/analyze_first_conv_bn_relu_fidelity.py. That existing script is a
Python NUMERICAL FIDELITY study: it rescales the INT32 convolution result
back to FLOAT (`int32_accum * scale_x * scale_w_folded + b_folded`, all in
float64) before comparing against the true PyTorch Conv-BN-ReLU output.
It does NOT define any fixed-point (integer-only) arithmetic, so it cannot
be used directly as a bit-exact VHDL golden reference -- VHDL has no
floating-point rescale step. This script instead performs the ENTIRE
rescale + bias + ReLU computation using explicit fixed-point (Q-format)
integer arithmetic, matching exactly what the new VHDL module
stream_conv3x3_3chan_kernel0_bn_relu.vhd computes.

---- Relationship to the existing BatchNorm folding -------------------------
Uses the SAME BatchNorm folding formula, SAME checkpoint, SAME weight
tensor key, and SAME per-output-channel symmetric INT8 folded-weight
quantization convention as scripts/analyze_first_conv_bn_relu_fidelity.py,
restricted to kernel (output channel) 0 only:
    scale_bn[oc]  = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc]  = w[oc] * scale_bn[oc]
    b_folded[oc]  = beta[oc] - running_mean[oc] * scale_bn[oc]
    scale_w_folded[oc] = max(|w_folded[oc]|) / 127
    w_folded_int8[oc]  = round(w_folded[oc] / scale_w_folded[oc]), clipped

---- Deliberate simplification: input activation scale ----------------------
scripts/analyze_first_conv_bn_relu_fidelity.py computes a PER-PATCH
activation scale (scale_x = max(|x_patch|)/127) for arbitrary real-valued
patches. This script instead reuses the same convention already used by
EVERY existing VHDL testbench and every existing multi-kernel export
script in this repo (scale_x = 1.0 exactly) for the canonical 5x5x3 toy
patch. This is exact and lossless here because the toy patch's maximum
magnitude is 50 (channel 1 = 2x channel 0, max 2*25=50), which fits inside
INT8 range [-128, 127] with no rounding -- so scale_x=1.0 introduces zero
quantization error for THIS SPECIFIC toy patch, and keeps this script's
raw INT32 convolution accumulation bit-identical to every other VHDL
testbench's pixel stimulus in this repo (pixel_c0=1..25, pixel_c1=2..50,
pixel_c2=-1..-25, driven as literal INT8 values). This is NOT the general
per-patch scale_x convention used for real (non-integer, larger-magnitude)
UAVSAR tiles in the fidelity study, and this script does not claim it is.

---- Fixed-point (Q-format) arithmetic, exactly what VHDL computes ----------
Let F = FRAC_BITS = 16 (fractional bits).
    SCALE_FX = round(scale_w_folded[0] * scale_x * 2^F)     -- unsigned constant
    BIAS_FX  = round(b_folded[0]                * 2^F)      -- signed constant
    raw_conv_int32   = sum over 3 channels, 9 taps of (x_int8 * w_folded_int8)  -- EXACT integer
    product_fx       = raw_conv_int32 * SCALE_FX             -- EXACT integer, Q.F format
    biased_fx         = product_fx + BIAS_FX                  -- EXACT integer, Q.F format
    relu_fx            = biased_fx if biased_fx > 0 else 0     -- EXACT integer, Q.F format
Because raw_conv_int32 has 0 fractional bits (integer) and SCALE_FX has F
fractional bits, their product is automatically in Q.F format with no
shift needed; BIAS_FX is generated directly in the same Q.F format, so no
shift is needed to add it either. This is why the VHDL datapath (see
stream_conv3x3_3chan_kernel0_bn_relu.vhd) contains no runtime shift
operations -- only one multiply, one add, and one ReLU compare-and-select,
all on constants/values already in a consistent Q.F fixed-point format.

relu_fx / 2^F (float) is also reported for human-readable comparison
against the true PyTorch Conv-BN-ReLU reference, but the VHDL testbench
compares against the EXACT INTEGER relu_fx value, not this float
conversion -- avoiding any silent float-vs-fixed-point comparison.

---- Scope --------------------------------------------------------------
- Kernel (output channel) 0 ONLY.
- Canonical 5x5x3 toy input, 9 valid output positions (5x5 -> 3x3), same
  as every other VHDL testbench in this repo.
- This is ANALYSIS/GOLDEN-VECTOR-GENERATION ONLY -- it does not retrain,
  modify the model, or touch any dataset splits. It does NOT imply that
  VHDL implements padding, pooling, downstream layers, or the full model
  pipeline.

---- Outputs -----------------------------------------------------------
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_vectors.json
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_vectors.csv
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_summary.md
hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_pkg.vhd
"""

import csv
import json
import pathlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
CONV_KEY_EXPECTED = "enc1.block.0.weight"
BN_PREFIX_EXPECTED = "enc1.block.1"
BN_EPS = 1e-5   # torch.nn.BatchNorm2d default; verified in the existing fidelity study

KERNEL_ID = 0
FRAC_BITS = 16   # fixed-point fractional bits (Q.F format) for scale/bias/output

OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors" / "conv_bn_relu_kernel0"
JSON_PATH = OUT_DIR / "kernel0_bn_relu_fixed_point_vectors.json"
CSV_PATH  = OUT_DIR / "kernel0_bn_relu_fixed_point_vectors.csv"
MD_PATH   = OUT_DIR / "kernel0_bn_relu_fixed_point_summary.md"
PKG_PATH  = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "first_conv_bn_relu_kernel0_pkg.vhd"

# ---------------------------------------------------------------------------
# 1. Load checkpoint, extract kernel 0's Conv2d weight + BatchNorm params
# ---------------------------------------------------------------------------
print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]

w_tensor = state_dict[CONV_KEY_EXPECTED]           # [32, 3, 3, 3]
w0_float = w_tensor[KERNEL_ID].float().numpy()     # (3, 3, 3) = (in_ch, ky, kx)
print(f"    Tensor key   : {CONV_KEY_EXPECTED}")
print(f"    Kernel shape : {list(w0_float.shape)}  (kernel {KERNEL_ID} only)")

bn_gamma = state_dict[f"{BN_PREFIX_EXPECTED}.weight"].float().numpy()[KERNEL_ID]
bn_beta = state_dict[f"{BN_PREFIX_EXPECTED}.bias"].float().numpy()[KERNEL_ID]
bn_mean = state_dict[f"{BN_PREFIX_EXPECTED}.running_mean"].float().numpy()[KERNEL_ID]
bn_var = state_dict[f"{BN_PREFIX_EXPECTED}.running_var"].float().numpy()[KERNEL_ID]
print(f"    BatchNorm[{KERNEL_ID}]: gamma={bn_gamma:.6f} beta={bn_beta:.6f} "
      f"running_mean={bn_mean:.6f} running_var={bn_var:.6f}")

# ---------------------------------------------------------------------------
# 2. Fold BatchNorm into kernel 0's weights/bias (same formula as the
#    existing fidelity study)
# ---------------------------------------------------------------------------
print("\n[2] Folding BatchNorm into kernel 0's weights/bias ...")
scale_bn = float(bn_gamma / np.sqrt(bn_var + BN_EPS))
w0_folded = w0_float * scale_bn                     # (3, 3, 3)
b0_folded = float(bn_beta - bn_mean * scale_bn)
print(f"    scale_bn[0] = {scale_bn:.6f}")
print(f"    b_folded[0] = {b0_folded:.6f}")

# ---------------------------------------------------------------------------
# 3. Symmetric per-channel INT8 quantization of the FOLDED kernel-0 weights
#    (identical formula to analyze_first_conv_bn_relu_fidelity.py)
# ---------------------------------------------------------------------------
print("\n[3] Quantizing folded kernel-0 weights to INT8 (symmetric) ...")
max_abs_w = float(np.abs(w0_folded).max())
scale_w_folded = max_abs_w / 127.0
w0_folded_int8 = np.clip(np.round(w0_folded / scale_w_folded), -127, 127).astype(np.int8)
print(f"    scale_w_folded[0] = {scale_w_folded:.8f}")
print(f"    INT8 weights ch0:\n{w0_folded_int8[0]}")
print(f"    INT8 weights ch1:\n{w0_folded_int8[1]}")
print(f"    INT8 weights ch2:\n{w0_folded_int8[2]}")

# ---------------------------------------------------------------------------
# 4. Canonical 5x5x3 toy input patch -- SAME literal INT8 pixel values as
#    every other VHDL testbench in this repo (scale_x = 1.0 exactly; see
#    module docstring for why this is lossless for this specific patch).
# ---------------------------------------------------------------------------
print("\n[4] Building canonical 5x5x3 toy input patch (scale_x = 1.0, exact) ...")
base = np.arange(1, 26, dtype=np.int64).reshape(5, 5)
x_int8 = np.stack([base, base * 2, -base], axis=0).astype(np.int64)   # (3, 5, 5), exact
scale_x = 1.0
print(f"    Channel 0 range: {int(x_int8[0].min())} .. {int(x_int8[0].max())}")
print(f"    Channel 1 range: {int(x_int8[1].min())} .. {int(x_int8[1].max())}  (2x ch0)")
print(f"    Channel 2 range: {int(x_int8[2].min())} .. {int(x_int8[2].max())}  (-1x ch0)")
assert int(np.abs(x_int8).max()) <= 127, "toy patch does not fit INT8 -- scale_x=1.0 invalid"

# ---------------------------------------------------------------------------
# 5. Exact INT32 valid (no-padding) 3x3 convolution accumulation, kernel 0
#    only, using the FOLDED INT8 weights (NOT the raw unfolded weights
#    used by every prior VHDL prototype in this repo).
# ---------------------------------------------------------------------------
print("\n[5] Computing exact INT32 raw convolution accumulation (folded weights) ...")


def valid_conv_int32(x_i8: np.ndarray, w_i8: np.ndarray) -> np.ndarray:
    """5x5x3 -> 3x3 valid convolution, exact INT accumulation (matches VHDL)."""
    y = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            acc = 0
            for ch in range(3):
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x_i8[ch, r + ky, c + kx]) * int(w_i8[ch, ky, kx])
            y[r, c] = acc
    return y


raw_conv_int32 = valid_conv_int32(x_int8, w0_folded_int8)   # (3, 3), exact integers
print(f"    raw_conv_int32 (3x3):\n{raw_conv_int32}")

# ---------------------------------------------------------------------------
# 6. Fixed-point (Q.F) constants and datapath, EXACTLY what VHDL computes
# ---------------------------------------------------------------------------
print(f"\n[6] Computing Q.{FRAC_BITS} fixed-point constants ...")
combined_scale = scale_x * scale_w_folded   # = scale_w_folded since scale_x = 1.0
SCALE_FX = int(round(combined_scale * (2 ** FRAC_BITS)))
BIAS_FX = int(round(b0_folded * (2 ** FRAC_BITS)))
print(f"    combined_scale (scale_x * scale_w_folded) = {combined_scale:.8f}")
print(f"    SCALE_FX (unsigned, Q.{FRAC_BITS})  = {SCALE_FX}")
print(f"    BIAS_FX  (signed,   Q.{FRAC_BITS})  = {BIAS_FX}")

product_fx = raw_conv_int32.astype(np.int64) * SCALE_FX   # exact integer, Q.F
biased_fx = product_fx + BIAS_FX                            # exact integer, Q.F
relu_fx = np.maximum(biased_fx, 0)                          # exact integer, Q.F

max_abs_biased_fx = int(np.abs(biased_fx).max())
bits_needed = max_abs_biased_fx.bit_length() + 1   # +1 for sign
print(f"    biased_fx (3x3, Q.{FRAC_BITS}):\n{biased_fx}")
print(f"    relu_fx   (3x3, Q.{FRAC_BITS}):\n{relu_fx}")
print(f"    max |biased_fx| = {max_abs_biased_fx}  "
      f"(needs >= {bits_needed} signed bits; VHDL uses 48)")

# Human-readable float conversion (NOT used for the VHDL golden comparison,
# informative only -- confirms this is in the right ballpark vs. the
# existing fidelity study's float-rescaled result for kernel 0).
relu_float_equiv = relu_fx.astype(np.float64) / (2 ** FRAC_BITS)
print(f"    relu_fx / 2^{FRAC_BITS} (float, informative only):\n{relu_float_equiv}")

# ---------------------------------------------------------------------------
# 7. Write JSON + CSV test vectors
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[7] Writing test vectors to {OUT_DIR} ...")

positions = []
idx = 0
for r in range(3):
    for c in range(3):
        positions.append({
            "output_index": idx,
            "row": r,
            "col": c,
            "raw_conv_int32": int(raw_conv_int32[r, c]),
            "product_fx": int(product_fx[r, c]),
            "biased_fx": int(biased_fx[r, c]),
            "relu_fx": int(relu_fx[r, c]),
            "relu_float_equiv": float(relu_float_equiv[r, c]),
        })
        idx += 1

data = {
    "checkpoint_path": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
    "kernel_id": KERNEL_ID,
    "conv_tensor_key": CONV_KEY_EXPECTED,
    "bn_prefix": BN_PREFIX_EXPECTED,
    "bn_eps": BN_EPS,
    "scale_bn": scale_bn,
    "b_folded": b0_folded,
    "scale_w_folded": scale_w_folded,
    "w_folded_int8": {f"channel_{c}": w0_folded_int8[c].flatten().tolist() for c in range(3)},
    "scale_x": scale_x,
    "fixed_point_format": f"Q.{FRAC_BITS} (signed, {FRAC_BITS} fractional bits)",
    "frac_bits": FRAC_BITS,
    "scale_fx": SCALE_FX,
    "bias_fx": BIAS_FX,
    "input_patch_description": (
        "5x5 toy patch: ch0=1..25 row-major, ch1=2xch0, ch2=-1xch0. "
        "scale_x=1.0 exact (matches every other VHDL testbench's literal "
        "INT8 pixel stimulus in this repo)."
    ),
    "positions": positions,
    "notes": [
        "This is a fixed-point (integer-only) golden reference, distinct "
        "from scripts/analyze_first_conv_bn_relu_fidelity.py's float "
        "rescale-based fidelity study.",
        "Uses FOLDED (BatchNorm-absorbed) INT8 weights for kernel 0, NOT "
        "the raw unfolded weights used by every prior VHDL prototype in "
        "this repo.",
        "relu_fx is the EXACT integer VHDL golden output (Q.16 fixed-point).",
        "relu_float_equiv is for human-readable comparison only; the VHDL "
        "testbench compares against the exact integer relu_fx, not this "
        "float value.",
    ],
}
with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"    Written: {JSON_PATH.relative_to(REPO_ROOT)}")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["output_index", "row", "col", "raw_conv_int32",
                      "product_fx", "biased_fx", "relu_fx", "relu_float_equiv"])
    for p in positions:
        writer.writerow([p["output_index"], p["row"], p["col"], p["raw_conv_int32"],
                          p["product_fx"], p["biased_fx"], p["relu_fx"],
                          f"{p['relu_float_equiv']:.8f}"])
print(f"    Written: {CSV_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 8. Write Markdown summary
# ---------------------------------------------------------------------------
pos_rows = "\n".join(
    f"| {p['output_index']} | {p['row']} | {p['col']} | {p['raw_conv_int32']} | "
    f"{p['product_fx']} | {p['biased_fx']} | **{p['relu_fx']}** | {p['relu_float_equiv']:.6f} |"
    for p in positions
)
md_lines = f"""\
# Kernel 0 Folded Conv-BN-ReLU -- Fixed-Point Test Vectors

Generated by `scripts/generate_kernel0_bn_relu_fixed_point_vectors.py`.

This is a **fixed-point (integer-only) golden reference**, distinct from
`scripts/analyze_first_conv_bn_relu_fidelity.py`'s float-rescale-based
numerical fidelity study. It performs the entire rescale + bias + ReLU
computation using Q.{FRAC_BITS} fixed-point integer arithmetic -- the exact
arithmetic the new VHDL module `stream_conv3x3_3chan_kernel0_bn_relu.vhd`
computes.

## Checkpoint / kernel

| Field | Value |
|---|---|
| Checkpoint | `{CHECKPOINT_PATH.relative_to(REPO_ROOT)}` |
| Conv tensor key | `{CONV_KEY_EXPECTED}` |
| Kernel (output channel) | {KERNEL_ID} |
| BatchNorm prefix | `{BN_PREFIX_EXPECTED}` |
| BN eps | {BN_EPS} |

## BatchNorm folding (kernel 0 only)

```
scale_bn[0]  = gamma[0] / sqrt(running_var[0] + eps) = {scale_bn:.6f}
b_folded[0]  = beta[0] - running_mean[0] * scale_bn[0] = {b0_folded:.6f}
```

## Folded weight quantization (symmetric INT8)

`scale_w_folded[0] = max(|w_folded[0]|) / 127 = {scale_w_folded:.8f}`

INT8 folded weights (row-major, top-left to bottom-right):

**Channel 0:** `{w0_folded_int8[0].flatten().tolist()}`
**Channel 1:** `{w0_folded_int8[1].flatten().tolist()}`
**Channel 2:** `{w0_folded_int8[2].flatten().tolist()}`

## Input patch

5x5x3 toy patch (identical to every other VHDL testbench in this repo):

| Channel | Values |
|---|---|
| 0 | 1..25 row-major |
| 1 | 2 x channel 0 (2..50) |
| 2 | -1 x channel 0 (-1..-25) |

`scale_x = 1.0` exactly (lossless for this integer-valued patch; see
script docstring for why this differs from the general per-patch scale_x
used in the float fidelity study).

## Fixed-point format

Q.{FRAC_BITS} (signed, {FRAC_BITS} fractional bits): `real_value ~= fixed_value / 2^{FRAC_BITS}`.

| Constant | Value | Meaning |
|---|---:|---|
| `SCALE_FX` | {SCALE_FX} | `round(scale_x * scale_w_folded[0] * 2^{FRAC_BITS})`, unsigned |
| `BIAS_FX` | {BIAS_FX} | `round(b_folded[0] * 2^{FRAC_BITS})`, signed |

Datapath (exact integers, no floating point, no runtime shifts):

```
raw_conv_int32 = sum(x_int8 * w_folded_int8)     -- exact INT32 accumulation
product_fx      = raw_conv_int32 * SCALE_FX        -- exact, already Q.{FRAC_BITS}
biased_fx        = product_fx + BIAS_FX              -- exact, Q.{FRAC_BITS}
relu_fx           = biased_fx if biased_fx > 0 else 0  -- exact, Q.{FRAC_BITS}
```

Max `|biased_fx|` observed: {max_abs_biased_fx} (needs >= {bits_needed} signed bits;
the VHDL module uses a 48-bit signed datapath for headroom).

## Expected outputs (9 valid positions, row-major)

| idx | row | col | raw_conv_int32 | product_fx | biased_fx | relu_fx (golden) | relu_fx / 2^{FRAC_BITS} (informative) |
|---|---|---|---:|---:|---:|---:|---:|
{pos_rows}

**The VHDL testbench compares against the exact integer `relu_fx` column,
not the float-equivalent column.**

## Limitations

- Kernel 0 only, not all 32 output channels.
- `scale_x = 1.0` is exact for this specific toy patch only (see script
  docstring); this is not the general per-patch activation scale used for
  real UAVSAR tiles in `analyze_first_conv_bn_relu_fidelity.py`.
- This is a golden-vector generation script, not a VHDL simulation; GHDL
  results are reported separately.
"""
with open(MD_PATH, "w") as f:
    f.write(md_lines)
print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 9. Write VHDL-2008 package of constants
# ---------------------------------------------------------------------------
print(f"\n[8] Writing VHDL package to {PKG_PATH} ...")


def vhdl_int8_array(vals) -> str:
    return "(" + ", ".join(str(int(v)) for v in vals) + ")"


relu_fx_flat = relu_fx.flatten().tolist()

pkg_text = f"""\
-- first_conv_bn_relu_kernel0_pkg.vhd
-- VHDL-2008 package of INT8 folded weights, Q.{FRAC_BITS} fixed-point
-- SCALE_FX/BIAS_FX constants, and Q.{FRAC_BITS} fixed-point golden output
-- constants for the kernel-0 folded Conv-BN-ReLU proof-of-concept.
--
-- Generated by: scripts/generate_kernel0_bn_relu_fixed_point_vectors.py
-- Source data : hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_vectors.json
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}
-- Tensor key : {CONV_KEY_EXPECTED} (kernel 0 only), folded with
--              {BN_PREFIX_EXPECTED} (BatchNorm2d, eval-mode running stats)
--
-- ---- BatchNorm folding ----------------------------------------------------
-- scale_bn[0] = gamma[0] / sqrt(running_var[0] + eps) = {scale_bn:.6f}
-- b_folded[0] = beta[0] - running_mean[0] * scale_bn[0] = {b0_folded:.6f}
-- (eps = {BN_EPS}, torch.nn.BatchNorm2d default)
--
-- ---- Folded weight quantization ------------------------------------------
-- scale_w_folded[0] = max(|w_folded[0]|) / 127 = {scale_w_folded:.8f}
-- symmetric per-channel INT8, SAME convention as
-- scripts/analyze_first_conv_bn_relu_fidelity.py, applied to kernel 0 only.
--
-- ---- Fixed-point format (Q.{FRAC_BITS}, signed, {FRAC_BITS} fractional bits) --------------
-- real_value ~= fixed_value / 2^{FRAC_BITS}
-- SCALE_FX = round(scale_x * scale_w_folded[0] * 2^{FRAC_BITS}), scale_x = 1.0 exact
--            for this toy patch (see script docstring)
-- BIAS_FX  = round(b_folded[0] * 2^{FRAC_BITS})
-- Datapath: product_fx = raw_conv_int32 * SCALE_FX (already Q.{FRAC_BITS}, no shift);
--           biased_fx = product_fx + BIAS_FX (same format, no shift);
--           relu_fx = biased_fx if biased_fx > 0 else 0.
--
-- ---- What this validates --------------------------------------------------
-- This package supplies the folded INT8 weights, the fixed-point scale/bias
-- constants, and the exact Q.{FRAC_BITS} fixed-point expected outputs so a
-- testbench can confirm stream_conv3x3_3chan_kernel0_bn_relu produces the
-- SAME fixed-point integer values as this Python script for kernel 0 on
-- the canonical 5x5x3 toy input. It does NOT validate the other 31 output
-- channels, does NOT implement padding, and does NOT claim board-tested
-- or measured performance.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package first_conv_bn_relu_kernel0_pkg is

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;

    -- FOLDED (BatchNorm-absorbed) INT8 weights, kernel 0
    constant K0_FOLDED_CH0_W : int8_kernel_t := {vhdl_int8_array(w0_folded_int8[0].flatten())};
    constant K0_FOLDED_CH1_W : int8_kernel_t := {vhdl_int8_array(w0_folded_int8[1].flatten())};
    constant K0_FOLDED_CH2_W : int8_kernel_t := {vhdl_int8_array(w0_folded_int8[2].flatten())};

    -- Fixed-point Q.{FRAC_BITS} constants
    constant FRAC_BITS : integer := {FRAC_BITS};
    constant SCALE_FX  : integer := {SCALE_FX};   -- unsigned, Q.{FRAC_BITS}
    constant BIAS_FX   : integer := {BIAS_FX};   -- signed,   Q.{FRAC_BITS}

    -- 9 Q.{FRAC_BITS} fixed-point expected outputs (row-major, output_index 0..8),
    -- EXACT integers -- this is what the VHDL testbench compares against.
    type fx_outputs_t is array (0 to 8) of integer;
    constant K0_BN_RELU_EXPECTED_FX : fx_outputs_t := (
        {relu_fx_flat[0]}, {relu_fx_flat[1]}, {relu_fx_flat[2]},
        {relu_fx_flat[3]}, {relu_fx_flat[4]}, {relu_fx_flat[5]},
        {relu_fx_flat[6]}, {relu_fx_flat[7]}, {relu_fx_flat[8]}
    );

end package first_conv_bn_relu_kernel0_pkg;
"""
with open(PKG_PATH, "w") as f:
    f.write(pkg_text)
print(f"    Written: {PKG_PATH.relative_to(REPO_ROOT)}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Kernel {KERNEL_ID}: scale_bn={scale_bn:.6f} b_folded={b0_folded:.6f} "
      f"scale_w_folded={scale_w_folded:.8f}")
print(f"SCALE_FX={SCALE_FX}  BIAS_FX={BIAS_FX}  (Q.{FRAC_BITS})")
print("Expected relu_fx (row-major):", relu_fx_flat)
print("\nOutput files:")
print(f"  {JSON_PATH.relative_to(REPO_ROOT)}")
print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
print(f"  {PKG_PATH.relative_to(REPO_ROOT)}")
