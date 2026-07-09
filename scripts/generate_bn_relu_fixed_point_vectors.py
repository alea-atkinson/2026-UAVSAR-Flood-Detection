"""
generate_bn_relu_fixed_point_vectors.py

Generalized version of scripts/generate_kernel0_bn_relu_fixed_point_vectors.py:
generates a GENUINE fixed-point (integer-only) golden reference for a VHDL
Conv -> folded-BatchNorm-bias/scale -> ReLU proof-of-concept, for ANY ONE
selected output kernel of the first U-Net Conv2d layer, via --kernel-id.

The original kernel0 script is left UNCHANGED -- this is a new, separate
script, not a modification of it. Running this script with --kernel-id 0
reproduces the kernel0 script's numeric results exactly (same formulas,
same conventions), but writes to kernel-specific paths
(hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>/,
hardware/vhdl_conv3x3/first_conv_bn_relu_kernel<N>_pkg.vhd) rather than
overwriting the existing kernel0 outputs.

---- Why this exists ------------------------------------------------------
The kernel0 VHDL Conv-BN-ReLU proof-of-concept (unpipelined and pipelined)
was verified only for kernel 0's specific SCALE_FX/BIAS_FX constants. This
script lets a DIFFERENT, harder kernel (selected from
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_fidelity/first_conv_bn_relu_per_channel_metrics.csv)
be regenerated with the exact same fixed-point method, to test whether the
kernel0 pipelined timing-fix structure generalizes.

---- Relationship to the existing BatchNorm folding -------------------------
Uses the SAME BatchNorm folding formula, SAME checkpoint, SAME weight
tensor key, and SAME per-output-channel symmetric INT8 folded-weight
quantization convention as scripts/analyze_first_conv_bn_relu_fidelity.py
and scripts/generate_kernel0_bn_relu_fixed_point_vectors.py:
    scale_bn[oc]  = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc]  = w[oc] * scale_bn[oc]
    b_folded[oc]  = beta[oc] - running_mean[oc] * scale_bn[oc]
    scale_w_folded[oc] = max(|w_folded[oc]|) / 127
    w_folded_int8[oc]  = round(w_folded[oc] / scale_w_folded[oc]), clipped

---- Deliberate simplification: input activation scale ----------------------
Same as the kernel0 script: scale_x = 1.0 exactly for the canonical 5x5x3
toy patch (lossless here since the patch's max magnitude, 50, fits inside
INT8 range). This is NOT the general per-patch scale_x convention used for
real UAVSAR tiles in the fidelity study.

---- Fixed-point (Q-format) arithmetic, exactly what VHDL computes ----------
Let F = FRAC_BITS = 16 (fractional bits), same as kernel 0, unless a given
kernel's constants would overflow the VHDL widths already used by the
kernel0 design (48-bit signed datapath, 18-bit signed SCALE_FX) -- this
script checks that and reports it, it does not silently widen anything.
    SCALE_FX = round(scale_w_folded[k] * scale_x * 2^F)     -- unsigned constant
    BIAS_FX  = round(b_folded[k]                * 2^F)      -- signed constant
    raw_conv_int32   = sum over 3 channels, 9 taps of (x_int8 * w_folded_int8)  -- EXACT integer
    product_fx       = raw_conv_int32 * SCALE_FX             -- EXACT integer, Q.F format
    biased_fx         = product_fx + BIAS_FX                  -- EXACT integer, Q.F format
    relu_fx            = biased_fx if biased_fx > 0 else 0     -- EXACT integer, Q.F format

---- Scope --------------------------------------------------------------
- ONE selected kernel per run (via --kernel-id).
- Canonical 5x5x3 toy input, 9 valid output positions (5x5 -> 3x3), same
  as every other VHDL testbench in this repo.
- ANALYSIS/GOLDEN-VECTOR-GENERATION ONLY -- does not retrain, modify the
  model, or touch any dataset splits. Does NOT imply VHDL implements
  padding, pooling, downstream layers, or the full model pipeline.

---- Usage ----------------------------------------------------------------
    python3 scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id 2

---- Outputs (kernel-id substituted for <N>) -----------------------------
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>/kernel<N>_bn_relu_fixed_point_vectors.json
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>/kernel<N>_bn_relu_fixed_point_vectors.csv
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>/kernel<N>_bn_relu_fixed_point_summary.md
hardware/vhdl_conv3x3/first_conv_bn_relu_kernel<N>_pkg.vhd

---- Optional synthetic ReLU-clamp test mode (--input-mode synthetic_relu_clamp) --
Default behavior (--input-mode toy_5x5, or omitting --input-mode entirely)
is UNCHANGED from the original generator: same canonical 5x5x3 toy patch,
same output paths as before. Passing --input-mode synthetic_relu_clamp
instead drives the SAME kernel (same folded INT8 weights, same SCALE_FX,
same BIAS_FX, same Q.F arithmetic, same 3x3 valid-convolution structure)
with an ALL-ZERO 5x5x3 synthetic input patch instead of the toy patch.

This is a SYNTHETIC UNIT TEST, not a real UAVSAR tile or model inference
example: an all-zero input makes raw_conv_int32 = 0 for every one of the 9
valid output positions (trivially, since sum(0 * w) = 0 regardless of
weights), so biased_fx = BIAS_FX exactly for all 9 positions. Since
BIAS_FX is negative for every kernel generated by this script so far
(kernel 0: -936, kernel 2: -786), this deterministically forces ALL 9
positions to relu_fx = 0 -- directly exercising the ReLU clamp
(negative-branch) logic that was never exercised by either kernel 0's or
kernel 2's toy-patch tests (where every biased_fx happened to be
positive). Outputs for this mode are written to
kernel-and-mode-specific paths (suffixed with "_relu_clamp") so they never
collide with or overwrite the existing toy-patch outputs for the same
kernel.

    python3 scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id 0 --input-mode synthetic_relu_clamp

---- Outputs in synthetic_relu_clamp mode (kernel-id substituted for <N>) --
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>_relu_clamp/kernel<N>_relu_clamp_bn_relu_fixed_point_vectors.json
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>_relu_clamp/kernel<N>_relu_clamp_bn_relu_fixed_point_vectors.csv
hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel<N>_relu_clamp/kernel<N>_relu_clamp_bn_relu_fixed_point_summary.md
hardware/vhdl_conv3x3/first_conv_bn_relu_kernel<N>_relu_clamp_pkg.vhd
"""

import argparse
import csv
import json
import pathlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel-id", type=int, required=True,
                     help="Output channel index (0-31) of enc1.block.0.weight to process.")
parser.add_argument("--frac-bits", type=int, default=16,
                     help="Fixed-point fractional bits (Q.F format). Default 16, same as kernel 0.")
parser.add_argument("--input-mode", type=str, default="toy_5x5",
                     choices=["toy_5x5", "synthetic_relu_clamp"],
                     help="'toy_5x5' (default, unchanged original behavior) or "
                          "'synthetic_relu_clamp' (all-zero synthetic input, forces "
                          "biased_fx = BIAS_FX for every position -- a targeted ReLU "
                          "clamp unit test, NOT a real UAVSAR tile).")
args = parser.parse_args()

KERNEL_ID = args.kernel_id
FRAC_BITS = args.frac_bits
INPUT_MODE = args.input_mode
if not (0 <= KERNEL_ID <= 31):
    raise SystemExit(f"ERROR: --kernel-id must be 0-31 (enc1.block.0.weight has 32 output channels), got {KERNEL_ID}")

# Suffix applied to all output paths/identifiers in synthetic clamp-test
# mode, so it never collides with or overwrites the default toy-patch
# outputs for the same kernel.
MODE_SUFFIX = "" if INPUT_MODE == "toy_5x5" else "_relu_clamp"

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

K = KERNEL_ID   # short alias used in generated identifiers below

OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors" / f"conv_bn_relu_kernel{K}{MODE_SUFFIX}"
JSON_PATH = OUT_DIR / f"kernel{K}{MODE_SUFFIX}_bn_relu_fixed_point_vectors.json"
CSV_PATH  = OUT_DIR / f"kernel{K}{MODE_SUFFIX}_bn_relu_fixed_point_vectors.csv"
MD_PATH   = OUT_DIR / f"kernel{K}{MODE_SUFFIX}_bn_relu_fixed_point_summary.md"
PKG_PATH  = REPO_ROOT / "hardware" / "vhdl_conv3x3" / f"first_conv_bn_relu_kernel{K}{MODE_SUFFIX}_pkg.vhd"

# Widths already used by the kernel0 VHDL design (unpipelined and
# pipelined) -- this script CHECKS a new kernel's constants against these,
# it does not silently widen anything.
SCALE_FX_BITS = 18   # signed(17 downto 0) in the kernel0 VHDL
OUTPUT_BITS = 48     # signed(47 downto 0) in the kernel0 VHDL

# ---------------------------------------------------------------------------
# 1. Load checkpoint, extract the selected kernel's Conv2d weight + BN params
# ---------------------------------------------------------------------------
print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]

w_tensor = state_dict[CONV_KEY_EXPECTED]           # [32, 3, 3, 3]
wk_float = w_tensor[K].float().numpy()             # (3, 3, 3) = (in_ch, ky, kx)
print(f"    Tensor key   : {CONV_KEY_EXPECTED}")
print(f"    Kernel shape : {list(wk_float.shape)}  (kernel {K} only)")

bn_gamma = state_dict[f"{BN_PREFIX_EXPECTED}.weight"].float().numpy()[K]
bn_beta = state_dict[f"{BN_PREFIX_EXPECTED}.bias"].float().numpy()[K]
bn_mean = state_dict[f"{BN_PREFIX_EXPECTED}.running_mean"].float().numpy()[K]
bn_var = state_dict[f"{BN_PREFIX_EXPECTED}.running_var"].float().numpy()[K]
print(f"    BatchNorm[{K}]: gamma={bn_gamma:.6f} beta={bn_beta:.6f} "
      f"running_mean={bn_mean:.6f} running_var={bn_var:.6f}")

# ---------------------------------------------------------------------------
# 2. Fold BatchNorm into the selected kernel's weights/bias
# ---------------------------------------------------------------------------
print(f"\n[2] Folding BatchNorm into kernel {K}'s weights/bias ...")
scale_bn = float(bn_gamma / np.sqrt(bn_var + BN_EPS))
wk_folded = wk_float * scale_bn                     # (3, 3, 3)
bk_folded = float(bn_beta - bn_mean * scale_bn)
print(f"    scale_bn[{K}] = {scale_bn:.6f}")
print(f"    b_folded[{K}] = {bk_folded:.6f}")

# ---------------------------------------------------------------------------
# 3. Symmetric per-channel INT8 quantization of the FOLDED weights
# ---------------------------------------------------------------------------
print(f"\n[3] Quantizing folded kernel-{K} weights to INT8 (symmetric) ...")
max_abs_w = float(np.abs(wk_folded).max())
scale_w_folded = max_abs_w / 127.0
wk_folded_int8 = np.clip(np.round(wk_folded / scale_w_folded), -127, 127).astype(np.int8)
print(f"    scale_w_folded[{K}] = {scale_w_folded:.8f}")
print(f"    INT8 weights ch0:\n{wk_folded_int8[0]}")
print(f"    INT8 weights ch1:\n{wk_folded_int8[1]}")
print(f"    INT8 weights ch2:\n{wk_folded_int8[2]}")

# ---------------------------------------------------------------------------
# 4. Input patch. Two modes:
#    - "toy_5x5" (default): the SAME canonical 5x5x3 toy patch used by
#      every other VHDL testbench in this repo (scale_x = 1.0 exactly).
#    - "synthetic_relu_clamp": an ALL-ZERO 5x5x3 SYNTHETIC patch, used
#      ONLY to force biased_fx <= 0 and exercise the ReLU clamp branch.
#      This is NOT a real UAVSAR tile and does NOT represent real model
#      inference input -- it is a targeted unit test input only.
# ---------------------------------------------------------------------------
if INPUT_MODE == "toy_5x5":
    print("\n[4] Building canonical 5x5x3 toy input patch (scale_x = 1.0, exact) ...")
    base = np.arange(1, 26, dtype=np.int64).reshape(5, 5)
    x_int8 = np.stack([base, base * 2, -base], axis=0).astype(np.int64)   # (3, 5, 5), exact
    input_patch_description = (
        "5x5 toy patch: ch0=1..25 row-major, ch1=2xch0, ch2=-1xch0. "
        "scale_x=1.0 exact (matches every other VHDL testbench's literal "
        "INT8 pixel stimulus in this repo)."
    )
else:
    print("\n[4] Building SYNTHETIC ALL-ZERO 5x5x3 input patch "
          "(--input-mode synthetic_relu_clamp; scale_x = 1.0, exact) ...")
    print("    *** SYNTHETIC UNIT TEST INPUT -- NOT a real UAVSAR tile, NOT model inference. ***")
    x_int8 = np.zeros((3, 5, 5), dtype=np.int64)
    input_patch_description = (
        "SYNTHETIC ReLU clamp test patch: all-zero 5x5x3 input (NOT a real "
        "UAVSAR tile). raw_conv_int32 = 0 for every position by construction "
        "(sum(0 * w) = 0 regardless of weights), so biased_fx = BIAS_FX "
        "exactly for all 9 positions -- deterministically forces the ReLU "
        "clamp branch whenever BIAS_FX < 0. scale_x=1.0 exact."
    )
scale_x = 1.0
print(f"    Channel 0 range: {int(x_int8[0].min())} .. {int(x_int8[0].max())}")
print(f"    Channel 1 range: {int(x_int8[1].min())} .. {int(x_int8[1].max())}"
      + ("  (2x ch0)" if INPUT_MODE == "toy_5x5" else ""))
print(f"    Channel 2 range: {int(x_int8[2].min())} .. {int(x_int8[2].max())}"
      + ("  (-1x ch0)" if INPUT_MODE == "toy_5x5" else ""))
assert int(np.abs(x_int8).max()) <= 127, "input patch does not fit INT8 -- scale_x=1.0 invalid"

# ---------------------------------------------------------------------------
# 5. Exact INT32 valid (no-padding) 3x3 convolution accumulation, selected
#    kernel only, using the FOLDED INT8 weights.
# ---------------------------------------------------------------------------
print(f"\n[5] Computing exact INT32 raw convolution accumulation for kernel {K} (folded weights) ...")


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


raw_conv_int32 = valid_conv_int32(x_int8, wk_folded_int8)   # (3, 3), exact integers
print(f"    raw_conv_int32 (3x3):\n{raw_conv_int32}")

# ---------------------------------------------------------------------------
# 6. Fixed-point (Q.F) constants and datapath, EXACTLY what VHDL computes
# ---------------------------------------------------------------------------
print(f"\n[6] Computing Q.{FRAC_BITS} fixed-point constants ...")
combined_scale = scale_x * scale_w_folded   # = scale_w_folded since scale_x = 1.0
SCALE_FX = int(round(combined_scale * (2 ** FRAC_BITS)))
BIAS_FX = int(round(bk_folded * (2 ** FRAC_BITS)))
print(f"    combined_scale (scale_x * scale_w_folded) = {combined_scale:.8f}")
print(f"    SCALE_FX (unsigned, Q.{FRAC_BITS})  = {SCALE_FX}")
print(f"    BIAS_FX  (signed,   Q.{FRAC_BITS})  = {BIAS_FX}")

product_fx = raw_conv_int32.astype(np.int64) * SCALE_FX   # exact integer, Q.F
biased_fx = product_fx + BIAS_FX                            # exact integer, Q.F
relu_fx = np.maximum(biased_fx, 0)                          # exact integer, Q.F

max_abs_biased_fx = int(np.abs(biased_fx).max())
bits_needed = max_abs_biased_fx.bit_length() + 1   # +1 for sign
relu_clamped = bool(np.any(biased_fx <= 0))
clamped_mask = biased_fx <= 0
clamped_positions = [(int(r), int(c)) for r in range(3) for c in range(3) if clamped_mask[r, c]]
num_clamped = len(clamped_positions)
print(f"    biased_fx (3x3, Q.{FRAC_BITS}):\n{biased_fx}")
print(f"    relu_fx   (3x3, Q.{FRAC_BITS}):\n{relu_fx}")
print(f"    max |biased_fx| = {max_abs_biased_fx}  "
      f"(needs >= {bits_needed} signed bits; kernel0 VHDL uses {OUTPUT_BITS})")
print(f"    ReLU clamping exercised (any biased_fx <= 0)? {relu_clamped}")
print(f"    Clamped output count: {num_clamped} / 9   positions (row,col): {clamped_positions}")

# ---------------------------------------------------------------------------
# 6b. Width checks against the kernel0 VHDL design's fixed widths -- this
#     script CHECKS, it does not silently widen anything.
# ---------------------------------------------------------------------------
scale_fx_bits_needed = SCALE_FX.bit_length() + 1   # unsigned magnitude, +1 sign-bit margin since stored signed
output_bits_needed = bits_needed
width_warnings = []
if scale_fx_bits_needed > SCALE_FX_BITS:
    width_warnings.append(
        f"SCALE_FX={SCALE_FX} needs >= {scale_fx_bits_needed} signed bits, "
        f"exceeds the kernel0 VHDL's {SCALE_FX_BITS}-bit SCALE_FX_SIGNED constant width."
    )
if output_bits_needed > OUTPUT_BITS:
    width_warnings.append(
        f"max |biased_fx|={max_abs_biased_fx} needs >= {output_bits_needed} signed bits, "
        f"exceeds the kernel0 VHDL's {OUTPUT_BITS}-bit datapath width."
    )
if width_warnings:
    print("\n    WIDTH WARNING(S) -- kernel0's fixed VHDL widths may not be sufficient:")
    for w in width_warnings:
        print(f"      - {w}")
else:
    print(f"\n    Width check OK: SCALE_FX fits in {SCALE_FX_BITS} bits, "
          f"biased_fx fits in {OUTPUT_BITS} bits (same widths as the kernel0 VHDL design).")

# Human-readable float conversion (informative only, not used for the VHDL
# golden comparison).
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
    "kernel_id": K,
    "input_mode": INPUT_MODE,
    "conv_tensor_key": CONV_KEY_EXPECTED,
    "bn_prefix": BN_PREFIX_EXPECTED,
    "bn_eps": BN_EPS,
    "scale_bn": scale_bn,
    "b_folded": bk_folded,
    "scale_w_folded": scale_w_folded,
    "w_folded_int8": {f"channel_{c}": wk_folded_int8[c].flatten().tolist() for c in range(3)},
    "scale_x": scale_x,
    "fixed_point_format": f"Q.{FRAC_BITS} (signed, {FRAC_BITS} fractional bits)",
    "frac_bits": FRAC_BITS,
    "scale_fx": SCALE_FX,
    "bias_fx": BIAS_FX,
    "relu_clamping_exercised": relu_clamped,
    "num_clamped_outputs": num_clamped,
    "clamped_positions_row_col": clamped_positions,
    "width_warnings": width_warnings,
    "input_patch_description": input_patch_description,
    "positions": positions,
    "notes": [
        "This is a fixed-point (integer-only) golden reference, distinct "
        "from scripts/analyze_first_conv_bn_relu_fidelity.py's float "
        "rescale-based fidelity study.",
        f"Uses FOLDED (BatchNorm-absorbed) INT8 weights for kernel {K}, NOT "
        "the raw unfolded weights used by every prior VHDL prototype in "
        "this repo.",
        "relu_fx is the EXACT integer VHDL golden output.",
        "relu_float_equiv is for human-readable comparison only; the VHDL "
        "testbench compares against the exact integer relu_fx, not this "
        "float value.",
    ] + (
        ["*** SYNTHETIC UNIT TEST INPUT (all-zero) -- NOT a real UAVSAR tile, "
         "NOT a real model inference example. Used only to force at least one "
         "biased_fx <= 0 and exercise the ReLU clamp (negative) branch. ***"]
        if INPUT_MODE == "synthetic_relu_clamp" else []
    ),
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
width_note = (
    "\n".join(f"- {w}" for w in width_warnings) if width_warnings
    else "None -- fits within the same fixed widths used by the kernel0 VHDL design."
)
synthetic_banner = (
    "\n> **SYNTHETIC UNIT TEST -- NOT A REAL UAVSAR TILE.** This uses an "
    "all-zero 5x5x3 synthetic input patch chosen only to force `biased_fx "
    "<= 0` and exercise the ReLU clamp (negative) branch. It is not a real "
    "model inference example and does not represent real sensor data.\n"
    if INPUT_MODE == "synthetic_relu_clamp" else ""
)
clamp_summary = (
    f"\n## ReLU clamp summary\n\n"
    f"**{num_clamped} / 9** output positions clamped to zero "
    f"(`biased_fx <= 0`). Clamped positions (row, col): "
    f"{clamped_positions if clamped_positions else 'none'}.\n"
    if INPUT_MODE == "synthetic_relu_clamp" else ""
)
title_suffix = " -- Synthetic ReLU Clamp Test" if INPUT_MODE == "synthetic_relu_clamp" else ""
md_lines = f"""\
# Kernel {K} Folded Conv-BN-ReLU -- Fixed-Point Test Vectors{title_suffix}
{synthetic_banner}
Generated by `scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id {K}{" --input-mode synthetic_relu_clamp" if INPUT_MODE == "synthetic_relu_clamp" else ""}`.

This is a **fixed-point (integer-only) golden reference**, distinct from
`scripts/analyze_first_conv_bn_relu_fidelity.py`'s float-rescale-based
numerical fidelity study. It performs the entire rescale + bias + ReLU
computation using Q.{FRAC_BITS} fixed-point integer arithmetic -- the same
method used for kernel 0
(`scripts/generate_kernel0_bn_relu_fixed_point_vectors.py`), generalized
to any selected kernel via `--kernel-id`.
{clamp_summary}
## Checkpoint / kernel

| Field | Value |
|---|---|
| Checkpoint | `{CHECKPOINT_PATH.relative_to(REPO_ROOT)}` |
| Conv tensor key | `{CONV_KEY_EXPECTED}` |
| Kernel (output channel) | {K} |
| BatchNorm prefix | `{BN_PREFIX_EXPECTED}` |
| BN eps | {BN_EPS} |

## BatchNorm folding (kernel {K} only)

```
scale_bn[{K}]  = gamma[{K}] / sqrt(running_var[{K}] + eps) = {scale_bn:.6f}
b_folded[{K}]  = beta[{K}] - running_mean[{K}] * scale_bn[{K}] = {bk_folded:.6f}
```

## Folded weight quantization (symmetric INT8)

`scale_w_folded[{K}] = max(|w_folded[{K}]|) / 127 = {scale_w_folded:.8f}`

INT8 folded weights (row-major, top-left to bottom-right):

**Channel 0:** `{wk_folded_int8[0].flatten().tolist()}`
**Channel 1:** `{wk_folded_int8[1].flatten().tolist()}`
**Channel 2:** `{wk_folded_int8[2].flatten().tolist()}`

## Input patch

{input_patch_description}
"""
if INPUT_MODE == "toy_5x5":
    md_lines += f"""\
| Channel | Values |
|---|---|
| 0 | 1..25 row-major |
| 1 | 2 x channel 0 (2..50) |
| 2 | -1 x channel 0 (-1..-25) |

`scale_x = 1.0` exactly (lossless for this integer-valued patch; same
convention as kernel 0).
"""
else:
    md_lines += """\
| Channel | Values |
|---|---|
| 0 | all zero |
| 1 | all zero |
| 2 | all zero |

`scale_x = 1.0` exactly. This input is a deliberate synthetic construction,
not derived from any real UAVSAR tile.
"""
md_lines += f"""\

## Fixed-point format

Q.{FRAC_BITS} (signed, {FRAC_BITS} fractional bits): `real_value ~= fixed_value / 2^{FRAC_BITS}`.

| Constant | Value | Meaning |
|---|---:|---|
| `SCALE_FX` | {SCALE_FX} | `round(scale_x * scale_w_folded[{K}] * 2^{FRAC_BITS})`, unsigned |
| `BIAS_FX` | {BIAS_FX} | `round(b_folded[{K}] * 2^{FRAC_BITS})`, signed |

Datapath (exact integers, no floating point, no runtime shifts):

```
raw_conv_int32 = sum(x_int8 * w_folded_int8)     -- exact INT32 accumulation
product_fx      = raw_conv_int32 * SCALE_FX        -- exact, already Q.{FRAC_BITS}
biased_fx        = product_fx + BIAS_FX              -- exact, Q.{FRAC_BITS}
relu_fx           = biased_fx if biased_fx > 0 else 0  -- exact, Q.{FRAC_BITS}
```

Max `|biased_fx|` observed: {max_abs_biased_fx} (needs >= {bits_needed} signed bits).

**Width check against kernel0's fixed VHDL widths (SCALE_FX: {SCALE_FX_BITS} bits,
output: {OUTPUT_BITS} bits)**:
{width_note}

**ReLU clamping exercised by this toy input?** {"**Yes** -- at least one of the 9 positions has biased_fx <= 0, so relu_fx differs from biased_fx for that position." if relu_clamped else "**No** -- all 9 biased_fx values are already positive, so relu_fx == biased_fx for every position (same limitation as kernel 0's toy-input result)."}

## Expected outputs (9 valid positions, row-major)

| idx | row | col | raw_conv_int32 | product_fx | biased_fx | relu_fx (golden) | relu_fx / 2^{FRAC_BITS} (informative) |
|---|---|---|---:|---:|---:|---:|---:|
{pos_rows}

**The VHDL testbench compares against the exact integer `relu_fx` column,
not the float-equivalent column.**

## Limitations

- Kernel {K} only, not all 32 output channels.
- `scale_x = 1.0` is exact for this specific toy patch only; this is not
  the general per-patch activation scale used for real UAVSAR tiles in
  `analyze_first_conv_bn_relu_fidelity.py`.
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
pkg_name = f"first_conv_bn_relu_kernel{K}{MODE_SUFFIX}_pkg"
gen_cli_suffix = " --input-mode synthetic_relu_clamp" if INPUT_MODE == "synthetic_relu_clamp" else ""

pkg_synthetic_banner_line = (
    "-- *** SYNTHETIC RELU CLAMP UNIT TEST -- NOT a real UAVSAR tile, NOT real model inference. ***\n"
    if INPUT_MODE == "synthetic_relu_clamp" else ""
)
pkg_text = f"""\
-- {pkg_name}.vhd
{pkg_synthetic_banner_line}-- VHDL-2008 package of INT8 folded weights, Q.{FRAC_BITS} fixed-point
-- SCALE_FX/BIAS_FX constants, and Q.{FRAC_BITS} fixed-point golden output
-- constants for the kernel-{K} folded Conv-BN-ReLU proof-of-concept.
--
-- Generated by: scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id {K}{gen_cli_suffix}
-- Source data : {JSON_PATH.relative_to(REPO_ROOT)}
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}
-- Tensor key : {CONV_KEY_EXPECTED} (kernel {K} only), folded with
--              {BN_PREFIX_EXPECTED} (BatchNorm2d, eval-mode running stats)
--
-- ---- BatchNorm folding ----------------------------------------------------
-- scale_bn[{K}] = gamma[{K}] / sqrt(running_var[{K}] + eps) = {scale_bn:.6f}
-- b_folded[{K}] = beta[{K}] - running_mean[{K}] * scale_bn[{K}] = {bk_folded:.6f}
-- (eps = {BN_EPS}, torch.nn.BatchNorm2d default)
--
-- ---- Folded weight quantization ------------------------------------------
-- scale_w_folded[{K}] = max(|w_folded[{K}]|) / 127 = {scale_w_folded:.8f}
-- symmetric per-channel INT8, SAME convention as
-- scripts/analyze_first_conv_bn_relu_fidelity.py, applied to kernel {K} only.
--
-- ---- Fixed-point format (Q.{FRAC_BITS}, signed, {FRAC_BITS} fractional bits) --------------
-- real_value ~= fixed_value / 2^{FRAC_BITS}
-- SCALE_FX = round(scale_x * scale_w_folded[{K}] * 2^{FRAC_BITS}), scale_x = 1.0 exact
--            for this input patch (see generator script docstring)
-- BIAS_FX  = round(b_folded[{K}] * 2^{FRAC_BITS})
-- Datapath: product_fx = raw_conv_int32 * SCALE_FX (already Q.{FRAC_BITS}, no shift);
--           biased_fx = product_fx + BIAS_FX (same format, no shift);
--           relu_fx = biased_fx if biased_fx > 0 else 0.
--
-- ---- What this validates --------------------------------------------------
-- This package supplies the folded INT8 weights, the fixed-point scale/bias
-- constants, and the exact Q.{FRAC_BITS} fixed-point expected outputs so a
-- testbench can confirm a kernel-{K} folded Conv-BN-ReLU VHDL design produces
-- the SAME fixed-point integer values as this Python script for kernel {K} on
-- {"a SYNTHETIC all-zero unit-test input designed to force the ReLU clamp branch (NOT a real UAVSAR tile)" if INPUT_MODE == "synthetic_relu_clamp" else "the canonical 5x5x3 toy input"}.
-- It does NOT validate the other 31 output
-- channels, does NOT implement padding, and does NOT claim board-tested
-- or measured performance.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package {pkg_name} is

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;

    -- FOLDED (BatchNorm-absorbed) INT8 weights, kernel {K}
    constant K{K}_FOLDED_CH0_W : int8_kernel_t := {vhdl_int8_array(wk_folded_int8[0].flatten())};
    constant K{K}_FOLDED_CH1_W : int8_kernel_t := {vhdl_int8_array(wk_folded_int8[1].flatten())};
    constant K{K}_FOLDED_CH2_W : int8_kernel_t := {vhdl_int8_array(wk_folded_int8[2].flatten())};

    -- Fixed-point Q.{FRAC_BITS} constants
    constant FRAC_BITS : integer := {FRAC_BITS};
    constant SCALE_FX  : integer := {SCALE_FX};   -- unsigned, Q.{FRAC_BITS}
    constant BIAS_FX   : integer := {BIAS_FX};   -- signed,   Q.{FRAC_BITS}

    -- 9 Q.{FRAC_BITS} fixed-point expected outputs (row-major, output_index 0..8),
    -- EXACT integers -- this is what the VHDL testbench compares against.
    type fx_outputs_t is array (0 to 8) of integer;
    constant K{K}_BN_RELU_EXPECTED_FX : fx_outputs_t := (
        {relu_fx_flat[0]}, {relu_fx_flat[1]}, {relu_fx_flat[2]},
        {relu_fx_flat[3]}, {relu_fx_flat[4]}, {relu_fx_flat[5]},
        {relu_fx_flat[6]}, {relu_fx_flat[7]}, {relu_fx_flat[8]}
    );

end package {pkg_name};
"""
with open(PKG_PATH, "w") as f:
    f.write(pkg_text)
print(f"    Written: {PKG_PATH.relative_to(REPO_ROOT)}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Kernel {K}: scale_bn={scale_bn:.6f} b_folded={bk_folded:.6f} "
      f"scale_w_folded={scale_w_folded:.8f}")
print(f"SCALE_FX={SCALE_FX}  BIAS_FX={BIAS_FX}  (Q.{FRAC_BITS})")
print(f"ReLU clamping exercised: {relu_clamped}")
print(f"Width warnings: {width_warnings if width_warnings else 'none'}")
print("Expected relu_fx (row-major):", relu_fx_flat)
print("\nOutput files:")
print(f"  {JSON_PATH.relative_to(REPO_ROOT)}")
print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
print(f"  {PKG_PATH.relative_to(REPO_ROOT)}")
