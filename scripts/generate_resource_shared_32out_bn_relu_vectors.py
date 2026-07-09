"""
generate_resource_shared_32out_bn_relu_vectors.py

Generates exact Q.16 fixed-point golden outputs for the resource-shared
(time-multiplexed) folded Conv-BN-ReLU first-layer prototype, for a
SELECTABLE number of output kernels (--num-kernels, default 4) on the
canonical 5x5x3 toy input.

Named "..._32out_..." because the design PLAN
(hardware/vhdl_conv3x3/resource_shared_first_layer_conv_bn_relu_plan.md)
targets the complete 32-output first Conv2d layer, and this script is
written to scale to --num-kernels 32 without structural changes -- but
the SMALLEST SERIOUS VALIDATED PROTOTYPE actually built and verified in
this task uses a smaller --num-kernels (see the design summary for the
exact count and why). This script does not claim to have generated or
verified vectors for all 32 kernels unless actually run with
--num-kernels 32.

---- Relationship to existing scripts --------------------------------------
Uses the SAME BatchNorm folding formula, SAME checkpoint, SAME per-channel
symmetric INT8 folded-weight quantization, and SAME Q.16 fixed-point
rescale+bias+ReLU arithmetic already established by
scripts/generate_bn_relu_fixed_point_vectors.py (which handles ONE kernel
at a time). This script generalizes that to a full SET of kernels
(0..N-1) in a single run, with a single combined VHDL package and a
single combined golden-output array in WINDOW-MAJOR order (see below) --
matching what the resource-shared hardware actually needs: one flat
weight/scale/bias LUT covering all N kernels, and one flat expected-output
array covering all NUM_WINDOWS x N outputs in emission order.

---- Output ordering (IMPORTANT, matches the RTL's emission order) ---------
WINDOW-MAJOR, KERNEL-MINOR: for each of the 9 valid window positions (in
window3x3_stream's row-major order), all N kernel outputs are listed
together, before moving to the next window. This matches
stream_conv3x3_3chan_8out_time_mux.vhd's existing valid_out pulse order
(y0..y7 together, once per window) and the direct-parallel 32-output
design's valid_out order (y0..y31 together, once per window) -- chosen so
this design's outputs are directly, position-by-position comparable to
both.

---- Fixed-point (Q.16) arithmetic, exactly what the VHDL computes --------
Per kernel k, per window position:
    SCALE_FX[k] = round(scale_w_folded[k] * scale_x * 2^16)     -- unsigned
    BIAS_FX[k]  = round(b_folded[k]                * 2^16)      -- signed
    raw_conv_int32   = sum over 3 channels, 9 taps of (x_int8 * w_folded_int8[k])
    product_fx       = raw_conv_int32 * SCALE_FX[k]
    biased_fx         = product_fx + BIAS_FX[k]
    relu_fx            = biased_fx if biased_fx > 0 else 0

scale_x = 1.0 exactly for the canonical toy patch (same deliberate
simplification as every other fixed-point golden script in this repo --
lossless here since the patch's max magnitude, 50, fits inside INT8).

---- Scope ------------------------------------------------------------------
- Selectable kernel COUNT (0..N-1, --num-kernels), canonical 5x5x3 toy
  input, 9 valid output positions only.
- ANALYSIS/GOLDEN-VECTOR-GENERATION ONLY -- does not retrain, modify the
  model, or touch any dataset splits. Does NOT imply VHDL implements
  padding, pooling, downstream layers, or the full model pipeline.

---- Usage ----------------------------------------------------------------
    python3 scripts/generate_resource_shared_32out_bn_relu_vectors.py --num-kernels 4

---- Outputs (num-kernels substituted for <N>) -----------------------------
hardware/vhdl_conv3x3/test_vectors/resource_shared_<N>out_bn_relu/resource_shared_<N>out_bn_relu_vectors.json
hardware/vhdl_conv3x3/test_vectors/resource_shared_<N>out_bn_relu/resource_shared_<N>out_bn_relu_vectors.csv
hardware/vhdl_conv3x3/test_vectors/resource_shared_<N>out_bn_relu/resource_shared_<N>out_bn_relu_summary.md
hardware/vhdl_conv3x3/first_layer_<N>out_bn_relu_resource_shared_pkg.vhd
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
parser.add_argument("--num-kernels", type=int, default=4,
                     help="Number of output kernels (0..N-1) to generate. Default 4 "
                          "(the smallest serious resource-shared prototype's scope). "
                          "The design plan targets 32; this script can be re-run with "
                          "--num-kernels 32 to scale up without code changes.")
parser.add_argument("--frac-bits", type=int, default=16,
                     help="Fixed-point fractional bits (Q.F format). Default 16, same as kernel 0/2.")
args = parser.parse_args()

N = args.num_kernels
FRAC_BITS = args.frac_bits
if not (1 <= N <= 32):
    raise SystemExit(f"ERROR: --num-kernels must be 1-32 (enc1.block.0.weight has 32 output channels), got {N}")

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

OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "test_vectors" / f"resource_shared_{N}out_bn_relu"
JSON_PATH = OUT_DIR / f"resource_shared_{N}out_bn_relu_vectors.json"
CSV_PATH  = OUT_DIR / f"resource_shared_{N}out_bn_relu_vectors.csv"
MD_PATH   = OUT_DIR / f"resource_shared_{N}out_bn_relu_summary.md"
PKG_PATH  = REPO_ROOT / "hardware" / "vhdl_conv3x3" / f"first_layer_{N}out_bn_relu_resource_shared_pkg.vhd"
PKG_NAME  = f"first_layer_{N}out_bn_relu_resource_shared_pkg"

# ---------------------------------------------------------------------------
# 1. Load checkpoint, extract kernels 0..N-1's Conv2d weights + BN params
# ---------------------------------------------------------------------------
print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]

w_tensor = state_dict[CONV_KEY_EXPECTED]           # [32, 3, 3, 3]
print(f"    Tensor key   : {CONV_KEY_EXPECTED}")
print(f"    Tensor shape : {list(w_tensor.shape)}  (using kernels 0..{N-1})")

bn_gamma_all = state_dict[f"{BN_PREFIX_EXPECTED}.weight"].float().numpy()
bn_beta_all = state_dict[f"{BN_PREFIX_EXPECTED}.bias"].float().numpy()
bn_mean_all = state_dict[f"{BN_PREFIX_EXPECTED}.running_mean"].float().numpy()
bn_var_all = state_dict[f"{BN_PREFIX_EXPECTED}.running_var"].float().numpy()

# ---------------------------------------------------------------------------
# 2-3. Fold BatchNorm + quantize, per kernel
# ---------------------------------------------------------------------------
print(f"\n[2] Folding BatchNorm and quantizing folded weights for kernels 0..{N-1} ...")

kernels = []  # list of dicts, one per kernel
for k in range(N):
    w_float = w_tensor[k].float().numpy()   # (3, 3, 3)
    scale_bn = float(bn_gamma_all[k] / np.sqrt(bn_var_all[k] + BN_EPS))
    w_folded = w_float * scale_bn
    b_folded = float(bn_beta_all[k] - bn_mean_all[k] * scale_bn)

    max_abs_w = float(np.abs(w_folded).max())
    scale_w_folded = max_abs_w / 127.0
    w_folded_int8 = np.clip(np.round(w_folded / scale_w_folded), -127, 127).astype(np.int8)

    kernels.append(dict(
        k=k, scale_bn=scale_bn, b_folded=b_folded,
        scale_w_folded=scale_w_folded, w_folded_int8=w_folded_int8,
    ))
    print(f"    kernel {k}: scale_bn={scale_bn:.6f} b_folded={b_folded:.6f} "
          f"scale_w_folded={scale_w_folded:.8f}")

# ---------------------------------------------------------------------------
# 4. Canonical 5x5x3 toy input patch -- SAME literal INT8 pixel values as
#    every other VHDL testbench in this repo (scale_x = 1.0 exactly).
# ---------------------------------------------------------------------------
print("\n[3] Building canonical 5x5x3 toy input patch (scale_x = 1.0, exact) ...")
base = np.arange(1, 26, dtype=np.int64).reshape(5, 5)
x_int8 = np.stack([base, base * 2, -base], axis=0).astype(np.int64)   # (3, 5, 5), exact
scale_x = 1.0
assert int(np.abs(x_int8).max()) <= 127, "toy patch does not fit INT8 -- scale_x=1.0 invalid"

# ---------------------------------------------------------------------------
# 5. Fixed-point constants per kernel, and exact per-window computation
# ---------------------------------------------------------------------------
print(f"\n[4] Computing Q.{FRAC_BITS} fixed-point constants and per-window outputs ...")


def valid_conv_int32_one_window(x_i8: np.ndarray, w_i8: np.ndarray, r: int, c: int) -> int:
    """Exact INT accumulation for ONE 3x3x3 window at valid position (r, c)."""
    acc = 0
    for ch in range(3):
        for ky in range(3):
            for kx in range(3):
                acc += int(x_i8[ch, r + ky, c + kx]) * int(w_i8[ch, ky, kx])
    return acc


for kd in kernels:
    combined_scale = scale_x * kd["scale_w_folded"]
    kd["SCALE_FX"] = int(round(combined_scale * (2 ** FRAC_BITS)))
    kd["BIAS_FX"] = int(round(kd["b_folded"] * (2 ** FRAC_BITS)))

# Window-major, kernel-minor enumeration (matches window3x3_stream's
# row-major window order and the RTL's emission order).
positions = []
win_idx = 0
for r in range(3):
    for c in range(3):
        for kd in kernels:
            raw = valid_conv_int32_one_window(x_int8, kd["w_folded_int8"], r, c)
            product_fx = raw * kd["SCALE_FX"]
            biased_fx = product_fx + kd["BIAS_FX"]
            relu_fx = max(biased_fx, 0)
            positions.append({
                "window_index": win_idx,
                "row": r,
                "col": c,
                "kernel_id": kd["k"],
                "raw_conv_int32": raw,
                "product_fx": product_fx,
                "biased_fx": biased_fx,
                "relu_fx": relu_fx,
            })
        win_idx += 1

total_outputs = len(positions)
num_clamped = sum(1 for p in positions if p["biased_fx"] <= 0)
print(f"    Total outputs: {total_outputs} (9 windows x {N} kernels)")
print(f"    Clamped (relu_fx == 0) count: {num_clamped} / {total_outputs}")

# ---------------------------------------------------------------------------
# 6. Write JSON + CSV test vectors
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[5] Writing test vectors to {OUT_DIR} ...")

data = {
    "checkpoint_path": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
    "num_kernels": N,
    "kernel_ids": list(range(N)),
    "conv_tensor_key": CONV_KEY_EXPECTED,
    "bn_prefix": BN_PREFIX_EXPECTED,
    "bn_eps": BN_EPS,
    "scale_x": scale_x,
    "fixed_point_format": f"Q.{FRAC_BITS} (signed, {FRAC_BITS} fractional bits)",
    "frac_bits": FRAC_BITS,
    "output_ordering": "window-major, kernel-minor: for each of 9 windows (row-major), "
                        "kernel outputs 0..N-1 together, before the next window",
    "num_clamped_outputs": num_clamped,
    "kernels": [
        {
            "kernel_id": kd["k"],
            "scale_bn": kd["scale_bn"],
            "b_folded": kd["b_folded"],
            "scale_w_folded": kd["scale_w_folded"],
            "SCALE_FX": kd["SCALE_FX"],
            "BIAS_FX": kd["BIAS_FX"],
            "w_folded_int8": {f"channel_{c}": kd["w_folded_int8"][c].flatten().tolist() for c in range(3)},
        }
        for kd in kernels
    ],
    "positions": positions,
    "notes": [
        "This is a fixed-point (integer-only) golden reference for the "
        "RESOURCE-SHARED (time-multiplexed) folded Conv-BN-ReLU prototype, "
        "distinct from the per-kernel scripts used for kernel 0/kernel 2.",
        "Output ordering is WINDOW-MAJOR, KERNEL-MINOR -- see 'output_ordering' above.",
        "relu_fx is the EXACT integer VHDL golden output.",
        f"Covers kernels 0-{N-1} only, not all 32 first-layer output channels "
        "unless run with --num-kernels 32.",
    ],
}
with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"    Written: {JSON_PATH.relative_to(REPO_ROOT)}")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["window_index", "row", "col", "kernel_id", "raw_conv_int32",
                      "product_fx", "biased_fx", "relu_fx"])
    for p in positions:
        writer.writerow([p["window_index"], p["row"], p["col"], p["kernel_id"],
                          p["raw_conv_int32"], p["product_fx"], p["biased_fx"], p["relu_fx"]])
print(f"    Written: {CSV_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 7. Write Markdown summary
# ---------------------------------------------------------------------------
kernel_rows = "\n".join(
    f"| {kd['k']} | {kd['scale_bn']:.6f} | {kd['b_folded']:.6f} | {kd['scale_w_folded']:.8f} | "
    f"{kd['SCALE_FX']} | {kd['BIAS_FX']} |"
    for kd in kernels
)
pos_rows = "\n".join(
    f"| {p['window_index']} | {p['row']} | {p['col']} | {p['kernel_id']} | {p['raw_conv_int32']} | "
    f"{p['product_fx']} | {p['biased_fx']} | **{p['relu_fx']}** |"
    for p in positions
)
md_lines = f"""\
# Resource-Shared {N}-Output Folded Conv-BN-ReLU -- Fixed-Point Test Vectors

Generated by `scripts/generate_resource_shared_32out_bn_relu_vectors.py --num-kernels {N}`.

This is a **fixed-point (integer-only) golden reference** for the
resource-shared (time-multiplexed) folded Conv-BN-ReLU prototype, using
the SAME BatchNorm folding formula, SAME INT8 quantization convention,
and SAME Q.{FRAC_BITS} fixed-point arithmetic already verified for kernel 0
and kernel 2 (`scripts/generate_bn_relu_fixed_point_vectors.py`),
generalized to a SET of {N} kernels in one run.

## Output ordering

**Window-major, kernel-minor**: for each of the 9 valid window positions
(row-major), kernel outputs 0..{N-1} are listed together, before the next
window. Total: 9 x {N} = {total_outputs} outputs.

## Per-kernel constants

| kernel | scale_bn | b_folded | scale_w_folded | SCALE_FX (Q.{FRAC_BITS}) | BIAS_FX (Q.{FRAC_BITS}) |
|---:|---:|---:|---:|---:|---:|
{kernel_rows}

## Clamping summary

**{num_clamped} / {total_outputs}** outputs have `biased_fx <= 0` (ReLU clamps to zero).

## Expected outputs ({total_outputs} total, window-major/kernel-minor order)

| idx | row | col | kernel | raw_conv_int32 | product_fx | biased_fx | relu_fx (golden) |
|---|---|---|---:|---:|---:|---:|---:|
{pos_rows}

## Limitations

- Kernels 0-{N-1} only, not all 32 output channels (unless run with `--num-kernels 32`).
- `scale_x = 1.0` is exact for this specific toy patch only.
- This is a golden-vector generation script, not a VHDL simulation; GHDL
  results are reported separately.
"""
with open(MD_PATH, "w") as f:
    f.write(md_lines)
print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------------
# 8. Write VHDL-2008 package: folded weight LUT, SCALE_FX/BIAS_FX LUTs,
#    and the flat expected-output array (window-major, kernel-minor).
# ---------------------------------------------------------------------------
print(f"\n[6] Writing VHDL package to {PKG_PATH} ...")


def vhdl_int8_array(vals) -> str:
    return "(" + ", ".join(str(int(v)) for v in vals) + ")"


pkg_lines = []
pkg_lines.append(f"""\
-- {PKG_NAME}.vhd
-- VHDL-2008 package for the RESOURCE-SHARED (time-multiplexed) folded
-- Conv-BN-ReLU prototype covering kernels 0-{N-1} of enc1.block.0.weight.
--
-- Generated by: scripts/generate_resource_shared_32out_bn_relu_vectors.py --num-kernels {N}
-- Source data : {JSON_PATH.relative_to(REPO_ROOT)}
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : {CHECKPOINT_PATH.relative_to(REPO_ROOT)}
-- Tensor key : {CONV_KEY_EXPECTED} (kernels 0-{N-1}), folded with
--              {BN_PREFIX_EXPECTED} (BatchNorm2d, eval-mode running stats)
--
-- ---- Fixed-point format (Q.{FRAC_BITS}, signed, {FRAC_BITS} fractional bits) --------------
-- real_value ~= fixed_value / 2^{FRAC_BITS}
-- SCALE_FX_LUT(k) = round(scale_x * scale_w_folded[k] * 2^{FRAC_BITS}), scale_x=1.0 exact
-- BIAS_FX_LUT(k)  = round(b_folded[k] * 2^{FRAC_BITS})
-- Datapath: product_fx = raw_conv_int32 * SCALE_FX_LUT(k) (no shift);
--           biased_fx = product_fx + BIAS_FX_LUT(k) (no shift);
--           relu_fx = biased_fx if biased_fx > 0 else 0.
--
-- ---- Output ordering (matches this package's EXPECTED_RELU_FX array) ------
-- WINDOW-MAJOR, KERNEL-MINOR: for each of 9 valid window positions
-- (row-major), kernel outputs 0..{N-1} together, before the next window.
--
-- ---- What this validates --------------------------------------------------
-- This package supplies the folded INT8 weight LUT, per-kernel SCALE_FX/
-- BIAS_FX LUTs, and the exact Q.{FRAC_BITS} fixed-point expected outputs so a
-- testbench can confirm a resource-shared VHDL design produces the SAME
-- fixed-point integer values as this Python script for kernels 0-{N-1} on
-- the canonical 5x5x3 toy input. It does NOT validate the other
-- {32-N} output channels, does NOT implement padding, and does NOT claim
-- board-tested or measured performance.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package {PKG_NAME} is

    constant NUM_KERNELS : integer := {N};
    constant NUM_WINDOWS : integer := 9;
    constant FRAC_BITS   : integer := {FRAC_BITS};

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;

    -- Folded INT8 weight LUT: NUM_KERNELS entries per channel.
    type weight_lut_t is array (0 to {N - 1}) of int8_kernel_t;

""")

for c in range(3):
    entries = ", ".join(vhdl_int8_array(kd["w_folded_int8"][c].flatten()) for kd in kernels)
    pkg_lines.append(f"    constant CH{c}_WEIGHT_LUT : weight_lut_t := ({entries});\n\n")

scale_entries = ", ".join(str(kd["SCALE_FX"]) for kd in kernels)
bias_entries = ", ".join(str(kd["BIAS_FX"]) for kd in kernels)
pkg_lines.append(f"""\
    -- Per-kernel Q.{FRAC_BITS} fixed-point scale/bias LUTs.
    type fx_lut_t is array (0 to {N - 1}) of integer;
    constant SCALE_FX_LUT : fx_lut_t := ({scale_entries});
    constant BIAS_FX_LUT  : fx_lut_t := ({bias_entries});

    -- {total_outputs} Q.{FRAC_BITS} fixed-point expected outputs, WINDOW-MAJOR /
    -- KERNEL-MINOR order (index = window_idx * NUM_KERNELS + kernel_idx).
    -- EXACT integers -- this is what the VHDL testbench compares against.
    type fx_outputs_t is array (0 to {total_outputs - 1}) of integer;
    constant EXPECTED_RELU_FX : fx_outputs_t := (
""")

relu_vals = [p["relu_fx"] for p in positions]
rows_of_8 = [relu_vals[i:i + 8] for i in range(0, len(relu_vals), 8)]
for i, row in enumerate(rows_of_8):
    sep = "," if i < len(rows_of_8) - 1 else ""
    pkg_lines.append("        " + ", ".join(str(v) for v in row) + sep + "\n")

pkg_lines.append(f"""\
    );

end package {PKG_NAME};
""")

with open(PKG_PATH, "w") as f:
    f.write("".join(pkg_lines))
print(f"    Written: {PKG_PATH.relative_to(REPO_ROOT)}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"num_kernels={N}  total_outputs={total_outputs}  clamped={num_clamped}")
print("\nOutput files:")
print(f"  {JSON_PATH.relative_to(REPO_ROOT)}")
print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
print(f"  {PKG_PATH.relative_to(REPO_ROOT)}")
