#!/usr/bin/env python3
"""
generate_first_layer_32out_folded_arithmetic_core_vectors.py

Generates the VHDL package and golden test vectors for a NEW arithmetic-core
prototype: hardware/vhdl_conv3x3/first_layer_32out_folded_arithmetic_core.vhd,
a single-shot (not streamed) module that accepts one already-flattened
27-value 3x3x3 window and computes all 32 folded Conv-BN-ReLU output
channels of the first learned Conv2d+BatchNorm+ReLU stage in one pipelined
pass -- the concrete hardware test of the "one window per cycle, all 32
channels in parallel" assumption used by
scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py.

---- Why this does NOT re-derive anything from the checkpoint ---------------
This repo already has, per output channel (0-31), independently generated
and GHDL-verified:
  - raw INT8 Conv2d weights, in
    hardware/vhdl_conv3x3/first_layer_kernels0_to31_pkg.vhd
    (used by the already-synthesized 740/740-DSP 32-output direct-parallel
    design), and
  - real-tile Q.20 SCALE_FX/BIAS_FX BatchNorm-fold constants plus 16 golden
    Q.20 expected outputs (for a real 6x6 UAVSAR sub-block), in
    hardware/vhdl_conv3x3/first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd
    (used by the all-32-kernel real-tile Q.20 verification).

This script does NOT touch the checkpoint, the tile file, torch, or
rasterio. It PARSES those two already-verified VHDL package families
(32 files each) with simple regexes, asserts internal consistency (the
raw INT8 weights must match exactly between the two families -- they are
mathematically expected to, since a strictly positive per-channel
BatchNorm scale cancels out of symmetric max-abs INT8 quantization; see
the "Weight consistency" note below), and MERGES them into one new,
consolidated package with 32-wide LUT arrays -- mirroring the existing
Q.16/toy-patch `first_layer_32out_bn_relu_resource_shared_pkg.vhd`'s
`weight_lut_t`/`fx_lut_t` shape, but at Q.20/real-tile scale. No numeric
value in the merged package is newly invented; every constant is copied
verbatim from an existing, already-GHDL-verified source file.

---- Weight consistency (why raw and "folded" INT8 weights are identical) ---
first_layer_kernels0_to31_pkg.vhd quantizes the RAW Conv2d weight w[k]
(scale_w = max(|w[k]|)/127, no BatchNorm). The real-tile Q.20 packages
quantize w_folded[k] = w[k] * scale_bn[k] (scale_bn[k] > 0, a per-channel
BatchNorm scale). Since INT8 quantization here is symmetric max-abs
scaling, and scale_bn[k] is a single positive scalar applied uniformly to
all 27 weights of kernel k:
    round(w_folded / (max(|w_folded|)/127)) = round(w*scale_bn / (scale_bn*max(|w|)/127))
                                             = round(w / (max(|w|)/127))
i.e. scale_bn cancels exactly, so the two families' INT8 weight values are
identical by construction. This script ASSERTS that equality (a real
consistency check, not an assumption) rather than silently trusting it.

---- New test vectors (this is the ONLY new data this script computes) ------
5 flattened 27-value windows, chosen to exercise the merged arithmetic
across all 32 kernels without over-expanding the test set:
  1. real_window_0    -- a genuine 3x3x3 window sliced directly from the
                          existing real_tile_stimulus_pkg.vhd's 6x6 REAL
                          UAVSAR-tile-derived block (top-left valid window,
                          output position (row=0, col=0)). Its expected
                          output is cross-checked against
                          first_conv_bn_relu_kernel{K}_real_tile_q20_pkg.vhd's
                          K{K}_RT_BN_RELU_EXPECTED_FX(0) for ALL 32 kernels
                          -- an independent tie-back to the already
                          GHDL-verified all-32-kernel Q.20 report.
  2. all_zero         -- every one of the 27 inputs = 0 (exercises BIAS_FX/
                          ReLU alone).
  3. all_max_pos       -- every input = +127 (max positive INT8).
  4. all_max_neg       -- every input = -128 (max-magnitude negative INT8).
  5. checkerboard      -- alternating +100/-100 by flattened index.
For each window, this script computes the EXACT integer datapath (identical
formulas to every other Q.20 script in this repo):
    raw_conv_int32[k] = sum over 27 taps of (window[i] * weight[k][i])
    product_fx[k]      = raw_conv_int32[k] * SCALE_FX[k]     (Q.20, exact)
    biased_fx[k]       = product_fx[k] + BIAS_FX[k]           (Q.20, exact)
    y[k]               = biased_fx[k] if biased_fx[k] > 0 else 0

---- Scope ------------------------------------------------------------------
- Golden-vector / package generation only -- does NOT run GHDL or Vivado
  (see run_ghdl_32out_folded_arithmetic_core.sh /
  synth_first_layer_32out_folded_arithmetic_core_200t.tcl for those).
- Arithmetic-core prototype only: no image streaming, no line buffers, no
  full U-Net, no board test, no measured hardware speedup.

---- Usage ------------------------------------------------------------------
    python3 scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py

---- Outputs -----------------------------------------------------------------
hardware/vhdl_conv3x3/first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
hardware/vhdl_conv3x3/tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.vhd
hardware/vhdl_conv3x3/test_vectors/first_layer_32out_folded_arithmetic_core/
    first_layer_32out_folded_arithmetic_core_vectors.json
    first_layer_32out_folded_arithmetic_core_vectors.csv
    first_layer_32out_folded_arithmetic_core_vectors_summary.md
"""

from __future__ import annotations

import csv
import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HW_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3"

RAW_WEIGHTS_PKG = HW_DIR / "first_layer_kernels0_to31_pkg.vhd"
REAL_TILE_STIMULUS_PKG = HW_DIR / "real_tile_stimulus_pkg.vhd"

MERGED_PKG_PATH = HW_DIR / "first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd"
MERGED_PKG_NAME = "first_layer_32out_folded_bn_relu_real_tile_q20_pkg"

TB_VECTORS_PKG_PATH = HW_DIR / "tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.vhd"
TB_VECTORS_PKG_NAME = "tb_first_layer_32out_folded_arithmetic_core_vectors_pkg"

OUT_VECTORS_DIR = HW_DIR / "test_vectors" / "first_layer_32out_folded_arithmetic_core"

NUM_KERNELS = 32
FRAC_BITS = 20


def parse_int_array(text: str, const_name: str) -> list[int]:
    """Parses `constant <const_name> : <type> := (a, b, c, ...);` -> [a, b, c, ...]."""
    pattern = rf"constant\s+{re.escape(const_name)}\s*:\s*\S+\s*:=\s*\(([^;]*?)\)\s*;"
    m = re.search(pattern, text, re.DOTALL)
    if m is None:
        raise SystemExit(f"ERROR: could not find constant '{const_name}'")
    return [int(v.strip()) for v in m.group(1).split(",")]


def parse_scalar_int(text: str, const_name: str) -> int:
    pattern = rf"constant\s+{re.escape(const_name)}\s*:\s*integer\s*:=\s*(-?\d+)\s*;"
    m = re.search(pattern, text)
    if m is None:
        raise SystemExit(f"ERROR: could not find scalar constant '{const_name}'")
    return int(m.group(1))


def main() -> None:
    print("=" * 78)
    print("[1] Parsing existing raw INT8 weight package (already-synthesized "
          "740/740-DSP design's weight source)")
    print("=" * 78)
    print(f"    {RAW_WEIGHTS_PKG.relative_to(REPO_ROOT)}")
    raw_pkg_text = RAW_WEIGHTS_PKG.read_text()

    raw_ch_weights: dict[int, dict[int, list[int]]] = {}
    for k in range(NUM_KERNELS):
        raw_ch_weights[k] = {
            c: parse_int_array(raw_pkg_text, f"KERNEL{k}_CH{c}_W") for c in range(3)
        }
    print(f"    Parsed raw INT8 weights for {NUM_KERNELS} kernels x 3 channels.")

    print("\n" + "=" * 78)
    print("[2] Parsing existing per-kernel real-tile Q.20 packages (already "
          "GHDL-verified all-32-kernel Q.20 report)")
    print("=" * 78)
    rt_ch_weights: dict[int, dict[int, list[int]]] = {}
    scale_fx: dict[int, int] = {}
    bias_fx: dict[int, int] = {}
    expected_fx0: dict[int, int] = {}  # output_index 0 (row=0,col=0) golden value, per kernel
    for k in range(NUM_KERNELS):
        pkg_path = HW_DIR / f"first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.vhd"
        text = pkg_path.read_text()
        rt_ch_weights[k] = {
            c: parse_int_array(text, f"K{k}_RT_FOLDED_CH{c}_W") for c in range(3)
        }
        frac_bits_k = parse_scalar_int(text, "FRAC_BITS")
        assert frac_bits_k == FRAC_BITS, f"kernel {k}: FRAC_BITS={frac_bits_k}, expected {FRAC_BITS}"
        scale_fx[k] = parse_scalar_int(text, "SCALE_FX")
        bias_fx[k] = parse_scalar_int(text, "BIAS_FX")
        expected_all = parse_int_array(text, f"K{k}_RT_BN_RELU_EXPECTED_FX")
        expected_fx0[k] = expected_all[0]  # output_index 0 = (row=0, col=0)
    print(f"    Parsed SCALE_FX/BIAS_FX and expected outputs for {NUM_KERNELS} kernels.")

    print("\n" + "=" * 78)
    print("[3] Consistency check: raw INT8 weights must be IDENTICAL between "
          "the two existing families (scale_bn cancels in symmetric "
          "max-abs INT8 quantization -- see module docstring)")
    print("=" * 78)
    mismatches = []
    for k in range(NUM_KERNELS):
        for c in range(3):
            if raw_ch_weights[k][c] != rt_ch_weights[k][c]:
                mismatches.append((k, c))
    if mismatches:
        raise SystemExit(f"ERROR: weight mismatch between raw and real-tile-Q20 "
                          f"packages at (kernel, channel) = {mismatches}")
    print(f"    OK: all {NUM_KERNELS} x 3 = {NUM_KERNELS * 3} channel-weight arrays "
          f"match exactly between the two existing, independently-generated package families.")

    print("\n" + "=" * 78)
    print("[4] Parsing existing real-tile pixel stimulus "
          "(real_tile_stimulus_pkg.vhd)")
    print("=" * 78)
    stim_text = REAL_TILE_STIMULUS_PKG.read_text()
    block = 6
    real_tile_ch = {c: parse_int_array(stim_text, f"REAL_TILE_CH{c}") for c in range(3)}
    for c in range(3):
        assert len(real_tile_ch[c]) == block * block

    # Top-left valid 3x3 window (output position row=0, col=0): rows 0-2, cols 0-2
    # of the 6x6 grid, row-major within each channel, channels concatenated 0,1,2 --
    # SAME flatten order used everywhere else in this repo (channel-major, then
    # row-major within the 3x3 window).
    def real_window(row0: int, col0: int) -> list[int]:
        window = []
        for c in range(3):
            for r in range(row0, row0 + 3):
                for col in range(col0, col0 + 3):
                    window.append(real_tile_ch[c][r * block + col])
        return window

    real_window_0 = real_window(0, 0)
    print(f"    Extracted real_window_0 (output position row=0,col=0): {real_window_0}")

    print("\n" + "=" * 78)
    print("[5] Building the 5-window golden test vector set")
    print("=" * 78)
    test_windows: dict[str, list[int]] = {
        "real_window_0": real_window_0,
        "all_zero": [0] * 27,
        "all_max_pos": [127] * 27,
        "all_max_neg": [-128] * 27,
        "checkerboard": [100 if (i % 2 == 0) else -100 for i in range(27)],
    }
    for name, w in test_windows.items():
        assert len(w) == 27, f"{name}: expected 27 values, got {len(w)}"
    print(f"    Windows: {list(test_windows.keys())}")

    def compute_outputs(window: list[int]) -> list[int]:
        outputs = []
        for k in range(NUM_KERNELS):
            raw = 0
            idx = 0
            for c in range(3):
                for w in raw_ch_weights[k][c]:
                    raw += window[idx] * w
                    idx += 1
            product_fx = raw * scale_fx[k]
            biased_fx = product_fx + bias_fx[k]
            y = biased_fx if biased_fx > 0 else 0
            outputs.append(y)
        return outputs

    test_expected: dict[str, list[int]] = {
        name: compute_outputs(window) for name, window in test_windows.items()
    }

    print("\n" + "=" * 78)
    print("[6] Cross-checking real_window_0 against the EXISTING, already "
          "GHDL-verified all-32-kernel real-tile Q.20 expected outputs "
          "(K{k}_RT_BN_RELU_EXPECTED_FX(0))")
    print("=" * 78)
    cross_check_mismatches = []
    for k in range(NUM_KERNELS):
        computed = test_expected["real_window_0"][k]
        existing = expected_fx0[k]
        if computed != existing:
            cross_check_mismatches.append((k, computed, existing))
    if cross_check_mismatches:
        raise SystemExit(
            f"ERROR: real_window_0 cross-check against existing all-32-kernel "
            f"Q.20 golden data FAILED at kernels: {cross_check_mismatches}"
        )
    print(f"    OK: all 32 kernels' computed outputs for real_window_0 EXACTLY "
          f"match the existing, already GHDL-verified all-32-kernel real-tile "
          f"Q.20 report's expected output at position (row=0, col=0).")

    for name, outputs in test_expected.items():
        n_nonzero = sum(1 for v in outputs if v != 0)
        print(f"    {name:16s}: {n_nonzero}/32 kernels produced a nonzero "
              f"(post-ReLU) output")

    # ------------------------------------------------------------------
    # Write the merged VHDL package (weights + SCALE_FX_LUT/BIAS_FX_LUT)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"[7] Writing merged package: {MERGED_PKG_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)

    def vhdl_kernel_array(vals: list[int]) -> str:
        return "(" + ", ".join(str(v) for v in vals) + ")"

    ch_lut_blocks = []
    for c in range(3):
        entries = ",\n        ".join(
            vhdl_kernel_array(raw_ch_weights[k][c]) for k in range(NUM_KERNELS)
        )
        ch_lut_blocks.append(
            f"    constant CH{c}_WEIGHT_LUT : weight_lut_t := (\n        {entries}\n    );"
        )

    scale_fx_entries = ", ".join(str(scale_fx[k]) for k in range(NUM_KERNELS))
    bias_fx_entries = ", ".join(str(bias_fx[k]) for k in range(NUM_KERNELS))

    merged_pkg_text = f"""\
-- {MERGED_PKG_NAME}.vhd
-- Consolidated, all-32-kernel Q.20 real-tile folded Conv-BN-ReLU constants
-- for first_layer_32out_folded_arithmetic_core.vhd.
--
-- Generated by: scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py
--
-- ---- This is a MERGE, not a new derivation ---------------------------------
-- Every constant below is copied VERBATIM from two existing, independently
-- generated, already-verified VHDL package families -- no new value is
-- invented here:
--   - Raw INT8 weights: hardware/vhdl_conv3x3/first_layer_kernels0_to31_pkg.vhd
--     (weight source for the already-synthesized 740/740-DSP 32-output
--     direct-parallel design, first_layer_32out_dsp_200t_summary.md).
--   - Q.20 SCALE_FX/BIAS_FX: hardware/vhdl_conv3x3/
--     first_conv_bn_relu_kernel{{0..31}}_real_tile_q20_pkg.vhd (already
--     GHDL-verified, real_tile_all32_kernels_q20_verification_summary.md).
-- The generator script ASSERTS that the two families' raw INT8 weights are
-- bit-for-bit identical (expected: a strictly positive per-channel
-- BatchNorm scale cancels out of symmetric max-abs INT8 quantization) before
-- writing this file.
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight (all 32 output channels), folded with
--              enc1.block.1 (BatchNorm2d, eval-mode running stats)
--
-- ---- Fixed-point format (Q.20, signed, 20 fractional bits) ----------------
-- real_value ~= fixed_value / 2^20
-- Datapath per kernel k: raw_conv_int32 = sum over 27 taps of
--   (window[i] * weight[k][i]);
--   product_fx = raw_conv_int32 * SCALE_FX_LUT(k)  (Q.20, exact, no shift);
--   biased_fx  = product_fx + BIAS_FX_LUT(k)        (Q.20, exact, no shift);
--   y[k]       = biased_fx if biased_fx > 0 else 0  (ReLU).
-- SCALE_FX_LUT/BIAS_FX_LUT already bake in the REAL per-tile activation
-- scale (scale_x, from tile_16_42.tif), NOT scale_x=1.0.
--
-- ---- What this validates ----------------------------------------------------
-- This package supplies ALL 32 output channels' folded-datapath constants
-- in ONE file, for a NEW single-shot (not streamed) arithmetic-core
-- prototype that accepts one already-flattened 27-value window and
-- produces all 32 Conv-BN-ReLU outputs in a single pipelined pass. It does
-- NOT implement padding, does NOT implement image streaming/line buffers,
-- and does NOT claim board-tested or measured performance.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package {MERGED_PKG_NAME} is

    constant NUM_KERNELS : integer := {NUM_KERNELS};
    constant FRAC_BITS   : integer := {FRAC_BITS};

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;
    type weight_lut_t  is array (0 to NUM_KERNELS - 1) of int8_kernel_t;
    type fx_lut_t       is array (0 to NUM_KERNELS - 1) of integer;

    -- Raw INT8 weights, one LUT per input channel, indexed by output kernel 0-31.
    -- IDENTICAL values to first_layer_kernels0_to31_pkg.vhd's KERNEL{{k}}_CH{{c}}_W.
{ch_lut_blocks[0]}

{ch_lut_blocks[1]}

{ch_lut_blocks[2]}

    -- Q.20 BatchNorm-fold scale/bias, one entry per output kernel 0-31.
    -- IDENTICAL values to first_conv_bn_relu_kernel{{k}}_real_tile_q20_pkg.vhd's
    -- SCALE_FX/BIAS_FX.
    constant SCALE_FX_LUT : fx_lut_t := ({scale_fx_entries});
    constant BIAS_FX_LUT  : fx_lut_t := ({bias_fx_entries});

end package {MERGED_PKG_NAME};
"""
    MERGED_PKG_PATH.write_text(merged_pkg_text)
    print(f"    Written: {MERGED_PKG_PATH.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Write the testbench stimulus/expected package
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"[8] Writing testbench vectors package: {TB_VECTORS_PKG_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)

    window_names = list(test_windows.keys())
    num_windows = len(window_names)

    windows_entries = ",\n        ".join(
        "(" + ", ".join(str(v) for v in test_windows[name]) + ")" for name in window_names
    )
    expected_entries = ",\n        ".join(
        vhdl_kernel_array(test_expected[name]) for name in window_names
    )
    labels_comment = "\n".join(
        f"--   window {i}: {name}" for i, name in enumerate(window_names)
    )

    tb_vectors_text = f"""\
-- {TB_VECTORS_PKG_NAME}.vhd
-- Golden test-vector stimulus/expected constants for
-- tb_first_layer_32out_folded_arithmetic_core.vhd.
--
-- Generated by: scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py
-- Source vectors: hardware/vhdl_conv3x3/test_vectors/
--                 first_layer_32out_folded_arithmetic_core/first_layer_32out_folded_arithmetic_core_vectors.json
--
-- {num_windows} flattened 27-value test windows (channel-major, then
-- row-major within the 3x3 window -- window[0..8]=channel0, window[9..17]=
-- channel1, window[18..26]=channel2):
{labels_comment}
--
-- Window 0 (real_window_0) is a genuine 3x3x3 window sliced from the
-- EXISTING real_tile_stimulus_pkg.vhd's real UAVSAR-tile-derived 6x6
-- block (top-left valid window). Its expected 32-output vector was
-- CROSS-CHECKED by the generator script against the existing, already
-- GHDL-verified first_conv_bn_relu_kernel{{0..31}}_real_tile_q20_pkg.vhd's
-- K{{k}}_RT_BN_RELU_EXPECTED_FX(0) constants for all 32 kernels, and matches
-- exactly. Windows 1-4 are synthetic edge cases (all-zero, max-positive,
-- max-negative, checkerboard) computed directly from
-- first_layer_32out_folded_bn_relu_real_tile_q20_pkg's weight/scale/bias
-- LUTs.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package {TB_VECTORS_PKG_NAME} is

    constant NUM_TEST_WINDOWS : integer := {num_windows};
    constant NUM_KERNELS      : integer := {NUM_KERNELS};

    type window27_t     is array (0 to 26) of integer range -128 to 127;
    type windows_lut_t  is array (0 to NUM_TEST_WINDOWS - 1) of window27_t;
    type y32_t           is array (0 to NUM_KERNELS - 1) of integer;
    type expected_lut_t is array (0 to NUM_TEST_WINDOWS - 1) of y32_t;

    constant TEST_WINDOWS : windows_lut_t := (
        {windows_entries}
    );

    -- Q.20 fixed-point expected outputs (ReLU already applied), one 32-wide
    -- row per test window, in the same order as TEST_WINDOWS above.
    constant EXPECTED_Y : expected_lut_t := (
        {expected_entries}
    );

end package {TB_VECTORS_PKG_NAME};
"""
    TB_VECTORS_PKG_PATH.write_text(tb_vectors_text)
    print(f"    Written: {TB_VECTORS_PKG_PATH.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Write JSON/CSV/MD test vector summary (transparency, matches
    # existing test_vectors/ convention)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"[9] Writing test-vector summary under {OUT_VECTORS_DIR.relative_to(REPO_ROOT)}")
    print("=" * 78)
    OUT_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_VECTORS_DIR / "first_layer_32out_folded_arithmetic_core_vectors.json"
    csv_path = OUT_VECTORS_DIR / "first_layer_32out_folded_arithmetic_core_vectors.csv"
    md_path = OUT_VECTORS_DIR / "first_layer_32out_folded_arithmetic_core_vectors_summary.md"

    data = {
        "frac_bits": FRAC_BITS,
        "num_kernels": NUM_KERNELS,
        "weight_source": "hardware/vhdl_conv3x3/first_layer_kernels0_to31_pkg.vhd",
        "scale_bias_source": "hardware/vhdl_conv3x3/first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd",
        "weight_consistency_check": "PASS (raw INT8 weights identical across both source families)",
        "real_window_0_cross_check": "PASS (matches existing all-32-kernel Q.20 golden output at position 0)",
        "test_windows": {name: w for name, w in test_windows.items()},
        "expected_outputs": {name: test_expected[name] for name in window_names},
        "scale_fx": scale_fx,
        "bias_fx": bias_fx,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Written: {json_path.relative_to(REPO_ROOT)}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_name", "kernel_id", "expected_y_q20"])
        for name in window_names:
            for k in range(NUM_KERNELS):
                writer.writerow([name, k, test_expected[name][k]])
    print(f"    Written: {csv_path.relative_to(REPO_ROOT)}")

    windows_md_rows = "\n".join(
        f"| {name} | `{test_windows[name][:9]}`... | "
        f"{sum(1 for v in test_expected[name] if v != 0)}/32 |"
        for name in window_names
    )
    md_text = f"""# First-Layer 32-Output Folded Arithmetic-Core: Golden Test Vectors

Generated by `scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py`.

## What this is

5 flattened 27-value test windows (channel-major, row-major within each
3x3 channel plane) and their Q.20 fixed-point, ReLU-applied, 32-output
golden vectors, for `first_layer_32out_folded_arithmetic_core.vhd`'s GHDL
testbench (`tb_first_layer_32out_folded_arithmetic_core.vhd`).

**No new weight, scale, or bias value was invented for this report.** All
32 kernels' raw INT8 weights and Q.20 SCALE_FX/BIAS_FX constants are
copied verbatim from two existing, already-verified package families (see
`first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd`'s header for
exact source paths), after an automated consistency check confirmed the
two families' raw INT8 weights match exactly.

## Test windows

| Window | First 9 values (ch0) | Nonzero outputs (of 32) |
|---|---|---:|
{windows_md_rows}

## Cross-check against prior GHDL-verified results

`real_window_0` is a genuine 3x3x3 window sliced from the existing
`real_tile_stimulus_pkg.vhd`'s real UAVSAR-tile-derived 6x6 block (the
same block used by the all-32-kernel real-tile Q.20 verification). Its
computed 32-output vector was checked against that existing verification's
`K{{k}}_RT_BN_RELU_EXPECTED_FX(0)` constants for all 32 kernels and
**matches exactly** -- an independent tie-back confirming this merged
package's constants are consistent with already-verified prior results,
not a fresh, unchecked derivation.

## Scope

- Golden-vector generation only; GHDL/Vivado results are reported
  separately in `hardware/vhdl_conv3x3/reports/`.
- Arithmetic-core prototype only: no image streaming, no line buffers, no
  full U-Net, no board-tested or measured performance.
"""
    md_path.write_text(md_text)
    print(f"    Written: {md_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
