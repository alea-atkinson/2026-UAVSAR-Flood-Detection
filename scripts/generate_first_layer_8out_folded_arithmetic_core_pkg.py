#!/usr/bin/env python3
"""
generate_first_layer_8out_folded_arithmetic_core_pkg.py

Generates hardware/vhdl_conv3x3/first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd,
the weight/scale/bias package for output channels 0-7 of the first learned
Conv2d+BatchNorm+ReLU stage, used by the NEW 2-window x 8-output spatial-
parallelism arithmetic core
(hardware/vhdl_conv3x3/first_layer_2win_8out_folded_arithmetic_core.vhd).

---- Why this is a SLICE, not a new derivation -------------------------------
Mirrors scripts/generate_first_layer_16out_folded_arithmetic_core_pkg.py
exactly, just keeping 8 kernels instead of 16. This repo already has a
consolidated, already-verified, all-32-kernel Q.20 package:
hardware/vhdl_conv3x3/first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
(itself a verified MERGE of two independent, already-GHDL-verified source
families -- see that file's own header). This script does NOT re-derive
anything from the checkpoint, the tile file, torch, or rasterio. It parses
ONLY the existing 32-kernel merged package, SLICES out kernels 0-7, and
cross-checks that slice against the existing, independently-generated
per-kernel real-tile Q.20 packages
(first_conv_bn_relu_kernel{0..7}_real_tile_q20_pkg.vhd) as a genuine
consistency check before writing anything.

---- Source of weights --------------------------------------------------
Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
Tensor key : enc1.block.0.weight (output channels 0-7 of 32), folded with
             enc1.block.1 (BatchNorm2d, eval-mode running stats)

---- Usage ------------------------------------------------------------------
    python3 scripts/generate_first_layer_8out_folded_arithmetic_core_pkg.py

---- Output -------------------------------------------------------------------
hardware/vhdl_conv3x3/first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HW_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3"

SOURCE_32OUT_PKG = HW_DIR / "first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd"

OUT_PKG_PATH = HW_DIR / "first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd"
OUT_PKG_NAME = "first_layer_8out_folded_bn_relu_real_tile_q20_pkg"

NUM_KERNELS = 8
FRAC_BITS = 20


def parse_weight_lut(text: str, const_name: str) -> list[list[int]]:
    """Parses `constant <const_name> : weight_lut_t := ( (a,..,i), (a,..,i), ... );`
    -> list of 32 9-tuples."""
    pattern = rf"constant\s+{re.escape(const_name)}\s*:\s*weight_lut_t\s*:=\s*\((.*?)\)\s*;"
    m = re.search(pattern, text, re.DOTALL)
    if m is None:
        raise SystemExit(f"ERROR: could not find constant '{const_name}'")
    body = m.group(1)
    tuples = re.findall(r"\(([^()]*)\)", body)
    return [[int(v.strip()) for v in t.split(",")] for t in tuples]


def parse_int_array(text: str, const_name: str) -> list[int]:
    pattern = rf"constant\s+{re.escape(const_name)}\s*:\s*fx_lut_t\s*:=\s*\(([^;]*?)\)\s*;"
    m = re.search(pattern, text, re.DOTALL)
    if m is None:
        raise SystemExit(f"ERROR: could not find constant '{const_name}'")
    return [int(v.strip()) for v in m.group(1).split(",")]


def parse_int8_kernel(text: str, const_name: str) -> list[int]:
    pattern = rf"constant\s+{re.escape(const_name)}\s*:\s*int8_kernel_t\s*:=\s*\(([^;]*?)\)\s*;"
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
    print("[1] Parsing the EXISTING, already-verified all-32-kernel merged "
          "package")
    print("=" * 78)
    print(f"    {SOURCE_32OUT_PKG.relative_to(REPO_ROOT)}")
    text32 = SOURCE_32OUT_PKG.read_text()

    ch_weights_32 = {c: parse_weight_lut(text32, f"CH{c}_WEIGHT_LUT") for c in range(3)}
    scale_fx_32 = parse_int_array(text32, "SCALE_FX_LUT")
    bias_fx_32 = parse_int_array(text32, "BIAS_FX_LUT")
    for c in range(3):
        assert len(ch_weights_32[c]) == 32, f"CH{c}_WEIGHT_LUT: expected 32 entries"
    assert len(scale_fx_32) == 32 and len(bias_fx_32) == 32

    print(f"    Parsed 32-kernel weight/scale/bias constants.")

    print("\n" + "=" * 78)
    print(f"[2] Slicing to kernels 0-{NUM_KERNELS - 1}")
    print("=" * 78)
    ch_weights_8 = {c: ch_weights_32[c][:NUM_KERNELS] for c in range(3)}
    scale_fx_8 = scale_fx_32[:NUM_KERNELS]
    bias_fx_8 = bias_fx_32[:NUM_KERNELS]

    print("\n" + "=" * 78)
    print("[3] Cross-checking the slice against the EXISTING, independently "
          "generated per-kernel real-tile Q.20 packages "
          f"(first_conv_bn_relu_kernel{{0..{NUM_KERNELS - 1}}}_real_tile_q20_pkg.vhd)")
    print("=" * 78)
    mismatches = []
    for k in range(NUM_KERNELS):
        pkg_path = HW_DIR / f"first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.vhd"
        ktext = pkg_path.read_text()
        for c in range(3):
            rt_w = parse_int8_kernel(ktext, f"K{k}_RT_FOLDED_CH{c}_W")
            if rt_w != ch_weights_8[c][k]:
                mismatches.append((k, f"CH{c}_W"))
        frac_bits_k = parse_scalar_int(ktext, "FRAC_BITS")
        if frac_bits_k != FRAC_BITS:
            mismatches.append((k, "FRAC_BITS"))
        rt_scale = parse_scalar_int(ktext, "SCALE_FX")
        if rt_scale != scale_fx_8[k]:
            mismatches.append((k, "SCALE_FX"))
        rt_bias = parse_scalar_int(ktext, "BIAS_FX")
        if rt_bias != bias_fx_8[k]:
            mismatches.append((k, "BIAS_FX"))
    if mismatches:
        raise SystemExit(f"ERROR: 8-kernel slice mismatch against per-kernel "
                          f"real-tile Q.20 packages at: {mismatches}")
    print(f"    OK: all {NUM_KERNELS} kernels' weights/SCALE_FX/BIAS_FX in the "
          f"32-kernel-package slice are bit-for-bit identical to the "
          f"existing, independently-generated per-kernel real-tile Q.20 "
          f"packages.")

    # ------------------------------------------------------------------
    # Write the sliced 8-kernel VHDL package
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"[4] Writing sliced package: {OUT_PKG_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)

    def vhdl_kernel_array(vals: list[int]) -> str:
        return "(" + ", ".join(str(v) for v in vals) + ")"

    ch_lut_blocks = []
    for c in range(3):
        entries = ",\n        ".join(
            vhdl_kernel_array(ch_weights_8[c][k]) for k in range(NUM_KERNELS)
        )
        ch_lut_blocks.append(
            f"    constant CH{c}_WEIGHT_LUT : weight_lut_t := (\n        {entries}\n    );"
        )

    scale_fx_entries = ", ".join(str(v) for v in scale_fx_8)
    bias_fx_entries = ", ".join(str(v) for v in bias_fx_8)

    pkg_text = f"""\
-- {OUT_PKG_NAME}.vhd
-- 8-kernel (output channels 0-{NUM_KERNELS - 1} of 32) Q.20 real-tile folded
-- Conv-BN-ReLU constants for first_layer_2win_8out_folded_arithmetic_core.vhd.
--
-- Generated by: scripts/generate_first_layer_8out_folded_arithmetic_core_pkg.py
--
-- ---- This is a SLICE of an already-verified merge, not a new derivation ---
-- Every constant below is copied VERBATIM from
-- hardware/vhdl_conv3x3/first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
-- (itself an already-GHDL-verified MERGE of two independent source
-- families -- see that file's header), keeping only output channels
-- 0-{NUM_KERNELS - 1}. No new weight, scale, or bias value is invented here. The
-- generator script additionally cross-checks this slice, kernel by kernel,
-- against the existing, independently-generated
-- first_conv_bn_relu_kernel{{0..{NUM_KERNELS - 1}}}_real_tile_q20_pkg.vhd files before
-- writing this file.
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Tensor key : enc1.block.0.weight (output channels 0-{NUM_KERNELS - 1} of 32),
--              folded with enc1.block.1 (BatchNorm2d, eval-mode running stats)
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
-- This package supplies output channels 0-{NUM_KERNELS - 1}'s folded-datapath
-- constants in ONE file, for the NEW 2-window x 8-output spatial-parallelism
-- arithmetic core, which accepts TWO already-flattened 27-value windows per
-- cycle (one per spatial lane) and produces {NUM_KERNELS} Conv-BN-ReLU outputs
-- PER LANE in a single pipelined pass. It does NOT implement padding, does
-- NOT implement image streaming/line buffers, and does NOT claim
-- board-tested or measured performance.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package {OUT_PKG_NAME} is

    constant NUM_KERNELS : integer := {NUM_KERNELS};
    constant FRAC_BITS   : integer := {FRAC_BITS};

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;
    type weight_lut_t  is array (0 to NUM_KERNELS - 1) of int8_kernel_t;
    type fx_lut_t       is array (0 to NUM_KERNELS - 1) of integer;

    -- Raw INT8 weights, one LUT per input channel, indexed by output kernel 0-{NUM_KERNELS - 1}.
    -- IDENTICAL values to first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd's
    -- CH{{c}}_WEIGHT_LUT(0..{NUM_KERNELS - 1}).
{ch_lut_blocks[0]}

{ch_lut_blocks[1]}

{ch_lut_blocks[2]}

    -- Q.20 BatchNorm-fold scale/bias, one entry per output kernel 0-{NUM_KERNELS - 1}.
    -- IDENTICAL values to first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd's
    -- SCALE_FX_LUT/BIAS_FX_LUT(0..{NUM_KERNELS - 1}).
    constant SCALE_FX_LUT : fx_lut_t := ({scale_fx_entries});
    constant BIAS_FX_LUT  : fx_lut_t := ({bias_fx_entries});

end package {OUT_PKG_NAME};
"""
    OUT_PKG_PATH.write_text(pkg_text)
    print(f"    Written: {OUT_PKG_PATH.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
