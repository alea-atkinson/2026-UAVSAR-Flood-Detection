"""
generate_all32_kernels_real_tile_q20_vhdl.py

Generates the VHDL entity files, combined testbench, and GHDL run script
needed to extend the real-tile Q.20 verification from the original
kernel0/kernel2 subset
(hardware/vhdl_conv3x3/reports/real_tile_first_layer_vhdl_verification_summary.md,
hardware/vhdl_conv3x3/reports/real_tile_fixed_point_precision_sensitivity_summary.md)
to ALL 32 output channels of the first Conv-BN-ReLU stage.

This is a THIN, MECHANICAL templating step, not a new hardware design:
every per-kernel entity generated here is structurally IDENTICAL to the
already-verified
hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20.vhd
(reusing the same, unmodified stream_conv3x3_3chan_cell sub-component and
the same two-stage registered Q.20 BatchNorm-fold + ReLU pipeline) --
only the kernel index and which Q.20 package it reads its
FOLDED-weight/SCALE_FX/BIAS_FX constants from changes between kernels.
Kernels 0 and 2 ALREADY have committed, GHDL-verified entity files
(stream_conv3x3_3chan_kernel{0,2}_bn_relu_real_tile_q20.vhd) -- this
script does NOT regenerate or touch those; it only creates the 30 new
per-kernel entity files (kernels 1, 3-31) that did not exist before, plus
ONE new combined all-32-kernel testbench and run script.

Prerequisite: scripts/generate_real_tile_bn_relu_fixed_point_vectors.py
must already have been run for --kernel-id 0 through 31 (--frac-bits 20
--variant-suffix _q20), producing all 32
first_conv_bn_relu_kernel{K}_real_tile_q20_pkg.vhd package files. This
script does not itself run that generator; it only assumes those 32
packages already exist (it will error out clearly if any is missing).

---- What this is NOT ----------------------------------------------------
- Does NOT modify any existing committed VHDL file (stream_conv3x3_3chan_cell,
  window3x3_stream, conv3x3_dot_pipelined, the existing kernel0/kernel2
  Q.16 or Q.20 entities/packages, or the existing 2-kernel combined
  testbench) -- all outputs are NEW, clearly-named files.
- Does NOT invent a new hardware architecture -- every generated entity
  is a mechanical copy of the already-verified kernel0 Q.20 entity's
  structure.
- Is NOT full FPGA U-Net, NOT board deployment, NOT full-image streaming
  -- this is simulation-only (GHDL) verification of the same real
  6x6x3 UAVSAR sub-block and 16 valid 3x3 windows already used for the
  kernel0/kernel2 subset, extended to all 32 output channels.

---- Usage ------------------------------------------------------------------
    python3 scripts/generate_all32_kernels_real_tile_q20_vhdl.py

---- Outputs -----------------------------------------------------------
hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel{K}_bn_relu_real_tile_q20.vhd
    for K in {1, 3, 4, ..., 31} (30 new files; K=0,2 already existed)
hardware/vhdl_conv3x3/tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd
hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VHDL_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3"

NUM_KERNELS = 32
ALREADY_EXISTING_KERNELS = {0, 2}   # already committed, GHDL-verified Q.20 entities
KERNELS_TO_GENERATE = [k for k in range(NUM_KERNELS) if k not in ALREADY_EXISTING_KERNELS]

TEMPLATE_KERNEL_FOR_ENTITY_TEXT = 0  # copy the structure of kernel 0's existing entity


def entity_text_for_kernel(k: int, template_text: str) -> str:
    """Produces kernel k's entity file text by substituting the template
    kernel's index/names throughout -- a mechanical rename, not a new
    architecture. Order of replacement matters (longer/more specific
    strings first) to avoid partial-match corruption."""
    t = template_text
    replacements = [
        ("first_conv_bn_relu_kernel0_real_tile_q20_pkg", f"first_conv_bn_relu_kernel{k}_real_tile_q20_pkg"),
        ("stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20", f"stream_conv3x3_3chan_kernel{k}_bn_relu_real_tile_q20"),
        ("stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd", f"stream_conv3x3_3chan_kernel{k}_bn_relu_pipelined.vhd (structurally, if it existed for this kernel)"),
        ("K0_RT_FOLDED_CH0_W", f"K{k}_RT_FOLDED_CH0_W"),
        ("K0_RT_FOLDED_CH1_W", f"K{k}_RT_FOLDED_CH1_W"),
        ("K0_RT_FOLDED_CH2_W", f"K{k}_RT_FOLDED_CH2_W"),
        ("kernel 0's FOLDED INT8 weights", f"kernel {k}'s FOLDED INT8 weights"),
        ("kernel-0 folded INT8", f"kernel-{k} folded INT8"),
        ("Kernel 0 ONLY (not all 32 output channels)", f"Kernel {k} ONLY (of the complete first Conv2d layer's 32 output channels; verified TOGETHER with all other 31 kernels by the combined all-32-kernel testbench)"),
        ("first_conv_bn_relu_kernel0_real_tile_q20_pkg", f"first_conv_bn_relu_kernel{k}_real_tile_q20_pkg"),
    ]
    for old, new in replacements:
        t = t.replace(old, new)
    return t


def main() -> None:
    print(f"[1] Checking all {NUM_KERNELS} kernels' Q.20 real-tile packages already exist ...")
    missing_pkgs = []
    for k in range(NUM_KERNELS):
        pkg_path = VHDL_DIR / f"first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.vhd"
        if not pkg_path.exists():
            missing_pkgs.append(str(pkg_path.relative_to(REPO_ROOT)))
    if missing_pkgs:
        sys.exit(
            "ERROR: the following kernel Q.20 packages do not exist yet. Run "
            "scripts/generate_real_tile_bn_relu_fixed_point_vectors.py --kernel-id <K> "
            "--frac-bits 20 --variant-suffix _q20 for each missing kernel first:\n"
            + "\n".join(f"  {p}" for p in missing_pkgs)
        )
    print(f"    All {NUM_KERNELS} packages found.")

    template_path = VHDL_DIR / f"stream_conv3x3_3chan_kernel{TEMPLATE_KERNEL_FOR_ENTITY_TEXT}_bn_relu_real_tile_q20.vhd"
    print(f"\n[2] Loading template entity: {template_path.relative_to(REPO_ROOT)}")
    template_text = template_path.read_text()

    print(f"\n[3] Generating {len(KERNELS_TO_GENERATE)} new per-kernel entity files "
          f"(kernels {KERNELS_TO_GENERATE}) ...")
    entity_paths_by_kernel = {}
    for k in range(NUM_KERNELS):
        entity_paths_by_kernel[k] = VHDL_DIR / f"stream_conv3x3_3chan_kernel{k}_bn_relu_real_tile_q20.vhd"

    for k in KERNELS_TO_GENERATE:
        out_path = entity_paths_by_kernel[k]
        if out_path.exists():
            print(f"    SKIP (already exists): {out_path.relative_to(REPO_ROOT)}")
            continue
        text = entity_text_for_kernel(k, template_text)
        out_path.write_text(text)
        print(f"    Written: {out_path.relative_to(REPO_ROOT)}")

    # Sanity check: every one of the 32 entity files must now exist (30
    # newly written here + 2 pre-existing).
    missing_entities = [
        str(p.relative_to(REPO_ROOT)) for p in entity_paths_by_kernel.values() if not p.exists()
    ]
    if missing_entities:
        sys.exit("ERROR: entity generation incomplete, missing: " + ", ".join(missing_entities))
    print(f"    All {NUM_KERNELS} per-kernel entity files present "
          f"({len(ALREADY_EXISTING_KERNELS)} pre-existing + {len(KERNELS_TO_GENERATE)} new).")

    # -- Combined all-32-kernel testbench --
    print("\n[4] Generating combined all-32-kernel testbench ...")
    tb_path = VHDL_DIR / "tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd"

    use_clauses = "\n".join(
        f"use work.first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.all;" for k in range(NUM_KERNELS)
    )
    dut_signals = "\n".join(
        f"    signal valid_out_k{k}    : std_logic;\n"
        f"    signal y_bn_relu_fx_k{k} : signed(47 downto 0);"
        for k in range(NUM_KERNELS)
    )
    dut_instances = "\n\n".join(
        f"    dut_k{k} : entity work.stream_conv3x3_3chan_kernel{k}_bn_relu_real_tile_q20\n"
        f"        generic map (IMG_WIDTH => BLOCK_SIZE)\n"
        f"        port map (\n"
        f"            clk => clk, rst => rst, valid_in => valid_in,\n"
        f"            pixel_c0 => pixel_c0, pixel_c1 => pixel_c1, pixel_c2 => pixel_c2,\n"
        f"            valid_out => valid_out_k{k}, y_bn_relu_fx => y_bn_relu_fx_k{k}\n"
        f"        );"
        for k in range(NUM_KERNELS)
    )
    per_kernel_out_count_decls = "\n".join(
        f"        variable out_count_k{k} : integer := 0;" for k in range(NUM_KERNELS)
    )
    per_kernel_checks_main_loop = "\n".join(
        f"            if valid_out_k{k} = '1' then\n"
        f"                out_count_k{k} := out_count_k{k} + 1;\n"
        f"                expected := K{k}_RT_BN_RELU_EXPECTED_FX(out_count_k{k} - 1);\n"
        f"                got      := to_integer(y_bn_relu_fx_k{k});\n"
        f"                check_output({k}, out_count_k{k}, expected, got);\n"
        f"            end if;"
        for k in range(NUM_KERNELS)
    )
    per_kernel_structural_asserts = "\n".join(
        f"        assert out_count_k{k} = N_POSITIONS\n"
        f"            report \"FAIL: kernel {k} expected exactly \" & integer'image(N_POSITIONS) &\n"
        f"                   \" valid output positions, observed \" & integer'image(out_count_k{k})\n"
        f"            severity warning;\n"
        f"        if out_count_k{k} /= N_POSITIONS then\n"
        f"            fail_count := fail_count + 1;\n"
        f"        end if;"
        for k in range(NUM_KERNELS)
    )

    tb_text = f'''-- tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd
-- Self-checking testbench: REAL-TILE Q.20 verification of ALL 32 output
-- channels of the first Conv-BN-ReLU stage.
--
-- This EXTENDS the original real-tile verification subset (kernel 0 and
-- kernel 2 only, Q.16 then Q.20 --
-- hardware/vhdl_conv3x3/reports/real_tile_first_layer_vhdl_verification_summary.md,
-- hardware/vhdl_conv3x3/reports/real_tile_fixed_point_precision_sensitivity_summary.md)
-- to the COMPLETE first Conv2d layer's 32 output channels, using the
-- IDENTICAL real 6x6x3 UAVSAR sub-block (tile_16_42.tif, row_offset=125,
-- col_offset=125) and the SAME 16 valid 3x3 windows per kernel already
-- used for the kernel0/kernel2 subset.
--
-- Every one of the 32 DUTs instantiated below
-- (stream_conv3x3_3chan_kernel{{0..31}}_bn_relu_real_tile_q20) is
-- structurally IDENTICAL: the SAME existing, unmodified
-- stream_conv3x3_3chan_cell sub-component, the SAME two-stage registered
-- Q.20 BatchNorm-fold + ReLU pipeline, differing ONLY in which kernel's
-- folded INT8 weights and Q.20 SCALE_FX/BIAS_FX constants they use. This
-- is NOT a new hardware architecture and NOT full 32-output resource
-- sharing (that is a DIFFERENT, separate design,
-- stream_conv3x3_3chan_32out_bn_relu_time_mux, not touched here) -- this
-- is 32 INDEPENDENT single-kernel real-tile pipelines, all fed the SAME
-- real pixel stream in parallel within this one simulation, each checked
-- against its own kernel's Python Q.20 golden vectors.
--
-- ---- Source of the real patch --------------------------------------------
-- Tile        : tile_16_42.tif (first row of heldout_fp2_test.csv)
-- Checkpoint  : models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
-- Generated by: scripts/generate_real_tile_bn_relu_fixed_point_vectors.py
--               (--kernel-id 0 through 31, --frac-bits 20, --variant-suffix _q20)
-- Normalization: FloodTileDataset._normalize_per_tile (train_unet_baseline_tuned.py)
-- Quantization : per-tile scale_x (analyze_first_layer_fixed_point_segmentation_impact.py
--                convention), applied to the FULL tile, then a 6x6 sub-block
--                sliced from the already-quantized INT8 tile -- IDENTICAL
--                real pixel stimulus to the kernel0/kernel2 subset
--                (real_tile_stimulus_pkg, not regenerated for this task).
--
-- ---- All 32 DUTs see the IDENTICAL pixel stream --------------------------
-- All 32 kernels are different output channels of the SAME trained
-- Conv2d layer, evaluated on the SAME real patch -- so all 32 DUTs are
-- driven by the same pixel_c0/c1/c2 signals in this testbench, and
-- (since all share the identical stream_conv3x3_3chan_cell/window3x3_stream
-- architecture and IMG_WIDTH) their valid_out pulses occur on IDENTICAL
-- clock cycles; only their y_bn_relu_fx values differ (different folded
-- weights, different per-kernel Q.20 SCALE_FX/BIAS_FX).
--
-- ---- Checking style (deliberately NOT severity failure per-check) --------
-- Same convention as the original kernel0/kernel2 combined testbench:
-- mismatches are reported as WARNINGS (not immediate FAILUREs) so the
-- simulation continues through all 32*16=512 checks and this testbench
-- can report a max-absolute-integer-difference summary across every
-- check, not just the first mismatch. A single final assertion (severity
-- failure) fires at the very end if any check failed.
--
-- ---- What this validates -----------------------------------------------
-- Confirms the EXISTING, UNMODIFIED (per-kernel, mechanically templated)
-- folded Conv-BN-ReLU pipelined datapath produces the SAME Q.20
-- fixed-point integer values as the Python golden-vector generator, for
-- ALL 32 output channels, on the SAME REAL UAVSAR-tile-derived patch
-- already used for the kernel0/kernel2 subset. It does NOT implement
-- padding, has NOT been run on real FPGA hardware, is NOT full FPGA
-- U-Net, NOT board deployment, and NOT full-image streaming -- this is
-- simulation-only verification of one small real patch's 16 valid
-- windows, for all 32 output channels.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.real_tile_stimulus_pkg.all;
{use_clauses}

entity tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 is
end entity tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20;

architecture sim of tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 is

    constant CLK_PERIOD   : time    := 10 ns;
    constant N_POSITIONS  : integer := (BLOCK_SIZE - 2) * (BLOCK_SIZE - 2);
    constant DRAIN_CYCLES : integer := 15;   -- generous margin beyond the fixed
                                              -- 6-cycle datapath latency
    constant NUM_KERNELS  : integer := {NUM_KERNELS};

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal valid_in : std_logic := '0';

    signal pixel_c0 : signed(7 downto 0) := (others => '0');
    signal pixel_c1 : signed(7 downto 0) := (others => '0');
    signal pixel_c2 : signed(7 downto 0) := (others => '0');

{dut_signals}

begin

    clk <= not clk after CLK_PERIOD / 2;

{dut_instances}

    stim : process
{per_kernel_out_count_decls}
        variable pass_count   : integer := 0;
        variable fail_count   : integer := 0;
        variable max_abs_diff : integer := 0;
        variable total_checked : integer := 0;
        variable expected     : integer;
        variable got          : integer;
        variable diff         : integer;

        procedure check_output(kernel_id : integer; out_num : integer;
                                expected_v : integer; got_v : integer) is
        begin
            diff := abs(got_v - expected_v);
            if diff > max_abs_diff then
                max_abs_diff := diff;
            end if;
            total_checked := total_checked + 1;
            if got_v = expected_v then
                pass_count := pass_count + 1;
            else
                fail_count := fail_count + 1;
                report "FAIL kernel " & integer'image(kernel_id) &
                       " output " & integer'image(out_num) &
                       ": y_bn_relu_fx = " & integer'image(got_v) &
                       "  expected " & integer'image(expected_v) &
                       "  abs_diff = " & integer'image(diff)
                    severity warning;
            end if;
        end procedure check_output;

    begin
        -- ---- Reset (3 clocks) ----------------------------------------
        rst      <= '1';
        valid_in <= '0';
        pixel_c0 <= (others => '0');
        pixel_c1 <= (others => '0');
        pixel_c2 <= (others => '0');
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst  <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;

        -- ---- Stream the REAL BLOCK_SIZE x BLOCK_SIZE patch (one pixel
        -- triplet per clock, row-major, matching real_tile_stimulus_pkg's
        -- flat layout) -- IDENTICAL stimulus to the kernel0/kernel2
        -- combined testbench. ----------------------------------------------------
        valid_in <= '1';
        for i in 0 to BLOCK_SIZE * BLOCK_SIZE - 1 loop
            pixel_c0 <= to_signed(REAL_TILE_CH0(i), 8);
            pixel_c1 <= to_signed(REAL_TILE_CH1(i), 8);
            pixel_c2 <= to_signed(REAL_TILE_CH2(i), 8);
            wait until rising_edge(clk);
            wait for 1 ns;

{per_kernel_checks_main_loop}
        end loop;

        -- ---- Drain: flush the pipeline for all 32 DUTs ------------------
        valid_in <= '0';
        for d in 1 to DRAIN_CYCLES loop
            wait until rising_edge(clk);
            wait for 1 ns;

{per_kernel_checks_main_loop}
        end loop;

        -- ---- Structural checks: exactly N_POSITIONS outputs per kernel --
{per_kernel_structural_asserts}

        -- ---- Concise summary report (PASS/FAIL count + max abs diff) --
        report "=== REAL-TILE ALL-32-KERNEL Q.20 first-layer Conv-BN-ReLU verification summary ===" severity note;
        report "    Total kernels checked  = " & integer'image(NUM_KERNELS) severity note;
        report "    Total output values checked = " & integer'image(total_checked) severity note;
        report "    PASS count = " & integer'image(pass_count) &
               "   FAIL count = " & integer'image(fail_count) severity note;
        report "    Max absolute integer difference (Q.20 fixed-point) = " &
               integer'image(max_abs_diff) severity note;
        report "    Real tile: tile_16_42.tif, 6x6 sub-block, row_offset=125, col_offset=125" severity note;

        if fail_count = 0 then
            report "=== All tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 tests PASSED ===" &
                   "  (" & integer'image(pass_count) & " / " & integer'image(pass_count) &
                   " outputs match Python Q.20 fixed-point golden vectors on REAL UAVSAR tile data, "
                   & "all 32 output channels)"
                severity note;
        else
            report "=== " & integer'image(fail_count) & " CHECK(S) FAILED ===" severity failure;
        end if;
        report "    All-32-kernel first Conv-BN-ReLU stage (REAL TILE tile_16_42.tif), enc1.block.0/1 from" &
               " alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt" severity note;
        report "    NOTE: simulation-only verification. NOT full FPGA U-Net, NOT board deployment," &
               " NOT full-image streaming." severity note;

        wait;
    end process stim;

end architecture sim;
'''
    tb_path.write_text(tb_text)
    print(f"    Written: {tb_path.relative_to(REPO_ROOT)}")

    # -- Run script --
    print("\n[5] Generating GHDL run script ...")
    run_script_path = VHDL_DIR / "run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh"
    analyze_lines = "\n".join(
        f'ghdl -a --std=08 first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.vhd\n'
        f'echo "  OK  first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.vhd"\n'
        f'ghdl -a --std=08 stream_conv3x3_3chan_kernel{k}_bn_relu_real_tile_q20.vhd\n'
        f'echo "  OK  stream_conv3x3_3chan_kernel{k}_bn_relu_real_tile_q20.vhd"'
        for k in range(NUM_KERNELS)
    )
    run_script_text = f'''#!/usr/bin/env bash
# run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh
# Simulate tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20
# using GHDL -- the ALL-32-KERNEL extension of
# run_ghdl_first_layer_real_tile_bn_relu_q20.sh (kernel 0 + kernel 2 only).
#
# This testbench drives 32 EXISTING/UNMODIFIED-STRUCTURE folded
# Conv-BN-ReLU pipelined datapaths (one per output channel of the first
# Conv2d layer, all structurally identical, differing only in their
# folded weights and Q.20 SCALE_FX/BIAS_FX constants) with the SAME REAL,
# per-tile-normalized-and-quantized 6x6x3 patch (tile_16_42.tif) already
# used for the kernel0/kernel2 subset.
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh --wave
#
# Scope note: simulation-only, real-data verification of all 32 first-layer
# output channels on ONE real 6x6 sub-block. NOT full FPGA U-Net, NOT
# board deployment, NOT full-image streaming.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v ghdl &>/dev/null; then
    echo ""
    echo "ERROR: GHDL not found in PATH."
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${{1:-}}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vcd  (open with GTKWave)"
elif [[ "${{1:-}}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_cell.vhd
echo "  OK  stream_conv3x3_3chan_cell.vhd"
ghdl -a --std=08 real_tile_stimulus_pkg.vhd
echo "  OK  real_tile_stimulus_pkg.vhd"

{analyze_lines}

ghdl -a --std=08 tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd
echo "  OK  tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 1000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 \\
    --stop-time=1000ns \\
    --assert-level=failure \\
    ${{EXTRA_RUN_FLAGS}}

echo ""
echo "=== Simulation complete ==="
'''
    run_script_path.write_text(run_script_text)
    run_script_path.chmod(0o755)
    print(f"    Written: {run_script_path.relative_to(REPO_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
