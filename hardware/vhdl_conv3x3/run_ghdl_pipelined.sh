#!/usr/bin/env bash
# run_ghdl_pipelined.sh
# Analyze, elaborate, and simulate the PIPELINED conv3x3_dot testbench using GHDL.
#
# Files analyzed:
#   conv3x3_dot_pipelined.vhd   -- 3-stage pipelined DUT
#   tb_conv3x3_dot_pipelined.vhd -- self-checking testbench
#
# Prerequisites:
#   GHDL installed (https://ghdl.github.io/ghdl/)
#   Ubuntu/Debian : sudo apt install ghdl
#   Fedora        : sudo dnf install ghdl
#   macOS (brew)  : brew install ghdl
#
# Usage (run from project root or hardware/vhdl_conv3x3/):
#   bash hardware/vhdl_conv3x3/run_ghdl_pipelined.sh
#
# Optional flags:
#   --vcd    generate tb_conv3x3_dot_pipelined.vcd (viewable with GTKWave)
#   --wave   generate tb_conv3x3_dot_pipelined.ghw (GHDL native format)
#
# Expected output:
#   PASS test 1 (pipelined): y = -6  (expected -6,  latency = 3 cycles)
#   PASS test 2 (pipelined): y = 19  (expected 19)
#   PASS test 3 (pipelined): y = 10  (expected 10)
#   === All conv3x3_dot_pipelined tests PASSED ===

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Check GHDL -------------------------------------------------------
if ! command -v ghdl &>/dev/null; then
    echo ""
    echo "ERROR: GHDL not found in PATH."
    echo ""
    echo "Install GHDL, then re-run this script:"
    echo "  Ubuntu/Debian : sudo apt install ghdl"
    echo "  Fedora/RHEL   : sudo dnf install ghdl"
    echo "  macOS (brew)  : brew install ghdl"
    echo "  Releases      : https://github.com/ghdl/ghdl/releases"
    echo ""
    echo "Alternatively, use Vivado xsim — see run_vivado_sim_pipelined.tcl"
    echo "  cd hardware/vhdl_conv3x3"
    echo "  xvhdl --2008 conv3x3_dot_pipelined.vhd tb_conv3x3_dot_pipelined.vhd"
    echo "  xelab tb_conv3x3_dot_pipelined -s tb_pip_sim"
    echo "  xsim tb_pip_sim --runall"
    echo ""
    exit 1
fi

GHDL_VER=$(ghdl --version | head -1)
echo "Using: ${GHDL_VER}"
echo ""

# ---- Parse optional flags ---------------------------------------------
EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_conv3x3_dot_pipelined.vcd"
    echo "Waveform output: tb_conv3x3_dot_pipelined.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_conv3x3_dot_pipelined.ghw"
    echo "Waveform output: tb_conv3x3_dot_pipelined.ghw"
fi

# ---- Analysis ---------------------------------------------------------
echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 tb_conv3x3_dot_pipelined.vhd
echo "  OK  tb_conv3x3_dot_pipelined.vhd"
echo ""

# ---- Elaboration ------------------------------------------------------
echo "=== [2/3] Elaborating tb_conv3x3_dot_pipelined ==="
ghdl -e --std=08 tb_conv3x3_dot_pipelined
echo "  OK"
echo ""

# ---- Simulation -------------------------------------------------------
echo "=== [3/3] Running simulation (stop after 500 ns) ==="
# 500 ns covers reset (40 ns) + 3 tests × ~60 ns each + drain cycles
# shellcheck disable=SC2086
ghdl -r --std=08 tb_conv3x3_dot_pipelined \
    --stop-time=500ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
if [[ -n "${EXTRA_RUN_FLAGS}" ]]; then
    echo "Waveform: ${SCRIPT_DIR}/${EXTRA_RUN_FLAGS#*=}"
fi
