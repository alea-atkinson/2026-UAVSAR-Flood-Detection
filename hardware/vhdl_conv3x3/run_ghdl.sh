#!/usr/bin/env bash
# run_ghdl.sh
# Analyze, elaborate, and simulate conv3x3_dot testbench using GHDL.
#
# Prerequisites:
#   GHDL installed (https://ghdl.github.io/ghdl/)
#   Ubuntu/Debian : sudo apt install ghdl
#   Fedora        : sudo dnf install ghdl
#   macOS (brew)  : brew install ghdl
#
# Usage (run from project root or from hardware/vhdl_conv3x3/):
#   bash hardware/vhdl_conv3x3/run_ghdl.sh
#
# Optional flags:
#   --vcd    generate tb_conv3x3_dot.vcd (viewable with GTKWave)
#   --wave   generate tb_conv3x3_dot.ghw (GHDL native format, smaller)

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
    echo "  From source   : https://github.com/ghdl/ghdl/releases"
    echo ""
    echo "Alternatively, use Vivado xsim — see run_vivado_sim.tcl"
    echo "  cd hardware/vhdl_conv3x3"
    echo "  xvhdl --2008 mac_unit.vhd conv3x3_dot.vhd tb_conv3x3_dot.vhd"
    echo "  xelab tb_conv3x3_dot -s tb_sim"
    echo "  xsim tb_sim --runall"
    echo ""
    exit 1
fi

GHDL_VER=$(ghdl --version | head -1)
echo "Using: ${GHDL_VER}"
echo ""

# ---- Parse optional flags ---------------------------------------------
EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_conv3x3_dot.vcd"
    echo "Waveform output: tb_conv3x3_dot.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_conv3x3_dot.ghw"
    echo "Waveform output: tb_conv3x3_dot.ghw  (open with gtkwave or GHDL viewer)"
fi

# ---- Analysis (compile) -----------------------------------------------
echo "=== [1/3] Analyzing VHDL sources (--std=08) ==="
ghdl -a --std=08 mac_unit.vhd
echo "  OK  mac_unit.vhd"
ghdl -a --std=08 conv3x3_dot.vhd
echo "  OK  conv3x3_dot.vhd"
ghdl -a --std=08 tb_conv3x3_dot.vhd
echo "  OK  tb_conv3x3_dot.vhd"
echo ""

# ---- Elaboration ------------------------------------------------------
echo "=== [2/3] Elaborating tb_conv3x3_dot ==="
ghdl -e --std=08 tb_conv3x3_dot
echo "  OK"
echo ""

# ---- Simulation -------------------------------------------------------
echo "=== [3/3] Running simulation (stop after 200 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_conv3x3_dot \
    --stop-time=200ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
if [[ -n "${EXTRA_RUN_FLAGS}" ]]; then
    echo "Waveform: ${SCRIPT_DIR}/${EXTRA_RUN_FLAGS#*=}"
fi
