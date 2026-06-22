#!/usr/bin/env bash
# run_ghdl_window.sh
# Analyze, elaborate, and simulate the window3x3_stream testbench using GHDL.
#
# Files analyzed:
#   window3x3_stream.vhd     -- sliding 3x3 window generator DUT
#   tb_window3x3_stream.vhd  -- self-checking testbench (5×5 image)
#
# Prerequisites:
#   GHDL: sudo apt install ghdl  (Ubuntu/Debian)
#          sudo dnf install ghdl  (Fedora)
#          brew install ghdl      (macOS)
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_window.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_window.sh --vcd   # generate waveform
#   bash hardware/vhdl_conv3x3/run_ghdl_window.sh --wave
#
# Expected output:
#   PASS window 1 (stream):  [1,2,3;  6,7,8;  11,12,13]
#   PASS window 2 (stream):  [2,3,4;  7,8,9;  12,13,14]
#   PASS window 3 (stream):  [3,4,5;  8,9,10;  13,14,15]
#   === All window3x3_stream tests PASSED ===  (9 valid windows total)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Check GHDL -------------------------------------------------------
if ! command -v ghdl &>/dev/null; then
    echo ""
    echo "ERROR: GHDL not found in PATH."
    echo ""
    echo "Install GHDL:"
    echo "  Ubuntu/Debian : sudo apt install ghdl"
    echo "  Fedora/RHEL   : sudo dnf install ghdl"
    echo "  macOS (brew)  : brew install ghdl"
    echo "  Releases      : https://github.com/ghdl/ghdl/releases"
    echo ""
    echo "Alternatively, use Vivado xsim — see run_vivado_sim_window.tcl"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_window3x3_stream.vcd"
    echo "Waveform: tb_window3x3_stream.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_window3x3_stream.ghw"
    echo "Waveform: tb_window3x3_stream.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 tb_window3x3_stream.vhd
echo "  OK  tb_window3x3_stream.vhd"
echo ""

echo "=== [2/3] Elaborating tb_window3x3_stream ==="
ghdl -e --std=08 tb_window3x3_stream
echo "  OK"
echo ""

# 5×5 image = 25 pixels + reset + drain ≈ 40 cycles × 10 ns = 400 ns; 600 ns is generous
echo "=== [3/3] Running simulation (stop after 600 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_window3x3_stream \
    --stop-time=600ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
