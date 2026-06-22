#!/usr/bin/env bash
# run_ghdl_stream_cell.sh
# Analyze, elaborate, and simulate the stream_conv3x3_cell testbench using GHDL.
#
# Analyzes (in dependency order):
#   window3x3_stream.vhd        -- sliding window generator
#   conv3x3_dot_pipelined.vhd   -- 3-stage pipelined dot product
#   stream_conv3x3_cell.vhd     -- top-level integrated cell
#   tb_stream_conv3x3_cell.vhd  -- self-checking testbench
#
# Prerequisites:
#   GHDL: sudo apt install ghdl  (Ubuntu/Debian)
#          sudo dnf install ghdl  (Fedora)
#          brew install ghdl      (macOS)
#          https://github.com/ghdl/ghdl/releases
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_stream_cell.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_stream_cell.sh --vcd   # + GTKWave waveform
#   bash hardware/vhdl_conv3x3/run_ghdl_stream_cell.sh --wave
#
# Expected output:
#   PASS output 1 (stream_conv3x3_cell): y = -6
#   PASS output 2 (stream_conv3x3_cell): y = -6
#   ...
#   PASS output 9 (stream_conv3x3_cell): y = -6
#   === All stream_conv3x3_cell tests PASSED ===  (9 / 9 outputs, all y = -6)

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
    echo "Alternatively, use Vivado xsim — see run_vivado_sim_stream_cell.tcl"
    echo "  xvhdl --2008 window3x3_stream.vhd conv3x3_dot_pipelined.vhd stream_conv3x3_cell.vhd tb_stream_conv3x3_cell.vhd"
    echo "  xelab -debug typical tb_stream_conv3x3_cell -s tb_cell_sim"
    echo "  xsim  tb_cell_sim --runall"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_cell.vcd"
    echo "Waveform: tb_stream_conv3x3_cell.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_cell.ghw"
    echo "Waveform: tb_stream_conv3x3_cell.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources (dependency order) ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_cell.vhd
echo "  OK  stream_conv3x3_cell.vhd"
ghdl -a --std=08 tb_stream_conv3x3_cell.vhd
echo "  OK  tb_stream_conv3x3_cell.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_cell ==="
ghdl -e --std=08 tb_stream_conv3x3_cell
echo "  OK"
echo ""

# Total simulation budget:
#   3 reset clocks + 1 idle + 25 pixel clocks + 3 drain clocks = 32 clocks × 10 ns = 320 ns
# Use 500 ns to be safe.
echo "=== [3/3] Running simulation (stop after 500 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_cell \
    --stop-time=500ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
