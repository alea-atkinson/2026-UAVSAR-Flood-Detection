#!/usr/bin/env bash
# run_ghdl_3chan_cell.sh
# Analyze, elaborate, and simulate the stream_conv3x3_3chan_cell testbench.
#
# Analyzes (in dependency order):
#   window3x3_stream.vhd             -- sliding window generator
#   conv3x3_dot_pipelined.vhd        -- 3-stage pipelined dot product
#   stream_conv3x3_3chan_cell.vhd     -- 3-channel integrated cell
#   tb_stream_conv3x3_3chan_cell.vhd  -- self-checking testbench
#
# Prerequisites:
#   GHDL: sudo apt install ghdl   (Ubuntu/Debian)
#          sudo dnf install ghdl   (Fedora)
#          brew install ghdl       (macOS)
#          https://github.com/ghdl/ghdl/releases
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_cell.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_cell.sh --vcd    # VCD waveform
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_cell.sh --wave   # GHW waveform
#
# Expected output:
#   PASS output 1 (stream_conv3x3_3chan_cell): y = -2
#   ...
#   PASS output 9 (stream_conv3x3_3chan_cell): y = -2
#   === All stream_conv3x3_3chan_cell tests PASSED ===  (9 / 9 outputs, all y = -2 ...)

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
    echo "Alternatively, use Vivado xsim:"
    echo "  xvhdl --2008 window3x3_stream.vhd conv3x3_dot_pipelined.vhd \\"
    echo "               stream_conv3x3_3chan_cell.vhd tb_stream_conv3x3_3chan_cell.vhd"
    echo "  xelab -debug typical tb_stream_conv3x3_3chan_cell -s tb_3chan_sim"
    echo "  xsim  tb_3chan_sim --runall"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_cell.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_cell.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_cell.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_cell.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources (dependency order) ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_cell.vhd
echo "  OK  stream_conv3x3_3chan_cell.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_cell.vhd
echo "  OK  tb_stream_conv3x3_3chan_cell.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_cell ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_cell
echo "  OK"
echo ""

# Simulation budget:
#   3 reset + 1 idle + 25 pixel + 4 drain = 33 clocks × 10 ns = 330 ns
# Use 600 ns.
echo "=== [3/3] Running simulation (stop after 600 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_cell \
    --stop-time=600ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
