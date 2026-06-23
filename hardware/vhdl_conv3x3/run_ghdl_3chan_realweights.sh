#!/usr/bin/env bash
# run_ghdl_3chan_realweights.sh
# Simulate tb_stream_conv3x3_3chan_realweights using GHDL.
#
# This testbench drives stream_conv3x3_3chan_cell with INT8 weights extracted
# from the real Alea-tuned U-Net checkpoint and checks the 9 expected INT32
# outputs against Python golden vectors.
#
# Weight source:
#   models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
#   tensor key: enc1.block.0.weight, output channel 0
#   scale_w = 0.00136063  (symmetric per-tensor INT8)
#
# Analyzes (dependency order):
#   window3x3_stream.vhd
#   conv3x3_dot_pipelined.vhd
#   stream_conv3x3_3chan_cell.vhd
#   tb_stream_conv3x3_3chan_realweights.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_realweights.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_realweights.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_realweights.sh --wave
#
# Expected output:
#   PASS output 1 (stream_conv3x3_3chan_realweights): y = 11287
#   PASS output 2 (stream_conv3x3_3chan_realweights): y = 12749
#   ...
#   PASS output 9 (stream_conv3x3_3chan_realweights): y = 28831
#   === All stream_conv3x3_3chan_realweights tests PASSED ===

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
    echo "               stream_conv3x3_3chan_cell.vhd tb_stream_conv3x3_3chan_realweights.vhd"
    echo "  xelab -debug typical tb_stream_conv3x3_3chan_realweights -s tb_rw_sim"
    echo "  xsim  tb_rw_sim --runall"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_realweights.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_realweights.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_realweights.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_realweights.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_cell.vhd
echo "  OK  stream_conv3x3_3chan_cell.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_realweights.vhd
echo "  OK  tb_stream_conv3x3_3chan_realweights.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_realweights ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_realweights
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 600 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_realweights \
    --stop-time=600ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
