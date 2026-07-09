#!/usr/bin/env bash
# run_ghdl_kernel0_bn_relu_pipelined.sh
# Simulate tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined using GHDL.
#
# This testbench drives stream_conv3x3_3chan_kernel0_bn_relu_pipelined --
# the PIPELINED variant of the kernel-0-only folded Conv -> BatchNorm
# bias/scale -> ReLU PROTOTYPE, which splits the unpipelined design's
# single combinational multiply/add/ReLU stage into two registered stages
# -- with the canonical 5x5x3 toy input and checks all 9 valid output
# positions (Q.16 fixed-point) against the SAME exact integer golden
# vectors from first_conv_bn_relu_kernel0_pkg used by the unpipelined
# design (no new golden arithmetic).
#
# Analyzes (dependency order):
#   window3x3_stream.vhd
#   conv3x3_dot_pipelined.vhd
#   stream_conv3x3_3chan_cell.vhd                     (existing, unmodified)
#   first_conv_bn_relu_kernel0_pkg.vhd                (existing, unmodified)
#   stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd
#   tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined.sh --wave
#
# Expected output:
#   PASS output 1: y_bn_relu_fx = 1522809  (Q.16 fixed-point, ReLU applied, pipelined)
#   ...
#   === All stream_conv3x3_3chan_kernel0_bn_relu_pipelined tests PASSED === (9 / 9 outputs match Python Q.16 fixed-point golden vectors)
#
# Scope note: kernel-0-only folded Conv-BN-ReLU PROTOTYPE. Not all 32
# output channels, not board-tested, not a measured speedup/power claim.

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
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_cell.vhd
echo "  OK  stream_conv3x3_3chan_cell.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel0_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel0_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd
echo "  OK  stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd
echo "  OK  tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 600 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined \
    --stop-time=600ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
