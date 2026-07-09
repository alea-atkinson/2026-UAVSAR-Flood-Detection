#!/usr/bin/env bash
# run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh
# *** SYNTHETIC RELU CLAMP UNIT TEST -- NOT a real UAVSAR tile. ***
#
# Simulate tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp
# using GHDL. This testbench drives an ALL-ZERO SYNTHETIC 5x5x3 input
# (not the canonical toy patch) through the pipelined kernel-0 folded
# Conv-BN-ReLU design, forcing every one of the 9 biased_fx values
# negative (BIAS_FX = -936), and checks that the VHDL ReLU clamp branch
# correctly zeroes all 9 outputs against the exact Q.16 fixed-point
# golden vectors from first_conv_bn_relu_kernel0_relu_clamp_pkg.
#
# Analyzes (dependency order):
#   window3x3_stream.vhd
#   conv3x3_dot_pipelined.vhd
#   stream_conv3x3_3chan_cell.vhd                                (existing, unmodified)
#   first_conv_bn_relu_kernel0_relu_clamp_pkg.vhd
#   stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd
#   tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh --wave
#
# Expected output:
#   PASS output 1: y_bn_relu_fx = 0  (Q.16 fixed-point, ReLU-clamped, kernel 0, SYNTHETIC input)
#   ...
#   === ReLU clamp count: 9 / 9 outputs clamped to zero ===
#   === All stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp tests PASSED === (9 / 9 outputs match Python Q.16 fixed-point golden vectors, ReLU clamp branch exercised)
#
# Scope note: SYNTHETIC unit test input, NOT a real UAVSAR tile, NOT real
# model inference. Kernel-0-only proof-of-concept. Not all 32 output
# channels, not board-tested, not a measured speedup/power claim.

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
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_cell.vhd
echo "  OK  stream_conv3x3_3chan_cell.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel0_relu_clamp_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel0_relu_clamp_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd
echo "  OK  stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd
echo "  OK  tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 600 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp \
    --stop-time=600ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
