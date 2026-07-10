#!/usr/bin/env bash
# run_ghdl_32out_bn_relu_resource_shared.sh
# Simulate tb_stream_conv3x3_3chan_32out_bn_relu_time_mux using GHDL.
#
# This runs the COMPLETE (32-of-32-kernel) resource-shared (time
# -multiplexed) folded Conv-BN-ReLU prototype -- the full scale-up of the
# smaller, already-verified 4-kernel prototype (preserved unmodified as
# conv3x3_3chan_4out_bn_relu_time_mux.vhd /
# stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd, still separately
# runnable via run_ghdl_4out_bn_relu_resource_shared.sh). This design
# covers ALL 32 first-layer output channels (the complete first Conv2d
# layer), reusing ONE dot-product engine and ONE Q.16 fixed-point BN+ReLU
# unit across all 32 kernels and all 9 valid window positions of the
# canonical 5x5x3 toy input.
#
# Analyzes (dependency order):
#   window3x3_stream.vhd                                (existing, unmodified)
#   conv3x3_dot_time_mux.vhd                             (existing, unmodified)
#   first_layer_32out_bn_relu_resource_shared_pkg.vhd
#   conv3x3_3chan_32out_bn_relu_time_mux.vhd
#   stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd
#   tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared.sh --wave
#
# Expected output:
#   PASS window 1: y0=1522809 y1=1988771 ... y31=... (@ cycle N)
#   ...
#   === All stream_conv3x3_3chan_32out_bn_relu_time_mux tests PASSED === (9 windows x 32 kernels = 288 / 288 outputs match Python Q.16 fixed-point golden vectors)
#
# Scope note: RESOURCE-SHARED prototype, complete 32-output first Conv2d
# layer. Not full U-Net inference, not board-tested, not a measured
# speedup/power claim, not a real UAVSAR inference example (canonical toy
# input only).

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
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_time_mux.vhd
echo "  OK  conv3x3_dot_time_mux.vhd"
ghdl -a --std=08 first_layer_32out_bn_relu_resource_shared_pkg.vhd
echo "  OK  first_layer_32out_bn_relu_resource_shared_pkg.vhd"
ghdl -a --std=08 conv3x3_3chan_32out_bn_relu_time_mux.vhd
echo "  OK  conv3x3_3chan_32out_bn_relu_time_mux.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd
echo "  OK  stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd
echo "  OK  tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_32out_bn_relu_time_mux ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_32out_bn_relu_time_mux
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 400000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_32out_bn_relu_time_mux \
    --stop-time=400000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
