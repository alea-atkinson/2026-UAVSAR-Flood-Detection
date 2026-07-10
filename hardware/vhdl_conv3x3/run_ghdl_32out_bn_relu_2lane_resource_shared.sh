#!/usr/bin/env bash
# run_ghdl_32out_bn_relu_2lane_resource_shared.sh
# Simulate tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux using GHDL.
#
# This runs the 2-LANE resource-shared (time-multiplexed) folded
# Conv-BN-ReLU design -- the first multi-lane implementation described in
# multi_lane_resource_shared_conv_bn_relu_design_memo.md. Lane 0 (kernels
# 0-15) and lane 1 (kernels 16-31) each have their own time-multiplexed
# dot-product engine and their own shared Q.16 BN+ReLU unit, and run IN
# PARALLEL on the same captured window (unlike the single-lane 32-output
# design, preserved unmodified and still separately runnable via
# run_ghdl_32out_bn_relu_resource_shared.sh, which processes all 32
# kernels sequentially through one shared engine).
#
# Analyzes (dependency order):
#   window3x3_stream.vhd                                     (existing, unmodified)
#   conv3x3_dot_time_mux.vhd                                  (existing, unmodified)
#   first_layer_32out_bn_relu_resource_shared_pkg.vhd         (existing, unmodified -- golden vectors reused)
#   conv3x3_3chan_16out_bn_relu_time_mux.vhd                  (new -- one lane engine, KERNEL_OFFSET generic)
#   stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd     (new -- 2-lane top wrapper + barrier)
#   tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd  (new)
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_2lane_resource_shared.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_2lane_resource_shared.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_2lane_resource_shared.sh --wave
#
# Expected output:
#   PASS window 1: y0=1522809 y1=1988771 ... y31=... (@ cycle N)
#   ...
#   === All stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux tests PASSED === (9 windows x 32 kernels = 288 / 288 outputs match Python Q.16 fixed-point golden vectors, REUSED UNCHANGED from the 1-lane design's package)
#
# Scope note: 2-LANE RESOURCE-SHARED prototype, complete 32-output first
# Conv2d layer. Not full U-Net inference, not board-tested, not a
# measured speedup/power claim, not a real UAVSAR inference example
# (canonical toy input only).

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
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_time_mux.vhd
echo "  OK  conv3x3_dot_time_mux.vhd"
ghdl -a --std=08 first_layer_32out_bn_relu_resource_shared_pkg.vhd
echo "  OK  first_layer_32out_bn_relu_resource_shared_pkg.vhd"
ghdl -a --std=08 conv3x3_3chan_16out_bn_relu_time_mux.vhd
echo "  OK  conv3x3_3chan_16out_bn_relu_time_mux.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd
echo "  OK  stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd
echo "  OK  tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 400000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux \
    --stop-time=400000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
