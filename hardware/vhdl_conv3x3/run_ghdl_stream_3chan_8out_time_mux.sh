#!/usr/bin/env bash
# run_ghdl_stream_3chan_8out_time_mux.sh
# Simulate tb_stream_conv3x3_3chan_8out_time_mux using GHDL.
#
# This testbench drives stream_conv3x3_3chan_8out_time_mux -- a STREAMING
# TIME-MULTIPLEXED 8-output PROTOTYPE -- with the canonical 5x5x3 toy image,
# streamed one pixel triplet per clock exactly as for the fully parallel
# 8-output cell, and checks all 9 windows x 8 kernels (72 INT32 values)
# against the same golden vectors baked into first_layer_kernels0_to7_pkg.
#
# This wraps the UNMODIFIED window3x3_stream generators and the UNMODIFIED
# one-window conv3x3_3chan_8out_time_mux scheduler in a simple two-phase
# capture-then-process FSM (see stream_conv3x3_3chan_8out_time_mux.vhd and
# stream_time_mux_8out_summary.md for why this is not a fully overlapped
# streaming pipeline).
#
# Weight source:
#   models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
#   tensor key: enc1.block.0.weight, output channels 0-7
#
# Analyzes (dependency order):
#   window3x3_stream.vhd
#   first_layer_kernels0_to7_pkg.vhd
#   conv3x3_dot_time_mux.vhd
#   conv3x3_3chan_8out_time_mux.vhd
#   stream_conv3x3_3chan_8out_time_mux.vhd
#   tb_stream_conv3x3_3chan_8out_time_mux.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_stream_3chan_8out_time_mux.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_stream_3chan_8out_time_mux.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_stream_3chan_8out_time_mux.sh --wave
#
# Expected output:
#   PASS window 1: y0=11287 y1=9214 ... (@ cycle N)
#   ...
#   PASS window 9: y0=28831 y1=20050 ... (@ cycle M)
#   Total cycles for streaming capture + sequential processing: <measured>
#   === All stream_conv3x3_3chan_8out_time_mux tests PASSED === ...
#
# Scope note: this is a STREAMING, TIME-MULTIPLEXED 8-output PROTOTYPE. It
# is NOT full U-Net FPGA inference, has NOT been run on real board hardware,
# does NOT overlap window capture with compute, does NOT implement
# backpressure into window3x3_stream, and does NOT include a measured
# speedup figure.

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
    echo "  xvhdl --2008 window3x3_stream.vhd first_layer_kernels0_to7_pkg.vhd \\"
    echo "               conv3x3_dot_time_mux.vhd conv3x3_3chan_8out_time_mux.vhd \\"
    echo "               stream_conv3x3_3chan_8out_time_mux.vhd \\"
    echo "               tb_stream_conv3x3_3chan_8out_time_mux.vhd"
    echo "  xelab -debug typical tb_stream_conv3x3_3chan_8out_time_mux -s tb_stream_tm_sim"
    echo "  xsim  tb_stream_tm_sim --runall"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_8out_time_mux.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_8out_time_mux.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_8out_time_mux.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_8out_time_mux.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 first_layer_kernels0_to7_pkg.vhd
echo "  OK  first_layer_kernels0_to7_pkg.vhd"
ghdl -a --std=08 conv3x3_dot_time_mux.vhd
echo "  OK  conv3x3_dot_time_mux.vhd"
ghdl -a --std=08 conv3x3_3chan_8out_time_mux.vhd
echo "  OK  conv3x3_3chan_8out_time_mux.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_8out_time_mux.vhd
echo "  OK  stream_conv3x3_3chan_8out_time_mux.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_8out_time_mux.vhd
echo "  OK  tb_stream_conv3x3_3chan_8out_time_mux.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_8out_time_mux ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_8out_time_mux
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 40000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_8out_time_mux \
    --stop-time=40000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
