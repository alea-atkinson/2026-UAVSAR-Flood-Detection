#!/usr/bin/env bash
# run_ghdl_3chan_8out_time_mux.sh
# Simulate tb_conv3x3_3chan_8out_time_mux using GHDL.
#
# This testbench drives conv3x3_3chan_8out_time_mux -- a time-multiplexed
# 8-output SCHEDULER PROTOTYPE for ONE 3-channel 3x3 spatial window -- with
# the canonical toy input's top-left 3x3 patch, and checks all 8 kernel
# outputs (y0..y7) against the same golden vectors used to verify the fully
# parallel 4-output and 8-output prototypes. It also measures the total
# clock-cycle count from start to done.
#
# Weight source:
#   models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
#   tensor key: enc1.block.0.weight, output channels 0-7
#
# Analyzes (dependency order):
#   first_layer_kernels0_to7_pkg.vhd
#   conv3x3_dot_time_mux.vhd
#   conv3x3_3chan_8out_time_mux.vhd
#   tb_conv3x3_3chan_8out_time_mux.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_8out_time_mux.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_8out_time_mux.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_3chan_8out_time_mux.sh --wave
#
# Expected output:
#   Scheduler done after 265 cycles (measured; see time_mux_8out_summary.md for the breakdown)
#   PASS kernel 0 (top-left position): y = 11287  (matches KERNEL0_EXPECTED(0))
#   ...
#   PASS kernel 7 (top-left position): y = -2419  (matches KERNEL7_EXPECTED(0))
#   === All conv3x3_3chan_8out_time_mux tests PASSED === ...
#
# Scope note: this is a ONE-WINDOW, 8-output, time-multiplexed SCHEDULER
# prototype, not a full streaming convolution engine. It is NOT full U-Net
# FPGA inference, has NOT been run on real board hardware, and does NOT
# include a measured speedup figure.

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
    echo "  xvhdl --2008 first_layer_kernels0_to7_pkg.vhd conv3x3_dot_time_mux.vhd \\"
    echo "               conv3x3_3chan_8out_time_mux.vhd \\"
    echo "               tb_conv3x3_3chan_8out_time_mux.vhd"
    echo "  xelab -debug typical tb_conv3x3_3chan_8out_time_mux -s tb_8out_tm_sim"
    echo "  xsim  tb_8out_tm_sim --runall"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_conv3x3_3chan_8out_time_mux.vcd"
    echo "Waveform: tb_conv3x3_3chan_8out_time_mux.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_conv3x3_3chan_8out_time_mux.ghw"
    echo "Waveform: tb_conv3x3_3chan_8out_time_mux.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 first_layer_kernels0_to7_pkg.vhd
echo "  OK  first_layer_kernels0_to7_pkg.vhd"
ghdl -a --std=08 conv3x3_dot_time_mux.vhd
echo "  OK  conv3x3_dot_time_mux.vhd"
ghdl -a --std=08 conv3x3_3chan_8out_time_mux.vhd
echo "  OK  conv3x3_3chan_8out_time_mux.vhd"
ghdl -a --std=08 tb_conv3x3_3chan_8out_time_mux.vhd
echo "  OK  tb_conv3x3_3chan_8out_time_mux.vhd"
echo ""

echo "=== [2/3] Elaborating tb_conv3x3_3chan_8out_time_mux ==="
ghdl -e --std=08 tb_conv3x3_3chan_8out_time_mux
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 10000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_conv3x3_3chan_8out_time_mux \
    --stop-time=10000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
