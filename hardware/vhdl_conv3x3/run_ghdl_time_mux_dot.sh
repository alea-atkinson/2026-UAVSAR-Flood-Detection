#!/usr/bin/env bash
# run_ghdl_time_mux_dot.sh
# Simulate tb_conv3x3_dot_time_mux using GHDL.
#
# This testbench drives conv3x3_dot_time_mux -- a single-lane, time-multiplexed
# 3x3 dot-product PROTOTYPE (one reusable MAC, 9 cycles per dot product) --
# with real trained-model weights from kernels 0, 1, and 7 of
# first_layer_kernels0_to7_pkg, and the canonical toy input's top-left 3x3
# window per channel. It checks that summing three time-multiplexed
# per-channel dot products reproduces the same golden top-left output value
# used to verify the fully parallel 4-output and 8-output prototypes, and
# that each dot product takes exactly 10 clock cycles (start to done).
#
# Weight source:
#   models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
#   tensor key: enc1.block.0.weight, output channels 0, 1, 7 (of 0-7 exported)
#
# Analyzes (dependency order):
#   first_layer_kernels0_to7_pkg.vhd
#   conv3x3_dot_time_mux.vhd
#   tb_conv3x3_dot_time_mux.vhd
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_time_mux_dot.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_time_mux_dot.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_time_mux_dot.sh --wave
#
# Expected output:
#   PASS kernel 0 (top-left position): summed y = 11287  (matches KERNEL0_EXPECTED(0) = 11287), 10 cycles per channel dot product
#   PASS kernel 1 (top-left position): summed y = 9214  (matches KERNEL1_EXPECTED(0) = 9214), 10 cycles per channel dot product
#   PASS kernel 7 (top-left position): summed y = -2419  (matches KERNEL7_EXPECTED(0) = -2419), 10 cycles per channel dot product
#   === All conv3x3_dot_time_mux tests PASSED === ...
#
# Scope note: this is a single dot-product time-multiplexing PROTOTYPE, not
# a full time-multiplexed convolution engine or scheduler. It is NOT full
# U-Net FPGA inference, has NOT been run on real board hardware, and does
# NOT include a measured speedup figure.

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
    echo "               tb_conv3x3_dot_time_mux.vhd"
    echo "  xelab -debug typical tb_conv3x3_dot_time_mux -s tb_time_mux_sim"
    echo "  xsim  tb_time_mux_sim --runall"
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_conv3x3_dot_time_mux.vcd"
    echo "Waveform: tb_conv3x3_dot_time_mux.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_conv3x3_dot_time_mux.ghw"
    echo "Waveform: tb_conv3x3_dot_time_mux.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 first_layer_kernels0_to7_pkg.vhd
echo "  OK  first_layer_kernels0_to7_pkg.vhd"
ghdl -a --std=08 conv3x3_dot_time_mux.vhd
echo "  OK  conv3x3_dot_time_mux.vhd"
ghdl -a --std=08 tb_conv3x3_dot_time_mux.vhd
echo "  OK  tb_conv3x3_dot_time_mux.vhd"
echo ""

echo "=== [2/3] Elaborating tb_conv3x3_dot_time_mux ==="
ghdl -e --std=08 tb_conv3x3_dot_time_mux
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 3000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_conv3x3_dot_time_mux \
    --stop-time=3000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
