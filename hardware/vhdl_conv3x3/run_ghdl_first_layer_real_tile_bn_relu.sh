#!/usr/bin/env bash
# run_ghdl_first_layer_real_tile_bn_relu.sh
# Simulate tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu using GHDL.
#
# This testbench drives the EXISTING, UNMODIFIED kernel-0 and kernel-2
# folded Conv-BN-ReLU pipelined datapaths (structurally identical to
# stream_conv3x3_3chan_kernel{0,2}_bn_relu_pipelined) with a REAL, per-tile
# -normalized-and-quantized 6x6x3 patch cut from a real fp2 held-out UAVSAR
# tile (tile_16_42.tif), instead of the synthetic 5x5x3 toy patch used by
# every other VHDL testbench in this repo. This is a REAL-DATA
# VERIFICATION SUBSET (2 of the complete first Conv2d layer's 32 output
# channels), not full 32-output real-tile verification.
#
# Analyzes (dependency order):
#   window3x3_stream.vhd                                    (existing, unmodified)
#   conv3x3_dot_pipelined.vhd                                (existing, unmodified)
#   stream_conv3x3_3chan_cell.vhd                            (existing, unmodified)
#   real_tile_stimulus_pkg.vhd                               (new)
#   first_conv_bn_relu_kernel0_real_tile_pkg.vhd             (new)
#   first_conv_bn_relu_kernel2_real_tile_pkg.vhd             (new)
#   stream_conv3x3_3chan_kernel0_bn_relu_real_tile.vhd       (new)
#   stream_conv3x3_3chan_kernel2_bn_relu_real_tile.vhd       (new)
#   tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.vhd (new)
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_first_layer_real_tile_bn_relu.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_first_layer_real_tile_bn_relu.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_first_layer_real_tile_bn_relu.sh --wave
#
# Expected output:
#   PASS kernel 0 output 1: y_bn_relu_fx = 0  (REAL TILE, Q.16 fixed-point, ReLU applied)
#   ...
#   === REAL-TILE first-layer Conv-BN-ReLU verification summary ===
#       Kernel 0: 16 / 16 outputs observed
#       Kernel 2: 16 / 16 outputs observed
#       PASS count = 32   FAIL count = 0
#       Max absolute integer difference (Q.16 fixed-point) = 0
#   === All tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu tests PASSED ===
#
# Scope note: real-data verification SUBSET (kernel 0 and kernel 2 of 32
# output channels), NOT board-tested, not a measured speedup/power claim.

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
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined.vhd
echo "  OK  conv3x3_dot_pipelined.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_cell.vhd
echo "  OK  stream_conv3x3_3chan_cell.vhd"
ghdl -a --std=08 real_tile_stimulus_pkg.vhd
echo "  OK  real_tile_stimulus_pkg.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel0_real_tile_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel0_real_tile_pkg.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel2_real_tile_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel2_real_tile_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel0_bn_relu_real_tile.vhd
echo "  OK  stream_conv3x3_3chan_kernel0_bn_relu_real_tile.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel2_bn_relu_real_tile.vhd
echo "  OK  stream_conv3x3_3chan_kernel2_bn_relu_real_tile.vhd"
ghdl -a --std=08 tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.vhd
echo "  OK  tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 1000 ns) ==="
# Note: --assert-level=failure only stops on severity=failure; per-check
# mismatches in this testbench are reported at severity=warning so the
# simulation can compute a max-absolute-difference summary across ALL
# checks. A single severity=failure assertion fires at the very end only
# if any check failed.
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu \
    --stop-time=1000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
