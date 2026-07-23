#!/usr/bin/env bash
# run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh
# Simulate tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20
# using GHDL -- the ALL-32-KERNEL extension of
# run_ghdl_first_layer_real_tile_bn_relu_q20.sh (kernel 0 + kernel 2 only).
#
# This testbench drives 32 EXISTING/UNMODIFIED-STRUCTURE folded
# Conv-BN-ReLU pipelined datapaths (one per output channel of the first
# Conv2d layer, all structurally identical, differing only in their
# folded weights and Q.20 SCALE_FX/BIAS_FX constants) with the SAME REAL,
# per-tile-normalized-and-quantized 6x6x3 patch (tile_16_42.tif) already
# used for the kernel0/kernel2 subset.
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_all32_kernels_real_tile_bn_relu_q20.sh --wave
#
# Scope note: simulation-only, real-data verification of all 32 first-layer
# output channels on ONE real 6x6 sub-block. NOT full FPGA U-Net, NOT
# board deployment, NOT full-image streaming.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v ghdl &>/dev/null; then
    echo ""
    echo "ERROR: GHDL not found in PATH."
    echo ""
    exit 1
fi

echo "Using: $(ghdl --version | head -1)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.ghw"
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

ghdl -a --std=08 first_conv_bn_relu_kernel0_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel0_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel1_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel1_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel1_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel1_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel2_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel2_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel2_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel2_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel3_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel3_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel3_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel3_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel4_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel4_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel4_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel4_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel5_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel5_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel5_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel5_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel6_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel6_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel6_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel6_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel7_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel7_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel7_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel7_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel8_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel8_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel8_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel8_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel9_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel9_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel9_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel9_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel10_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel10_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel10_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel10_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel11_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel11_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel11_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel11_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel12_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel12_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel12_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel12_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel13_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel13_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel13_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel13_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel14_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel14_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel14_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel14_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel15_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel15_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel15_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel15_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel16_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel16_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel16_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel16_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel17_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel17_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel17_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel17_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel18_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel18_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel18_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel18_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel19_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel19_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel19_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel19_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel20_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel20_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel20_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel20_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel21_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel21_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel21_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel21_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel22_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel22_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel22_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel22_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel23_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel23_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel23_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel23_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel24_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel24_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel24_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel24_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel25_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel25_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel25_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel25_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel26_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel26_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel26_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel26_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel27_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel27_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel27_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel27_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel28_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel28_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel28_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel28_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel29_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel29_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel29_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel29_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel30_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel30_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel30_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel30_bn_relu_real_tile_q20.vhd"
ghdl -a --std=08 first_conv_bn_relu_kernel31_real_tile_q20_pkg.vhd
echo "  OK  first_conv_bn_relu_kernel31_real_tile_q20_pkg.vhd"
ghdl -a --std=08 stream_conv3x3_3chan_kernel31_bn_relu_real_tile_q20.vhd
echo "  OK  stream_conv3x3_3chan_kernel31_bn_relu_real_tile_q20.vhd"

ghdl -a --std=08 tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd
echo "  OK  tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 ==="
ghdl -e --std=08 tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 1000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_stream_conv3x3_3chan_all32_kernels_real_tile_bn_relu_q20 \
    --stop-time=1000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
