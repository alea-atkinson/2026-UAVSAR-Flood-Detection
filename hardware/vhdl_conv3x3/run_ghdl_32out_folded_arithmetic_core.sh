#!/usr/bin/env bash
# run_ghdl_32out_folded_arithmetic_core.sh
# Simulate tb_first_layer_32out_folded_arithmetic_core using GHDL.
#
# Tests the single-shot (not streamed) 32-output folded Conv-BN-ReLU
# arithmetic-core prototype: one already-flattened 27-value window per
# cycle in, all 32 Q.20 fixed-point folded Conv-BN-ReLU outputs 6 cycles
# later, 1 window/cycle throughput -- the concrete hardware test of
# scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py's assumed
# FPGA architecture.
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_folded_arithmetic_core.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_folded_arithmetic_core.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_folded_arithmetic_core.sh --wave
#
# Scope: simulation-only. NOT image streaming, NOT full U-Net, NOT board
# deployment, NOT a measured speedup or power claim.

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
    EXTRA_RUN_FLAGS="--vcd=tb_first_layer_32out_folded_arithmetic_core.vcd"
    echo "Waveform: tb_first_layer_32out_folded_arithmetic_core.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_first_layer_32out_folded_arithmetic_core.ghw"
    echo "Waveform: tb_first_layer_32out_folded_arithmetic_core.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 conv3x3_dot_pipelined_dsp.vhd
echo "  OK  conv3x3_dot_pipelined_dsp.vhd"
ghdl -a --std=08 first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
echo "  OK  first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd"
ghdl -a --std=08 first_layer_32out_folded_arithmetic_core.vhd
echo "  OK  first_layer_32out_folded_arithmetic_core.vhd"
ghdl -a --std=08 tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.vhd
echo "  OK  tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.vhd"
ghdl -a --std=08 tb_first_layer_32out_folded_arithmetic_core.vhd
echo "  OK  tb_first_layer_32out_folded_arithmetic_core.vhd"
echo ""

echo "=== [2/3] Elaborating tb_first_layer_32out_folded_arithmetic_core ==="
ghdl -e --std=08 tb_first_layer_32out_folded_arithmetic_core
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 500 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_first_layer_32out_folded_arithmetic_core \
    --stop-time=500ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
