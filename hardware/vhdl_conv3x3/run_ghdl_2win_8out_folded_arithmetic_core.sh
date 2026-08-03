#!/usr/bin/env bash
# run_ghdl_2win_8out_folded_arithmetic_core.sh
# Simulate tb_first_layer_2win_8out_folded_arithmetic_core using GHDL --
# REAL-UAVSAR-ACTIVATION validation of the NEW 2-window x 8-output
# spatial-parallelism folded Conv-BN-ReLU arithmetic core.
#
# Analyzes (dependency order, into the shared default `work` library --
# no package-name collisions with any existing file in this directory):
#   conv3x3_dot_pipelined_dsp.vhd                           (existing, unmodified)
#   first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd   (NEW -- 8-kernel slice)
#   first_layer_2win_8out_folded_arithmetic_core.vhd        (NEW)
#   real_tile_stimulus_pkg.vhd                              (existing, unmodified -- REAL 6x6 block)
#   first_conv_bn_relu_kernel{0..7}_real_tile_q20_pkg.vhd   (existing, unmodified -- golden vectors)
#   tb_first_layer_2win_8out_folded_arithmetic_core.vhd     (NEW)
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_2win_8out_folded_arithmetic_core.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_2win_8out_folded_arithmetic_core.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_2win_8out_folded_arithmetic_core.sh --wave
#
# Scope: simulation-only, real-UAVSAR-activation check of the NEW
# arithmetic-core-only spatial-parallelism artifact. Pre-extracted windows
# only -- no streaming front end exists for this core yet (future work).
# Not full-tile streaming, not full U-Net inference, not board-tested, not
# a measured speedup/power claim.

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
    EXTRA_RUN_FLAGS="--vcd=tb_first_layer_2win_8out_folded_arithmetic_core.vcd"
    echo "Waveform: tb_first_layer_2win_8out_folded_arithmetic_core.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_first_layer_2win_8out_folded_arithmetic_core.ghw"
    echo "Waveform: tb_first_layer_2win_8out_folded_arithmetic_core.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 conv3x3_dot_pipelined_dsp.vhd
echo "  OK  conv3x3_dot_pipelined_dsp.vhd"
ghdl -a --std=08 first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd
echo "  OK  first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd"
ghdl -a --std=08 first_layer_2win_8out_folded_arithmetic_core.vhd
echo "  OK  first_layer_2win_8out_folded_arithmetic_core.vhd"
ghdl -a --std=08 real_tile_stimulus_pkg.vhd
echo "  OK  real_tile_stimulus_pkg.vhd"

for k in $(seq 0 7); do
    ghdl -a --std=08 "first_conv_bn_relu_kernel${k}_real_tile_q20_pkg.vhd"
done
echo "  OK  first_conv_bn_relu_kernel{0..7}_real_tile_q20_pkg.vhd (8 files)"

ghdl -a --std=08 tb_first_layer_2win_8out_folded_arithmetic_core.vhd
echo "  OK  tb_first_layer_2win_8out_folded_arithmetic_core.vhd"
echo ""

echo "=== [2/3] Elaborating tb_first_layer_2win_8out_folded_arithmetic_core ==="
ghdl -e --std=08 tb_first_layer_2win_8out_folded_arithmetic_core
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 500 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_first_layer_2win_8out_folded_arithmetic_core \
    --stop-time=500ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
