#!/usr/bin/env bash
# run_ghdl_window3x3_stream_3chan_flattened.sh
# Simulate tb_window3x3_stream_3chan_flattened using GHDL -- an END-TO-END
# feasibility check of streaming a real pixel image through a NEW 3-channel
# front-end wrapper (window3x3_stream_3chan_flattened.vhd, built from 3
# EXISTING, UNMODIFIED window3x3_stream instances) directly into the
# EXISTING, UNMODIFIED direct-parallel folded 32-output arithmetic core
# (first_layer_32out_folded_arithmetic_core.vhd), via a tiny integrated
# skeleton (first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd).
#
# No isolated work library is needed here (unlike the resource-shared
# real-tile run): every package referenced already has the name its
# consumer expects -- nothing is substituted under an existing name.
#
# Analyzes (dependency order):
#   window3x3_stream.vhd                                          (existing, unmodified)
#   window3x3_stream_3chan_flattened.vhd                          (NEW -- 3-channel front-end wrapper)
#   conv3x3_dot_pipelined_dsp.vhd                                  (existing, unmodified)
#   first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd        (existing, unmodified)
#   first_layer_32out_folded_arithmetic_core.vhd                  (existing, unmodified)
#   first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd  (NEW -- integrated skeleton)
#   real_tile_stimulus_pkg.vhd                                     (existing, unmodified -- REAL 6x6 block)
#   first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd         (existing, unmodified -- golden vectors)
#   tb_window3x3_stream_3chan_flattened.vhd                        (NEW)
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_window3x3_stream_3chan_flattened.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_window3x3_stream_3chan_flattened.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_window3x3_stream_3chan_flattened.sh --wave
#
# Scope note: simulation-only front-end feasibility check. Not full-tile
# board behavior, not full U-Net inference, not board-tested, not a
# measured speedup/power claim.

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
    EXTRA_RUN_FLAGS="--vcd=tb_window3x3_stream_3chan_flattened.vcd"
    echo "Waveform: tb_window3x3_stream_3chan_flattened.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_window3x3_stream_3chan_flattened.ghw"
    echo "Waveform: tb_window3x3_stream_3chan_flattened.ghw"
fi

echo "=== [1/3] Analyzing VHDL-2008 sources ==="
ghdl -a --std=08 window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
ghdl -a --std=08 window3x3_stream_3chan_flattened.vhd
echo "  OK  window3x3_stream_3chan_flattened.vhd"
ghdl -a --std=08 conv3x3_dot_pipelined_dsp.vhd
echo "  OK  conv3x3_dot_pipelined_dsp.vhd"
ghdl -a --std=08 first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
echo "  OK  first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd"
ghdl -a --std=08 first_layer_32out_folded_arithmetic_core.vhd
echo "  OK  first_layer_32out_folded_arithmetic_core.vhd"
ghdl -a --std=08 first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd
echo "  OK  first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd"
ghdl -a --std=08 real_tile_stimulus_pkg.vhd
echo "  OK  real_tile_stimulus_pkg.vhd"

for k in $(seq 0 31); do
    ghdl -a --std=08 "first_conv_bn_relu_kernel${k}_real_tile_q20_pkg.vhd"
done
echo "  OK  first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd (32 files)"

ghdl -a --std=08 tb_window3x3_stream_3chan_flattened.vhd
echo "  OK  tb_window3x3_stream_3chan_flattened.vhd"
echo ""

echo "=== [2/3] Elaborating tb_window3x3_stream_3chan_flattened ==="
ghdl -e --std=08 tb_window3x3_stream_3chan_flattened
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 1000 ns) ==="
# shellcheck disable=SC2086
ghdl -r --std=08 tb_window3x3_stream_3chan_flattened \
    --stop-time=1000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
