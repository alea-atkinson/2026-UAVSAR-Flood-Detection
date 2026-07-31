#!/usr/bin/env bash
# run_ghdl_32out_bn_relu_resource_shared_real_tile_q20.sh
# Simulate tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20
# using GHDL -- REAL-UAVSAR-ACTIVATION validation of the EXISTING,
# UNMODIFIED 2-lane resource-shared (time-multiplexed) folded Conv-BN-ReLU
# design (stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd).
#
# ---- Isolated GHDL work library (--workdir) --------------------------------
# This run analyzes a NEW package (resource_shared_32out_real_tile_q20_vectors_pkg.vhd)
# that is declared under the SAME package name the resource-shared design's
# dependencies already reference (`first_layer_32out_bn_relu_resource_shared_pkg`),
# but with REAL-TILE Q.20 constants instead of the existing file's Q.16
# toy-pattern constants. To avoid ANY chance of this overwriting the
# existing toy-pattern package's compiled unit in the shared default `work`
# library used by run_ghdl_32out_bn_relu_2lane_resource_shared.sh and
# run_ghdl_32out_bn_relu_resource_shared.sh in this same directory, this
# script analyzes EVERYTHING (including the existing, unmodified datapath
# source files, re-analyzed from their original .vhd text -- no source
# file is touched) into a SEPARATE, ISOLATED work library directory. This
# is a purely local GHDL compilation artifact directory; it does not
# affect any other script's compiled library state.
#
# Analyzes (dependency order, into the isolated library):
#   window3x3_stream.vhd                                          (existing, unmodified)
#   conv3x3_dot_time_mux.vhd                                       (existing, unmodified)
#   resource_shared_32out_real_tile_q20_vectors_pkg.vhd            (NEW -- real-tile Q.20 constants,
#                                                                     package name matches the existing
#                                                                     toy package's name)
#   conv3x3_3chan_16out_bn_relu_time_mux.vhd                       (existing, unmodified)
#   stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd          (existing, unmodified)
#   real_tile_stimulus_pkg.vhd                                     (existing, unmodified -- REAL 6x6 block)
#   first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd         (existing, unmodified -- golden vectors,
#                                                                     REUSED from the direct-parallel
#                                                                     single-kernel-pipeline verification)
#   tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.vhd  (NEW)
#
# Usage:
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared_real_tile_q20.sh
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared_real_tile_q20.sh --vcd
#   bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared_real_tile_q20.sh --wave
#
# Scope note: REAL-UAVSAR-ACTIVATION validation of the EXISTING 2-lane
# resource-shared prototype only. Not full-tile streaming, not full U-Net
# inference, not board-tested, not a measured speedup/power claim.

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

WORKDIR="ghdl_work_resource_shared_real_tile_q20"
mkdir -p "$WORKDIR"
echo "Isolated GHDL work library: $WORKDIR/ (does not affect the shared" \
     "default work library used by the toy-pattern resource-shared testbenches)"
echo ""

EXTRA_RUN_FLAGS=""
if [[ "${1:-}" == "--vcd" ]]; then
    EXTRA_RUN_FLAGS="--vcd=tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.vcd"
    echo "Waveform: tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.vcd  (open with GTKWave)"
elif [[ "${1:-}" == "--wave" ]]; then
    EXTRA_RUN_FLAGS="--wave=tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.ghw"
    echo "Waveform: tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.ghw"
fi

GHDL_A="ghdl -a --std=08 --workdir=$WORKDIR -P$WORKDIR"
GHDL_E="ghdl -e --std=08 --workdir=$WORKDIR -P$WORKDIR"
GHDL_R="ghdl -r --std=08 --workdir=$WORKDIR -P$WORKDIR"

echo "=== [1/3] Analyzing VHDL-2008 sources into the isolated work library ==="
$GHDL_A window3x3_stream.vhd
echo "  OK  window3x3_stream.vhd"
$GHDL_A conv3x3_dot_time_mux.vhd
echo "  OK  conv3x3_dot_time_mux.vhd"
$GHDL_A resource_shared_32out_real_tile_q20_vectors_pkg.vhd
echo "  OK  resource_shared_32out_real_tile_q20_vectors_pkg.vhd (real-tile Q.20 constants," \
     "package name first_layer_32out_bn_relu_resource_shared_pkg)"
$GHDL_A conv3x3_3chan_16out_bn_relu_time_mux.vhd
echo "  OK  conv3x3_3chan_16out_bn_relu_time_mux.vhd"
$GHDL_A stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd
echo "  OK  stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd"
$GHDL_A real_tile_stimulus_pkg.vhd
echo "  OK  real_tile_stimulus_pkg.vhd"

for k in $(seq 0 31); do
    $GHDL_A "first_conv_bn_relu_kernel${k}_real_tile_q20_pkg.vhd"
done
echo "  OK  first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd (32 files)"

$GHDL_A tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.vhd
echo "  OK  tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20.vhd"
echo ""

echo "=== [2/3] Elaborating tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20 ==="
$GHDL_E tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20
echo "  OK"
echo ""

echo "=== [3/3] Running simulation (stop after 700000 ns -- generous margin above" \
     "the testbench's own internal MAX_CYCLES safety-bound assertion; the 2-lane" \
     "design's measured toy-pattern latency was 5,085 cycles for 9 windows, so" \
     "~9,000-10,000 cycles are expected here for 16 real windows) ==="
# shellcheck disable=SC2086
$GHDL_R tb_stream_conv3x3_3chan_32out_bn_relu_resource_shared_real_tile_q20 \
    --stop-time=700000ns \
    --assert-level=failure \
    ${EXTRA_RUN_FLAGS}

echo ""
echo "=== Simulation complete ==="
