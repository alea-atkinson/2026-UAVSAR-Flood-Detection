# First-Layer 16-Output Folded Core: Comfortable FPGA Subproblem (Short Summary)

Full report: `hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md`.
Machine-readable comparison: `synthesis_comparison.csv` (this directory).

## The question

The existing 32-output direct-parallel folded Conv-BN-ReLU arithmetic core
(`first_layer_32out_folded_arithmetic_core.vhd`) saturates the Artix-7 200T
target at 740/740 (100%) DSP48E1 slices, leaving zero headroom. Does
computing only the first 16 of those 32 real, trained output channels
leave real headroom while still meeting 100 MHz timing?

## Result

**Yes.** The 16-output folded core (channels 0-15, real trained UAVSAR
flood-segmentation checkpoint weights, folded BatchNorm + ReLU, Q.20
fixed-point) uses **405/740 (54.73%) DSP48E1 slices** -- 335 DSPs
(45.27%) of headroom remaining on the same part -- and meets 100 MHz
timing with the SAME positive slack (WNS = +2.218 ns) as the core alone,
at every image width tested (6, 128, 256), including with the real
streaming front end (`window3x3_stream_3chan_flattened.vhd`, reused
unmodified) integrated in. GHDL: 256/256 outputs (16 real UAVSAR-derived
windows x 16 kernels) match already-verified real-tile Q.20 golden
vectors exactly.

## Claim boundaries

First-layer folded Conv-BN-ReLU scope only. Not full U-Net FPGA
inference. Not board-measured. Does not claim FPGA is faster than GPU.
Out-of-context Vivado synthesis estimates only.
