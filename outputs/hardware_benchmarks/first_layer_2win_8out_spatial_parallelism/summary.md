# First-Layer 2-Window x 8-Output Spatial-Parallelism Core (Short Summary)

Full report: `hardware/vhdl_conv3x3/reports/first_layer_2win_8out_folded_arithmetic_core_summary.md`.
Machine-readable comparison: `spatial_parallelism_comparison.csv` (this directory).

## The question

The 16-output comfortable subproblem (commit `fe79e096`) uses 405/740
DSPs for **1 spatial window/cycle x 16 output channels**. Can that same
per-cycle output-channel budget (16) be reshaped into **2 spatial
windows/cycle x 8 output channels each**, without hitting the 32-output
design's 100%-DSP wall?

## Result

**Yes.** The new 2-window x 8-output arithmetic core (real trained
UAVSAR checkpoint weights, channels 0-7, folded BatchNorm + ReLU, Q.20
fixed-point) uses **406/740 (54.86%) DSP48E1 slices** -- essentially
identical to the 16-output design's 405/740 (54.73%) -- and meets 100 MHz
timing with the SAME slack (WNS = +2.218 ns). GHDL: 128/128 outputs (8
window pairs x 2 lanes x 8 kernels) match already-verified real-tile Q.20
golden vectors exactly.

## Claim boundaries

Arithmetic-core only, pre-extracted windows (no 2-pixel/cycle streaming
front end built or claimed here -- explicitly future work). Not full
first layer. Not full U-Net FPGA inference. Not board-measured. Does not
claim FPGA is faster than GPU. Computes only 8 of 32 output channels per
window -- not directly comparable to the 16-output GPU benchmark without
normalizing for channel count.
