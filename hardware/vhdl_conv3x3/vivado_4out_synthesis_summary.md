# Vivado Synthesis Summary: 4-Output Streaming 3x3 Convolution Cell

Date: July 2, 2026
Tool: Vivado 2025.2
Target part: xc7a35tcpg236-1
Top module: `stream_conv3x3_3chan_4out_cell`
Synthesis mode: out-of-context
Clock constraint: 10.000 ns, 100 MHz

## Result

Vivado synthesis completed successfully with:

- 0 errors
- 0 critical warnings
- 0 warnings

(One informational `WARNING: [Timing 38-242]` about `HD.CLK_SRC` not being set appears during `report_timing_summary`; it is expected for out-of-context synthesis with no clock buffer instantiated and is not counted among the "0 warnings" reported by `synth_design` itself. The 1-output flow produces the identical message.)

## Resource Utilization

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 5020 | 20800 | 24.13% |
| Slice Registers | 3214 | 41600 | 7.73% |
| DSPs | 0 | 90 | 0.00% |
| Block RAM Tiles | 0 | 50 | 0.00% |

## Timing

The design met the 100 MHz timing constraint.

| Timing metric | Value |
|---|---:|
| Clock period constraint | 10.000 ns |
| Worst setup slack | +5.117 ns |
| Worst hold slack | +0.262 ns |
| Worst pulse-width slack | +4.500 ns |

All user-specified timing constraints are met (0 failing endpoints for setup, hold, and pulse-width across 3406/3406, 3406/3406, and 3214/3214 endpoints respectively).

## Power Estimate

| Power metric | Value |
|---|---:|
| Total on-chip power | 0.177 W |
| Dynamic power | 0.108 W |
| Device static power | 0.069 W |
| Junction temperature | 25.9 C |

Vector-less activity propagation was used (no simulation activity file supplied); Vivado reports "Medium" confidence for this estimate, same caveat as the 1-output flow.

## Comparison: 1-Output vs. 4-Output Cell

| Metric | 1-output (`stream_conv3x3_3chan_cell`) | 4-output (`stream_conv3x3_3chan_4out_cell`) | Ratio |
|---|---:|---:|---:|
| Slice LUTs | 2731 | 5020 | 1.84x |
| Slice Registers | 1324 | 3214 | 2.43x |
| DSPs | 0 | 0 | -- |
| Block RAM Tiles | 0 | 0 | -- |
| Worst setup slack | +4.456 ns | +5.117 ns | both meet 100 MHz |
| Worst hold slack | +0.262 ns | +0.262 ns | unchanged |
| Total on-chip power | 0.120 W | 0.177 W | 1.48x |
| Dynamic power | 0.051 W | 0.108 W | 2.12x |
| Device static power | 0.069 W | 0.069 W | unchanged (device-intrinsic, not design-dependent) |

### Interpretation

The 4-output cell computes four kernels' worth of convolution but shares one
`window3x3_stream` instance per input channel (3 total) across all four
kernels instead of instantiating 3 per kernel (12 total). This is why LUT and
register growth (1.84x, 2.43x) is well below a naive 4x: the three window
generators (which account for a large share of the 1-output design's LUTs
and registers) are amortized across all four output kernels, and only the
twelve `conv3x3_dot_pipelined` dot-product units scale linearly with kernel
count.

Both designs meet the 100 MHz timing constraint with comfortable positive
setup and hold slack. The worst hold slack is numerically identical between
the two designs (+0.262 ns), consistent with the hold-critical path being
inside the shared `window3x3_stream` structure rather than the replicated
dot-product datapath.

**This is a resource/timing/power comparison from static synthesis reports
only.** It is not a measured throughput or latency speedup (both designs
process one pixel triplet per clock and share the same 4-cycle pipeline
latency, as confirmed by GHDL simulation), and it has not been validated on
real board hardware.

## Notes

This synthesis run targets a 4-output first-layer convolution HARDWARE
PROTOTYPE covering kernels 0-3 of `enc1.block.0.weight` (4 of the 32 output
channels in the full first Conv2d layer). It is:

- **Not** full U-Net FPGA inference.
- **Not** a synthesis result for all 32 first-layer output channels.
- **Not** board-tested -- these are Vivado out-of-context synthesis
  estimates (utilization, static timing analysis, vector-less power
  estimation), not measurements from a programmed device.
- **Not** a measured speedup claim -- no throughput or latency benchmarking
  was performed; the comparison above is limited to synthesized resource
  counts, static timing slack, and estimated power.
- Built on the same INT8, non-BatchNorm-folded weights as the GHDL-verified
  prototype (`stream_conv3x3_3chan_4out_cell.vhd`, verified 36/36 outputs
  against Python golden vectors in commit b62885e1).

Vivado did not infer DSP or BRAM usage for either design; both map entirely
to LUT/register/carry-chain fabric. A larger scaled design (e.g. more output
channels, or all 32 first-layer channels) would likely need explicit
DSP-aware architecture choices to control LUT growth and meet tighter
resource budgets.
