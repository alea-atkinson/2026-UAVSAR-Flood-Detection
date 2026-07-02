# Vivado Synthesis Summary: 8-Output Streaming 3x3 Convolution Cell

Date: July 2, 2026
Tool: Vivado 2025.2
Target part: xc7a35tcpg236-1
Top module: `stream_conv3x3_3chan_8out_cell`
Synthesis mode: out-of-context
Clock constraint: 10.000 ns, 100 MHz

## Result

Vivado synthesis completed successfully with:

- 0 errors
- 0 critical warnings
- 0 warnings

(The same informational `WARNING: [Timing 38-242]` about `HD.CLK_SRC` not being set appears during `report_timing_summary`, identical to the 1-output and 4-output flows -- expected for out-of-context synthesis with no clock buffer instantiated.)

## Resource Utilization

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 9374 | 20800 | 45.07% |
| Slice Registers | 5704 | 41600 | 13.71% |
| DSPs | 0 | 90 | 0.00% |
| Block RAM Tiles | 0 | 50 | 0.00% |

## Timing

The design met the 100 MHz timing constraint.

| Timing metric | Value |
|---|---:|
| Clock period constraint | 10.000 ns |
| Worst setup slack | +5.105 ns |
| Worst hold slack | +0.262 ns |
| Worst pulse-width slack | +4.500 ns |

All user-specified timing constraints are met (0 failing endpoints for setup, hold, and pulse-width across 5896/5896, 5896/5896, and 5704/5704 endpoints respectively).

## Power Estimate

| Power metric | Value |
|---|---:|
| Total on-chip power | 0.271 W |
| Dynamic power | 0.202 W |
| Device static power | 0.069 W |
| Junction temperature | 26.4 C |

Vector-less activity propagation was used (no simulation activity file supplied); Vivado reports "Medium" confidence for this estimate, same caveat as the 1-output and 4-output flows.

## Comparison: 1-Output vs. 4-Output vs. 8-Output

| Metric | 1-output | 4-output | 8-output | 8-out / 1-out | 8-out / 4-out |
|---|---:|---:|---:|---:|---:|
| Slice LUTs | 2731 | 5020 | 9374 | 3.43x | 1.87x |
| Slice Registers | 1324 | 3214 | 5704 | 4.31x | 1.78x |
| DSPs | 0 | 0 | 0 | -- | -- |
| Block RAM Tiles | 0 | 0 | 0 | -- | -- |
| Worst setup slack | +4.456 ns | +5.117 ns | +5.105 ns | all meet 100 MHz | all meet 100 MHz |
| Worst hold slack | +0.262 ns | +0.262 ns | +0.262 ns | unchanged | unchanged |
| Total on-chip power | 0.120 W | 0.177 W | 0.271 W | 2.26x | 1.53x |
| Dynamic power | 0.051 W | 0.108 W | 0.202 W | 3.96x | 1.87x |
| Device static power | 0.069 W | 0.069 W | 0.069 W | unchanged (device-intrinsic) | unchanged |

### Interpretation

Kernel count doubled from 4 to 8, but LUTs grew 1.87x and registers grew
1.78x (not 2x) going from the 4-output to the 8-output cell. This is
consistent with the window-sharing structure: all designs instantiate
exactly 3 `window3x3_stream` generators regardless of kernel count, so their
fixed LUT/register cost is amortized across more kernels as the count grows
(3 window generators / 4 kernels vs. 3 window generators / 8 kernels). The
`conv3x3_dot_pipelined` units, which scale linearly with kernel count (3
units per kernel), dominate more of the total as kernel count increases,
which is why the sub-linear effect is less pronounced between 4-to-8 than it
was between 1-to-4.

All three designs meet the 100 MHz timing constraint with comfortable
positive setup and hold slack, and the worst hold slack is numerically
identical (+0.262 ns) across all three -- consistent with the hold-critical
path living inside the shared `window3x3_stream` structure, which is
unchanged in all three designs.

Total on-chip power grows slightly faster than LUT/register count (2.26x
LUTs vs. 2.26x total power from 1-to-8; dynamic power alone grows faster at
3.96x), which is expected since dynamic power scales with switching activity
across the added dot-product logic, not just static cell count. Device
static power (0.069 W) is identical across all three designs because it
reflects the fixed part's leakage characteristics at the given junction
temperature, not the design's logic content.

**This is a resource/timing/power comparison from static synthesis reports
only.** It is not a measured throughput or latency speedup -- all three
designs process one pixel triplet per clock and share the identical 4-cycle
pipeline latency (confirmed by GHDL simulation for all three), and none of
this has been validated on real board hardware.

## Notes

This synthesis run targets an 8-output first-layer convolution HARDWARE
PROTOTYPE covering kernels 0-7 of `enc1.block.0.weight` (8 of the 32 output
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
  prototype (`stream_conv3x3_3chan_8out_cell.vhd`, verified 72/72 outputs
  against Python golden vectors).

Vivado did not infer DSP or BRAM usage for any of the three designs; all map
entirely to LUT/register/carry-chain fabric. At 45.07% LUT utilization for
8 outputs (vs. 13.13% for 1 output and 24.13% for 4 outputs), continuing this
trend toward the full 32-channel first layer would approach or exceed the
xc7a35t's available LUT budget under this architecture -- a larger scaled
design would likely need explicit DSP-aware restructuring (e.g. mapping the
INT8 multiplies to DSP48E1 slices instead of LUT fabric) to stay within
resource budget and control further growth.

## Extrapolation caveat

The LUT/register counts above follow a clearly sub-linear trend relative to
kernel count (1x -> 4x -> 8x kernels does not produce 1x -> 4x -> 8x
resource growth) because of the shared window generators. This is a
structural observation from three synthesized data points, not a fitted
model -- no attempt is made here to project resource usage at higher kernel
counts (e.g. 16 or 32), since the balance between the fixed
window-generator cost and the linearly-scaling dot-product cost would need
to be re-measured at each additional data point.
