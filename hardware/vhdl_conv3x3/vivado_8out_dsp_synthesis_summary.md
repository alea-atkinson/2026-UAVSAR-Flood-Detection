# Vivado Synthesis Summary: 8-Output Cell, LUT-Only vs. DSP-Aware

Date: July 2, 2026
Tool: Vivado 2025.2
Target part: xc7a35tcpg236-1
Clock constraint: 10.000 ns, 100 MHz
Synthesis mode: out-of-context

Compares two synthesis results for functionally identical 8-output
first-layer convolution designs (kernels 0-7 of `enc1.block.0.weight`):

- **LUT-only baseline**: `stream_conv3x3_3chan_8out_cell` (uses
  `conv3x3_dot_pipelined`, no synthesis attributes on the multiplies)
- **DSP-aware variant**: `stream_conv3x3_3chan_8out_cell_dsp` (uses
  `conv3x3_dot_pipelined_dsp`, which adds a `use_dsp = "yes"` Vivado
  synthesis attribute to each of the nine multiply-result signals per
  dot-product unit)

Both designs are cycle-accurate identical in simulation (GHDL confirms
72/72 bit-identical outputs against the same Python golden vectors); only
the multiplier-to-fabric mapping differs.

## Result

| | LUT-only baseline | DSP-aware variant |
|---|---|---|
| Errors | 0 | 0 |
| Critical warnings | 0 | 0 |
| Warnings | 0 | **1** |

The DSP-aware variant's one warning is:

```
WARNING: [Synth 8-3323] Resources of type DSP have been overutilized.
Used = 266, Available = 90. Use report_utilization command for details.
```

This fired during an intermediate Cross Boundary and Area Optimization pass.
The design requests 24 dot-product units x 9 multiplies = 216 multiplies (the
"266" figure in Vivado's own message reflects its internal accounting at
that optimization stage, not a number we independently derived). Vivado does
not fail or downgrade this to an error -- it legalizes the final netlist by
mapping only as many multiplies to DSP48E1 as the part has available (90)
and falls the remaining multiplies back to LUT fabric automatically. This is
confirmed by the final utilization report below: exactly 90/90 (100%) DSPs
used, with `synth_design completed successfully`.

## Resource Utilization

| Resource | LUT-only baseline | DSP-aware variant | Change |
|---|---:|---:|---:|
| Slice LUTs | 9374 | 6991 | -2383 (-25.4%) |
| Slice Registers | 5704 | 4405 | -1299 (-22.8%) |
| DSPs | 0 | 90 | +90 (0% -> 100% of part) |
| Block RAM Tiles | 0 | 0 | unchanged |

**DSP usage increased: yes** (0 -> 90, maxing out the part's entire DSP48E1 budget).
**LUT usage decreased: yes** (9374 -> 6991, a 25.4% reduction).

The register drop is notable: DSP48E1 slices include internal pipeline
registers, so some of the `pr0_s1..pr8_s1` stage-1 registers that would
otherwise consume slice flip-flops get absorbed into the DSP48E1 hardware
itself for the 90 multiplies that were DSP-mapped.

## Timing

| Timing metric | LUT-only baseline | DSP-aware variant |
|---|---:|---:|
| Worst setup slack | +5.105 ns | +2.371 ns |
| Worst hold slack | +0.262 ns | +0.262 ns |
| Worst pulse-width slack | +4.500 ns | +4.500 ns |

**Timing still passes: yes** -- both designs meet the 100 MHz constraint
with 0 failing endpoints for setup, hold, and pulse-width. However, the
DSP-aware variant's setup margin dropped substantially (+5.105 ns ->
+2.371 ns). This is consistent with a design that is now a mixed
DSP48E1/LUT implementation: paths through DSP48E1 slices and paths through
the remaining LUT-mapped multiplies do not have identical delay
characteristics, and the DSP-overutilization condition means Vivado's
resource-constrained legalization did not have a free choice of which 90
multiplies to map to DSP -- this can produce a less balanced critical path
than a design that fits its DSP request within the part's budget from the
start. Hold slack is unchanged, consistent with the hold-critical path
still living in the untouched `window3x3_stream` structure.

## Power

| Power metric | LUT-only baseline | DSP-aware variant | Change |
|---|---:|---:|---:|
| Total on-chip power | 0.271 W | 0.346 W | +0.075 W (+27.7%) |
| Dynamic power | 0.202 W | 0.277 W | +0.075 W (+37.1%) |
| Device static power | 0.069 W | 0.069 W | unchanged |

Despite using fewer LUTs, the DSP-aware variant's estimated power is
**higher**, not lower. This is a counterintuitive but plausible result:
Vivado's vector-less power estimator models DSP48E1 slices as drawing more
dynamic power per active multiply-accumulate at this design's toggle rates
than the equivalent narrow (8x8-bit) LUT-based multiplier, especially when
the DSPs are driven at high utilization (90/90) alongside the surviving
LUT-mapped multiplies. Device static power is identical, as expected (it
reflects the fixed part's leakage characteristics, not design content).

## Interpretation

For this specific 8-output design, forcing DSP mapping is **not a clean
win**: it trades a 25% LUT reduction for exhausting 100% of the part's DSP
budget, a meaningfully reduced timing margin, and higher estimated power.
More importantly, the DSP-overutilization warning is itself the key finding:
**this single 8-output design already requests more multiplies (216) than
the entire xc7a35t part has DSP48E1 slices (90)**. Even in the best case
where all 216 multiplies could be perfectly packed into DSPs, the part
physically cannot fit them -- DSP-based scaling of this architecture past
roughly 4 output kernels (4 kernels x 3 channels x 9 = 108 multiplies,
still over budget) is not possible on this specific part without a
fundamentally different architecture (e.g. time-multiplexing multiplies
across fewer physical DSP48E1 units, or a smaller part upgrade).

**This is a synthesis-level DSP-mapping/resource tradeoff experiment.** It
does not measure real power on hardware, does not represent a recommended
production configuration, and is not a claim that DSP-aware synthesis
generally reduces resource usage -- only that it does so here for LUTs,
at the cost of maxing out DSPs, reduced timing margin, and higher estimated
power.

## Notes

- Both designs cover only 8 of 32 first-layer output channels (kernels
  0-7); this is not full U-Net FPGA inference and does not represent all
  32 first-layer channels.
- Neither design has been run on real board hardware -- both are Vivado
  out-of-context synthesis estimates (utilization, static timing analysis,
  vector-less power estimation).
- No BatchNorm folding; bias is 0 for all eight kernels in both designs,
  consistent with every prior prototype in this series.
- No measured throughput or latency speedup is claimed -- both designs
  share the identical 4-cycle pipeline latency (GHDL-confirmed), and this
  comparison is limited to synthesized resource counts, static timing
  slack, and estimated power.
- The `use_dsp` attribute is a Vivado-specific synthesis hint (see Xilinx
  UG901); it does not guarantee DSP mapping when the part's DSP budget is
  exceeded, as demonstrated by this experiment's own overutilization
  warning and 100% DSP utilization result.

## Next technical implication

Since a single 8-output prototype already exceeds the part's 90-DSP budget
if fully DSP-mapped, and the LUT-only baseline was already at 45% LUT
utilization at 8 outputs, **scaling this architecture to 16 or 32 output
channels on this specific part (xc7a35tcpg236-1) is not feasible under
either the pure LUT-only or pure DSP-request approach** without an
architecture change -- e.g., time-multiplexing a small number of shared
DSP-based multiply-accumulate units across multiple kernels/channels per
clock cycle (trading throughput for resource usage), rather than
instantiating one dedicated multiplier per weight tap per kernel.
