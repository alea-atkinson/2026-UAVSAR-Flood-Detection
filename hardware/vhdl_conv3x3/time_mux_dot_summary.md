# Time-Multiplexed 3x3 Dot-Product Prototype

Date: July 2, 2026

## What this module does

`conv3x3_dot_time_mux.vhd` computes ONE 3x3 signed INT8 dot product
(`y = bias + sum(p_i * w_i)` for i = 0..8) using a **single reusable
multiply-accumulate (MAC) lane**, instead of the 9 parallel multipliers used
by every other dot-product module in this directory (`conv3x3_dot.vhd`,
`conv3x3_dot_pipelined.vhd`, `conv3x3_dot_pipelined_dsp.vhd`).

It uses a start/done handshake:

- While idle, asserting `start` for one cycle latches all 9 pixels, all 9
  weights, and the bias into internal registers and begins computing.
- One multiply-accumulate is performed per clock cycle, indexing into the
  latched pixel/weight arrays with a shared tap counter (`p_reg(tap) *
  w_reg(tap)`), so the same physical multiplier is reused for all 9 taps.
- `done` pulses for one cycle when the INT32 result `y` becomes valid.
- `start` is ignored while busy; the module completes its current dot
  product before it can accept a new one (no pipelining, no throughput
  overlap).

This is a **prototype of resource sharing via time multiplexing for a
single dot product only** -- it does not schedule multiple kernels,
channels, or output positions across the shared MAC lane. That would be a
follow-on "time-multiplexed 8-output cell," not built here.

## How many cycles one 3x3 dot product takes

**10 clock cycles** from the edge that samples `start='1'` to the edge that
asserts `done='1'`:

- Cycle 0 (load): latch p0..p8, w0..w8, bias; initialize the accumulator
- Cycles 1-8: process taps 0-7, one multiply-accumulate per cycle
- Cycle 9: process tap 8 (final), register `y`, assert `done`

This was not just designed but **measured empirically** by the testbench,
which counts clock edges between asserting `start` and observing `done`.
All 9 dot-product runs in the testbench (3 kernels x 3 channels) measured
exactly 10 cycles, confirming the design's documented latency.

## GHDL verification result

**PASS.** `tb_conv3x3_dot_time_mux.vhd` runs the time-multiplexed DUT three
times per kernel (once per input channel, using that channel's real
trained-model weights and the corresponding channel of the canonical toy
input's top-left 3x3 window), sums the three per-channel results plus the
kernel's bias, and checks the sum against `KERNEL{k}_EXPECTED(0)` -- the
same top-left golden value used to verify the fully parallel 4-output and
8-output prototypes.

Checked kernels 0, 1, and 7 (9 dot-product runs total):

| Kernel | Summed y (top-left) | Matches KERNEL{k}_EXPECTED(0) | Cycles per channel dot product |
|---|---:|---|---|
| 0 | 11287 | Yes | 10 / 10 / 10 |
| 1 | 9214 | Yes | 10 / 10 / 10 |
| 7 | -2419 | Yes | 10 / 10 / 10 |

All 9/9 checks passed. This confirms the time-multiplexed single-MAC
datapath produces numerically identical results to the fully parallel
designs for the same real trained-model kernels and input.

## Synthesis resource/timing/power result

Vivado 2025.2, target `xc7a35tcpg236-1`, out-of-context synthesis,
100 MHz / 10 ns clock constraint. Synthesis completed successfully: **0
errors, 0 critical warnings, 0 warnings** (the routine `HD.CLK_SRC` timing
estimation note common to all out-of-context runs in this directory still
appears, as usual).

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 205 | 20800 | 0.99% |
| Slice Registers | 214 | 41600 | 0.51% |
| DSPs | 0 | 90 | 0.00% |
| Block RAM Tiles | 0 | 50 | 0.00% |

| Timing metric | Value |
|---|---:|
| Worst setup slack | +0.656 ns |
| Worst hold slack | +0.290 ns |
| Worst pulse-width slack | +4.500 ns |

| Power metric | Value |
|---|---:|
| Total on-chip power | 0.074 W |
| Dynamic power | 0.006 W |
| Device static power | 0.068 W |

Timing still meets the 100 MHz constraint (0 failing endpoints for setup,
hold, and pulse-width), but with a much tighter setup margin (+0.656 ns)
than any of the fully parallel designs (+4.456 ns to +5.117 ns). This is
consistent with the module's datapath: selecting `p_reg(tap)` / `w_reg(tap)`
from a 9-element array with a variable index (`tap`) requires an 8:1
multiplexer ahead of the multiplier, adding combinational delay that the
fixed, statically-indexed multiplies in the parallel designs do not have.

## Why this matters

The fully parallel prototypes in this series demonstrated that resource
usage scales with kernel count: LUTs grow from 2731 (1 output) to 9374 (8
outputs), and forcing DSP mapping at 8 outputs already exhausts the part's
entire 90-slice DSP budget (`vivado_8out_dsp_synthesis_summary.md`). Neither
pure LUT-only nor pure DSP-request parallel scaling can reach significantly
higher kernel counts on this part without an architecture change.

This module is the **smallest possible step** toward the alternative:
reusing a small, fixed number of MAC resources across time instead of
instantiating one multiplier per weight tap per kernel. It proves, at the
single-dot-product level, that:

1. A dot product can be computed correctly with 1 multiplier instead of 9,
   producing numerically identical results to the parallel designs.
2. The resource cost of doing so is dramatically lower: 205 LUTs / 214
   registers / 0 DSPs, versus 2731 LUTs / 1324 registers for the smallest
   parallel dot-product-based design (the 1-output cell, which itself
   contains 3 dot-product units plus 3 window generators).
3. The cost of that resource reduction is latency: 10 cycles for one dot
   product, versus 1 cycle (combinational) or 3 cycles (pipelined, 1
   result/cycle throughput) for the parallel dot-product modules.

## Comparison to the fully parallel dot product (stated carefully)

| | Parallel (`conv3x3_dot_pipelined`) | Time-multiplexed (`conv3x3_dot_time_mux`) |
|---|---|---|
| Multipliers instantiated | 9 (one per tap) | 1 (reused for all 9 taps) |
| Cycles per dot product | 3 (pipelined latency; 1 result/cycle throughput once full) | 10 (latency; no pipelining, no overlap) |
| Standalone synthesis LUTs | not separately synthesized (only measured as part of a full cell) | 205 |
| Standalone synthesis DSPs | not separately synthesized | 0 |

**This is not a throughput or speed comparison.** The two modules were not
synthesized as part of the same larger design, are not directly
substitutable without a scheduler, and no clock-for-clock or
resource-for-resource equivalent comparison is being claimed. The only
claim here is: within its own right, this prototype shows one MAC lane can
correctly compute a 3x3 dot product over multiple cycles, at a small
fraction of the per-instance resource cost of a parallel dot-product unit,
and at the cost of substantially higher per-result latency and zero
pipelining/throughput.

## Limitations / safe-claim notes

- **Not a full time-multiplexed convolution engine.** This module computes
  exactly one dot product per start/done cycle. It does not implement a
  scheduler across multiple kernels, input channels, or spatial output
  positions -- extending it to reuse this MAC lane across an 8-output (or
  larger) design would require a new wrapper module, not built here.
- **Not full U-Net FPGA inference.** This validates one arithmetic
  primitive only.
- **Does not cover all 32 first-layer output channels.** Verified against
  kernels 0, 1, and 7 of the 8 exported so far.
- **Not board-tested.** All results are GHDL simulation and Vivado
  out-of-context synthesis estimates (utilization, static timing, and
  "Medium confidence" vector-less power estimation).
- **No measured throughput or latency speedup claim.** The 10-cycle latency
  is measured in simulation; no comparison to real hardware timing, and no
  claim that this approach is faster or slower in an end-to-end sense than
  the parallel designs -- only that it uses far fewer per-instance
  resources at the cost of far higher per-result latency.
- **No BatchNorm folding.** Bias is 0 for every kernel checked, consistent
  with every other prototype in this series.
- The 8:1 tap-select multiplexer's combinational delay is the likely reason
  for this module's reduced setup timing margin relative to the parallel
  designs; this has not been root-caused with a timing path report, only
  noted as a plausible explanation.

## Next recommended technical step

Design a small **time-multiplexed multi-kernel scheduler** that reuses this
same single MAC lane (or a small pool of a few lanes) across multiple
output kernels and input channels sequentially, with a controller that
cycles through the 24 (kernel, channel) combinations needed for an 8-output
cell one at a time. That would be the first real test of whether
time-multiplexing can deliver an 8-output (or larger) design within a LUT
and DSP budget the fully parallel and DSP-aware approaches could not meet,
at the necessary cost of reduced throughput -- and it is the natural
next increment before attempting a full streaming time-multiplexed
convolution engine.
