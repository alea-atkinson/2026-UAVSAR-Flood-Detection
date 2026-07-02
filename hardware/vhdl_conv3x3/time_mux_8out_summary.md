# Time-Multiplexed 8-Output Scheduler Prototype (One 3x3x3 Window)

Date: July 2, 2026

## What the scheduler does

`conv3x3_3chan_8out_time_mux.vhd` computes all **eight** first-layer
output-channel values (kernels 0-7 of `enc1.block.0.weight`) for **one**
already-captured 3-channel 3x3 spatial window, by reusing **one**
`conv3x3_dot_time_mux` instance sequentially across the 24 (kernel, channel)
combinations needed -- 8 kernels x 3 input channels -- instead of
instantiating 24 parallel dot-product units the way
`stream_conv3x3_3chan_8out_cell.vhd` does.

This is the natural next step after `conv3x3_dot_time_mux.vhd` (a single
time-multiplexed dot product, previously driven only by testbench-side
orchestration). Here, that orchestration moves **into hardware**: a small
FSM schedules the 24 dot operations in order (step = kernel_idx*3 +
channel_idx, for step = 0..23), muxes in the correct pixel channel and
kernel weights for each one from LUTs built at elaboration time from
`first_layer_kernels0_to7_pkg`, accumulates each kernel's three per-channel
results, adds that kernel's bias (0 for every kernel here), and reports
y0..y7 with a single `done` pulse.

**This is not a full streaming convolution engine.** The patch
(`c0_p0..c2_p8`, 27 pixel inputs) must be presented and held stable for the
whole run; the module does not slide a window across an image, does not
stream pixels in, and does not implement padding, activation, or BatchNorm.

## How many cycles it takes to produce y0 through y7

**265 clock cycles**, measured empirically by the testbench (which counts
clock edges from `start` to `done` directly, exactly as the standalone
`conv3x3_dot_time_mux` testbench did for its own 10-cycle figure).

The 265 total is close to, but higher than, the naive "24 dot operations x
10 cycles = 240" estimate. The extra cycles come from an unavoidable
one-cycle registration gap at every redispatch: the scheduler's `dot_start`
signal is registered, so a redispatch decision made on the same edge the
previous operation's `done` fires can only become visible to the shared
`conv3x3_dot_time_mux` instance on the *following* edge -- one gap cycle
between every pair of consecutive dot operations, plus one more before the
very first operation's dispatch becomes visible. This is not a bug specific
to this FSM: a standalone probe test built during development, using a
hand-written testbench driving `conv3x3_dot_time_mux` directly with the
exact same "redispatch immediately after observing done" pattern, shows the
identical one-cycle gap. The exact cycle-by-cycle accounting was not fully
reconciled by hand (an earlier attempt to do so produced an incorrect
240-cycle design estimate); the number that matters is the one measured
directly by the testbench: **265**.

An earlier design-comment draft assumed a "back-to-back, zero-bubble"
240-cycle total; that assumption turned out to be incorrect and was
corrected after the testbench's own measurement caught the discrepancy --
which is exactly why the cycle count is measured here rather than only
asserted.

## GHDL verification result

**PASS.** `tb_conv3x3_3chan_8out_time_mux.vhd` drives the scheduler with the
top-left 3x3 patch of the canonical 5x5x3 toy input (channel 0 = 1..25
row-major, channel 1 = 2x channel 0, channel 2 = -1x channel 0), pulses
`start` once, and checks all eight outputs:

| Kernel | y (top-left) | Matches KERNEL{k}_EXPECTED(0) |
|---|---:|---|
| 0 | 11287 | Yes |
| 1 | 9214 | Yes |
| 2 | 1522 | Yes |
| 3 | 2636 | Yes |
| 4 | -9630 | Yes |
| 5 | -2734 | Yes |
| 6 | -5077 | Yes |
| 7 | -2419 | Yes |

All 8/8 outputs matched the same golden vectors used to verify the fully
parallel 4-output and 8-output prototypes and the standalone time-mux dot
product. The testbench also confirmed `done` is a clean one-cycle pulse
(returns to '0' the following cycle).

## Synthesis resource/timing/power result

Vivado 2025.2, target `xc7a35tcpg236-1`, out-of-context synthesis,
100 MHz / 10 ns clock constraint. Synthesis completed successfully: **0
errors, 0 critical warnings, 0 warnings** (the routine `HD.CLK_SRC` timing
estimation note common to all out-of-context runs in this directory still
appears).

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 376 | 20800 | 1.81% |
| Slice Registers | 510 | 41600 | 1.23% |
| DSPs | 0 | 90 | 0.00% |
| Block RAM Tiles | 0 | 50 | 0.00% |

| Timing metric | Value |
|---|---:|
| Worst setup slack | +0.655 ns |
| Worst hold slack | +0.275 ns |
| Worst pulse-width slack | +4.500 ns |

| Power metric | Value |
|---|---:|
| Total on-chip power | 0.076 W |
| Dynamic power | 0.007 W |
| Device static power | 0.068 W |

Timing still meets the 100 MHz constraint (0 failing endpoints for setup,
hold, and pulse-width), with a setup margin (+0.655 ns) nearly identical to
the standalone `conv3x3_dot_time_mux`'s own +0.656 ns -- consistent with the
scheduler's added logic (step counter, LUTs, kernel/channel muxing) not
being on the critical path; the tap-select multiplexer inside the shared
dot-product instance still dominates.

## Comparison against the fully parallel 8-output LUT-only design

| Metric | Parallel 8-output (`stream_conv3x3_3chan_8out_cell`) | Time-mux scheduler (`conv3x3_3chan_8out_time_mux`) | Ratio |
|---|---:|---:|---:|
| Slice LUTs | 9374 | 376 | 4.0% (24.9x fewer) |
| Slice Registers | 5704 | 510 | 8.9% (11.2x fewer) |
| DSPs | 0 | 0 | -- |
| Total on-chip power | 0.271 W | 0.076 W | 28.0% (3.6x lower) |

**This is not a throughput comparison.** The parallel 8-output cell is a
*streaming* design: it accepts one new pixel triplet per clock and produces
a full 9-position x 8-kernel result set (72 values) in roughly the time it
takes to stream a 5x5 image (a few hundred nanoseconds total, pipelined).
The time-mux scheduler computes 8 values for **one** pre-loaded window in
265 cycles (2650 ns at 100 MHz) and must be re-triggered for every new
window -- it has no streaming input, no pipelining, and no window-sliding
logic. The resource numbers above are directly comparable (both are
standalone Vivado syntheses of complete modules); the cycle counts are not,
because the two designs solve different problems (one window vs. an
implicit assumption of continuous streaming).

## Safe interpretation

- **Resource sharing reduces hardware footprint substantially at the
  single-window level**: roughly 25x fewer LUTs, 11x fewer registers, and
  3.6x lower estimated power than the fully parallel 8-output cell, using
  exactly one multiplier instead of 24.
- **Latency increases correspondingly**: 265 cycles to produce one
  window's 8 outputs, versus a few cycles of pipeline latency (with 1
  output cycle throughput) for the parallel design once its pipeline is
  full.
- **This is not yet a streaming convolution accelerator.** The scheduler
  computes exactly one 3x3x3 window per `start` pulse and requires the
  window to be externally re-loaded and re-triggered for every spatial
  position. It does not slide across an image, does not manage line
  buffers, and does not overlap the compute of one window with the
  loading of the next.
- **Not full U-Net FPGA inference.** Only kernels 0-7 of 32 first-layer
  output channels are computed, for a single window, with no padding,
  activation, or BatchNorm folding (bias is 0 for all 8 kernels).
- **Not board-tested.** All results are GHDL simulation and Vivado
  out-of-context synthesis estimates (utilization, static timing, and
  "Medium confidence" vector-less power estimation).
- **No measured throughput or latency speedup claim.** The comparison
  above is resource/timing/power from static synthesis reports and a
  single simulated run; no claim is made that this scheduler is faster or
  slower than the parallel design in any end-to-end sense -- only that it
  uses far fewer resources per instance at the cost of far higher latency
  per window.

## Next recommended technical step

Extend the scheduler to accept a **new window each time it's re-triggered**,
combined with a window-generator (like `window3x3_stream.vhd`, already used
by the parallel designs) feeding it a stream of 3x3 windows as pixels
arrive -- i.e., build the first genuinely *streaming* time-multiplexed
convolution prototype, where the scheduler processes window N+1 while (or
immediately after) reporting window N's results, rather than requiring
external re-load-and-retrigger for every position. That is the natural
increment before this approach could be compared meaningfully, in
throughput terms, against the parallel 4-output and 8-output designs.
