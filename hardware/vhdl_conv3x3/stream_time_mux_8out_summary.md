# Streaming Time-Multiplexed 8-Output Prototype (5x5 Toy Image)

Date: July 3, 2026

## What the streaming prototype does

`stream_conv3x3_3chan_8out_time_mux.vhd` is a first streaming wrapper
around the existing one-window scheduler
(`conv3x3_3chan_8out_time_mux.vhd`). It accepts the same pixel-stream
interface as the fully parallel `stream_conv3x3_3chan_8out_cell.vhd` (clk,
rst, valid_in, pixel_c0/pixel_c1/pixel_c2 -- one pixel triplet per clock)
and produces the same eight kernel outputs (y0..y7) per valid 3x3 window,
using the SAME real trained-model weights (kernels 0-7 of
`enc1.block.0.weight`).

Internally it reuses, unmodified:
- Three `window3x3_stream` instances (one per input channel), the exact
  same window generators the fully parallel cell uses.
- One `conv3x3_3chan_8out_time_mux` instance (the one-window,
  time-multiplexed scheduler), the exact same module verified previously
  against a single pre-loaded patch.

A small new capture-then-process FSM ties them together in **two
non-overlapped phases**:

- **Phase 1 (capture)**: pixels stream in at 1/clock. The three window
  generators produce valid 3x3 windows exactly as they do for the parallel
  cell; each one is captured into an internal buffer (9 slots for the 5x5
  toy image) as it arrives. No compute happens yet.
- **Phase 2 (process)**: once all 9 windows are captured, the wrapper
  autonomously starts feeding them into the one shared scheduler instance,
  one window at a time -- present the window, pulse the scheduler's
  `start`, wait for its `done`, latch y0..y7, pulse this module's own
  `valid_out`, then move to the next window. This is explicitly **not**
  overlapped: window N+1 does not begin until window N's outputs are valid.

### Why two non-overlapped phases

`window3x3_stream` has no back-pressure support and cannot be paused
mid-row (documented in its own header). The one-window scheduler takes
~265 cycles to process a single window. Streaming windows in at 1/clock
while the scheduler can only consume one window per ~265 clocks would
require either dropping windows or a real backpressure/FIFO redesign of the
window generator -- explicitly out of scope for this first prototype (per
the task's own guidance: "do not over-engineer a full backpressure/AXI-style
streaming system unless absolutely necessary"). Buffering all 9 windows
first, then processing them sequentially, is the simplest correct design
that fits the stated scope.

## Does it verify 72/72 outputs?

**Yes.** `tb_stream_conv3x3_3chan_8out_time_mux.vhd` streams the canonical
5x5x3 toy image (channel 0 = 1..25 row-major, channel 1 = 2x channel 0,
channel 2 = -1x channel 0) and checks all 9 windows x 8 kernels = 72 INT32
values against the same `KERNEL{0..7}_EXPECTED` golden vectors used by the
fully parallel 8-output testbench and the one-window scheduler testbench.
All 72/72 outputs matched. `all_done` was confirmed to pulse on the exact
same cycle as the 9th window's `valid_out`.

## Total cycles / throughput behavior

Measured directly by the testbench (not asserted from a precomputed
formula):

| Event | Cycle |
|---|---:|
| Window 1 ready | 293 |
| Window 2 ready | 559 |
| Window 3 ready | 825 |
| Window 4 ready | 1091 |
| Window 5 ready | 1357 |
| Window 6 ready | 1623 |
| Window 7 ready | 1889 |
| Window 8 ready | 2155 |
| Window 9 ready (all_done) | 2421 |

**Total: 2421 clock cycles** for streaming capture (25 cycles) plus
sequential processing of all 9 windows. After the first window (which
includes the ~25-cycle capture phase plus its own processing), every
subsequent window arrives **exactly 266 cycles** later -- consistent with
(and 1 cycle more than) the ~265-cycle-per-window figure measured for the
standalone one-window scheduler, the extra cycle plausibly coming from this
wrapper's own FSM adding one more layer of registered start/done handshake
on top of the scheduler's internal one (the same kind of one-cycle
registration gap documented for the scheduler itself in
`time_mux_8out_summary.md`).

**This confirms the design has no pipelining or overlap between windows**:
each window's full ~266-cycle cost is paid serially, with zero throughput
improvement from streaming multiple windows through -- exactly as expected
for a first, intentionally non-overlapped prototype.

## Resource/timing/power results

Vivado 2025.2, target `xc7a35tcpg236-1`, out-of-context synthesis,
100 MHz / 10 ns clock constraint. Synthesis completed successfully: **0
errors, 0 critical warnings, 1 warning**. The warning
(`[Synth 8-3332] Sequential element (FSM_onehot_state_reg[2]) is unused and
will be removed`) is a benign netlist-optimization note about the 3-state
capture/process FSM's one-hot encoding being trimmed to fewer bits than
requested -- not a functional issue (the design still passed GHDL 72/72 and
`synth_design completed successfully`). The routine `HD.CLK_SRC` timing
note common to every out-of-context run in this directory also appears
during `report_timing_summary`.

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 1409 | 20800 | 6.77% |
| Slice Registers | 3328 | 41600 | 8.00% |
| DSPs | 0 | 90 | 0.00% |
| Block RAM Tiles | 0 | 50 | 0.00% |

| Timing metric | Value |
|---|---:|
| Worst setup slack | +0.600 ns |
| Worst hold slack | +0.262 ns |
| Worst pulse-width slack | +4.500 ns |

| Power metric | Value |
|---|---:|
| Total on-chip power | 0.086 W |
| Dynamic power | 0.017 W |
| Device static power | 0.068 W |

Timing still meets the 100 MHz constraint (0 failing endpoints for setup,
hold, and pulse-width). Setup margin (+0.600 ns) is close to the one-window
scheduler's own +0.655 ns, consistent with the same tap-select multiplexer
inside the shared dot-product instance still dominating the critical path.

## Comparison

### a) vs. the fully parallel 8-output streaming design (`stream_conv3x3_3chan_8out_cell`)

| Metric | Parallel streaming | Streaming time-mux | Ratio |
|---|---:|---:|---:|
| Slice LUTs | 9374 | 1409 | 6.7x fewer |
| Slice Registers | 5704 | 3328 | 1.7x fewer |
| DSPs | 0 | 0 | -- |
| Total on-chip power | 0.271 W | 0.086 W | 3.2x lower |

**Not a throughput comparison.** The parallel cell accepts one new pixel
triplet per clock and produces all 9 windows' 72 outputs in roughly the
time it takes to stream the 5x5 image (~29 clocks total including drain,
per its own testbench). The streaming time-mux prototype takes 2421
cycles for the same 72 outputs -- about 83x more clock cycles for this
specific toy image. This is the direct, expected cost of trading 24
parallel multipliers for 1 shared one, with zero pipelining between windows
in this first version.

### b) vs. the one-window time-multiplexed scheduler (`conv3x3_3chan_8out_time_mux`)

| Metric | One-window scheduler | Streaming time-mux | Ratio |
|---|---:|---:|---:|
| Slice LUTs | 376 | 1409 | 3.7x more |
| Slice Registers | 510 | 3328 | 6.5x more |
| DSPs | 0 | 0 | -- |
| Total on-chip power | 0.076 W | 0.086 W | 1.1x more |

The added resources over the bare one-window scheduler come from: three
`window3x3_stream` instances (each with their own internal 3-row line
buffers), the 9-window capture buffer (9 windows x 27 INT8 pixels = 243
bytes of registers), the y-output latch registers, and the wrapper's own
capture/process FSM. The per-window latency also grew slightly (266 cycles
here vs. 265 measured for the bare scheduler processing one pre-loaded
patch), consistent with one more layer of registered handshake logic
sitting on top of the same underlying scheduler.

## Safe interpretation

- **Resource sharing still reduces hardware footprint substantially** even
  once wrapped in a streaming interface: roughly 6.7x fewer LUTs and 3.2x
  lower estimated power than the fully parallel streaming design, for the
  same 72 verified outputs.
- **Latency increases correspondingly, and there is currently zero
  pipelining/overlap between windows**: this first version pays the full
  ~266-cycle scheduler cost once per window, serially, with no throughput
  benefit from streaming multiple windows through back-to-back.
- **This is still not a fully streaming, backpressure-aware convolution
  accelerator.** Windows must all be buffered before any processing
  begins (a consequence of `window3x3_stream` having no backpressure and
  the scheduler being far slower than the pixel arrival rate); there is no
  handshake back into the window generators, and this design has only been
  verified for one fixed 5x5 image size.
- **Not full U-Net FPGA inference.** Only 8 of 32 first-layer output
  channels, one 5x5 toy image, no padding, activation, or BatchNorm
  folding (bias is 0 for all 8 kernels, consistent with every prior
  prototype in this series).
- **Not board-tested.** All results are GHDL simulation and Vivado
  out-of-context synthesis estimates (utilization, static timing, and
  "Medium confidence" vector-less power estimation).
- **No measured throughput or latency speedup claim.** The comparisons
  above are resource/timing/power from static synthesis reports and
  measured simulation cycle counts; they are not a claim that this design
  is faster, slower, or more efficient than the parallel design in any
  deployed sense -- only that it trades resources for latency in the
  directions shown.

## Next recommended technical step

Add backpressure between the window generators and the capture buffer (or
redesign `window3x3_stream` to support a pause/valid-ready handshake), so
that windows can be captured continuously for images larger than the
9-slot buffer used here, and investigate overlapping the LAST few cycles of
one window's processing with the START of loading the next window's
weights/pixels into the scheduler -- the natural next increment toward
closing the throughput gap with the fully parallel design, before this
approach could be evaluated on a real (non-toy) UAVSAR tile size.
