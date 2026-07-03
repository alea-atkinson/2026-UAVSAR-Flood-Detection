# FPGA Architecture Comparison Summary

Date: July 3, 2026

This is a documentation and analysis report, not a new RTL or synthesis
task. All figures below are copied from existing markdown summaries and
Vivado report files already in this repository; no new synthesis or GHDL
runs were performed to produce this document. Source files are cited under
each entry.

## 1. Purpose

This is a **first-layer FPGA feasibility and architecture tradeoff study**
for trained UAVSAR flood-segmentation U-Net inference. Its goal is to
compare several candidate hardware architectures -- fully parallel,
DSP-aware, and time-multiplexed -- for computing the first convolution
layer of the trained U-Net on an Artix-7 FPGA, using real learned weights,
so that resource/timing/power tradeoffs are visible before committing to a
scaling direction for larger channel counts.

## 2. Scope

- **First convolution layer only** (`enc1.block.0.weight`, shape
  `[32, 3, 3, 3]`). No other U-Net layers are implemented.
- **Real learned kernels** extracted from the trained checkpoint
  `models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`,
  channels 0-7 of the 32 available first-layer output channels.
- **INT8 weights** (symmetric per-tensor quantization) and **INT32
  golden-output checks** against Python-computed reference values, verified
  in GHDL simulation for every design in this comparison.
- **Artix-7 `xc7a35tcpg236-1`** synthesis target (Digilent Cmod A7-35T),
  Vivado 2025.2, out-of-context synthesis mode.
- **100 MHz (10.000 ns) clock target** for every design compared here.
- **Not full U-Net FPGA inference.** Only the first Conv2d layer, 8 of 32
  output channels at most, no padding, no activation, no BatchNorm folding,
  no downstream encoder/decoder stages.
- **Not board-tested.** Every number below comes from GHDL simulation
  (functional correctness) and Vivado out-of-context synthesis reports
  (utilization, static timing analysis, vector-less power estimation) --
  none of these designs have been programmed onto real hardware.

## 3. Main comparison table

| Design | Output channels / scope | GHDL verification | LUTs | Registers | DSPs | BRAM | Setup slack | Est. total power | Cycle / throughput behavior | Main interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1-output parallel baseline | 1 kernel, 9 spatial positions | 9/9 outputs passed | 2731 | 1324 | 0 | 0 | +4.456 ns | 0.120 W | 1 pixel/clock streamed in; ~4-cycle pipeline latency, 1 result/clock throughput once full | Smallest working proof-of-concept; establishes the per-kernel resource baseline |
| 4-output parallel prototype | 4 kernels (0-3), 9 spatial positions | 36/36 outputs passed | 5020 | 3214 | 0 | 0 | +5.117 ns | 0.177 W | Same streaming/pipeline behavior as 1-output; shared window generators | LUT/register growth sub-linear (1.84x/2.43x, not 4x) due to shared window generators |
| 8-output parallel prototype | 8 kernels (0-7), 9 spatial positions | 72/72 outputs passed | 9374 | 5704 | 0 | 0 | +5.105 ns | 0.271 W | Same streaming/pipeline behavior; 45.07% of part's LUTs already used | Confirms feasibility at 100 MHz but shows LUT growth trending toward the part's ceiling |
| DSP-aware 8-output prototype | 8 kernels (0-7), 9 spatial positions | 72/72 outputs passed (bit-identical to LUT-only) | 6991 | 4405 | 90/90 | 0 | +2.371 ns | 0.346 W | Same cycle-accurate behavior as 8-output LUT-only (only multiplier fabric mapping differs) | 25% LUT reduction, but exhausts 100% of the part's DSP budget, reduces timing margin, and *increases* power |
| Time-mux 3x3 dot-product prototype | 1 dot product at a time (single MAC lane), tested on kernels 0, 1, 7 | 9/9 channel-dot checks passed | 205 | 214 | 0 | 0 | +0.656 ns | 0.074 W | 10 cycles per channel dot product (measured); no pipelining | Smallest possible resource-sharing step: 1 multiplier instead of 9, ~13x fewer LUTs than the 1-output baseline |
| Time-mux 8-output one-window scheduler | 8 kernels (0-7), 1 pre-loaded 3x3x3 patch | 8/8 outputs passed | 376 | 510 | 0 | 0 | +0.655 ns | 0.076 W | 265 cycles (measured) to produce all 8 outputs for one window; no streaming input | Reuses 1 dot-product lane across 24 (kernel, channel) operations; ~25x fewer LUTs than 8-output parallel, at ~265x the per-window latency |
| Streaming time-mux 8-output prototype | 8 kernels (0-7), 9 spatial positions, streamed 5x5 image | 72/72 outputs passed | 1409 | 3328 | 0 | 0 | +0.600 ns | 0.086 W | 2421 cycles (measured) for streaming capture + sequential processing of all 9 windows; no overlap between windows | Wraps the one-window scheduler with the existing window generators; verifies the full 72-output set end-to-end from a pixel stream, at ~83x the parallel design's cycle count |

Sources: `vivado_synthesis_summary.md` (1-output), `vivado_4out_synthesis_summary.md`
(4-output), `vivado_8out_synthesis_summary.md` (8-output),
`vivado_8out_dsp_synthesis_summary.md` (DSP-aware 8-output),
`time_mux_dot_summary.md` (time-mux dot product), `time_mux_8out_summary.md`
(one-window scheduler), `stream_time_mux_8out_summary.md` (streaming
time-mux), plus the underlying Vivado report files in
`hardware/vhdl_conv3x3/vivado_synth_reports/`.

## 4. Interpretation

### A. Parallel scaling

Going from 1 to 4 to 8 output channels shows that fully parallel first-layer
convolution is functionally correct and meets the 100 MHz timing target at
every step (setup slack stays comfortably positive: +4.456 ns, +5.117 ns,
+5.105 ns). However, LUT usage grows quickly: 2731 -> 5020 -> 9374, so the
8-output design already consumes **45.07%** of the Artix-7 xc7a35t's
available LUTs. Growth is sub-linear relative to kernel count because the
three `window3x3_stream` window generators are shared across all output
kernels rather than replicated per kernel, but the dot-product units still
scale linearly with kernel count. Extrapolating this trend, **fully
parallel 32-output scaling would likely become LUT-limited** on this
specific part well before reaching all 32 first-layer channels.

### B. DSP-aware mapping

Adding a `use_dsp` synthesis attribute to the 8-output design's multiplies
reduced LUTs by 25% (9374 -> 6991) and registers by 23% (5704 -> 4405), but
the design requested far more multiplies (216, from 8 kernels x 3 channels
x 9 taps) than the part's entire DSP budget (90 DSP48E1 slices) -- Vivado
mapped exactly 90 to DSP and fell the rest back to LUT fabric automatically.
The result: **100% of the part's DSP slices used**, a reduced timing margin
(+5.105 ns -> +2.371 ns), and **higher**, not lower, estimated power
(0.271 W -> 0.346 W). DSP-aware mapping is therefore not a clean scaling
solution on this part -- it trades one resource pressure (LUTs) for another
(DSPs), with a timing and power cost, and this single 8-output design
already exceeds the part's DSP capacity before even considering 16 or 32
output channels.

### C. Time-multiplexing

The three time-multiplexed designs (dot-product, one-window scheduler,
streaming prototype) all use a single reusable multiply-accumulate lane
instead of one multiplier per weight tap. This dramatically reduces
resource usage and estimated power at every step: the streaming time-mux
8-output prototype uses **1409 LUTs vs. 9374** for the fully parallel
8-output design (6.7x fewer) and **0.086 W vs. 0.271 W** (3.2x lower), while
verifying the **exact same 72 outputs** against the same golden vectors.
The cost is latency: the streaming time-mux prototype takes **2421 cycles**
to produce those 72 outputs for the 5x5 toy image, versus roughly 29 cycles
for the fully parallel streaming design (per its own testbench) -- about
83x more clock cycles for the same result, with zero pipelining or overlap
between windows in this first version.

### D. Best current architecture story

This set of experiments gives a clear, three-way tradeoff for the first
convolution layer on this specific part:

- **Parallel** = high throughput (1 pixel/clock streaming, few-cycle
  pipeline latency), but high and fast-growing LUT usage that approaches
  the part's ceiling well before 32 output channels.
- **DSP-aware parallel** = somewhat lower LUT usage, but DSP-limited (a
  single 8-output design already exhausts the part's entire DSP budget) and
  *higher* estimated power, not a clean win.
- **Time-multiplexed** = low LUT/register usage and low estimated power at
  every scale tested, verified to produce numerically identical outputs to
  the parallel designs, at the direct cost of substantially higher latency
  and (in the current prototypes) no pipelining across windows or kernels.

No single architecture in this comparison is unconditionally "best" --
the right choice depends on whether the eventual deployment target
prioritizes throughput (favoring parallel or a hybrid) or resource/power
budget (favoring time-multiplexing), a decision that requires knowing the
real-time processing requirement for UAVSAR tiles, which has not been
established in this study.

## 5. Safe claims

- Real, trained-model INT8 weights from the first U-Net convolution layer
  (`enc1.block.0.weight`, kernels 0-7 of 32) have been used and verified in
  every design compared here.
- All seven designs have been functionally verified in GHDL simulation
  against Python-computed INT32 golden outputs, with 100% pass rates on
  every check performed (9/9 through 72/72 depending on the design's scope).
- All seven designs have been synthesized with Vivado 2025.2 for the Artix-7
  `xc7a35tcpg236-1` part at a 100 MHz clock target, and all meet that timing
  constraint with positive setup and hold slack.
- Time-multiplexed architectures measurably and substantially reduce LUT,
  register, and estimated-power usage relative to fully parallel designs
  for the same verified output set, at a measured, substantially higher
  cycle-count cost.
- DSP-aware synthesis of the 8-output design measurably reduces LUT usage
  but requires 100% of the part's DSP48E1 budget and does not reduce
  estimated power for this specific design.
- A streaming interface (pixel stream in, sequential per-window processing
  out) has been demonstrated on top of the time-multiplexed one-window
  scheduler, producing the same 72 verified outputs as the fully parallel
  streaming design.

## 6. Claims to avoid right now

The following are **not** established by the work completed so far and
should not be claimed:

- Full U-Net FPGA inference (only the first convolution layer has been
  implemented).
- A complete first layer with all 32 output channels (at most 8 of 32 have
  been built and verified in any design).
- End-to-end flood-segmentation hardware (no downstream layers, no
  activation, no output stage, no full inference pipeline).
- Board testing of any kind (every result is GHDL simulation or Vivado
  out-of-context synthesis estimation; nothing has been programmed onto a
  physical FPGA).
- Measured speedup of any design relative to another, or relative to a
  software baseline (all cycle counts are simulation measurements of
  isolated modules, not end-to-end system throughput comparisons).
- Measured board power (all power figures are Vivado's vector-less,
  "Medium confidence" static estimates, not measurements from a running
  device).
- BatchNorm-folded quantized inference (every design uses `bias = 0`
  because the first Conv2d layer has no direct bias and BatchNorm
  parameters have not been folded into the weights).

Several of these (full 32-channel scaling, BatchNorm folding, board testing,
measured power/speedup) are reasonable candidates for future work, but they
are not completed as of this report.

## 7. Recommended next steps

Ranked by suggested priority:

1. **Add backpressure / larger-image support to the streaming time-mux
   prototype.** The current streaming design buffers all windows before
   processing begins because `window3x3_stream` has no back-pressure and
   the one-window scheduler is far slower than pixel arrival; a real
   streaming buffer (e.g. a FIFO with backpressure into the window
   generator) would be needed for images larger than the 5x5 toy case.
2. **Add a 32-output first-layer scaling estimate without necessarily
   implementing all 32 in parallel.** The 1/4/8-output LUT trend and the
   8-output DSP-overutilization result both suggest that neither pure
   parallel nor pure DSP-mapped scaling reaches 32 channels on this part;
   an estimate (extrapolation or a partial synthesis at an intermediate
   channel count) would clarify how much of a gap remains, and whether a
   time-multiplexed or hybrid approach is required.
3. **Optionally synthesize selected designs for a larger FPGA target** as a
   scaling comparison, to separate "this specific part is too small" from
   "this architecture does not scale" as explanations for the observed LUT
   and DSP pressure.
4. **Prepare a slide-ready architecture comparison table** summarizing the
   tradeoffs in this report for non-hardware audiences, once the above
   scaling questions are better characterized.

## Do not commit

This report is documentation only. Per task instructions, it has not been
committed to the repository.
