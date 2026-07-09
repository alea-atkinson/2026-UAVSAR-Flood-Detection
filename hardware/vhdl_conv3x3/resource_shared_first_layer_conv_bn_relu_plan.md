# Resource-Shared 32-Output First-Layer Conv-BN-ReLU: Design Plan

Date: July 9, 2026

This is a design plan for the complete (32-output) first-layer folded
Conv-BN-ReLU stage using a resource-shared (time-multiplexed) datapath,
plus (see the accompanying summary,
`first_layer_32out_bn_relu_resource_shared_summary.md`) a smaller
implemented prototype that validates the architecture at reduced scale.
This document does not claim the full U-Net fits, does not claim board
testing, measured speedup, or measured power, and does not claim
deployment readiness.

## 1. Purpose

`first_layer_32out_dsp_200t_summary.md` showed that a **direct-parallel**
32-output DSP-aware first Conv2d layer synthesizes and meets 100 MHz
timing on Artix-7 200T, but only by fully saturating the device's DSP
budget (740/740, 100.00%) and falling back to LUT/carry-chain logic for
the final cross-channel summation stage, for every one of the 32 kernels
(`first_layer_24out_dsp_fallback_analysis.md` and its 32-output
follow-on documented this fallback in detail). This result demonstrates
that direct-parallel mapping of the complete first layer is
synthesis-feasible on this specific larger target, but is not a
scalable strategy -- there is no DSP headroom left for anything else, and
a smaller/cheaper target would not support it at all (the existing
`larger_fpga_target_comparison_summary.md` and 24-output fallback
analysis already showed the same architecture badly overshoots the
90-DSP Artix-7 35T).

This plan proposes and (at reduced scale) implements a **resource-shared**
alternative: reuse a small, fixed amount of multiply/accumulate and
BN+ReLU hardware across all 32 output channels (and, eventually, across
spatial positions for larger tiles), trading latency for dramatically
lower DSP and LUT pressure.

## 2. Why direct-parallel mapping is resource-expensive

The direct-parallel 32-output design instantiates one
`conv3x3_dot_pipelined_dsp` per (kernel, input-channel) pair -- 32 x 3 =
96 dot-product datapaths, each with 9 taps, for 96 x 9 = 864 raw
multiplies, plus 32 independent final-summation adders (one per kernel).
Every one of those 96 dot-product units and 32 adders exists as separate,
simultaneously-active hardware, permanently occupying its own DSP48E1
slices (or LUT fabric, once DSPs run out) for the entire time the design
is powered, whether or not that specific kernel's result is needed at
that instant. This is fast (all 32 kernels' results become available
together, every clock, once the pipeline is full) but resource cost
scales **linearly with the number of output channels** -- exactly the
scaling problem the 8/16/24/32-output DSP-aware synthesis series
(`first_layer_16out_dsp_200t_summary.md`,
`first_layer_24out_dsp_200t_summary.md`,
`first_layer_32out_dsp_200t_summary.md`) tracked in increasing severity.

## 3. Existing direct-parallel 32-output result, for comparison

From `first_layer_32out_dsp_200t_summary.md` (commit `af222b1a`):

| Metric | Value |
|---|---:|
| GHDL | PASS 288/288 |
| LUTs | 10,461 / 134,600 (7.77%) |
| Registers | 4,618 / 269,200 (1.72%) |
| DSP48E1 | **740 / 740 (100.00%)** |
| BRAM | 0 / 365 |
| WNS | +2.343 ns |
| WHS | +0.262 ns |
| Total power (estimate) | 1.324 W (dynamic 1.196 W, static 0.128 W) |
| Warnings | 1 (`DSP overutilized: Used = 1044, Available = 740`) |

This is the baseline this plan aims to improve on: same functional scope
(complete 32-channel first Conv2d layer, raw -- not yet BN/ReLU-folded --
in that specific design), zero DSP headroom remaining.

## 4. Existing time-mux/resource-shared prototypes already in the repo

Three existing, GHDL-verified prototypes already establish the
resource-sharing pattern this plan extends:

1. **`conv3x3_dot_time_mux.vhd`** -- ONE reusable 8x8 signed multiplier,
   time-shared across the 9 taps of a single 3x3 dot product (10 cycles
   per dot product: 1 load cycle + 9 tap cycles).
2. **`conv3x3_3chan_8out_time_mux.vhd`** -- a scheduler FSM that reuses
   ONE `conv3x3_dot_time_mux` instance sequentially across 8 kernels x 3
   input channels = 24 (kernel, channel) steps, for a single
   already-captured 3x3x3 window. Measured: 265 cycles per window
   (`time_mux_8out_summary.md`).
3. **`stream_conv3x3_3chan_8out_time_mux.vhd`** -- a streaming wrapper: 3
   unmodified `window3x3_stream` instances capture all 9 valid windows of
   a 5x5 image into a small buffer (phase 1, ~25 cycles), then the single
   shared scheduler from (2) processes the 9 buffered windows one at a
   time, non-overlapped (phase 2). Measured: 2,421 total cycles for the
   full 5x5x3 toy image, 8 kernels (`stream_time_mux_8out_summary.md`).

**These existing prototypes cover only 8 of 32 output channels, and
compute the RAW convolution only (bias = 0, no BatchNorm, no ReLU) --
they do not yet apply folded scale/bias/ReLU.** This plan's contribution
is: (a) extending the kernel count toward all 32, and (b) adding a
resource-shared, Q.16 fixed-point folded BN+ReLU stage on top of the
existing resource-shared raw-convolution pattern, using the same folding
methodology already verified for kernel 0 and kernel 2
(`first_conv_bn_relu_kernel0_pipelined_summary.md`,
`first_conv_bn_relu_harder_kernel_pipelined_summary.md`).

## 5. Proposed resource-shared architecture

**Three-level structure, directly extending the existing pattern (item 4
above), scaled to 32 kernels and augmented with a resource-shared
BN+ReLU stage:**

```
Level 1: conv3x3_dot_time_mux (UNCHANGED, reused as-is)
  ONE 8x8 signed multiplier, time-shared over 9 taps. 10 cycles/dot product.

Level 2: conv3x3_3chan_32out_bn_relu_time_mux (NEW; extends
  conv3x3_3chan_8out_time_mux's pattern from 8 to 32 kernels)
  ONE shared Level-1 instance, sequenced by an FSM over
  32 kernels x 3 channels = 96 (kernel, channel) steps for a SINGLE
  already-captured window. After each kernel's 3rd channel completes,
  a SECOND shared resource -- ONE Q.16 fixed-point multiply-add-ReLU
  unit (the same two-stage structure already verified in
  stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd) -- is reused to
  rescale that kernel's raw INT32 sum by its own SCALE_FX, add its own
  BIAS_FX, and apply ReLU, producing that kernel's final Q.16 output.
  This BN+ReLU unit is itself resource-shared: only ONE exists, reused
  32 times per window (once per kernel), not 32 in parallel.

Level 3: stream_conv3x3_3chan_32out_bn_relu_time_mux (NEW; extends
  stream_conv3x3_3chan_8out_time_mux's two-phase capture/process pattern)
  3 UNCHANGED window3x3_stream instances capture all NUM_WINDOWS valid
  windows of the input tile (9 for the 5x5x3 toy input). The single
  shared Level-2 engine then processes the buffered windows
  sequentially, non-overlapped, producing 32 Q.16 fixed-point outputs
  per window.
```

**Total unique compute resources, regardless of kernel count or tile
size**: one 8x8 multiplier (raw convolution), one Q.16 fixed-point
multiply-add-ReLU unit (folded BN+ReLU), and the fixed control/weight-LUT
logic to sequence them. This is the core resource-sharing claim: DSP/LUT
pressure no longer scales with the number of output channels.

## 6. Datapath

Per (kernel, channel) step, `conv3x3_dot_time_mux` computes
`sum(p_i * w_i for i in 0..8)` using its one shared multiplier (bias
input tied to 0, since the per-channel dot product itself carries no
bias -- consistent with every prior design in this repo). The Level-2
FSM accumulates the three per-channel results for a kernel into a
registered `partial_sum`. On the third channel's completion, that
kernel's raw INT32 sum is exact and complete. It is then fed through the
same Q.16 fixed-point sequence already verified for kernel 0/kernel 2:

```
product_fx = raw_sum * SCALE_FX(kernel_idx)      -- Q.16, exact, no shift
biased_fx  = product_fx + BIAS_FX(kernel_idx)      -- Q.16, no shift
relu_fx    = biased_fx if biased_fx > 0 else 0      -- Q.16
```

registered across two pipeline stages (mirroring
`stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd`'s Stage 1/Stage 2
split, which was shown to close 100 MHz timing where a single
combinational stage did not -- see
`first_conv_bn_relu_kernel0_pipelined_summary.md`). `relu_fx` is then
latched into that kernel's output register.

## 7. Control/scheduling

A single FSM per Level-2 instance, extending
`conv3x3_3chan_8out_time_mux`'s `step = kernel_idx * 3 + channel_idx`
scheme from `step in 0..23` (8 kernels) to `step in 0..95` (32 kernels):

```
for step in 0 to (NUM_KERNELS*3 - 1):
    kernel_idx  = step / 3
    channel_idx = step mod 3
    dispatch conv3x3_dot_time_mux with (pixel channel, weight[kernel_idx][channel_idx])
    on its done pulse:
        accumulate into partial_sum(kernel_idx)
        if channel_idx == 2:
            dispatch the 2-stage BN+ReLU unit on partial_sum(kernel_idx)
            on its 2-cycle completion: latch relu_fx into y(kernel_idx)
            advance to next kernel
```

Level 3 adds an outer loop over `proc_idx in 0..NUM_WINDOWS-1`, dispatching
the Level-2 engine once per buffered window, exactly as
`stream_conv3x3_3chan_8out_time_mux.vhd` already does for its 8-kernel,
raw-only engine.

## 8. Weight storage strategy

Same convention as `conv3x3_3chan_8out_time_mux.vhd`'s `WEIGHT_LUT`: a
VHDL constant array of `NUM_KERNELS * 3` entries (one `signed8_kernel_t`
per (kernel, channel) pair), built once at elaboration time from a
generated VHDL package's plain-integer constants (via the same
`to_signed_kernel` conversion function already used throughout this
repo). For 32 kernels this is a 96-entry constant LUT of INT8 weights
(96 x 9 = 864 INT8 values, same total weight count as the direct-parallel
design -- weight storage cost is unchanged by resource sharing, only the
*compute* hardware is shared). Two additional small constant arrays hold
`SCALE_FX` (32 entries) and `BIAS_FX` (32 entries), analogous to the
existing `BIAS_LUT`.

Because these are VHDL `constant`s (not registers), Vivado is free to
implement them as LUT-ROM, block RAM, or distributed logic as it sees
fit during synthesis -- this plan does not mandate BRAM usage, though
BRAM is a plausible outcome for a 96-entry weight table and is worth
watching for in the synthesis report (see Section 16).

## 9. Bias/ReLU placement

Folded bias (`BIAS_FX`) and ReLU are applied **once per kernel per
window**, after that kernel's three-channel raw sum is complete -- not
once per (kernel, channel) step. This mirrors exactly where bias and
ReLU are applied in every existing folded design in this repo (kernel 0,
kernel 2): after summing all input channels' contributions for one
output channel, never partway through.

## 10. Fixed-point format

**Identical to the existing kernel0/kernel2 work**: Q.16 (signed, 16
fractional bits), `SCALE_FX` and `BIAS_FX` per kernel generated by the
same BatchNorm-folding and symmetric per-output-channel INT8
quantization formulas already used by
`scripts/generate_bn_relu_fixed_point_vectors.py`, with the same
deliberate `scale_x = 1.0` simplification for the toy input (exact and
lossless for this specific integer-valued patch). No new fixed-point
convention is introduced; a resource-shared design should not also
introduce new numerical risk.

## 11. Expected output ordering

**Window-major, kernel-minor**: for each of the `NUM_WINDOWS` valid
spatial positions (in the same row-major order `window3x3_stream`
already produces), emit all `NUM_KERNELS` kernel outputs together as one
`valid_out` pulse, before moving to the next window. This is the SAME
ordering already used by `stream_conv3x3_3chan_8out_time_mux.vhd` (whose
`valid_out` pulses carry `y0..y7` together, once per window) and by the
direct-parallel 32-output design's `valid_out` pulses (`y0..y31`
together, once per window) -- chosen specifically so this new design's
per-window output tuple is directly comparable to both existing
designs' output order, position by position.

The alternative (kernel-major: finish all 9 positions for kernel 0
before starting kernel 1) does not fit this architecture naturally,
since windows are captured once and reused for all kernels within the
capture-then-process phase structure already established -- window-major
order is both the natural hardware behavior and the more directly
comparable choice.

## 12. Latency estimate for 5x5x3 toy input

Extrapolating directly from the measured 8-kernel figures
(`time_mux_8out_summary.md`, `stream_time_mux_8out_summary.md`):

- Level 2 (one window, N kernels): `N * 3` dot ops x 10 cycles, plus
  `N * 3` one-cycle registration gaps (one per dot-op dispatch, per the
  8-kernel design's measured overhead), plus `N * 2` cycles for the
  2-stage BN+ReLU unit, once per kernel. For **N=32**:
  `32*3*10 + 32*3*1 + 32*2 = 960 + 96 + 64 = 1120` cycles per window
  (estimate; the accompanying implemented prototype measures the actual
  figure for its own kernel count rather than asserting this exact
  number -- see Section 15).
- Level 3 (9 windows, capture + sequential process): `~25` cycles capture
  + `9 * (per-window figure)`. For N=32: `25 + 9*1120 ≈ 10,105` cycles
  total (**estimate**, to be confirmed empirically once/if a full
  32-kernel implementation is built; see Section 18).

For comparison, the 8-kernel raw-only design measured 2,421 total cycles
for the same 5x5x3 image (`stream_time_mux_8out_summary.md`); the direct
-parallel design's equivalent (once pipeline-primed) is a handful of
cycles for all 9 windows combined. The resource-sharing tradeoff is
explicit and large: **roughly 3-4 orders of magnitude more clock cycles
for the same output, in exchange for compute hardware that does not grow
with kernel count.**

## 13. Latency estimate formula for HxW input

For a square `S x S` valid-convolution image (`S = IMG_WIDTH`, matching
`window3x3_stream`'s existing square-image assumption), with `N` output
kernels:

```
NUM_WINDOWS(S)        = (S - 2)^2
cycles_per_window(N)  = N*3*10 + N*3*1 + N*2   =  N*(30 + 3 + 2) = 35*N
capture_cycles(S)     ~= S^2                      (one pixel-triplet per clock)
total_cycles(S, N)    ~= S^2 + (S - 2)^2 * 35*N
```

For the canonical toy input (S=5, N=32): `25 + 9*35*32 = 25 + 10,080 =
10,105` cycles, matching Section 12's estimate. For a realistic UAVSAR
tile (e.g. S=256, N=32, not attempted in this task):
`65,536 + 64,516 * 1,120 ≈ 7.2e7` cycles -- at 100 MHz that is roughly
0.72 seconds **per first-layer pass alone**, for this specific
single-engine sharing degree. This is the central scaling limitation of
the "share everything down to one engine" extreme and is discussed
further in Section 17/18.

## 14. Expected resource tradeoff versus direct parallel

| Resource | Direct-parallel 32-out (measured) | Resource-shared 32-out (expected) |
|---|---:|---|
| DSP48E1 | 740/740 (100%) | A small, roughly constant number (1-2 multipliers' worth), independent of kernel count |
| LUTs | 10,461 | Expected substantially lower once weight-LUT/BRAM overhead is accounted for; not yet measured at 32-kernel scale |
| Registers | 4,618 | Expected lower (one accumulator/pipeline set, not 32) |
| Latency | ~4-5 cycles (pipelined, all windows in parallel once primed) | ~10,000+ cycles for the same 5x5x3 image (Section 12) |
| Warnings | 1 (DSP overutilization) | Expected 0, since DSP demand is no longer proportional to kernel count |

**This table states an expectation, not a measured result for 32
kernels** -- the accompanying implementation (Deliverable B) measures
actual resource/timing numbers at a smaller, tractable kernel count (see
the summary document) to validate this expectation before committing to
a full 32-kernel build.

## 15. Verification plan

1. **Python golden generation**: extend the existing per-kernel folding
   methodology to a full kernel *set* in one script run (not one
   kernel at a time), producing exact Q.16 fixed-point golden outputs in
   window-major order for all `NUM_WINDOWS * NUM_KERNELS` output values.
2. **GHDL**: a self-checking testbench streams the canonical 5x5x3 toy
   image into the DUT and checks every produced output (in emission
   order) against the Python golden array, exactly as
   `tb_stream_conv3x3_3chan_8out_time_mux.vhd` already does for the raw
   8-kernel case.
3. **Regression**: re-run every existing folded Conv-BN-ReLU and
   direct-parallel 32-output testbench unmodified, to confirm this new
   design does not disturb any prior verified result.
4. Measure actual total cycle count directly from the testbench
   (matching the existing convention of *measuring*, not *asserting*, an
   exact cycle count for time-multiplexed designs), and compare against
   the Section 12/13 estimate.

## 16. Synthesis plan

Synthesize on `xc7a200tsbg484-1` (same 100 MHz / 10.000 ns, out-of-context
style as every other design in this repo), and specifically inspect
whether Vivado maps the weight LUT to BRAM, distributed LUT-ROM, or
registers -- this affects whether "resource-shared" claims should be
qualified further (a large BRAM-mapped weight table would be a different
resource-cost story than a small LUT-ROM one). Collect the same
utilization/timing/power metrics as every other design in this repo.

## 17. Limitations

- **This plan targets 32 kernels on the 5x5x3 toy input; the
  accompanying implementation validates the architecture at a smaller,
  explicitly scoped kernel count** (see the summary document for the
  exact count implemented and why). It does not claim the full 32-kernel
  resource-shared design has been built, synthesized, or measured.
- **Single-engine sharing is the most extreme point on the
  latency/resource tradeoff curve.** Section 13's scaling formula shows
  latency grows linearly with both image area and kernel count under
  this "share everything" strategy -- for realistic tile sizes this
  would be far too slow for any real-time framing, which is exactly why
  this is explored as an architecture *option*, not a recommended final
  design. A small number of parallel lanes (e.g. 2, 4, or 8 shared
  engines instead of 1) is the natural next tradeoff point and is
  flagged as the recommended next step (Section 18).
- **This is still first-layer convolution only.** No padding, pooling,
  BatchNorm folding beyond what is already established for kernel 0/2,
  downstream layers, or full model pipeline. Does not claim the full
  U-Net fits.
- **No board testing, no measured speedup, no measured board power.**
  All figures are Vivado `synth_design`/GHDL simulation results or
  documented estimates, never measurements from real hardware.
- **Toy input only.** The 5x5x3 canonical patch, not a real UAVSAR tile;
  this is not a real-world inference benchmark.
- **Weight-LUT resource behavior at 32-kernel scale (BRAM vs. LUT-ROM
  vs. distributed) is not yet measured** -- only inferred from the
  smaller implemented prototype's own synthesis result.

## 18. Recommended implementation path

1. **Now (this task)**: implement and verify the smallest serious
   resource-shared folded Conv-BN-ReLU prototype at reduced kernel count
   (see the accompanying summary document for exact scope), reusing the
   existing `conv3x3_dot_time_mux` engine unchanged and extending the
   existing `conv3x3_3chan_8out_time_mux` / `stream_conv3x3_3chan_8out_time_mux`
   pattern with a new resource-shared BN+ReLU stage.
2. **Next**: scale the same architecture's kernel count toward the full
   32 (widening the weight LUT and step counter, no structural change),
   and measure actual resource/timing/latency at that full scale to
   replace Section 12-14's estimates with measured figures.
3. **After that**: explore a small-lane variant (2, 4, or 8 parallel
   resource-shared engines, each responsible for a subset of the 32
   kernels) to find a better point on the latency/resource curve than
   either the single-engine extreme (this plan) or the fully-parallel
   extreme (the existing direct-parallel 32-output design) -- this is
   the more realistic target for any eventual larger-tile deployment
   discussion, though still far short of a real-time claim.
4. **Not yet planned**: extending beyond the toy input to realistic tile
   sizes, which (per Section 13's formula) would require either far more
   parallel lanes or a fundamentally different windowing/buffering
   strategy (true streaming with backpressure, rather than the
   capture-then-process buffering used here) before latency becomes
   practical.
