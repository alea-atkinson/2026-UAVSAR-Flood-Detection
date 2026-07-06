# First-Layer 32-Output Scaling Estimate

Date: July 6, 2026

This is an **analysis document, not a new RTL implementation**. No 32-output
VHDL design was created, and no new Vivado synthesis was run to produce
this report. All numbers below are either copied from existing measured
results already in this repository, or explicitly derived/estimated from
those measurements, with the derivation shown and the confidence level
labeled at each step.

## 1. Purpose and scope

This document estimates what would happen if the current first-layer
convolution prototypes were scaled from their verified output-channel
counts (1, 4, and 8) up to **all 32 output channels** of
`enc1.block.0.weight` (shape `[32, 3, 3, 3]`) -- the full first Conv2d
layer of the trained UAVSAR flood-segmentation U-Net.

**No design in this repository has implemented, synthesized, or verified a
32-output first-layer convolution.** The most output channels verified in
any working prototype so far is 8 (both fully parallel and time-multiplexed
variants). This document only projects from that measured data; it does not
replace an actual 32-output implementation and synthesis run.

Target part for all estimates: Artix-7 `xc7a35tcpg236-1`
(20800 LUTs, 41600 registers, 90 DSPs, 50 BRAM tiles), 100 MHz clock target,
matching every synthesis run referenced in this repository.

## 2. Parallel LUT-only estimate

### Why not just multiply by 4

Naively scaling the 8-output LUT-only design by 4x (to reach 32 outputs)
would overstate resource growth, because the fully parallel designs share
fixed overhead across all output kernels: exactly **3**
`window3x3_stream` instances are instantiated regardless of kernel count (1,
4, or 8 kernels all reuse the same 3 window generators), so that fixed cost
is amortized over more kernels as the design grows. Only the
`conv3x3_dot_pipelined` units scale linearly with kernel count (3 per
kernel: one per input channel). This is exactly why the measured 1->4->8
LUT growth (2731 -> 5020 -> 9374) is sub-linear rather than proportional to
kernel count, as documented in `vivado_8out_synthesis_summary.md` and
`fpga_architecture_comparison_summary.md`.

### Incremental trend from 4-output to 8-output (measured)

Using the 4-output and 8-output measured results directly:

| Metric | 4-output (measured) | 8-output (measured) | Delta for +4 outputs |
|---|---:|---:|---:|
| LUTs | 5020 | 9374 | **4354** |
| Registers | 3214 | 5704 | **2490** |
| Total power | 0.177 W | 0.271 W | **0.094 W** |

This delta captures the *marginal* cost of 4 more output kernels once the
fixed window-generator/control overhead is already paid for by the 4-output
design. It is a better basis for projecting further growth than either a
naive per-kernel average (which would over-count the shared overhead) or a
flat multiply of the 8-output total (which double-counts the fixed
overhead every time).

### 16-output and 32-output estimates (incremental-trend method)

Applying the 4-to-8 incremental delta forward, assuming the SAME per-4-output
marginal cost continues to hold (an assumption, not a certainty -- see
caveats below):

| Metric | 16-output estimate | 32-output estimate |
|---|---:|---:|
| LUTs | 9374 + 2 x 4354 = **18082** | 9374 + 6 x 4354 = **35498** |
| Registers | 5704 + 2 x 2490 = **10684** | 5704 + 6 x 2490 = **20644** |
| Total power | 0.271 + 2 x 0.094 = **0.459 W** | 0.271 + 6 x 0.094 = **0.835 W** |

Utilization against the target part (20800 LUTs, 41600 registers):

| Metric | 16-output estimate | 32-output estimate |
|---|---:|---:|
| LUT utilization | 18082 / 20800 = **86.9%** | 35498 / 20800 = **170.7%** |
| Register utilization | 10684 / 41600 = **25.7%** | 20644 / 41600 = **49.6%** |

**Confidence: estimate from 4-to-8 incremental trend.** This is a two-point
linear extrapolation (only the 4-output and 8-output data points were used
to derive the marginal per-4-output cost), not a fitted model over many
data points, and not a synthesized measurement at 16 or 32 outputs.

### Naive 8-output x N estimate (labeled explicitly as naive, for comparison only)

A simpler but **overstated** method — multiplying the full 8-output totals
by 2x or 4x, ignoring the fact that window-generator overhead is already
shared — gives:

| Metric | Naive 16-output (8-out x 2) | Naive 32-output (8-out x 4) |
|---|---:|---:|
| LUTs | 18748 | 37496 |
| Registers | 11408 | 22816 |
| Total power | 0.542 W | 1.084 W |

This naive method is included only to show that even the more optimistic
incremental-trend estimate is in a similar ballpark for LUTs at 32 outputs
(35498 vs. 37496 -- both exceed the part), while it diverges more visibly
at 16 outputs (18082 incremental vs. 18748 naive) and for registers
throughout. **The incremental-trend numbers above are the primary estimate
used in this report; the naive numbers are shown only as an upper-bound
sanity check, not as the recommended projection.**

### Interpretation

- **16-output parallel might barely fit by LUT count** (86.9% utilization
  under the incremental-trend estimate) but would leave very little margin
  for place-and-route, clock buffering, or any other logic sharing the
  device.
- **32-output parallel does not fit by LUT count** on this Artix-7 target
  under either estimation method (170.7% incremental-trend, or 180.3% naive)
  -- both exceed the part's 20800 available LUTs.
- **Timing/routing could get harder even before the absolute LUT count is
  exceeded.** All three measured designs (1/4/8-output) show setup slack
  holding roughly steady (+4.456 ns, +5.117 ns, +5.105 ns), but that
  measurement says nothing about place-and-route difficulty at 87-100%+
  LUT utilization, where routing congestion typically degrades achievable
  Fmax well before the raw resource count is the binding constraint. This
  is a known general FPGA implementation risk, not something measured in
  this repository's out-of-context synthesis-only results.

## 3. DSP-aware estimate

The 8-output DSP-aware design (`vivado_8out_dsp_synthesis_summary.md`)
already demonstrates the ceiling for this approach:

- The 8-output DSP-aware design requested DSP mapping for its multiplies
  and **already uses all 90 DSP slices** on the target part after Vivado's
  fallback legalization (some multiplies that could not fit in the 90
  available DSPs were automatically remapped back to LUT fabric).
- The theoretical multiplier count for 8 outputs is:
  **8 outputs x 3 channels x 9 taps = 216 multiplies.**
- The target part has only **90 DSPs** -- fewer than half of what 8 outputs
  alone would need if every multiply were DSP-mapped.
- The theoretical multiplier count for 32 outputs is:
  **32 outputs x 3 channels x 9 taps = 864 multiplies.**
- **864 multiplies vs. 90 available DSPs: fully parallel DSP mapping for
  32 outputs is impossible on this part** -- not tight, not marginal,
  impossible by a factor of ~9.6x.
- Even the 8-output case (216 multiplies) already exceeds the 90-DSP
  budget by more than 2x before Vivado's fallback legalization kicks in.

**Confidence: theoretical, impossible by DSP count.** This is a direct
arithmetic comparison (multiplies needed vs. DSPs available), not an
extrapolation -- the 8-output case has already been measured and confirms
the fallback behavior Vivado uses when the requested DSP count exceeds the
part's supply.

DSP-aware mapping, based on this evidence, is useful only as a **partial or
fallback mapping strategy** (as Vivado already does automatically when
over-requested) or as one building block inside a **time-multiplexed or
hybrid architecture** where a small, fixed number of DSP-backed
multiply-accumulate lanes are reused across many kernels/channels/positions
-- not as a route to fully parallel 32-output DSP mapping on this part.

## 4. Time-mux scaling estimate

### One-window scheduler

The measured 8-output one-window time-multiplexed scheduler
(`time_mux_8out_summary.md`) takes **265 cycles** to produce all 8 outputs
for one pre-loaded 3-channel 3x3 patch, using a single reusable
dot-product lane sequenced across 24 (kernel, channel) operations (8
kernels x 3 channels).

A simple 32-output version using the SAME single dot-product lane would
need to sequence through 4x as many (kernel, channel) combinations (96
instead of 24), so a rough estimate is:

- **~4 x 265 = ~1060 cycles** per 3x3x3 window, plus whatever additional
  controller overhead is needed to address 32 kernels instead of 8 (a
  larger weight LUT and a wider kernel-index counter, not a fundamentally
  different control structure).
- **Resource growth could stay modest**: reusing the same single
  dot-product lane means the multiplier/accumulator hardware itself does
  not grow with kernel count. The main added cost would be: a larger
  weight lookup table (32 x 3 x 9 INT8 values instead of 8 x 3 x 9), 32
  output registers instead of 8, and a slightly wider kernel-select
  control path. None of these are expected to approach the part's LUT or
  register ceiling based on how modestly the 8-output scheduler's resource
  usage (376 LUTs, 510 registers) compares to the fully parallel 8-output
  design (9374 LUTs, 5704 registers).
- **This would likely fit on the target part** by LUT/register/DSP count,
  but would be roughly 4x slower per window than the already-slow 8-output
  one-window scheduler.

**Confidence: rough time-mux extrapolation.** The 4x cycle-count scaling
follows directly from the scheduler's own documented behavior (one
dot-product operation per (kernel, channel) pair, ~10-11 cycles each,
sequenced with no overlap), but the exact resource growth for a 32-kernel
weight LUT and wider control path has not been measured or synthesized.

### Streaming time-mux

The measured 8-output streaming time-multiplexed prototype
(`stream_time_mux_8out_summary.md`) takes **2421 cycles** for 9 spatial
windows x 8 outputs on the 5x5 toy image (streaming capture of all 9
windows, then sequential processing of each through the one-window
scheduler, with no overlap between windows).

A simple 32-output version with the same single compute lane would likely
need roughly 4x the compute work per window (matching the one-window
scheduler's own 4x estimate above), giving a rough estimate of:

- **~4 x 2421 = ~9684 cycles** for the same 9 windows on the same 5x5 toy
  image.
- This is **an estimate only, not a measured result** -- no 32-output
  streaming design has been built or simulated.
- It would likely **fit by LUT/DSP count** (following the same reasoning as
  the one-window scheduler: the shared dot-product lane and window
  generators do not scale with kernel count, only the weight LUT, output
  registers, and control width do), **but would be far too slow** for any
  application with a real-time or near-real-time throughput requirement,
  unless the throughput requirement is loose or the design is extended
  with multiple lanes (see Section 5).

**Confidence: rough time-mux extrapolation.** Same caveat as above -- the
cycle-count scaling is a direct consequence of the design's own measured
sequential behavior, but no 32-output version has been synthesized or
simulated to confirm resource usage or the exact cycle count.

## 5. Hybrid architecture discussion

The estimates above point to a three-way tradeoff, none of which is
individually satisfying for a 32-output target on this part:

- **Fully parallel 32-output**: likely does not fit by LUT count (~171%
  utilization under the incremental-trend estimate).
- **Fully parallel DSP-mapped 32-output**: impossible by DSP count (864
  multiplies needed vs. 90 available).
- **Single-lane time-multiplexed 32-output**: likely fits comfortably by
  LUT/register/DSP count, but at roughly 4x the already-high latency of
  the 8-output time-mux prototypes (~1060 cycles/window for the one-window
  scheduler, ~9684 cycles for a full 5x5 image pass in the streaming
  version) -- a large latency cost for full resource headroom.

A **hybrid architecture using multiple reusable time-mux lanes** (e.g. 2,
4, or 8 parallel dot-product lanes, each time-multiplexed across a subset
of the 32 kernels) is a likely better middle ground:

- Using **N** lanes instead of 1 would divide the per-window cycle count by
  roughly N (each lane handles 32/N kernels' worth of (kernel, channel)
  operations), while multiplying the lane-specific resource cost
  (multiplier, accumulator, tap-select logic -- on the order of the 205
  LUTs / 214 registers measured for the standalone
  `conv3x3_dot_time_mux` primitive) by roughly N.
- This trades some of the time-mux resource savings for reduced latency,
  aiming for a point between the two extremes above: meaningfully faster
  than a single lane, while still using dramatically fewer LUTs/DSPs than
  a fully parallel 32-output design.
- **A reasonable future experiment** would be to actually synthesize a
  2-lane or 4-lane time-mux design (reusing the existing
  `conv3x3_dot_time_mux` primitive, replicated a small number of times with
  a wider scheduler) to get a real measured data point for this middle
  ground, rather than continuing to extrapolate from the single-lane
  result.

No lane count has been synthesized or simulated for this hybrid direction;
it is presented here as the most promising next step, not as a result.

## 6. Summary table

| Design | LUTs | LUT util. | Registers | DSPs | Cycle behavior | Likely fits on xc7a35tcpg236-1? | Confidence |
|---|---:|---:|---:|---:|---|---|---|
| 8-output parallel | 9374 | 45.07% | 5704 | 0 | ~4-cycle pipeline latency, 1 result/clock throughput | Yes (measured) | measured |
| 16-output parallel | ~18082 | ~86.9% | ~10684 | 0 | (same streaming/pipeline style, not measured) | Marginal -- little headroom left | estimate from 4-to-8 incremental trend |
| 32-output parallel | ~35498 | ~170.7% | ~20644 | 0 | (same streaming/pipeline style, not measured) | **No** -- exceeds available LUTs | estimate from 4-to-8 incremental trend |
| 8-output DSP-aware | 6991 | 33.6% | 4405 | 90/90 | Same cycle-accurate behavior as 8-out LUT-only | Yes, but at 100% DSP budget already | measured |
| 32-output full DSP (theoretical) | -- | -- | -- | 864 needed vs. 90 available | -- | **No** -- impossible by DSP count | theoretical, impossible by DSP count |
| 8-output one-window time-mux | 376 | 1.81% | 510 | 0 | 265 cycles for one 3x3x3 window's 8 outputs | Yes (measured) | measured |
| 32-output one-window time-mux | (modest growth expected; not computed) | (low; not computed) | (modest growth expected; not computed) | 0 | ~1060 cycles per window (~4x) | Likely yes by resource count, but very slow | rough time-mux extrapolation |
| 8-output streaming time-mux | 1409 | 6.77% | 3328 | 0 | 2421 cycles for 9 windows x 8 outputs (5x5 toy image) | Yes (measured) | measured |
| 32-output streaming time-mux | (modest growth expected; not computed) | (low; not computed) | (modest growth expected; not computed) | 0 | ~9684 cycles for the same 5x5 toy image (~4x) | Likely yes by resource count, but very slow | rough time-mux extrapolation |

LUT and register figures for the 32-output time-mux rows are intentionally
left as qualitative ("modest growth expected") rather than a specific
number, because -- unlike the parallel design's LUT growth, which has three
measured data points (1/4/8 outputs) to extrapolate from -- there is only
ONE measured time-mux data point (8 outputs) for each time-mux design, which
is not enough to establish a reliable per-kernel resource-growth rate. Only
the cycle-count scaling (which follows directly from the sequential,
one-operation-at-a-time control structure) is estimated numerically here.

## 7. Final conclusion

- **Fully parallel 32-output first-layer convolution likely does not fit**
  on the Artix-7 `xc7a35tcpg236-1` by LUT count, under the incremental
  trend estimated from the measured 1/4/8-output data (~171% LUT
  utilization).
- **Fully parallel DSP mapping for 32 outputs is impossible by DSP count**:
  864 multiplies would be needed against only 90 available DSP48E1 slices
  on this part -- not a close call.
- **Single-lane time-multiplexing likely fits comfortably by resource
  count** at 32 outputs (extrapolating from the 8-output time-mux designs'
  very low LUT/register/DSP usage), **but would be much slower** --
  roughly 4x the already-substantial cycle counts measured for the
  8-output time-mux prototypes.
- **The most promising future direction is a hybrid architecture** using a
  small number of reusable time-mux lanes (e.g. 2-4), trading some of the
  single-lane resource savings for reduced latency -- or, alternatively,
  evaluating a larger FPGA target if fully parallel throughput is required.
- **These are estimates, not replacements for actual synthesis.** No
  16-output or 32-output design (parallel, DSP-aware, or time-multiplexed)
  has been built, simulated, or synthesized. All numbers above beyond the
  measured 1/4/8-output data points are projections with explicitly labeled
  confidence levels, and should be confirmed by real synthesis before being
  used for any final architecture decision.

## 8. Safe claims / claims to avoid

### Safe to claim

- Based on measured 1/4/8-output trends, 32-output fully parallel scaling
  is likely LUT-limited on this target (~171% LUT utilization under the
  incremental-trend estimate, and still over budget under the naive
  estimate).
- The 8-output DSP-aware result already exhausts DSP resources (90/90
  used), so full 32-output DSP mapping is not feasible on this target
  (864 multiplies needed vs. 90 available, a ~9.6x shortfall).
- Time-mux architectures are resource-efficient (measured: 376-1409 LUTs
  for 8 outputs, vs. 9374 for the fully parallel 8-output design) but
  latency-heavy (measured: 265-2421 cycles, vs. a few cycles of pipeline
  latency for the fully parallel design).

### Avoid claiming

- Do **not** claim that a 32-output design was implemented -- it was not;
  this document is a projection from 1/4/8-output measured data.
- Do **not** claim exact 32-output resource usage -- all 16/32-output
  parallel figures are two-point linear extrapolations, and all 32-output
  time-mux figures are qualitative/cycle-count-only estimates, not
  synthesized measurements.
- Do **not** claim full U-Net FPGA inference -- this remains limited to
  first-layer convolution channel-count scaling only.
- Do **not** claim board testing -- every number in this document and its
  source summaries comes from GHDL simulation and Vivado out-of-context
  synthesis estimation, not a programmed device.
- Do **not** claim measured speedup -- the cycle-count comparisons in
  Sections 4-6 are simulation-measured latencies for isolated modules,
  not an end-to-end throughput benchmark, and the 32-output figures are
  estimates, not measurements at all.

## Do not commit

This report is documentation/analysis only. Per task instructions, it has
not been committed to the repository.
