# Multi-Lane Resource-Shared Folded Conv-BN-ReLU: Design Memo

Date: July 10, 2026

**This is a design memo only. No VHDL was written or modified for this
task.** It compares 1-lane, 2-lane, 4-lane, and 8-lane resource-shared
architectures for the complete 32-output first-layer folded Conv-BN-ReLU
stage, using the two real measured data points available (4-kernel and
32-kernel single-lane designs) to build an evidence-based resource-scaling
estimate, and ends with a recommendation for the next implementation.

## 1. Purpose

The single-lane (1-lane) 32-output resource-shared design
(`first_layer_32out_bn_relu_resource_shared_summary.md`, commit
`6d9e9b4a`) proved that DSP demand can be made independent of kernel
count: 2 DSP48E1 slices instead of 740, at the cost of 10,125 cycles of
latency for the 5x5x3 toy input versus ~4-5 cycles for the direct
-parallel design. That latency is real and large -- roughly 2,000x more
clock cycles for the same output. A single shared engine is the most
extreme point on the latency/resource tradeoff curve; the design plan's
own Section 17/18 already flagged "a small number of parallel lanes (e.g.
2, 4, or 8 shared engines instead of 1)" as the natural next tradeoff
point, without committing to a specific lane count. This memo works out
that comparison in detail, using the two real data points now available
(the 4-kernel and 32-kernel single-lane syntheses) rather than a single
extrapolation point, before any multi-lane code is written.

## 2. Current baseline results

| Design | Kernels | LUTs | Registers | DSP48E1 | BRAM | WNS | Power | Latency (5x5x3 toy) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct-parallel raw Conv2d | 32 (complete) | 10,461 | 4,618 | 740/740 (100%) | 0 | +2.343 ns | 1.324 W | ~4-5 cycles |
| Resource-shared folded Conv-BN-ReLU (1 lane) | 4 | 1,500 | 3,180 | 2/740 (0.27%) | 0 | +0.597 ns | 0.148 W | 1,305 cycles (measured) |
| Resource-shared folded Conv-BN-ReLU (1 lane) | 32 (complete) | 1,837 | 5,588 | 2/740 (0.27%) | 0 | +0.662 ns | 0.165 W | 10,125 cycles (measured) |

All three are GHDL-verified (288/288, 36/36, 288/288 respectively) and
synthesized on `xc7a200tsbg484-1` at 100 MHz / 10.000 ns, out-of-context.
One fact from this table is easy to miss and matters for the rest of this
memo: **the 32-kernel single-lane resource-shared design already uses
more registers (5,588) than the direct-parallel design (4,618)**, even
though its LUTs, DSPs, and power are all dramatically lower. Register
count, not LUTs or DSPs, is the resource most at risk of becoming the
multi-lane design's binding constraint -- see Section 5.

## 3. What "lane" means

One **lane** is a complete, independent instance of the existing Level-2
resource-shared engine (structurally identical to
`conv3x3_3chan_32out_bn_relu_time_mux.vhd`, just with `NUM_KERNELS` set
to a fraction of 32), consisting of:

- **One time-multiplexed dot-product engine** (`conv3x3_dot_time_mux`,
  unmodified) -- the single 8x8 signed multiplier, reused across all
  (kernel, channel) steps assigned to this lane.
- **One shared Q.16 BN+ReLU unit** -- the two-stage
  `product_fx = raw_sum * SCALE_FX; biased_fx = product_fx + BIAS_FX;
  relu_fx = max(biased_fx, 0)` sequence, reused once per kernel assigned
  to this lane (not once per output).
- **An assigned, fixed subset of the 32 output kernels** -- for `L` lanes
  dividing 32 evenly, each lane owns `32/L` contiguous kernels (lane 0:
  kernels `0..(32/L)-1`, lane 1: kernels `(32/L)..(64/L)-1`, etc.). Each
  lane's own small FSM loops `step = 0 .. (32/L)*3 - 1` exactly as the
  existing single-lane FSM does, just over its own smaller kernel range
  -- **no FSM logic changes**, only the `NUM_KERNELS`-equivalent constant
  per lane, exactly as already proven true going from 4 to 32 kernels in
  a single lane.
- **Reused across spatial windows**: within a lane, the SAME two shared
  resources (dot-product engine, BN+ReLU unit) are time-multiplexed
  across both kernels *and* windows -- one lane still processes windows
  sequentially (phase 2, non-overlapped), exactly as the existing
  1-lane design does. Multi-lane parallelism is across **kernels within
  the same window**, not across windows.

With `L` lanes, all lanes read the **same** currently-presented window
(from the shared phase-1 capture buffer, which is not duplicated -- it is
simply read by `L` lane instances instead of 1) and run **in parallel**
on that window, each producing its own `32/L`-kernel slice of the
window's output tuple. The top-level wrapper must wait for **all** `L`
lanes to finish before advancing to the next window (a synchronization
barrier not present in the 1-lane design; see Section 4).

## 4. Architecture options

| | 1 lane (existing) | 2 lanes | 4 lanes | 8 lanes |
|---|---|---|---|---|
| Kernels per lane | 32 | 16 | 8 | 4 |
| New Level-2 engine needed? | No (built) | New (widen from 4-kernel pattern) | New (matches existing raw 8-kernel time-mux's kernel count, but needs BN+ReLU added) | **None -- can literally reuse `conv3x3_3chan_4out_bn_relu_time_mux.vhd` as-is, once per lane, with a different weight-package instance per lane** |
| Expected DSP count (Section 5) | 2 (measured) | ~4 | ~8 | ~16 |
| Expected latency reduction | 1x (baseline, measured 10,125 cycles) | ~2x (~5,065 cycles, estimate) | ~4x (~2,545 cycles, estimate) | ~8x (~1,285 cycles, estimate -- notably, this is almost exactly the ALREADY-MEASURED 1,305-cycle figure for the existing 4-kernel single-lane design, since an 8-lane/4-kernels-per-lane design's per-lane workload is identical in shape to that already-built-and-verified design) |
| Expected LUT count (Section 5 model) | 1,837 (measured) | ~3,288 | ~6,192 | ~12,000 (**exceeds the direct-parallel design's 10,461**) |
| Expected register count (Section 5 model) | 5,588 (measured) | ~8,424 | ~14,096 | ~25,440 (**~5.5x the direct-parallel design's 4,618**) |
| Control complexity | Existing, simple (one FSM, `S_IDLE/S_RUN/S_BN1/S_BN2`) | Add: `L` engine instances + an all-lanes-done barrier before emitting `valid_out` | Same barrier pattern, more instances to wire (mechanical, not conceptually harder) | Same barrier pattern; largest instance count, most port-mapping/wiring, still conceptually the same barrier |
| Output ordering changes? | N/A (baseline) | **No, if the wrapper waits for all lanes and emits the combined 32-kernel tuple per window** (recommended) | Same | Same |
| Existing golden vectors reusable? | Yes (baseline) | **Yes, unchanged** -- `first_layer_32out_bn_relu_resource_shared_pkg.vhd`'s existing `EXPECTED_RELU_FX` (288 entries, window-major/kernel-minor) remains the correct reference as long as output ordering is preserved per lane's kernel slice; no new Python generation needed, only slicing the SAME existing per-kernel weight/SCALE_FX/BIAS_FX arrays across lanes | Same | Same |

**Why output ordering does not need to change**: each lane owns a fixed,
known, contiguous kernel range. If the top-level wrapper collects all `L`
lanes' outputs into the correct global kernel positions (lane 0's outputs
into `y0..y(32/L - 1)`, lane 1's into `y(32/L)..y(64/L - 1)`, etc.) and
only asserts `valid_out` once every lane has signaled done for the
current window, the emitted 32-value tuple per window is bit-identical in
order to the existing 1-lane design's output -- the SAME `EXPECTED_RELU_FX`
array already generated and verified can be reused with zero
regeneration. This is a deliberate design constraint recommended for the
next implementation specifically to avoid touching Python golden
generation at all.

**A more aggressive alternative** (not recommended as the first
multi-lane step) would let each lane emit its own `valid_out` pulse
independently, without a synchronization barrier -- this would reduce
control complexity slightly and could reduce the barrier's tail latency
(the whole window waits for the *slowest* lane, which for an even kernel
split should be all lanes simultaneously, so this saving is expected to
be small), but it changes the output granularity and would require either
a new, more complex golden-vector comparison order in the testbench or a
buffering scheme to reassemble the full 32-tuple downstream. Given the
existing golden vectors and testbench infrastructure are a valuable,
already-verified asset, the barrier-synchronized, unchanged-output-order
approach is the clearly preferable starting point.

## 5. Estimated resource scaling table

**Method**: the only two real 1-lane data points available (4 kernels and
32 kernels, same architecture, same target) are used to fit a simple
`fixed_per_lane_cost + per_kernel_cost * kernels_in_lane` model for LUTs
and registers, then multiplied by lane count `L` (holding total kernels
at 32, so `kernels_in_lane = 32/L`). This is a **linear extrapolation
from 2 points**, not a synthesis result -- Section 9 and the Limitations
section restate this explicitly.

Fitting from the two measured single-lane points:

```
LUT(K)  = LUT_fixed + lut_per_kernel * K
Reg(K)  = Reg_fixed + reg_per_kernel * K

From K=4:  1,500 = LUT_fixed + lut_per_kernel * 4
From K=32: 1,837 = LUT_fixed + lut_per_kernel * 32
  => lut_per_kernel ~= 12.0,  LUT_fixed ~= 1,452

From K=4:  3,180 = Reg_fixed + reg_per_kernel * 4
From K=32: 5,588 = Reg_fixed + reg_per_kernel * 32
  => reg_per_kernel ~= 86.0,  Reg_fixed ~= 2,836
```

For `L` lanes covering all 32 kernels (`K = 32/L` kernels/lane):

```
Total_LUT(L) ~= L * (1,452 + 12 * (32/L))  = 1,452*L + 384
Total_Reg(L) ~= L * (2,836 + 86 * (32/L))  = 2,836*L + 2,752
Total_DSP(L) ~= 2 * L    (measured DSP count per lane is a near-constant
                          2, independent of kernels/lane, at both tested
                          points -- this is the most confident estimate here)
```

| Lanes | Kernels/lane | Est. LUTs | Est. Registers | Est. DSP48E1 | Est. latency (cycles, 5x5x3 toy) |
|---:|---:|---:|---:|---:|---:|
| 1 (measured) | 32 | 1,837 | 5,588 | 2 | 10,125 (measured) |
| 2 | 16 | ~3,288 | ~8,424 | ~4 | ~5,065 |
| 4 | 8 | ~6,192 | ~14,096 | ~8 | ~2,545 |
| 8 | 4 | **~12,000** | **~25,440** | ~16 | ~1,285 |

**The DSP estimate is the most trustworthy figure in this table** -- it
is a simple, near-constant per-lane cost measured identically at two very
different kernel-per-lane counts (2 DSPs at K=4 and 2 DSPs at K=32),
giving high confidence that `~2*L` holds for intermediate lane counts
too, and it directly confirms the task's own expectation (~8 DSPs at 4
lanes).

**The LUT and, especially, register estimates carry a real caution the
task explicitly asked not to gloss over**: because each lane pays a
large, roughly kernel-count-*independent* fixed overhead (~1,452 LUTs and
~2,836 registers per lane, from the FSM, pipeline registers, and per
-lane output-holding array), resource cost grows **close to linearly with
lane count**, not favorably with the smaller per-lane kernel count. At
`L=8`, the estimated LUT count (~12,000) would **exceed** the
direct-parallel design's 10,461 LUTs, and the estimated register count
(~25,440) would be roughly **5.5x** the direct-parallel design's 4,618 --
even though DSP usage would still be dramatically lower (~16 vs. 740).
This is a genuinely different conclusion than a naive "more lanes only
helps" intuition, and it directly informs the recommendation in Section
9.

## 6. Latency estimates

Using the design plan's own scaling formula
(`resource_shared_first_layer_conv_bn_relu_plan.md` Section 13),
generalized to `L` lanes by dividing the per-window compute-cycle term by
`L` (capture-phase cycles are unaffected by lane count, since window
capture is not paralleled by lanes in this architecture):

```
cycles_per_window(N, L) ~= 35 * N / L    (N = 32 total kernels)
total_cycles(S, N, L)   ~= S^2 + (S - 2)^2 * 35 * N / L
```

**5x5x3 toy input (S=5, N=32)** -- restating Section 5's latency column:

| Lanes | Estimated total cycles | Estimated time @ 100 MHz |
|---:|---:|---:|
| 1 (measured) | 10,125 | 101.25 us |
| 2 | ~5,065 | ~50.65 us |
| 4 | ~2,545 | ~25.45 us |
| 8 | ~1,285 | ~12.85 us |

**256x256 tile (S=256, N=32, not attempted, not measured)** -- extending
the same formula to a realistic UAVSAR tile size, purely as an order-of
-magnitude estimate:

| Lanes | Estimated total cycles | Estimated time @ 100 MHz |
|---:|---:|---:|
| 1 | ~7.2e7 (design plan's own estimate) | ~0.72 s |
| 2 | ~3.6e7 | ~0.36 s |
| 4 | ~1.8e7 | ~0.18 s |
| 8 | ~9.1e6 | ~0.09 s |

**These are estimates only**, extrapolated from a formula itself derived
from small-scale measurements (8-kernel time-mux design,
`time_mux_8out_summary.md`) and not yet confirmed at any lane count above
1 or any tile size above 5x5. They should not be read as timing claims
until a multi-lane design is actually built and its cycle count measured
directly by a testbench, exactly as done for the 1-lane 4-kernel and
32-kernel designs.

## 7. Verification plan

For a future multi-lane implementation:

1. **Reuse Python Q.16 golden outputs.** `first_layer_32out_bn_relu_resource_shared_pkg.vhd`'s
   existing `EXPECTED_RELU_FX` array (288 entries, generated by
   `scripts/generate_resource_shared_32out_bn_relu_vectors.py
   --num-kernels 32`, unchanged) remains the correct golden reference **if
   and only if** output ordering is preserved (Section 4). No new Python
   script or regeneration is expected to be necessary.
2. **Preserve deterministic output order.** If a future implementation
   changes to per-lane independent `valid_out` pulses (the "more
   aggressive alternative" flagged in Section 4) instead of the
   recommended all-lanes-barrier approach, this must be documented
   explicitly and the testbench's comparison order updated accordingly --
   this memo recommends avoiding that path specifically to keep this
   verification step trivial.
3. **GHDL must check all 288 outputs**, exactly as the 1-lane 32-kernel
   testbench (`tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd`)
   already does -- a multi-lane testbench should follow the same
   structure (stream the toy image, wait for 9 window pulses, check all
   32 kernel values per pulse against `EXPECTED_RELU_FX`).
4. **Regressions to rerun, unmodified:**
   - `run_ghdl_kernel0_bn_relu_pipelined.sh`
   - `run_ghdl_kernel2_bn_relu_pipelined.sh`
   - `run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh`
   - `run_ghdl_32out_dsp.sh` (direct-parallel 32-output DSP-aware)
   - `run_ghdl_32out_bn_relu_resource_shared.sh` (single-lane 32-output
     resource-shared -- the new baseline this memo is built on)

## 8. Synthesis plan

Target `xc7a200tsbg484-1`, 100 MHz / 10.000 ns, out-of-context
`synth_design`, matching every existing design in this repo (see
`synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_200t.tcl` as
the direct template -- a multi-lane synthesis script would only need
additional `read_vhdl` lines for the extra lane-engine instances/packages
and possibly a top-level lane-synchronization wrapper file). Collect, as
for every prior design:

- LUT count and utilization
- Register (FF) count and utilization
- DSP48E1 count and utilization
- Block RAM tile count and utilization (**specifically watch for a shift
  from LUT-ROM to BRAM** for the weight storage once multiple
  smaller-per-lane weight tables exist side by side -- not expected to
  change behavior at these table sizes based on the 1-lane result already
  showing 0 BRAM at a 32-entry table, but worth confirming per lane
  count)
- WNS (setup slack)
- WHS (hold slack)
- Total on-chip power estimate
- Dynamic power estimate
- Static power estimate
- Warnings/errors (watch specifically for any *new* warning class beyond
  the benign "unused sequential element" and `FSM_onehot_state_reg`
  notes already seen at 1 lane -- a genuinely new warning type would be
  the first sign the barrier-synchronization logic introduced an
  unexpected issue)

## 9. Recommended next implementation

**Recommendation: build the 2-lane design next, not 4-lane.**

The task's own working expectation was 4-lane, reasoning that it "cuts
latency by about 4x while still likely using only about 8 DSPs." The DSP
part of that reasoning is well-supported by Section 5's analysis (~8 DSPs
at 4 lanes is a solid estimate). But evaluating the *full* resource
picture, not just DSPs, changes the recommendation:

- **2 lanes already delivers a meaningful ~2x latency cut** (10,125 ->
  ~5,065 cycles) while keeping the LUT estimate (~3,288) comfortably
  below the direct-parallel design's 10,461, and the register estimate
  (~8,424) at a more modest ~1.8x the direct-parallel design's 4,618 --
  a growth rate similar in character to what was already observed and
  accepted going from 4 to 32 kernels in a single lane.
- **4 lanes' register estimate (~14,096, ~3x direct-parallel) is a
  larger jump** than the DSP savings narrative alone suggests, and would
  be the point where "resource-shared" starts requiring a real caveat
  about *which* resource is being saved (DSPs, clearly; LUTs, still yes;
  registers, no longer by as comfortable a margin).
- **8 lanes' estimated LUT and register counts equal or exceed the
  direct-parallel design's**, which would undercut the core motivating
  claim of this whole architecture direction for at least two of the
  three resource types being compared, even though DSP usage would still
  be far lower. Building 8 lanes first, without first confirming the
  2-lane step's actual (not estimated) resource growth, risks a much
  more expensive detour if the linear extrapolation underestimates real
  synthesis overhead (e.g. routing/control-set costs that do not appear
  in a simple two-point linear fit).

**2 lanes is also the lowest-risk next build**: it requires exactly one
new thing (a 16-kernel Level-2 engine, following the identical
zero-FSM-change widening pattern already proven going from 4 to 32
kernels) plus the smallest possible version of the new
all-lanes-done barrier logic (waiting for 2 signals, not 4 or 8). It is
the smallest step that actually tests the previously-unverified part of
this architecture -- parallel lanes and the synchronization barrier --
without simultaneously making the biggest, least-validated resource-model
extrapolation (8 lanes) the very first thing built.

If, after building and measuring the 2-lane design, its actual LUT/
register growth tracks close to this memo's linear model (rather than
being worse), 4-lane becomes a well-justified follow-on step with
real data backing it, and the task's original 4-lane instinct would
likely be vindicated as the better "middle ground" once one data point
past 1-lane exists to calibrate the model. This memo does not conclude
4-lane is a bad idea -- only that 2-lane is the more defensible *next*
step given what is measured today.

## 10. Safe claims

**Safe to make now (already measured or directly follows from measured
data):**

- "The 1-lane, complete 32-output resource-shared design uses 2 DSP48E1
  slices versus the direct-parallel design's 740, at the cost of roughly
  2,000x more clock cycles for the same 5x5x3 toy input."
- "A multi-lane design is expected to reduce latency roughly in
  proportion to lane count, based on the existing single-lane
  architecture's per-window cycle formula."
- "DSP count is expected to scale linearly with lane count (~2 DSPs per
  lane), based on identical per-lane DSP usage measured at two very
  different kernel-per-lane counts (4 and 32)."
- "LUT and register growth with lane count is expected to be dominated
  by a large, roughly kernel-count-independent fixed cost per lane, not
  by the (smaller) per-kernel cost -- meaning higher lane counts do not
  automatically stay resource-cheap."
- "At an 8-lane estimate, LUT and register usage could rival or exceed
  the direct-parallel design's usage, based on a linear extrapolation
  from two measured single-lane data points."

**Safe only after a future multi-lane implementation is built, verified,
and synthesized:**

- Any specific LUT, register, DSP, power, or WNS figure for a 2-, 4-, or
  8-lane design (all such figures in this memo are estimates, explicitly
  labeled as such).
- Any claim that a specific lane count "is" the best tradeoff point --
  only that it is the recommended *next experiment* given current
  evidence.
- Any claim about actual measured latency for a 256x256 (or any
  non-5x5x3) tile at any lane count.
- Any claim that the barrier-synchronization control logic adds
  negligible overhead -- this is expected but not yet measured.
- Any claim about BRAM usage at higher lane counts.

## 11. Limitations

- **First-layer convolution only.** This memo, and every design it
  discusses, covers the first Conv2d layer's folded Conv-BN-ReLU stage
  only -- not the full U-Net, not any downstream layer.
- **Toy input only for all currently verified hardware.** The canonical
  5x5x3 patch is the only input any built-and-measured design in this
  repo has been run against; the 256x256 figures in Section 6 are
  formula-based estimates only.
- **No full U-Net.** Nothing in this memo or the designs it references
  claims the complete model fits on this or any target.
- **No board testing.** All figures (measured and estimated) are Vivado
  `synth_design` / GHDL simulation results, never measurements from
  physical hardware.
- **No measured speedup.** No throughput or latency comparison in this
  memo constitutes a measured speedup claim; all multi-lane figures are
  estimates pending implementation.
- **No measured board power.** All power figures referenced (1-lane
  baselines) are Vivado vector-less power estimates ("Medium confidence"
  per Vivado's own report), not measurements; no power figures exist yet
  for any multi-lane design.
- **Estimates in this memo are not measurements until implemented and
  synthesized.** The Section 5 resource-scaling model is a linear fit
  through exactly two data points (4 and 32 kernels, both at 1 lane) and
  has not been validated against any actual multi-lane synthesis result.
  Real synthesis behavior (place-and-route optimization, shared logic
  across near-identical lane instances, routing congestion at higher
  instance counts) could make actual multi-lane resource usage
  meaningfully better or worse than this linear model predicts in either
  direction.

## 12. Final recommendation

**Build the 2-lane resource-shared folded Conv-BN-ReLU design next.**
Reuse the existing, unmodified `conv3x3_dot_time_mux` engine and the
existing `first_layer_32out_bn_relu_resource_shared_pkg.vhd`'s per-kernel
weight/SCALE_FX/BIAS_FX data (sliced into two 16-kernel halves, no new
Python generation needed); create a new 16-kernel Level-2 engine by
widening the existing pattern exactly as already proven from 4 to 32
kernels (zero FSM logic changes expected); wrap two instances of it in a
new streaming wrapper that waits for both lanes' `done` signals before
combining their outputs into the existing 32-value, window-major/
kernel-minor order and asserting `valid_out`; verify against the
existing, unmodified `EXPECTED_RELU_FX` golden array (288 checks); rerun
all five regressions listed in Section 7; and synthesize on
`xc7a200tsbg484-1` at 100 MHz to replace this memo's 2-lane estimates
with measured figures. Use that result to decide whether 4-lane is worth
building next, rather than committing directly to 4- or 8-lane based on
estimates alone.
