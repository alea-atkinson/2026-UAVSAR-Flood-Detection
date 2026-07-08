# Larger FPGA Target Comparison for First-Layer Prototypes

Date: July 8, 2026

This is a **synthesis-target comparison only**. No new convolution
architecture was created for this comparison -- every design synthesized
here is an existing, already GHDL-verified VHDL prototype from
`hardware/vhdl_conv3x3/`. No board testing, no board deployment, no
measured speedup, and no measured board power are claimed anywhere in this
document.

## 1. Purpose

The Artix-7 35T (`xc7a35tcpg236-1`, the Digilent Cmod A7-35T target) has
been the only synthesis target used throughout this FPGA feasibility study
so far. It has been useful as a constrained baseline, but treating it as a
permanent ceiling would conflate two different questions:

1. Is the **architecture** (fully parallel, DSP-aware, or time-multiplexed
   first-layer convolution) fundamentally resource-hungry?
2. Or is the **35T device itself** simply too small to show these
   architectures' real headroom?

This document re-synthesizes the same 7 existing, verified first-layer
prototypes against a larger Xilinx target and compares utilization,
timing, and power estimates side by side with the existing Artix-7 35T
results, to help separate those two questions.

## 2. Why Artix-7 35T remains useful as a constrained baseline

The 35T is not being discarded -- it remains the **relevant constrained
baseline** for this project because:

- It corresponds to a real, inexpensive, physically available class of
  device (Digilent Cmod A7-35T) that represents a realistic lower bound
  for an embedded/edge UAVSAR deployment scenario.
- Its tight resource budget (20,800 LUTs, 41,600 registers, 90 DSPs, 50
  BRAM tiles) is exactly what exposed the scaling problems this whole
  study has been investigating: the 8-output parallel design already used
  45.07% of its LUTs, and the DSP-aware 8-output design already
  overutilized its entire DSP budget (see
  `hardware/vhdl_conv3x3/vivado_8out_dsp_synthesis_summary.md` and
  `hardware/vhdl_conv3x3/first_layer_32out_scaling_estimate.md`).
- Comparing against it is precisely what makes the larger-target results
  in this document meaningful: without the constrained baseline, "this fits
  comfortably on a bigger chip" would not distinguish "the architecture is
  efficient" from "we picked an easy device."

## 3. Why the larger target was chosen

Vivado 2025.2 in this environment does not have any **Kintex-7** parts
available (`get_parts -filter {FAMILY == kintex7}` returned an empty list
-- consistent with a WebPACK-limited installation that does not include
Kintex-7 device support). Per the task's stated preference order (a
7-series target first, to keep the comparison closest to the existing
Artix-7 flow), the next best option is a **larger Artix-7** part, which
**is** available.

**Chosen target: `xc7a200tsbg484-1`** -- the largest standard Artix-7
device (XC7A200T), with the same **-1 speed grade** and same Artix-7 family
as the existing baseline (`xc7a35tcpg236-1`). This keeps the synthesis flow
close to the existing baseline while substantially increasing available
resources. This specific part number also corresponds to a real,
well-known development board (Digilent Nexys Video), though **this
document makes no claim of owning, testing, or deploying to that or any
other physical board** -- it is used purely as a valid, available Vivado
synthesis target.

Device resource totals (read directly from this run's own Vivado
utilization reports, not from external datasheets):

| Resource | xc7a35tcpg236-1 (baseline) | xc7a200tsbg484-1 (larger) | Ratio |
|---|---:|---:|---:|
| LUTs | 20,800 | 134,600 | 6.47x |
| Registers | 41,600 | 269,200 | 6.47x |
| DSP48E1 slices | 90 | 740 | 8.22x |
| Block RAM tiles | 50 | 365 | 7.30x |

## 4. Designs compared

All 7 existing, GHDL-verified first-layer prototypes, unmodified:

1. `stream_conv3x3_3chan_cell` -- 1-output parallel
2. `stream_conv3x3_3chan_4out_cell` -- 4-output parallel
3. `stream_conv3x3_3chan_8out_cell` -- 8-output parallel
4. `stream_conv3x3_3chan_8out_cell_dsp` -- DSP-aware 8-output parallel
5. `conv3x3_dot_time_mux` -- time-multiplexed single dot-product prototype
6. `conv3x3_3chan_8out_time_mux` -- time-multiplexed 8-output one-window scheduler
7. `stream_conv3x3_3chan_8out_time_mux` -- streaming time-multiplexed 8-output prototype

Each was synthesized via a new **target-parameterized** TCL script,
`hardware/vhdl_conv3x3/synth_target_parameterized.tcl`, which takes the
target part, top module name, report directory, and VHDL source file list
as command-line arguments and performs the exact same `synth_design`
options (100 MHz / 10.000 ns clock, out-of-context mode) and report types
(`report_utilization`, `report_timing_summary`, `report_power`,
`write_checkpoint`) as every original per-design `synth_*.tcl` script. The
same VHDL source files already used by each design's original Artix-7
script were reused unchanged -- no new RTL was written.

**All 7 designs synthesized successfully on the larger target with 0
errors.**

## 5. Table: Artix-7 35T vs. Artix-7 200T results

Full detail (including exact percentages) is in
`hardware/vhdl_conv3x3/larger_fpga_target_comparison.csv`. Summary:

| Design | Target | LUTs (util%) | Regs (util%) | DSPs (util%) | BRAM | WNS (ns) | WHS (ns) | Total power (W) | Warnings |
|---|---|---|---|---|---|---:|---:|---:|---|
| 1-output parallel | 35T | 2731 (13.13%) | 1324 (3.18%) | 0 (0%) | 0 | +4.456 | +0.262 | 0.120 | 0 |
| 1-output parallel | 200T | 2731 (2.03%) | 1324 (0.49%) | 0 (0%) | 0 | +4.456 | +0.262 | 0.175 | 0 |
| 4-output parallel | 35T | 5020 (24.13%) | 3214 (7.73%) | 0 (0%) | 0 | +5.117 | +0.262 | 0.177 | 0 |
| 4-output parallel | 200T | 5020 (3.73%) | 3214 (1.19%) | 0 (0%) | 0 | +5.117 | +0.262 | 0.232 | 0 |
| 8-output parallel | 35T | 9374 (45.07%) | 5704 (13.71%) | 0 (0%) | 0 | +5.105 | +0.262 | 0.271 | 0 |
| 8-output parallel | 200T | 9374 (6.96%) | 5704 (2.12%) | 0 (0%) | 0 | +5.105 | +0.262 | 0.332 | 0 |
| DSP-aware 8-output | 35T | 6991 (33.61%) | 4405 (10.59%) | **90 (100%)** | 0 | +2.371 | +0.262 | 0.346 | **1 (DSP overutilized)** |
| DSP-aware 8-output | 200T | **668 (0.50%)** | **410 (0.15%)** | 203 (27.43%) | 0 | +3.848 | +0.262 | 0.373 | **0** |
| Time-mux dot product | 35T | 205 (0.99%) | 214 (0.51%) | 0 (0%) | 0 | +0.656 | +0.290 | 0.074 | 0 |
| Time-mux dot product | 200T | 205 (0.15%) | 214 (0.08%) | 0 (0%) | 0 | +0.656 | +0.290 | 0.128 | 0 |
| Time-mux 8-out scheduler | 35T | 376 (1.81%) | 510 (1.23%) | 0 (0%) | 0 | +0.655 | +0.275 | 0.076 | 0 |
| Time-mux 8-out scheduler | 200T | 376 (0.28%) | 510 (0.19%) | 0 (0%) | 0 | +0.655 | +0.275 | 0.129 | 0 |
| Streaming time-mux 8-out | 35T | 1409 (6.77%) | 3328 (8.00%) | 0 (0%) | 0 | +0.600 | +0.262 | 0.086 | 1 (benign FSM note) |
| Streaming time-mux 8-out | 200T | 1409 (1.05%) | 3328 (1.24%) | 0 (0%) | 0 | +0.600 | +0.262 | 0.139 | 1 (benign FSM note) |

**Key observation used throughout the interpretation below**: for every
design except the DSP-aware one, the **absolute LUT and register counts
are identical** between the two targets (only the utilization *percentage*
changes, because the denominator -- total device resources -- is larger).
This is expected for same-speed-grade, same-family (Artix-7) synthesis of
identical VHDL with identical primitives (LUT6, FDRE, CARRY4, etc. are the
same building blocks across the whole 7-series Artix-7 line) -- it confirms
these designs' *resource need* is fixed by the architecture and clock
target, not by the part.

## 6. Interpretation by architecture family

### A. Parallel (1/4/8-output)

Absolute LUT/register counts are **unchanged** between targets (e.g. the
8-output design uses exactly 9374 LUTs on both parts). Utilization drops
purely because the 200T has 6.47x more LUTs and registers available:
45.07% -> 6.96% for the 8-output design. Setup and hold slack are
**identical to three decimal places** across targets, confirming the
timing-critical paths (inside the shared `window3x3_stream` structure, per
earlier synthesis summaries) are unaffected by device size at the same
speed grade. **This demonstrates that the parallel designs' LUT pressure
observed at 8 outputs was a small-device utilization problem, not an
inherent per-output-channel LUT cost that grows on a larger device** -- the
same fixed LUT cost simply represents a much smaller fraction of a bigger
device.

### B. DSP-aware

This is the family where the larger target changes the outcome most
clearly. On the 35T, the DSP-aware 8-output design **overutilized the
entire 90-DSP budget** (requesting more multiplies than available, with
Vivado's own warning: "Resources of type DSP have been overutilized. Used
= 266, Available = 90") and had to fall back a large share of multiplies to
LUT fabric, consuming 6991 LUTs. On the 200T (740 DSP48E1 slices
available), the **same VHDL** synthesizes with **0 warnings**, using only
203 DSPs (27.43% of the part) and just **668 LUTs (0.50%)** -- a roughly
10x reduction in LUT usage, because essentially all multiplies now map
cleanly to dedicated DSP48E1 hardware instead of falling back to LUT-based
multipliers. Setup slack also improved (+2.371 ns on 35T -> +3.848 ns on
200T), consistent with a cleaner, non-resource-constrained DSP mapping.
**This is direct evidence that the DSP-aware design's LUT/timing pressure
on the 35T was caused by the small device's DSP shortage, not by the
DSP-aware architecture itself being inefficient.**

### C. Time-multiplexed (single dot product and 8-output one-window scheduler)

Both time-mux designs already used well under 2% of the 35T's LUTs and
registers, so there was essentially no small-device pressure to relieve.
Absolute LUT/register counts and timing slack are unchanged on the 200T,
as expected. These designs' value proposition (dramatically lower resource
use than the parallel designs, at the cost of much higher latency, as
already documented in `hardware/vhdl_conv3x3/time_mux_dot_summary.md` and
`time_mux_8out_summary.md`) is unaffected by target size in either
direction -- they were never device-constrained on the 35T, and remain
proportionally tiny on the 200T.

### D. Streaming time-multiplexed 8-output

Same pattern as the other time-mux designs: unchanged absolute resource
counts and timing (1409 LUTs / 3328 registers / +0.600 ns WNS on both
targets), low utilization on both parts (6.77% LUTs on 35T, 1.05% on
200T). The one non-critical warning (`FSM_onehot_state_reg[2]` unused, a
netlist-optimization note about the capture/process FSM's one-hot
encoding) reproduces identically on both targets, confirming it is a
property of the design's own state machine, not a target-specific issue.

## 7. Does the larger target change the 32-output scaling conclusion?

**Partially, and only for one of the three architecture families discussed
in `first_layer_32out_scaling_estimate.md`:**

- **Fully parallel scaling**: the earlier estimate concluded that fully
  parallel 32-output scaling would not fit on the 35T by LUT count
  (~171% of 20,800 LUTs under the incremental-trend estimate). On the
  200T (134,600 LUTs), that same incremental estimate (~35,498 LUTs)
  would only use about **26.4%** of the part -- comfortably fitting by LUT
  count. **The larger target changes this specific conclusion**: fully
  parallel 32-output scaling looks LUT-limited on the 35T specifically,
  not on 7-series Artix-7 devices in general.
- **Fully parallel DSP-aware scaling**: the earlier estimate found 32
  outputs would need roughly 864 multiplies, "impossible" against the
  35T's 90 DSPs. Against the 200T's 740 DSPs, 864 multiplies would still
  **exceed** the available DSP count (864 > 740), though far less
  dramatically than on the 35T (a ~1.17x shortfall vs. a ~9.6x shortfall).
  **This does not fully overturn the earlier conclusion** -- full DSP-only
  mapping for all 32 channels still would not fit on this specific larger
  part either, though it is much closer, and a mixed DSP+LUT mapping (as
  actually observed for 8 outputs on the 35T) would very plausibly fit.
  This has **not been synthesized or confirmed** for 32 outputs on either
  target -- it is an extrapolation, exactly as in the original scaling
  estimate.
- **Time-multiplexed scaling**: the earlier estimate already concluded
  single-lane time-multiplexed 32-output scaling would likely fit
  comfortably by resource count on the 35T (just slower). The 200T results
  reinforce that conclusion further -- there is even more headroom -- but
  do not change it qualitatively.

**None of this has been synthesized at 32 outputs on either target.** This
section describes how the existing 32-output *estimate* would change if
re-expressed against the 200T's resource budget, not a new measurement.

## 8. Limitations

- **Synthesis-target comparison only.** No physical board testing was
  performed for either target.
- **No measured board power.** All power figures (both targets) are
  Vivado's vector-less, "Medium confidence" static estimates from
  `report_power`, not measurements from a running device.
- **No measured speedup.** All designs share the same clock target (100
  MHz) and the same cycle-accurate behavior (already GHDL-verified);
  nothing about throughput or latency was measured or changed by this
  comparison.
- **Comparison is limited to the existing first-layer hardware
  prototypes.** No other layer, block, or model stage was synthesized.
- **No all-32-channel implementation was created or synthesized** for
  either target -- Section 7 above only re-expresses the existing
  32-output *estimate* against the larger device's resource budget.
- **No new RTL architecture was created for this comparison.** Every
  design synthesized here is an existing, unmodified, already
  GHDL-verified VHDL module.
- **Vivado power is a tool estimate, not measured power**, for both
  targets.
- **Power estimates across different FPGA families/packages should not be
  overinterpreted as direct apples-to-apples measurements.** The 200T's
  device static power (~0.122-0.123 W across all designs) is consistently
  higher than the 35T's (~0.068-0.069 W) purely because it is a larger
  die with more leakage current, essentially independent of which design
  is synthesized onto it -- this is a property of the silicon, not of the
  architecture. Total on-chip power on the 200T is higher for every design
  in this comparison, which is an artifact of device static power scaling
  with die size, not evidence that any design is less power-efficient on
  the larger part. **Utilization, timing headroom, and DSP-vs-LUT mapping
  behavior are the meaningful comparisons here; absolute power should be
  read with this caveat in mind.**

## Safe claims made in this document

- "Artix-7 35T is a constrained baseline."
- "The larger FPGA target reduces utilization pressure for the existing
  designs" (directly observed: identical absolute LUT/register counts,
  much lower utilization percentage).
- "This helps distinguish small-device limits from architecture scaling
  limits" (most clearly demonstrated by the DSP-aware design: the DSP
  overutilization warning and its associated LUT-fallback cost disappear
  entirely on the larger target, using the identical VHDL).
- "Results are Vivado synthesis/timing/power estimates only."
- "Architecture tradeoffs remain visible across targets" (parallel designs
  still use far more resources than time-multiplexed designs on both
  targets; the DSP-aware vs. LUT-only tradeoff is visible on both targets,
  just not DSP-constrained on the larger one).

This document does not claim board testing, real-time flood mapping,
that larger downstream model stages fit, measured power, or deployment
readiness.

## 9. Recommended next step

Given that DSP-aware mapping now has ample DSP headroom on the 200T
(203/740 used for 8 outputs), the most informative next experiment would
be to **synthesize a DSP-aware design for a higher output-channel count
(e.g. 16, or a full 32 if a design is built)** specifically on the 200T
target, to get a real synthesized data point (not just an estimate) for
whether full first-layer DSP-mapped parallelism becomes achievable once
the small-device DSP ceiling is removed -- directly following up on the
partial conclusion in Section 7 that 32-channel DSP demand (864) still
slightly exceeds the 200T's 740 DSPs, but a mixed DSP+LUT mapping (as
already observed to work automatically for 8 outputs on the 35T) would
very plausibly close that gap.
