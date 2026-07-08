# 24-Output DSP-Aware Artix-7 200T: DSP/LUT Fallback Analysis

Date: July 8, 2026

This is an **analysis-only** task: it inspects the already-synthesized
24-output DSP-aware checkpoint (and the existing 8- and 16-output
checkpoints) to understand why the 24-output design triggered a DSP
overutilization warning on the Artix-7 200T. No RTL was modified, no new
architecture was created, no 32-output design was built, and nothing was
committed.

## 1. Purpose

The 24-output DSP-aware first-layer design synthesized on the Artix-7
200T and met timing, but Vivado reported a DSP overutilization warning
during `synth_design`:

```
WARNING: [Synth 8-3323] Resources of type DSP have been overutilized. Used = 788, Available = 740.
```

with a final mapped result of DSP48E1 731/740, LUTs 3374, WNS +2.760 ns,
1 warning (`hardware/vhdl_conv3x3/first_layer_24out_dsp_200t_summary.md`).
This document opens the existing synthesized checkpoint
(`stream_conv3x3_3chan_24out_cell_dsp_synth.dcp`) to determine, as
concretely as the available reports allow, what got implemented in LUT
fabric instead of DSP48E1, and whether that fallback can be localized to
specific kernels or instances.

## 2. Files/reports inspected

Pre-existing (not modified):

- `hardware/vhdl_conv3x3/first_layer_24out_dsp_200t_summary.md`
- `hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t/stream_conv3x3_3chan_24out_cell_dsp_{utilization,timing_summary,power}.txt`
- `hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t/stream_conv3x3_3chan_24out_cell_dsp_synth.dcp`
- `hardware/vhdl_conv3x3/vivado_reports_16out_dsp_200t/stream_conv3x3_3chan_16out_cell_dsp_{utilization,synth.dcp}`
- `hardware/vhdl_conv3x3/vivado_reports_larger_target/stream_conv3x3_3chan_8out_cell_dsp_{utilization,synth.dcp}`
  (this is where the 8-output-on-200T results were saved, from the
  earlier larger-FPGA-target comparison task)

Newly generated (this task), under
`hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t_fallback_analysis/`,
one set per design (`stream_conv3x3_3chan_{8,16,24}out_cell_dsp_*`):

- `*_hierarchical_utilization.txt` (`report_utilization -hierarchical`)
- `*_hierarchical_utilization_depth4.txt` (`report_utilization -hierarchical -hierarchical_depth 4`)
- `*_dsp_utilization.txt` (attempted `report_dsp_utilization`; see Section 3)
- `*_timing_summary.txt` (re-derived from the reopened checkpoint, for consistency)
- `*_dsp48e1_cell_paths.txt` (every DSP48E1 cell's hierarchical instance path)
- `*_carry4_cell_paths.txt` (every CARRY4 cell's hierarchical instance path)
- `*_primitive_counts.txt` (whole-netlist cell counts by `REF_NAME`, via `get_cells -hierarchical`)

## 3. Commands run

A new, analysis-only Tcl script,
`hardware/vhdl_conv3x3/analyze_dsp_lut_fallback.tcl`, was written to open
an existing checkpoint and generate the reports above without re-running
synthesis or writing a new checkpoint. It was run once per design:

```
vivado -mode batch -source hardware/vhdl_conv3x3/analyze_dsp_lut_fallback.tcl \
  -tclargs hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t/stream_conv3x3_3chan_24out_cell_dsp_synth.dcp \
           hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t_fallback_analysis \
           stream_conv3x3_3chan_24out_cell_dsp

vivado -mode batch -source hardware/vhdl_conv3x3/analyze_dsp_lut_fallback.tcl \
  -tclargs hardware/vhdl_conv3x3/vivado_reports_16out_dsp_200t/stream_conv3x3_3chan_16out_cell_dsp_synth.dcp \
           hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t_fallback_analysis \
           stream_conv3x3_3chan_16out_cell_dsp

vivado -mode batch -source hardware/vhdl_conv3x3/analyze_dsp_lut_fallback.tcl \
  -tclargs hardware/vhdl_conv3x3/vivado_reports_larger_target/stream_conv3x3_3chan_8out_cell_dsp_synth.dcp \
           hardware/vhdl_conv3x3/vivado_reports_24out_dsp_200t_fallback_analysis \
           stream_conv3x3_3chan_8out_cell_dsp
```

(each run with the standard `LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"`
compat workaround used throughout this project).

**All three checkpoints opened successfully with `open_checkpoint`.**
`report_dsp_utilization` was attempted inside the script and **failed**:

```
invalid command name "report_dsp_utilization"
```

This command does not exist in this Vivado 2025.2 installation (it is not
a standard `report_*` command for this flow). The script caught the error
and wrote it to `*_dsp_utilization.txt` instead of stopping; all other
requested reports (`report_utilization -hierarchical`, both depth
variants, `report_timing_summary`, `get_cells`-based cell listings)
succeeded normally for all three designs.

## 4. 8-output vs. 16-output vs. 24-output resource comparison

Primitive (`REF_NAME`) counts from `report_utilization`'s "7. Primitives"
table (also cross-checked against this task's own `get_cells -hierarchical`
counts, which matched exactly):

| Primitive | 8-out 200T | 16-out 200T | 24-out 200T |
|---|---:|---:|---:|
| DSP48E1 | 203 | 405 | 731 |
| LUT6 | 466 | 711 | 1214 |
| LUT5 | 273 | 273 | 337 |
| LUT4 | 4 | 17 | 140 |
| LUT3 | 12 | 12 | 464 |
| LUT2 | 3 | 3 | **1750** |
| CARRY4 | **0** | **0** | **456** |
| FDRE | 410 | 418 | 706 |
| MUXF7 | 48 | 48 | 48 |
| Total LUTs | 668 | 929 | 3374 |
| Total Registers | 410 | 418 | 706 |

The two most striking changes at 24 outputs: **LUT2 usage jumps from 3 to
1750**, and **CARRY4 (fast carry-chain logic, used for LUT-based wide
adders) appears for the first time, at 456 instances**, having been
exactly **0** at both 8 and 16 outputs.

## 5. What changed at 24 outputs

- DSP48E1 usage scaled roughly linearly through 16 outputs (203 -> 405,
  about 2x for 2x the kernels) but the *requested* count reported in the
  overutilization warning (266 -> 529 -> 788, see
  `first_layer_24out_dsp_200t_summary.md` Section 9) crossed the 200T's
  740-DSP budget between 16 and 24 outputs. Once the requested demand
  exceeded the budget, Vivado's synthesis-time resource-sharing pass had
  to implement the shortfall in LUT fabric.
- The **shape** of the LUT growth is different from a simple proportional
  increase: LUT6 (the workhorse cell for the 16- and 8-output designs'
  small amount of non-DSP logic) only grew modestly (711 -> 1214, ~1.7x),
  while LUT2 grew nearly 600x (3 -> 1750) and LUT3 grew ~39x (12 -> 464).
  LUT2/LUT3 are the cell types typically produced by LUT-based
  add/compare logic feeding a carry chain, which matches the appearance
  of CARRY4 at the same point.
- Register count also grew disproportionately (418 -> 706, +288) compared
  with the near-flat 8-out -> 16-out change (410 -> 418, +8). This is
  consistent with pipeline/adder registers that would otherwise be
  absorbed into a DSP48E1's internal register stages instead needing
  separate Slice Registers when the surrounding arithmetic is implemented
  in LUT fabric.

## 6. Evidence for DSP-to-LUT fallback

**Whole-netlist DSP48E1 count matches the report exactly.** `get_cells
-hierarchical -filter {REF_NAME == DSP48E1}` on the reopened 24-output
checkpoint returned exactly **731** cells, matching
`stream_conv3x3_3chan_24out_cell_dsp_utilization.txt`'s reported
DSP48E1 count. This confirms the final synthesized 24-output 200T netlist
used 731 DSP48E1 blocks, not merely that the utilization report claimed
this.

**CARRY4 is present only in the 24-output netlist.** The same query for
`REF_NAME == CARRY4` returned 0 cells for both the 8-output and 16-output
200T checkpoints, and **456** cells for the 24-output checkpoint. CARRY4
is Xilinx's fast-carry-chain primitive, used by LUT-based adders/wide
arithmetic -- its appearance, exclusively at 24 outputs, is direct
evidence that some arithmetic which previously mapped entirely into
DSP48E1 hard macros (at 8 and 16 outputs) is now implemented as LUT +
carry-chain logic instead.

**The evidence is consistent with some arithmetic being implemented in
LUT fabric rather than DSP48E1, and points specifically at the top-level
final-summation adders** (the `final_sum` process in the RTL, which
computes `y{k}_r <= K{k}_BIAS32 + y{k}_c0 + y{k}_c1 + y{k}_c2` for each
kernel `k`), not at the per-tap multiply-accumulate datapath inside
`conv3x3_dot_pipelined_dsp`:

- At 8 and 16 outputs, `get_cells -hierarchical -filter {REF_NAME ==
  DSP48E1}` shows, in addition to the DSP48E1 cells nested under each
  `dot_k{k}_c{ch}` instance, exactly one **top-level** DSP48E1 cell named
  `y{k}_r_reg` for **every** kernel `k` (all 8 kernels for the 8-output
  design, all 16 for the 16-output design, with no exceptions). This is
  consistent with every kernel's final bias+channel-sum addition being
  folded into a DSP48E1 configured as an adder, in addition to the
  per-tap multiplies.
- At 24 outputs, only **22 of the 24** `y{k}_r_reg` names appear as
  top-level DSP48E1 cells; **kernel 7 and kernel 18 are absent** from
  that list. Cross-checking the CARRY4 cell list shows `y7_r_reg` and
  `y18_r_reg` logic present there instead (nested under
  `dot_k7_c0`/`dot_k7_c1` and `dot_k18_c0`/`dot_k18_c1`/`dot_k18_c2`
  respectively), and `report_utilization -hierarchical -hierarchical_depth
  4` shows these specific channel-instances retaining local flip-flops
  (32 FFs each) that are **not** present in the corresponding rows for
  any 8- or 16-output design's channel-instances (which show 0 local FFs,
  i.e., fully absorbed into DSP48E1 hard-macro registers). Kernels 7 and
  18's final-summation logic did not get a DSP48E1 at all in this run and
  is implemented entirely in LUT + CARRY4 fabric.
- A further 21 of the remaining 22 kernels (all except kernel 4) also
  show **some** CARRY4-tagged cells associated with their `y{k}_r_reg`
  name (ranging from 8 to 32 CARRY4 cells per kernel; kernel 4 shows
  none). Since these same kernels *also* have a top-level DSP48E1 cell
  for `y{k}_r_reg`, this is consistent with a **hybrid** mapping: part of
  the multi-term addition (bias + three channel partial sums) computed in
  LUT/carry-chain logic, feeding into a DSP48E1 configured as an adder for
  the remaining term(s) -- rather than a clean, single-resource mapping
  like the 8- and 16-output designs show for every kernel.
- The CARRY4-tagged cells are heavily concentrated under `c1`/`c2`
  channel-instance paths (37 of the associated instance paths are `c1` or
  `c2`, only 2 are `c0`), which is consistent with the RTL's
  left-to-right expression structure
  (`bias + y_c0 + y_c1 + y_c2`) -- the `c1` and `c2` terms are combined
  later in the expression tree, so post-synthesis-optimization naming
  tends to associate the merged adder logic with those instances' name
  space, though Vivado's netlist flattening does not preserve strict
  RTL-to-netlist correspondence, so this placement is suggestive rather
  than a certain causal explanation.

## 7. Localizability of the fallback

**Partially localizable, at kernel granularity, using cell-naming and
hierarchical-utilization evidence -- not fully localizable to individual
arithmetic operations.**

- **What can be stated with confidence:** the 731 DSP48E1 and 456 CARRY4
  cells were counted directly from the reopened netlist (not inferred
  from the utilization report alone). The final-summation stage is where
  the fallback is visible: 22 of 24 kernels' final-sum DSP48E1 mapping
  survived, kernels 7 and 18's final-sum logic did not get any DSP48E1 at
  all, and 21 of the remaining 22 kernels show a hybrid CARRY4+DSP48E1
  mapping for their final sum rather than the clean, single-DSP48E1
  mapping seen for every kernel at 8 and 16 outputs.
- **What cannot be stated with confidence:** whether any of the 72
  `conv3x3_dot_pipelined_dsp` instances' own internal 3x3=9-tap multiplies
  individually fell back to LUT fabric. Every one of the 72
  `dot_k{k}_c{ch}` instances shows at least 7 DSP48E1 cells locally
  (ranging 7-11 per instance out of a maximum of 9 raw taps per
  instance, since some instances also absorb part of the final-sum
  logic), so the per-tap multiply structure inside each channel instance
  does not show the same clean "instance had zero DSPs" signature that
  kernels 7 and 18's final-sum logic shows. The observed 7-11 range is
  reported as-is; this analysis does not determine, and does not claim to
  know, which specific weight values or taps within an instance drove
  that count away from a flat 9.
- In short: fallback is confidently localizable to **the top-level final
  cross-channel summation stage**, and to **two specific kernels (7 and
  18)** whose final-sum stage entirely avoided DSP48E1, but is only
  inferred at the resource/cell-count level (not proven at the
  individual-multiplier level) for the per-tap datapath inside
  `conv3x3_dot_pipelined_dsp`.

## 8. Timing impact

Re-running `report_timing_summary` against the reopened 24-output
checkpoint reproduced the same result already on file:
`WNS = +2.760 ns`, `WHS = +0.262 ns`, all endpoints meeting the 100 MHz /
10.000 ns clock constraint with 0 failing endpoints. The LUT/CARRY4
fallback did not cause a timing failure in this synthesis-level (not
placed-and-routed) result; the WNS did drop from 16-output's +3.848 ns to
24-output's +2.760 ns, consistent with more logic being on the critical
final-summation path once part of it is implemented in slower LUT/carry
fabric instead of a single hardened DSP48E1, though out-of-context
synthesis timing (no placement, no real routing delay) limits how much
weight this specific number should carry.

## 9. Power impact

No new power analysis was run in this task (`report_power` was not
re-invoked; the existing `first_layer_24out_dsp_200t_summary.md` figures
are unchanged: 24-output 200T total on-chip power estimate 1.041 W,
dynamic 0.914 W, static 0.127 W, vs. 16-output's 0.617 W). The dynamic
power increase is consistent with the additional active LUT/CARRY4 logic
in the fallback path, but this was not independently re-verified in this
task and remains a Vivado vector-less estimate, not a measurement.

## 10. Interpretation for a possible 32-output mixed DSP+LUT experiment

- Vivado already automatically produces a mixed DSP+LUT mapping once
  demand exceeds the 200T's DSP budget, rather than failing outright --
  the 24-output design still met timing at the synthesis level despite
  the DSP overutilization warning.
- If a 32-output design were built (not done in this task), it should be
  **framed as a mixed DSP+LUT mapping experiment from the outset**, not
  as a "does it fit cleanly" experiment: the 24-output evidence shows
  that once the DSP budget is exceeded, Vivado's fallback is not an
  all-or-nothing per-kernel decision -- most affected kernels get a
  *hybrid* CARRY4+DSP48E1 final-sum mapping, and only a couple of kernels
  (7, 18, in this specific 24-output run) lost DSP mapping for their
  final-sum stage entirely.
- The magnitude of LUT growth from this fallback is disproportionate to
  the number of affected kernels: only 2 of 24 kernels lost their
  top-level DSP48E1 entirely, and 21 more show a hybrid mapping, yet
  total LUTs grew more than 3.6x (929 -> 3374) between 16 and 24 outputs.
  This suggests a 32-output design, which the existing linear
  DSP-request trend projects to exceed the 740-DSP budget by a wider
  margin than 24 outputs did, could see LUT usage grow further still
  through the same mechanism -- but this is an extrapolation from one
  data point's internal structure, not a synthesized measurement.

## 11. Limitations

- This is an **analysis-only** task: no RTL was modified, no new
  architecture was created, and no 32-output design was built or
  synthesized.
- `report_dsp_utilization` is not available in this Vivado 2025.2
  installation; DSP/LUT fallback evidence here relies on
  `report_utilization -hierarchical` and `get_cells`-based primitive
  counts and hierarchical cell-path naming instead.
- Cell-path naming after synthesis optimization does not always preserve
  a strict 1:1 correspondence to the original RTL structure (Vivado may
  flatten and merge logic across instance boundaries), so statements
  about *which* instance's namespace a merged adder's cells appear under
  are structural observations, not proof of exactly which RTL expression
  produced them.
- The exact reason individual `conv3x3_dot_pipelined_dsp` instances show
  DSP48E1 counts ranging from 7 to 11 (rather than a flat 9, one per tap)
  was not determined in this task; this analysis does not claim to know
  which specific taps or weight values drove that variation.
- This is a synthesis-level (`synth_design`, out-of-context) result, not
  a placed-and-routed or board-tested result. Timing and power figures
  referenced here are the same Vivado estimates already reported in
  `first_layer_24out_dsp_200t_summary.md`, not measurements.
- No new synthesis was performed in this task -- the existing 24-output,
  16-output, and 8-output checkpoints were reopened read-only for
  reporting purposes; the underlying netlists are unchanged from the
  prior task's results.
- This analysis does not prove that a 32-output design will or will not
  fit on the Artix-7 200T; no 32-output design exists yet.

## Safe claims made in this document

- "Vivado reported DSP overutilization before fallback" (the `Used = 788,
  Available = 740` warning, Section 1, reproduced from the existing
  synthesis log).
- "The final synthesized 24-output 200T netlist used 731 DSP48E1 blocks"
  (confirmed directly via `get_cells -hierarchical`, Section 6).
- "The LUT count increased sharply compared with 16-output" (929 -> 3374,
  Section 4).
- "The evidence is consistent with some arithmetic being implemented in
  LUT fabric" (CARRY4 appearing only at 24 outputs, localized mainly to
  the final cross-channel summation stage, Sections 6-7).
- "A 32-output version should be framed as mixed DSP+LUT mapping if
  attempted" (Section 10).

This document does not claim to know exactly which mathematical
multipliers fell back to LUT fabric (only the final-summation stage's
fallback is well-localized; the per-tap multiply structure is not), does
not claim this proves a 32-output design will or will not fit (none was
built), and does not claim measured hardware power or board testing.

## 12. Recommended next step

Before attempting a 32-output design, use the same
`analyze_dsp_lut_fallback.tcl` approach against a small, targeted
experiment: re-synthesize the existing 24-output design with an explicit
`-directive` or DSP-packing constraint (e.g. testing whether forcing full
DSP packing via a stricter `synth_design` option changes which two
kernels lose DSP mapping, or whether it can be avoided at 24 outputs
specifically) to see whether the specific kernels affected are an
artifact of Vivado's resource-allocation order rather than an inherent
property of those kernels' weight values. This would clarify whether the
24-output fallback pattern is stable/reproducible or sensitive to
synthesis options, before committing effort to a full 32-output mixed
DSP+LUT design.
