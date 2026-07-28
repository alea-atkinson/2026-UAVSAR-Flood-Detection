# Kintex-7 Synthesis Comparison for the Learned First-Layer FPGA Prototypes

Date: July 28, 2026

This report compares the existing Artix-7 200T synthesis results against
a representative Kintex-7 target, using the SAME, UNMODIFIED VHDL source
files already synthesized and GHDL-verified elsewhere in this repo. No
new hardware architecture or datapath was created for this comparison --
only new synthesis TCL scripts and new report folders.

## Claim boundary (read this first)

- **Synthesis only.** Every figure in this report comes from Vivado
  `synth_design -mode out_of_context`, `report_utilization`,
  `report_timing_summary`, and `report_power`. Nothing was placed,
  routed, or run on physical hardware.
- **No board testing.** Neither the Artix-7 200T nor this new Kintex-7
  part has been programmed onto a physical device at any point in this
  project.
- **No measured speedup or measured power.** All timing and power
  figures are Vivado synthesis-level estimates (vector-less, "Medium
  confidence" for power), not measurements.
- **Same first-layer prototypes only.** This compares three ALREADY
  existing, ALREADY GHDL-verified first Conv2d-layer / first
  Conv-BN-ReLU-stage prototypes -- not the full U-Net, and not a new
  design.

## 1. Selected Kintex-7 part and why

`get_parts *xc7k*` on this Vivado 2025.2 installation returns **56
parts**, all belonging to exactly two Kintex-7 device sizes:
`xc7k70t` and `xc7k160t` (in various packages, speed grades, and
temperature grades -- no `xc7k325t`/`xc7k410t`/`xc7k480t` parts are
available under this license tier).

Per the task's preference for a larger part where available, **`xc7k160t`**
(the larger of the two exposed sizes) was selected. Package and speed
grade were chosen to mirror the existing Artix-7 200T target
(`xc7a200tsbg484-1`) as closely as possible: the `484`-pin package size
and `-1` speed grade, both already used for every prior synthesis
comparison in this repo.

**Selected part: `xc7k160tfbg484-1`.**

| Resource | xc7a200tsbg484-1 (existing) | xc7k160tfbg484-1 (this report) | Ratio |
|---|---:|---:|---:|
| LUTs | 134,600 | 101,400 | 0.75x |
| Registers | 269,200 | 202,800 | 0.75x |
| DSP48E1 slices | 740 | **600** | **0.81x (FEWER, not more)** |
| Block RAM tiles | 365 | 325 | 0.89x |

Device resource totals read directly from this run's own Vivado
utilization reports. **The selected Kintex-7 part has fewer LUTs,
registers, DSPs, and BRAM tiles than the Artix-7 200T** -- despite
Kintex-7 generally being marketed as the higher-performance 7-series
family, the specific free-tier part available here (`xc7k160t`) is
smaller across every resource category than the largest available
Artix-7 part (`xc7a200t`). This is stated up front because it directly
shapes the DSP-pressure result below.

## 2. Designs compared (unmodified from their existing Artix-7 200T synthesis)

| # | Design | VHDL top module |
|---|---|---|
| 1 | 32-output direct-parallel DSP-aware first Conv2d | `stream_conv3x3_3chan_32out_cell_dsp` |
| 2 | 32-output single-lane resource-shared folded Conv-BN-ReLU | `stream_conv3x3_3chan_32out_bn_relu_time_mux` |
| 3 | 2-lane resource-shared folded Conv-BN-ReLU | `stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux` |

## 3. Commands run

```bash
module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_32out_cell_dsp_kintex7.tcl
vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_kintex7.tcl
vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_32out_bn_relu_2lane_resource_shared_kintex7.tcl
```

Each: `synth_design -top <top> -part xc7k160tfbg484-1 -mode out_of_context`,
`create_clock -period 10.000` (100 MHz), matching every existing Artix-7
synthesis flow in this repo exactly. Reports written to:
- `hardware/vhdl_conv3x3/vivado_reports_32out_dsp_kintex7/`
- `hardware/vhdl_conv3x3/vivado_reports_32out_bn_relu_resource_shared_kintex7/`
- `hardware/vhdl_conv3x3/vivado_reports_32out_bn_relu_2lane_resource_shared_kintex7/`

## 4. Artix-7 200T vs. Kintex-7 160T comparison

| Design | Target | LUTs | Registers | DSPs | BRAM | WNS (ns) | WHS (ns) | Power (W) | Warnings/Crit/Err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 32-output DSP-aware direct-parallel | Artix-7 200T | 10,461 / 134,600 (7.77%) | 4,618 / 269,200 (1.72%) | 740 / 740 (100.00%) | 0 / 365 | +2.343 | +0.262 | 1.324 | 1 / 0 / 0 |
| 32-output DSP-aware direct-parallel | **Kintex-7 160T** | **15,723** / 101,400 (15.51%) | **8,354** / 202,800 (4.12%) | **603** / 600 (**100.50%**) | 0 / 325 | **+4.646** | +0.191 | 1.278 | 1 / 0 / 0 |
| 32-output single-lane resource-shared | Artix-7 200T | 1,837 / 134,600 (1.36%) | 5,588 / 269,200 (2.08%) | 2 / 740 (0.27%) | 0 / 365 | +0.662 | +0.262 | 0.165 | 32 / 0 / 0 |
| 32-output single-lane resource-shared | **Kintex-7 160T** | **1,837** / 101,400 (1.81%) | **5,588** / 202,800 (2.76%) | **2** / 600 (0.33%) | 0 / 325 | **+3.783** | +0.191 | 0.153 | 32 / 0 / 0 |
| 2-lane resource-shared | Artix-7 200T | 2,531 / 134,600 (1.88%) | 5,878 / 269,200 (2.18%) | 4 / 740 (0.54%) | 0 / 365 | +0.229 | +0.262 | 0.176 | 64 / 0 / 0 |
| 2-lane resource-shared | **Kintex-7 160T** | **2,531** / 101,400 (2.50%) | **5,878** / 202,800 (2.90%) | **4** / 600 (0.67%) | 0 / 325 | **+3.801** | +0.191 | 0.164 | 63 / 0 / 0 |

For the DSP-aware design, Vivado's synthesis-level DSP request (before
fallback legalization) was **1,044 multiplies** on both targets
(identical netlist, identical request) -- the *available* budget is what
differs (740 on Artix-7 200T vs. 600 on Kintex-7 160T).

Every warning on every design is the same class already documented for
these designs on Artix-7 (the DSP-aware design's single "DSP
overutilized" warning; the resource-shared designs' benign "unused
sequential element" / `FSM_onehot_state_reg` notes). 0 critical warnings
and 0 errors on all six syntheses (three designs x two targets).

## 5. Interpretation

**Did Kintex-7 reduce DSP pressure?** **No -- it got worse.** The
selected free-tier Kintex-7 part has FEWER DSP48E1 slices (600) than the
Artix-7 200T (740). The DSP-aware design's 1,044-multiply request now
exceeds an even smaller budget, so DSP utilization is **100.50%** (603
DSPs used, technically over the part's own 600-DSP total at the
synthesis-report level) and MORE of the design falls back to LUT/CARRY4
fabric than on Artix-7 200T: LUTs grew from 10,461 to 15,723 (+50.3%)
and registers from 4,618 to 8,354 (+80.9%). **Kintex-7 as a family name
does not automatically mean "more DSPs" -- the specific part matters
more than the family**, and this project's own tool license only exposes
smaller Kintex-7 parts than the largest available Artix-7 part.

**Did timing improve?** **Yes, substantially, for all three designs.**
Worst-case setup slack (WNS) improved on every design when moving to
Kintex-7 160T at the same nominal `-1` speed grade and same 100 MHz
target -- most dramatically for the resource-shared designs (1-lane:
+0.662 ns -> +3.783 ns; 2-lane: +0.229 ns -> +3.801 ns), and also for the
DSP-aware design (+2.343 ns -> +4.646 ns). Hold slack (WHS) is slightly
tighter on Kintex-7 (+0.191 ns vs. +0.262 ns) but remains comfortably
positive in every case -- all six syntheses meet 100 MHz timing with 0
failing endpoints. This is consistent with Kintex-7 being Xilinx's
higher-performance 7-series family relative to the cost/power-optimized
Artix-7 family, even though the specific `-1` speed-grade label is not an
absolute cross-family unit.

**Did resource mapping change?** For the two LUT-only, non-DSP
resource-shared designs, **no** -- LUT and register counts are IDENTICAL
between Artix-7 200T and Kintex-7 160T (1,837 LUTs / 5,588 registers for
1-lane; 2,531 LUTs / 5,878 registers for 2-lane, on BOTH targets). This
extends a pattern already established for same-family, different-size
Artix-7 comparisons
(`larger_fpga_target_comparison_summary.md`) to a DIFFERENT FPGA FAMILY:
because Kintex-7 and Artix-7 are both 7-series parts built from the same
underlying primitives (LUT6, FDRE, CARRY4) at the same speed grade, a
design that doesn't use DSPs or BRAM maps to identical absolute resource
counts regardless of family. For the DSP-aware design, resource mapping
DID change, and in the worse direction, exactly because that design's
resource need is NOT independent of the specific part's DSP budget (see
above).

**Does this change the project's conclusion?** **No -- it reinforces it,
with one new nuance.** This project's established conclusion has been:
(a) fully parallel, DSP-mapped designs saturate whatever DSP budget is
available and do not scale cleanly; (b) resource-shared,
time-multiplexed folded Conv-BN-ReLU designs are the resource-cheap path,
trading latency for a tiny, near-constant DSP/LUT footprint; (c) a larger
target mainly relieves utilization *percentage* pressure, not the
underlying architecture's fixed resource *need*. This Kintex-7
comparison confirms (b) and (c) hold across FPGA families, not just
across Artix-7 device sizes, and ALSO shows that (a)'s core problem
(DSP-mapped scaling) is not solved by simply changing family -- it can
even get worse, since the free-tier Kintex-7 part available here has
fewer DSPs than the Artix-7 200T already tried. The new nuance: Kintex-7
brings a real, "free" timing-margin improvement for these designs,
without changing which architecture family (resource-shared) is the
practical choice.

## 6. Safe claim boundaries

- Synthesis-only comparison; no placement, routing, or board testing was
  performed for either target.
- No measured speedup and no measured board power for either target --
  all timing and power figures are Vivado `synth_design` /
  `report_timing_summary` / `report_power` (vector-less, "Medium
  confidence") estimates.
- This compares the SAME first-layer / first Conv-BN-ReLU-stage
  prototypes already used throughout this repo -- not the full U-Net,
  and no new datapath was created.
- The Kintex-7 part used (`xc7k160tfbg484-1`) is the largest available
  under this specific Vivado license tier at the time of this report;
  larger Kintex-7 parts (325T/410T/480T) were checked and are NOT
  available, so this is not necessarily representative of the whole
  Kintex-7 family's DSP/resource capability.
- Absolute power comparisons across the two targets should be read with
  the same caution already documented in
  `larger_fpga_target_comparison_summary.md`: static (leakage) power
  differences partly reflect die-size/family characteristics, not
  design efficiency.

## 7. Files created for this comparison

- `hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_32out_cell_dsp_kintex7.tcl`
- `hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_kintex7.tcl`
- `hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_32out_bn_relu_2lane_resource_shared_kintex7.tcl`
- `hardware/vhdl_conv3x3/vivado_reports_32out_dsp_kintex7/`
- `hardware/vhdl_conv3x3/vivado_reports_32out_bn_relu_resource_shared_kintex7/`
- `hardware/vhdl_conv3x3/vivado_reports_32out_bn_relu_2lane_resource_shared_kintex7/`
- This report: `hardware/vhdl_conv3x3/reports/kintex7_synthesis_comparison_summary.md`

No existing VHDL file was modified.
