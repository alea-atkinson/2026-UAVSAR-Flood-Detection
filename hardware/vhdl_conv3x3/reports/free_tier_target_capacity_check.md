# Free-Tier FPGA Target Capacity Check (Vivado 2025.2, hpcl5-5)

Date: July 29, 2026

Before running any new Vivado synthesis of the project's learned
first-layer designs on Zynq-7000 or Spartan-7, this report checks
whether any newly-installed Zynq-7000 or Spartan-7 part is actually a
bigger or more useful synthesis target than the Artix-7 200T or
Kintex-7 160T already used in this project. Per the task's own
instruction, **the full project designs were NOT synthesized on any new
family** -- only a capacity check was performed, and it did not show any
new target to be more useful, so no further synthesis was warranted (see
Recommendation).

## Claim boundary (read this first)

- **This is a resource-capacity check only.** No project datapath
  (32-output DSP-aware, resource-shared, or 2-lane) was synthesized on
  Zynq-7000 or Spartan-7 for this report.
- **No board testing, no measured speedup, no measured power** anywhere
  in this report.
- **Free-tier / license-visible parts only.** Every number below reflects
  what THIS Vivado 2025.2 installation on `hpcl5-5.hslinux` exposes via
  `get_parts` today -- not the full Xilinx/AMD product catalog. A
  different license tier or a future Vivado version could expose
  different (likely larger) parts in any of these families.
- **No existing VHDL datapath was modified.** Only one new, trivial,
  clearly-named dummy top module was added (see below), used solely to
  make Vivado report each part's own resource capacity.

## 1. Confirmations

```
$ git status -sb
## data_testing...origin/data_testing
   (only pre-existing untracked items and this task's new files; see
   final git status at the end of this report)

$ hostname
hpcl5-5.hslinux

$ module load vivado/2025.2 ; vivado -version
vivado v2025.2 (64-bit)
```

`get_parts` counts by family on this installation:

| Family prefix | Part count |
|---|---:|
| `xc7a*` (Artix-7) | 210 |
| `xc7k*` (Kintex-7) | 56 |
| `xc7z*` (Zynq-7000) | 62 |
| `xc7s*` (Spartan-7) | 64 |

Device sizes actually present within each family (from the full
`get_parts` listing, not assumed from family name):

| Family | Device sizes visible |
|---|---|
| Kintex-7 | `xc7k70t`, `xc7k160t` only |
| Zynq-7000 | `xc7z007s`, `xc7z010`, `xc7z012s`, `xc7z014s`, `xc7z015`, `xc7z020`, `xc7z030` |
| Spartan-7 | `xc7s6`, `xc7s15`, `xc7s25`, `xc7s50`, `xc7s75`, `xc7s100` |

**`xc7k160t` is confirmed the largest visible Kintex-7 device** (no
`xc7k325t`/`xc7k410t`/`xc7k480t` present). **`xc7z030` is confirmed the
largest visible Zynq-7000 device.** **`xc7s100` is confirmed the largest
visible Spartan-7 device.**

## 2. Method: reliable, Vivado-verified capacity check (not name-based assumptions)

A trivial, single-flip-flop dummy top module
(`hardware/vhdl_conv3x3/capacity_check_dummy_top.vhd`) was synthesized
against each selected part using a new generic script
(`hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl`), and
`report_utilization`'s **Available** column was read directly -- this
column reflects the PART's own total resource capacity, independent of
the (deliberately trivial) design's own resource use. This avoids
inferring capacity from part names/family marketing, per the task's
requirement.

### Exact commands run

```bash
module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl \
  -tclargs xc7a200tsbg484-1 hardware/vhdl_conv3x3/vivado_reports_capacity_check_artix200t

vivado -mode batch -source hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl \
  -tclargs xc7k160tfbg484-1 hardware/vhdl_conv3x3/vivado_reports_capacity_check_kintex160t

vivado -mode batch -source hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl \
  -tclargs xc7z030fbg484-1 hardware/vhdl_conv3x3/vivado_reports_capacity_check_zynq030

vivado -mode batch -source hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl \
  -tclargs xc7s100fgga484-1 hardware/vhdl_conv3x3/vivado_reports_capacity_check_spartan100
```

Each run: `synth_design -top capacity_check_dummy_top -part <part> -mode
out_of_context` followed by `report_utilization`. All four runs
completed with **0 errors, 0 critical warnings, 0 warnings**.

### Selected part names

| Target | Selected part | Why |
|---|---|---|
| Artix-7 200T | `xc7a200tsbg484-1` | Same part already used throughout this project (largest Artix-7 tried). |
| Kintex-7 160T | `xc7k160tfbg484-1` | Same part already used in the prior Kintex-7 synthesis comparison (largest Kintex-7 visible). |
| Zynq-7000 (largest visible) | `xc7z030fbg484-1` | Largest visible Zynq-7000 device (`xc7z030`); normal commercial `-1` speed grade; `fbg484` package chosen to match the `484`-pin package used for the other three targets. |
| Spartan-7 (largest visible) | `xc7s100fgga484-1` | Largest visible Spartan-7 device (`xc7s100`); normal commercial `-1` speed grade; `fgga484` package (Spartan-7's `484`-pin package family) chosen to match pin-count convention. |

## 3. Resource capacity table (from Vivado `report_utilization`, "Available" column)

| Target | Part | LUTs | Registers (FF) | DSP48 blocks | BRAM tiles |
|---|---|---:|---:|---:|---:|
| Artix-7 200T | `xc7a200tsbg484-1` | **134,600** | **269,200** | **740** | **365** |
| Kintex-7 160T | `xc7k160tfbg484-1` | 101,400 | 202,800 | 600 | 325 |
| Zynq-7000 (xc7z030) | `xc7z030fbg484-1` | 78,600 | 157,200 | 400 | 265 |
| Spartan-7 (xc7s100) | `xc7s100fgga484-1` | 64,000 | 128,000 | 160 | 120 |

The Artix-7 200T and Kintex-7 160T figures above are **identical** to the
"Available" denominators already seen in this project's real DSP-aware
and resource-shared design utilization reports -- this cross-checks the
dummy-top method as reliable (device capacity is a property of the part,
not of whatever design is loaded onto it).

## 4. Comparison to the current Artix-7 200T target

| Resource | Artix-7 200T | Kintex-7 160T | Zynq-7000 xc7z030 | Spartan-7 xc7s100 |
|---|---:|---:|---:|---:|
| LUTs (ratio to Artix 200T) | 1.00x | 0.75x | 0.58x | 0.48x |
| Registers (ratio) | 1.00x | 0.75x | 0.58x | 0.48x |
| DSP48 blocks (ratio) | 1.00x | 0.81x | **0.54x** | **0.22x** |
| BRAM tiles (ratio) | 1.00x | 0.89x | 0.73x | 0.33x |

**Every visible Zynq-7000 and Spartan-7 part checked is smaller than the
Artix-7 200T already used in this project, in every resource category,
by a wide margin** -- most importantly DSP48 blocks, since DSP pressure
is the binding constraint for the 32-output direct-parallel design (this
project's own prior finding).

## 5. Answers to the specific questions

**Is any visible Kintex-7 part bigger than `xc7k160t`?**
**No.** Only `xc7k70t` and `xc7k160t` are visible in this installation;
`xc7k160t` is already the largest, and it was already used in the prior
Kintex-7 synthesis comparison
(`kintex7_synthesis_comparison_summary.md`).

**Is the largest visible Zynq-7000 (`xc7z030`) bigger/more useful than
Artix-7 200T for the DSP-heavy 32-output design?**
**No.** `xc7z030` has only **400 DSP48 blocks**, compared to the
Artix-7 200T's 740 and even the Kintex-7 160T's 600. The 32-output
direct-parallel DSP-aware design already requests ~1,044 multiplies
(per the existing project findings); moving to `xc7z030` would leave an
even LARGER DSP shortfall than either part already tried, forcing even
more of the design into LUT/CARRY4 fallback. `xc7z030` is also smaller
in LUTs, registers, and BRAM than both prior targets. There is no
resource dimension in which it is a better target.

**Is the largest visible Spartan-7 (`xc7s100`) bigger/more useful?**
**No, and by a much larger margin.** `xc7s100` has only **160 DSP48
blocks** (barely a fifth of the Kintex-7 160T's 600, and about a fifth of
the DSP-aware design's per-channel-group multiplier needs even at 8
output channels), 64,000 LUTs, and 120 BRAM tiles -- the smallest of all
four targets checked in every category. Spartan-7 is Xilinx's
cost/power-optimized, most resource-constrained 7-series family; this is
consistent with that positioning, not a surprising result.

## 6. Recommendation

**Keep only the existing Kintex-7 synthesis comparison. Do NOT run
Zynq-7000 or Spartan-7 synthesis of the project's learned first-layer
designs.**

The capacity check shows conclusively -- from Vivado's own reported
resource totals, not from part-name assumptions -- that neither the
largest visible Zynq-7000 part (`xc7z030`) nor the largest visible
Spartan-7 part (`xc7s100`) offers MORE of any resource (LUTs, registers,
DSP48 blocks, or BRAM) than the Artix-7 200T or Kintex-7 160T already
synthesized in this project. Both would only reproduce the same DSP-
saturation story already documented for Kintex-7 -- likely a worse one,
given their smaller DSP budgets -- while adding no new capability
insight. Per the task's own instruction ("do not synthesize the full
project designs unless the capacity check shows a target may actually be
useful"), this capacity check result means full synthesis on Zynq-7000
or Spartan-7 is not warranted.

## 7. Caveat

This entire capacity check is based on the **free-tier parts installed
and visible to Vivado 2025.2 on `hpcl5-5.hslinux` at the time of this
report** (July 2026). It does not reflect the full range of parts
available under a different Xilinx/AMD license tier, a different Vivado
version, or a different installation. If a future license or
installation exposes larger Zynq-7000 (e.g. `xc7z045`, `xc7z100`) or
Spartan-7 parts, or larger Kintex-7 parts (e.g. `xc7k325t` and above),
this capacity check would need to be repeated before drawing the same
conclusion for those parts.

## 8. Files created for this check

- `hardware/vhdl_conv3x3/capacity_check_dummy_top.vhd` (trivial dummy design, not a project datapath)
- `hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl`
- `hardware/vhdl_conv3x3/vivado_reports_capacity_check_artix200t/`
- `hardware/vhdl_conv3x3/vivado_reports_capacity_check_kintex160t/`
- `hardware/vhdl_conv3x3/vivado_reports_capacity_check_zynq030/`
- `hardware/vhdl_conv3x3/vivado_reports_capacity_check_spartan100/`
- This report: `hardware/vhdl_conv3x3/reports/free_tier_target_capacity_check.md`

No existing VHDL file was modified, and no project design (DSP-aware,
resource-shared, or 2-lane) was synthesized on any new part.
