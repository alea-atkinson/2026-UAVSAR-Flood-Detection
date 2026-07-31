# Direct-Parallel Streaming Front-End: Synthesis Result

Date: July 31, 2026

This report synthesizes the INTEGRATED skeleton added in commit
`77a71e35` --
`first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd`
(the new 3-channel streaming front end, chained directly into the
EXISTING, unmodified direct-parallel folded 32-output arithmetic core) --
on the same Artix-7 200T target used throughout this repo, to answer:
**does adding a real streaming front end change LUT/register/timing/power
compared with the arithmetic core alone?**

## Claim boundary (read this first)

- **Out-of-context Vivado synthesis only.** No board testing, no
  board-measured speedup, no board-measured power anywhere in this
  report.
- **No existing VHDL file was modified.** The arithmetic core, its
  weight/scale/bias package, and `window3x3_stream.vhd` are all reused
  byte-for-bit unchanged; only the already-existing (commit `77a71e35`)
  front-end wrapper and integrated skeleton were synthesized.
- **Still first-layer arithmetic-core scope only** -- not full-tile
  board behavior, not full U-Net inference, not a board demo.
- `.dcp` checkpoint files are written locally (matching this repo's
  existing synthesis-script convention) but are `.gitignore`d, matching
  every other synthesis result in this repository -- they are not staged.

## 1. Synthesis result

**PASS.** `synth_design` completed with 0 errors, 0 critical warnings.
**All user-specified timing constraints are met at 100 MHz.**

## 2. Commands run

```bash
module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"
vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_streaming_front_end_200t.tcl
```

## 3. Resource / timing / power numbers

| Metric | Streaming front end + core (this run) | Arithmetic core alone (prior result) | Delta |
|---|---:|---:|---:|
| LUTs | **13,713** (10.19%) | 12,039 (8.94%) | **+1,674 (+13.9%)** |
| Registers | **7,154** (2.66%) | 6,564 (2.44%) | **+590 (+9.0%)** |
| DSP48E1 | **740 / 740 (100.00%)** | 740 / 740 (100.00%) | **+0** |
| Block RAM tiles | **0 (0.00%)** | 0 (0.00%) | +0 |
| WNS (setup slack) | **+2.218 ns** | +2.218 ns | **+0 ns (bit-identical)** |
| WHS (hold slack) | **+0.262 ns** | +0.262 ns | +0 ns (bit-identical) |
| Total on-chip power (est.) | **1.441 W** (dynamic 1.313 W + static 0.129 W) | 1.334 W (dynamic 1.206 W + static 0.128 W) | **+0.107 W (+8.0%)** |
| Warnings | 2 (DSP demand-vs-available notice, out-of-context clock-skew notice) | 2 (same two) | +0 (identical) |

Full text reports: `hardware/vhdl_conv3x3/
vivado_reports_32out_folded_arithmetic_core_streaming_front_end_200t/
{first_layer_32out_folded_arithmetic_core_streaming_front_end_utilization.txt,
..._timing_summary.txt, ..._power.txt}`.

## 4. Comparison vs. arithmetic-core-only

Source: `hardware/vhdl_conv3x3/reports/
first_layer_32out_folded_arithmetic_core_summary.md`.

- **DSP usage is completely unchanged (740/740, 100%).** Expected:
  `window3x3_stream`'s line buffers are pure shift-register/control logic
  with no multiplies, so adding 3 instances of it does not touch the
  DSP-saturated arithmetic core's own multiplier usage at all.
- **LUTs increase by 1,674 (+13.9%) and registers by 590 (+9.0%)** -- the
  cost of the 3 line-buffer instances (each holding 3 rows x IMG_WIDTH
  pixels of shift-register state, plus column/row counters and the
  round-robin write-pointer control logic). This is a real, measurable,
  but modest addition -- total design utilization remains at only
  ~10% of the part's LUTs and ~2.7% of its registers, nowhere near a
  resource ceiling.
- **Timing is bit-identical (WNS/WHS unchanged to the nanosecond).** The
  critical path evidently remains entirely inside the already-DSP-heavy
  arithmetic core; the front end's shift-register-based line buffers add
  no additional combinational depth on that path. Adding the front end
  did not make timing closure any harder on this target.
- **Power increases by 0.107 W (+8.0%)**, consistent with ~1,674 more
  LUTs and ~590 more registers toggling -- a modest, not a dramatic,
  increase.
- **Warnings are identical** (the same pre-fallback DSP demand notice and
  the same out-of-context clock-skew notice already seen for the
  core-only synthesis) -- the front end introduces no new synthesis
  concerns.

## 5. Is the line-buffer/front-end cost small or meaningful?

**Small in absolute utilization terms, but not negligible in relative
terms.** A +13.9% LUT increase and +9.0% register increase are real,
measurable costs -- this is not "free" the way the front end's simulated
behavior (Section 6 of `direct_parallel_line_buffer_feasibility_summary.md`)
might suggest. But in absolute terms the combined design still uses only
~10% of the part's LUTs and ~2.7% of its registers, so this addition does
not threaten the design's overall feasibility on this target -- the
binding constraint remains the arithmetic core's own 100% DSP usage,
unchanged by the front end.

## 6. Does timing still meet 100 MHz?

**Yes, exactly as well as before** -- WNS = +2.218 ns, identical to the
arithmetic-core-only result to three decimal places. The streaming front
end adds resource cost but not timing risk on this target, at this clock.

## 7. Claim boundaries

- **Out-of-context synthesis only** -- no board testing, no board-measured
  speedup, no board-measured power. The 1.441 W figure is Vivado's
  vectorless, Medium-confidence synthesis-time estimate.
- **No claim that this combined design has been placed, routed, or run on
  a board.**
- **First-layer arithmetic-core-plus-front-end scope only** -- not full
  U-Net FPGA acceleration, not a full 256x256 tile-streaming board
  implementation.
- **This synthesis used `IMG_WIDTH = 6`** (the integrated skeleton's own
  default generic value, matching the 6x6 real-tile GHDL testbench from
  commit `77a71e35`). The line buffers inside `window3x3_stream` are
  sized to `IMG_WIDTH` (3 rows x `IMG_WIDTH` pixels, per channel), so
  **the +1,674 LUT / +590 register front-end cost reported here is
  specific to a 6-pixel-wide image and should NOT be assumed to hold at
  128 or 256 pixels wide** -- wider line buffers would need to store
  proportionally more pixels per row. This was NOT synthesized or
  measured at 128/256 width in this task; it is a natural next
  feasibility check, not a result claimed here.
- **No existing VHDL file was modified** to obtain this result; no bug
  was found that would have required touching the arithmetic core.
