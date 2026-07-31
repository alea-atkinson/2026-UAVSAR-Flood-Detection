# Streaming Front-End + Arithmetic Core: Realistic Line-Buffer Width Sweep (128, 256)

Date: July 31, 2026

`hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_streaming_front_end_synthesis_summary.md`
(commit `a3cb20c8`) synthesized the integrated front-end + arithmetic-core
skeleton at `IMG_WIDTH=6` only, and explicitly warned that the line
buffers' resource cost would grow at realistic tile widths (128, 256)
since `window3x3_stream`'s line buffers scale with `IMG_WIDTH`. This
report closes that gap: the SAME, UNMODIFIED integrated skeleton was
resynthesized at `IMG_WIDTH=128` and `IMG_WIDTH=256`, using Vivado's own
`-generic` override mechanism -- no new wrapper file, no VHDL edit.

## Claim boundary (read this first)

- **Out-of-context Vivado synthesis only.** No board testing, no
  board-measured speedup, no board-measured power anywhere in this
  report.
- **No existing VHDL file was modified or duplicated.** The exact same
  `first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd` top
  entity (commit `77a71e35`) was re-elaborated with `IMG_WIDTH` set via
  `synth_design -generic IMG_WIDTH=<value>` -- Vivado's standard
  top-level generic override mechanism, not a source-code change.
- **Still first-layer arithmetic-core scope only** -- not full-tile board
  behavior, not full U-Net inference, not a board demo.
- `.dcp` checkpoints are written locally per this repo's existing
  convention but are `.gitignore`d; not staged.

## 1. Method: `-generic` override, no new wrapper

The top-level entity already exposes `IMG_WIDTH` as a generic (default
6). `synth_first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_200t.tcl`
is a PARAMETERIZED script (mirroring this repo's existing
`synth_capacity_check_parameterized.tcl` convention) that accepts
`<img_width> <report_dir>` via `-tclargs` and passes the width straight
through to `synth_design -generic IMG_WIDTH=$img_width`. No width-specific
wrapper file was needed or created.

```bash
vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_200t.tcl \
  -tclargs 128 hardware/vhdl_conv3x3/vivado_reports_32out_folded_arithmetic_core_streaming_front_end_width128_200t

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_200t.tcl \
  -tclargs 256 hardware/vhdl_conv3x3/vivado_reports_32out_folded_arithmetic_core_streaming_front_end_width256_200t
```

## 2. Synthesis result: IMG_WIDTH=128

**PASS.** 0 errors, 0 critical warnings (the same 2 informational
warnings seen at every synthesis of this design: DSP pre-fallback demand
notice, out-of-context clock-skew notice). **All user-specified timing
constraints are met at 100 MHz.**

## 3. Synthesis result: IMG_WIDTH=256

**PASS.** 0 errors, 0 critical warnings (identical warning set).
**All user-specified timing constraints are met at 100 MHz.**

## 4. Resource / timing / power comparison

| Design | IMG_WIDTH | LUTs | Registers | DSP48E1 | BRAM | WNS (ns) | WHS (ns) | Power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core only (no front end) | N/A | 12,039 (8.9%) | 6,564 (2.4%) | 740/740 (100%) | 0 | +2.218 | +0.262 | 1.334 |
| Integrated (commit `a3cb20c8`) | 6 | 13,713 (10.2%) | 7,154 (2.7%) | 740/740 (100%) | 0 | +2.218 | +0.262 | 1.441 |
| Integrated -- this run | **128** | **30,120 (22.4%)** | **16,156 (6.0%)** | 740/740 (100%) | 0 | **+2.063** | +0.262 | **1.695** |
| Integrated -- this run | **256** | **47,542 (35.3%)** | **25,585 (9.5%)** | 740/740 (100%) | 0 | **+2.063** | +0.262 | **1.745** |

### Deltas from the IMG_WIDTH=6 integrated result

| | Width 128 vs. width 6 | Width 256 vs. width 6 |
|---|---:|---:|
| LUTs | +16,407 (+120%) | +33,829 (+247%) |
| Registers | +9,002 (+126%) | +18,431 (+258%) |
| DSP48E1 | +0 | +0 |
| WNS | -0.155 ns | -0.155 ns |
| Power | +0.254 W (+18%) | +0.304 W (+21%) |

### Front-end-only cost (total minus the core-only 12,039 LUT / 6,564 reg baseline)

| IMG_WIDTH | Front-end LUTs | Front-end registers | LUTs per pixel of width |
|---:|---:|---:|---:|
| 6 | 1,674 | 590 | 279.0 |
| 128 | 18,081 | 9,592 | 141.3 |
| 256 | 35,503 | 19,021 | 138.7 |

Fitting the 128/256 points: **front-end LUTs ~= 659 + 136 x IMG_WIDTH** --
consistent with `window3x3_stream`'s own documented storage requirement
(3 rows x IMG_WIDTH pixels x 3 channels), i.e. genuinely LINEAR growth in
image width, not a superlinear blow-up. The 6-pixel-wide point sits above
this line because the fixed control/counter overhead (row/column
counters, round-robin write pointer) is a much larger fraction of a tiny
line buffer's total cost.

## 5. Is shift-register line-buffer scaling small, meaningful, or problematic?

**Meaningful, not merely small, but not yet problematic on this part.**
DSPs remain completely unaffected (740/740, 100%, identical across every
width) -- the front end never touches a multiplier, so it cannot worsen
the design's binding DSP constraint. LUTs, however, grow linearly and
substantially: from 10.2% of the part at IMG_WIDTH=6 to **35.3% at
IMG_WIDTH=256** -- almost 3x the core-only design's own LUT footprint,
added purely for flip-flop-based line buffering. This is not
"negligible," but it is also not disqualifying: the combined design still
fits comfortably (35.3% LUT, 9.5% register utilization, both well under
100%), and timing still closes with positive margin at both realistic
widths.

## 6. Does timing change?

**Slightly, and it stabilizes.** WNS drops from +2.218 ns (IMG_WIDTH=6)
to +2.063 ns at BOTH 128 and 256 (identical between the two, suggesting
the critical path shifted once to a wider-address-range structure and
then stopped changing further as width grows). WHS is unchanged
(+0.262 ns) across all four rows. **100 MHz timing closure holds at every
width tested, with no new warnings or failing endpoints.**

## 7. Does the earlier direct-parallel tile-timing correction remain valid?

**Yes.** The corrected cycle-count formula from
`hardware/vhdl_conv3x3/reports/direct_parallel_line_buffer_feasibility_summary.md`
(`total_cycles = H*W + 6`) is a FUNCTIONAL/behavioral property, confirmed
by GHDL simulation, independent of the RESOURCE cost measured here by
synthesis. This report confirms that achieving that timing at 100 MHz is
ALSO synthesizable at realistic widths (128, 256) -- it does not change
the cycle-count formula itself, it adds the resource-cost half of the
picture that formula never claimed to cover.

## 8. Should a BRAM-backed line buffer be recommended for future real designs?

**Yes.** `window3x3_stream.vhd`'s own header comment already flags this
as a known limitation ("For large IMG_WIDTH, line buffers should be
implemented with BRAM instead of flip-flops") -- this synthesis result is
the first concrete evidence for WHY: at IMG_WIDTH=256, line buffering
alone costs ~35,500 LUTs (26.4% of the part) while BRAM sits at **0/365
(0%) utilization** across every width tested. Any future REAL streaming
design at 128x128/256x256 scale should trade this abundant, currently
unused BRAM for the LUT-heavy flip-flop line buffers -- a natural next
feasibility check, not attempted or claimed here (no BRAM-backed
line-buffer design was built or synthesized in this task).

## 9. Files created

| File | Purpose |
|---|---|
| `hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_200t.tcl` | Parameterized synthesis script (`-generic IMG_WIDTH=...`) |
| `hardware/vhdl_conv3x3/vivado_reports_32out_folded_arithmetic_core_streaming_front_end_width128_200t/` | Utilization/timing/power reports, IMG_WIDTH=128 |
| `hardware/vhdl_conv3x3/vivado_reports_32out_folded_arithmetic_core_streaming_front_end_width256_200t/` | Utilization/timing/power reports, IMG_WIDTH=256 |
| This report | `hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_summary.md` |
| `outputs/hardware_benchmarks/direct_parallel_line_buffer_feasibility/streaming_front_end_width_sweep.csv` | Machine-readable comparison table |
| `outputs/hardware_benchmarks/direct_parallel_line_buffer_feasibility/streaming_front_end_width_sweep_summary.md` | Short companion summary |

**No existing VHDL file was modified.** No bug was found that would have
required touching the arithmetic core or the front end.
