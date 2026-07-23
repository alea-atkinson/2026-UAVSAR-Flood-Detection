# Real-Tile Q.16 vs. Q.20 Synthesis Comparison (Artix-7 200T)

Date: July 23, 2026

This report quantifies whether the Q.20 fixed-point format
(recommended by
`hardware/vhdl_conv3x3/reports/real_tile_fixed_point_precision_sensitivity_summary.md`
over Q.16 for real-tile deployment accuracy) costs more FPGA resources
or timing margin than the existing Q.16 real-tile kernel0/kernel2
Conv-BN-ReLU designs, when actually synthesized.

## Claim boundary (read this first)

- **Synthesis-only comparison.** No board testing, no placed-and-routed
  implementation, no measured power -- all figures below are Vivado
  `synth_design` (out-of-context) utilization/timing/power estimates.
- **No datapath was changed.** All four designs synthesized here are the
  EXISTING, already GHDL-verified real-tile kernel0/kernel2 designs from
  `hardware/vhdl_conv3x3/reports/real_tile_first_layer_vhdl_verification_summary.md`
  (Q.16) and
  `hardware/vhdl_conv3x3/reports/real_tile_fixed_point_precision_sensitivity_summary.md`
  (Q.20) -- structurally identical to each other, differing only in
  which constants package (Q.16 vs. Q.20 `SCALE_FX`/`BIAS_FX`) they read.
  No existing committed VHDL file was modified for this comparison; only
  new, clearly-named synthesis TCL scripts and report folders were added.
- **Target confirmed today**: this Vivado 2025.2 installation exposes
  Artix-7 parts only (175 `artix7` parts; 0 `kintex7`, `virtex7`,
  `kintexu`, `kintexuplus`, `virtexu`, `virtexuplus`, `zynquplus`), so
  `xc7a200tsbg484-1` (the existing project's larger Artix-7 comparison
  target) is used, not a larger family.
- **Kernel 0 and kernel 2 only** -- the same real-data verification
  subset as the prior reports, not all 32 output channels.

## Designs compared

| # | Design | Format | Top module | Package |
|---|---|---|---|---|
| 1 | kernel 0 real-tile | Q.16 | `stream_conv3x3_3chan_kernel0_bn_relu_real_tile` | `first_conv_bn_relu_kernel0_real_tile_pkg` |
| 2 | kernel 0 real-tile | Q.20 | `stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20` | `first_conv_bn_relu_kernel0_real_tile_q20_pkg` |
| 3 | kernel 2 real-tile | Q.16 | `stream_conv3x3_3chan_kernel2_bn_relu_real_tile` | `first_conv_bn_relu_kernel2_real_tile_pkg` |
| 4 | kernel 2 real-tile | Q.20 | `stream_conv3x3_3chan_kernel2_bn_relu_real_tile_q20` | `first_conv_bn_relu_kernel2_real_tile_q20_pkg` |

Each design reuses the existing, UNMODIFIED `stream_conv3x3_3chan_cell.vhd`
/ `conv3x3_dot_pipelined.vhd` / `window3x3_stream.vhd` sub-components --
the only source file that differs between the Q.16 and Q.20 variant of
the same kernel is the constants package.

## Exact Vivado commands run

```bash
module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_kernel0_bn_relu_real_tile_200t.tcl
vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20_200t.tcl
vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_kernel2_bn_relu_real_tile_200t.tcl
vivado -mode batch -source hardware/vhdl_conv3x3/synth_stream_conv3x3_3chan_kernel2_bn_relu_real_tile_q20_200t.tcl
```

(each: `synth_design -mode out_of_context` against `xc7a200tsbg484-1`,
`create_clock -period 10.000` i.e. 100 MHz, matching every existing
Artix-7 synthesis flow in this repo; the `LD_LIBRARY_PATH` compat
workaround is the same one used throughout this project for this
Vivado 2025.2 installation.)

## Q.16 vs. Q.20 synthesis results

| Design | LUTs | Slice Registers | DSPs | BRAM Tiles | WNS (ns) | WHS (ns) | Total power (W) | Warnings/Crit/Errors | Timing met? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| kernel 0, Q.16 | 1,984 / 134,600 (1.47%) | 1,428 / 269,200 (0.53%) | 0 / 740 (0.00%) | 0 / 365 (0.00%) | +2.781 | +0.262 | 0.170 | 0 / 0 / 0 | **Yes** |
| kernel 0, Q.20 | 2,058 / 134,600 (1.53%) | 1,440 / 269,200 (0.53%) | 0 / 740 (0.00%) | 0 / 365 (0.00%) | +2.535 | +0.262 | 0.172 | 0 / 0 / 0 | **Yes** |
| kernel 2, Q.16 | 2,032 / 134,600 (1.51%) | 1,446 / 269,200 (0.54%) | 0 / 740 (0.00%) | 0 / 365 (0.00%) | +2.390 | +0.262 | 0.171 | 0 / 0 / 0 | **Yes** |
| kernel 2, Q.20 | 2,110 / 134,600 (1.57%) | 1,454 / 269,200 (0.54%) | 0 / 740 (0.00%) | 0 / 365 (0.00%) | +2.462 | +0.262 | 0.173 | 0 / 0 / 0 | **Yes** |

The one warning class present in every design's timing report (not
counted above as a design warning) is the standard, already-documented
benign out-of-context note: `WARNING: [Timing 38-242] The property
HD.CLK_SRC of clock port "clk" is not set` -- identical across every
out-of-context synthesis in this repo, a property of out-of-context mode
itself, not of these designs.

## Q.16 -> Q.20 deltas

| Metric | Kernel 0 delta | Kernel 2 delta |
|---|---:|---:|
| LUTs | +74 (+3.73%) | +78 (+3.84%) |
| Slice Registers | +12 (+0.84%) | +8 (+0.55%) |
| DSPs | 0 | 0 |
| BRAM | 0 | 0 |
| WNS | -0.246 ns (still +2.535 ns margin) | +0.072 ns (still +2.390->+2.462 ns margin) |
| WHS | 0 (unchanged) | 0 (unchanged) |
| Total power | +0.002 W (+1.18%) | +0.002 W (+1.17%) |

## Whether Q.20 meaningfully increases cost

**No.** Every resource delta is small in both absolute and relative
terms:

- **LUTs grow by only ~3.7-3.8%** (74-78 LUTs out of a 134,600-LUT
  part -- 0.06 percentage points of device utilization), consistent
  with the wider `SCALE_FX`/`BIAS_FX` constants (Q.20 needs a few more
  bits than Q.16, per the precision-sensitivity report's own width
  check) requiring marginally wider constant-multiply and adder logic,
  not a new datapath.
- **Registers grow by well under 1%** for both kernels (12 and 8
  registers respectively) -- the pipeline depth and register count are
  unchanged; only the bit-widths of the already-registered
  `product_fx_r`/`y_bn_relu_fx_r` signals' USED portion shifts slightly
  (both designs already use the same fixed 48-bit datapath declared
  width, so this is a small change in how much of that width synthesis
  actually needs to implement, not a width change).
- **DSP and BRAM usage are IDENTICAL (0 for both)** -- Q.20 does not
  push any part of this datapath into different primitive usage.
- **Timing margin remains comfortable at both formats** for both
  kernels -- WNS stays above +2.3 ns (i.e. more than 23% of the 10 ns
  clock period free) in all four cases, and WHS is unchanged (+0.262 ns)
  in all four cases. Kernel 0's margin shrinks slightly (2.781 ->
  2.535 ns) while kernel 2's margin slightly IMPROVES (2.390 ->
  2.462 ns) -- neither change is close to threatening 100 MHz closure.
- **Power increases by ~1.2%** (0.002 W) for both kernels -- negligible,
  within the normal run-to-run noise band of Vivado's vector-less power
  estimator.

**Conclusion: Q.20 is essentially free at this scale.** The
precision-sensitivity study's accuracy improvement (Q.16 -> Q.20 cut
max-abs-diff-vs-float by roughly 4-40x for these two kernels) comes at a
cost of a few dozen extra LUTs and a couple of extra registers per
kernel -- not a meaningfully different resource or timing picture.

## Recommendation

**Adopt Q.20 as the fixed-point format for any future real-tile-targeted
implementation of this kernel0/kernel2 datapath** (or its eventual
32-output generalization). The accuracy case for Q.20 over Q.16
(established in the precision-sensitivity report) is not offset by any
meaningful synthesis cost: LUTs, registers, DSPs, BRAM, and timing
margin all remain effectively unchanged. This recommendation reinforces,
rather than revises, the precision-sensitivity report's own
recommendation -- this synthesis comparison removes "it might cost more
hardware" as a reason not to adopt Q.20.

Q.24 was not synthesized here, since the precision-sensitivity study
already found it gives no further accuracy benefit over Q.20 for these
two kernels; there is no reason to expect it would synthesize to a
meaningfully different resource/timing picture than Q.20 either, but
this was not measured.

## Limitations

- **Kernel 0 and kernel 2 only**, one real tile
  (`tile_16_42.tif`), one 6x6 sub-block location -- identical scope
  limits to every prior real-tile report in this series.
- **Out-of-context synthesis only** (`synth_design -mode out_of_context`),
  not placed-and-routed, not board-tested. WNS/WHS/power are Vivado
  synthesis-level estimates, not measurements from a programmed device.
- **Power is Vivado's vector-less, "estimate" mode**, not a measurement.
- **Q.24 was not synthesized** for this comparison (see recommendation
  above for why).
- **DSP-aware multiplication was not used** -- both formats use the
  existing `conv3x3_dot_pipelined` LUT-only multiplier (0 DSPs used in
  all four designs), matching the existing kernel0/kernel2 pipelined
  designs' convention; a DSP-mapped variant was not synthesized or
  compared here.
- **This does not extend to the complete 32-output first-layer design**
  -- only the same 2-kernel real-tile verification subset used
  throughout this series.
- **No board testing, no measured speedup, no measured board power.**
