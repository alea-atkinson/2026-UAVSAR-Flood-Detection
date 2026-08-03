# First-Layer 16-Output Folded Arithmetic Core + Streaming Front End: A Comfortably Fitting FPGA Subproblem

Date: August 3, 2026

This report covers a NEW, concrete hardware artifact built to answer a
question the existing 32-output design could not: `first_layer_32out_folded_arithmetic_core.vhd`
(commit `313558e4`) computes all 32 real, trained first-layer
Conv-BN-ReLU output channels in parallel and saturates the Artix-7 200T
target at exactly 740/740 (100%) DSP48E1 slices -- zero headroom. This
artifact computes only the first **16** of those 32 real output channels
(same checkpoint, same real weights, same folded BatchNorm + ReLU, same
Q.20 fixed-point convention) to test whether a smaller, still-useful
real-model first-layer subproblem leaves real DSP headroom on the SAME
part while still meeting 100 MHz timing and streaming from a real pixel
input.

## Claim boundary (read this first)

- **First-layer folded Conv-BN-ReLU scope only.** This is one
  Conv2d+BatchNorm2d+ReLU stage, 16 of its 32 output channels. It is
  **NOT full U-Net FPGA inference**, not a full first-layer implementation
  (channels 16-31 are not computed by this artifact), no downstream
  layers.
- **NOT board-measured.** GHDL simulation + out-of-context Vivado
  synthesis estimates only -- no board testing, no measured hardware
  speedup, no measured board power anywhere in this report.
- **This does NOT claim FPGA is faster than GPU.** No GPU comparison is
  made in this report.
- Power figures are Vivado's vectorless, Medium-confidence synthesis-time
  estimate, not silicon-measured.
- **No existing file was modified.** The 32-output core, its package, the
  32-output streaming wrapper, and `window3x3_stream_3chan_flattened.vhd`
  are all reused byte-for-bit unchanged or referenced read-only; every new
  file introduced by this artifact uses a distinct `16out`-qualified name.

## 1. What was built

| File | Purpose |
|---|---|
| `scripts/generate_first_layer_16out_folded_arithmetic_core_pkg.py` | Slices the existing, already-verified 32-kernel merged Q.20 package down to channels 0-15; cross-checks the slice against the existing per-kernel real-tile Q.20 packages |
| `hardware/vhdl_conv3x3/first_layer_16out_folded_bn_relu_real_tile_q20_pkg.vhd` | 16-kernel Q.20 weight/scale/bias package (generated) |
| `hardware/vhdl_conv3x3/first_layer_16out_folded_arithmetic_core.vhd` | NEW single-shot 16-output folded Conv-BN-ReLU arithmetic core |
| `hardware/vhdl_conv3x3/first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd` | NEW integrated wrapper: existing streaming front end + new 16-output core |
| `hardware/vhdl_conv3x3/tb_first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd` | NEW self-checking, real-UAVSAR-activation GHDL testbench |
| `hardware/vhdl_conv3x3/run_ghdl_16out_folded_arithmetic_core_streaming_front_end.sh` | GHDL run script |
| `hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_200t.tcl` | Vivado synthesis script, core only, `xc7a200tsbg484-1`, 100 MHz |
| `hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl` | Parameterized Vivado synthesis script (`-tclargs <img_width> <report_dir>`), integrated design |
| `hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_200t/` | Utilization/timing/power reports, core only |
| `hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width128_200t/` | Utilization/timing/power reports, integrated, IMG_WIDTH=128 |
| `hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width256_200t/` | Utilization/timing/power reports, integrated, IMG_WIDTH=256 |
| `outputs/hardware_benchmarks/first_layer_16out_comfortable_fpga_subproblem/synthesis_comparison.csv` | Machine-readable comparison table (16-out vs. 32-out, all widths) |
| `outputs/hardware_benchmarks/first_layer_16out_comfortable_fpga_subproblem/summary.md` | Short companion summary |

### What was reused, unmodified

- `conv3x3_dot_pipelined_dsp.vhd` -- the SAME DSP-steered 3x3 dot-product
  IP already used by the 32-output core, just 48 instances (16 kernels x 3
  channels) instead of 96.
- `window3x3_stream.vhd` and `window3x3_stream_3chan_flattened.vhd` -- the
  SAME streaming front end already used by the 32-output integrated
  skeleton (commit `77a71e35`), completely unmodified.
- `real_tile_stimulus_pkg.vhd` -- the SAME real 6x6 UAVSAR-tile-derived
  pixel block (`tile_16_42.tif`, sub-block at row=125, col=125) already
  used by every other real-tile testbench in this repo.
- `first_conv_bn_relu_kernel{0..15}_real_tile_q20_pkg.vhd` -- the SAME
  16 (of 32) existing, independently-generated, already-GHDL-verified
  per-kernel real-tile Q.20 golden-vector packages already used by the
  direct-parallel single-kernel designs and the 32-output resource-shared
  real-tile Q.20 testbench.
- All weight/scale/bias constants in the new 16-kernel package are copied
  verbatim from the existing, already-verified 32-kernel merged package
  (`first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd`) -- **no new
  weight, scale, or bias value was invented anywhere in this task.** The
  generator script additionally cross-checked the slice against the
  independently-generated per-kernel packages before writing the file
  (see Section 3).

**The existing 32-output core, its package, its streaming wrapper, and the
existing streaming front end were all left completely unmodified.**

## 2. Architecture

Identical structure to the 32-output core (same 6-cycle pipeline: 3-cycle
`conv3x3_dot_pipelined_dsp` raw dot product, then registered
sum/scale/bias-ReLU stages), just looped over **16 kernels instead of
32** -- 48 dot-product instances total instead of 96. The integrated
streaming wrapper chains the existing, unmodified
`window3x3_stream_3chan_flattened` front end directly into this new
16-output core, exactly mirroring how the 32-output integrated skeleton
chains the same front end into the 32-output core.

## 3. Consistency check (before any hardware was built)

`scripts/generate_first_layer_16out_folded_arithmetic_core_pkg.py` parsed
the existing, already-verified 32-kernel merged package, sliced out
channels 0-15, and cross-checked every sliced weight/`SCALE_FX`/`BIAS_FX`
value against the existing, independently-generated
`first_conv_bn_relu_kernel{0..15}_real_tile_q20_pkg.vhd` files:

```
OK: all 16 kernels' weights/SCALE_FX/BIAS_FX in the 32-kernel-package
slice are bit-for-bit identical to the existing, independently-generated
per-kernel real-tile Q.20 packages.
```

## 4. GHDL results

`bash hardware/vhdl_conv3x3/run_ghdl_16out_folded_arithmetic_core_streaming_front_end.sh`
(VHDL-2008, GHDL 5.1.1):

```
PASS window 1 y0=0 y1=62052 y15=1142268 (@ cycle 21)
PASS window 2 y0=0 y1=0 y15=754100 (@ cycle 22)
PASS window 3 y0=0 y1=0 y15=0 (@ cycle 23)
PASS window 4 y0=1720531 y1=0 y15=0 (@ cycle 24)
PASS window 5 y0=88303 y1=87693 y15=549852 (@ cycle 27)
PASS window 6 y0=0 y1=995208 y15=905328 (@ cycle 28)
PASS window 7 y0=0 y1=0 y15=744772 (@ cycle 29)
PASS window 8 y0=0 y1=0 y15=343976 (@ cycle 30)
PASS window 9 y0=654700 y1=1682475 y15=240356 (@ cycle 33)
PASS window 10 y0=0 y1=257919 y15=1275060 (@ cycle 34)
PASS window 11 y0=0 y1=125367 y15=652504 (@ cycle 35)
PASS window 12 y0=0 y1=0 y15=538896 (@ cycle 36)
PASS window 13 y0=2501662 y1=756690 y15=0 (@ cycle 39)
PASS window 14 y0=0 y1=37797 y15=0 (@ cycle 40)
PASS window 15 y0=0 y1=999870 y15=117860 (@ cycle 41)
PASS window 16 y0=0 y1=52980 y15=688716 (@ cycle 42)
=== All tb_first_layer_16out_folded_arithmetic_core_streaming_front_end tests PASSED ===
  (16 REAL windows x 16 kernels = 256 / 256 outputs match the SAME
  real-tile Q.20 golden vectors already used by the direct-parallel
  single-kernel and 32-output resource-shared verifications)
```

**PASS count: 256/256 outputs match** (16 real UAVSAR-derived windows x 16
kernels). The design was driven end-to-end by a real, row-major pixel
stream (one triplet per clock, through the unmodified streaming front
end) rather than pre-extracted windows -- the first end-to-end,
streamed-pixel GHDL check of this front-end/core combination pattern in
this repo (the 32-output integrated skeleton was only verified indirectly:
front end alone, and core alone with pre-flattened windows, never
together with real streamed pixels).

**Cross-architecture consistency:** window 1's `y1=62052` here matches
`tb_first_layer_32out_folded_arithmetic_core`'s single-shot
`window 0 y1=62052` result exactly (same real top-left window, same
kernel 1) -- an independent tie-back confirming the 16-output core
computes the identical Q.20 result as the 32-output core for the channels
they share.

## 5. Vivado synthesis results (`xc7a200tsbg484-1`, 100 MHz)

Commands:
```bash
module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_200t.tcl

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl \
  -tclargs 128 hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width128_200t

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl \
  -tclargs 256 hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width256_200t
```

All three runs: **`synth_design` completed with 0 errors, 0 critical
warnings. All user-specified timing constraints are met at 100 MHz.**

| Design | IMG_WIDTH | LUTs | Registers | DSP48E1 | BRAM | WNS (ns) | WHS (ns) | Power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16-out core only | N/A | 1,796 (1.33%) | 1,238 (0.46%) | **405/740 (54.73%)** | 0 | +2.218 | +0.262 | 0.624 |
| 16-out integrated | 128 | 16,444 (12.22%) | 10,701 (3.98%) | **405/740 (54.73%)** | 0 | +2.218 | +0.262 | 0.888 |
| 16-out integrated | 256 | 30,660 (22.78%) | 20,046 (7.45%) | **405/740 (54.73%)** | 0 | +2.218 | +0.262 | 0.940 |

Full text reports: `hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_200t/`,
`.../vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width{128,256}_200t/`
(`*_utilization.txt`, `*_timing_summary.txt`, `*_power.txt`).

**Warnings:** core-only run had 2 (a floorplanning-hierarchy notice specific
to the flat generate-loop netlist, and the same out-of-context
clock-skew notice seen throughout this repo); both integrated (128, 256)
runs had 1 (the clock-skew notice only). **No DSP pre-fallback
demand/overutilization warning appeared in any of the three runs** --
unlike every 32-output synthesis in this repo, which always logged a DSP
demand notice ("Used = 1044, Available = 740") because that design's raw
demand exceeds the part's budget. The 16-output design's DSP demand never
approaches the part's ceiling, so Vivado never needed to reason about a
DSP fallback at all.

## 6. Comparison against the prior 32-output design

Source: `hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md`
and `.../first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_summary.md`.

| Design | Channels | IMG_WIDTH | LUTs | Registers | DSP48E1 | WNS (ns) | Power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core only | 32 | N/A | 12,039 (8.94%) | 6,564 (2.44%) | **740/740 (100.00%)** | +2.218 | 1.334 |
| Core only | **16** | N/A | **1,796 (1.33%)** | **1,238 (0.46%)** | **405/740 (54.73%)** | +2.218 | **0.624** |
| Integrated | 32 | 128 | 30,120 (22.4%) | 16,156 (6.0%) | 740/740 (100.00%) | +2.063 | 1.695 |
| Integrated | **16** | 128 | **16,444 (12.22%)** | **10,701 (3.98%)** | **405/740 (54.73%)** | **+2.218** | **0.888** |
| Integrated | 32 | 256 | 47,542 (35.3%) | 25,585 (9.5%) | 740/740 (100.00%) | +2.063 | 1.745 |
| Integrated | **16** | 256 | **30,660 (22.78%)** | **20,046 (7.45%)** | **405/740 (54.73%)** | **+2.218** | **0.940** |

Observations:

- **DSP usage roughly halves, but not exactly.** 16/32 of the output
  channels would predict 370 DSPs (50%); the measured value is 405
  (54.73%). This is a real, synthesis-measured number, not the naive
  linear estimate -- Vivado's DSP48E1 packing/cascading for the raw
  9-tap dot products is not perfectly proportional to kernel count. The
  headroom is still substantial: **335 DSPs (45.27% of the part)
  unused**, versus **zero** for the 32-output design.
  All three widths tested (core-only, 128, 256) show the identical
  405/740 DSP count -- confirming DSPs are entirely a property of the
  arithmetic core (kernel count), never touched by the streaming front
  end, exactly as already established for the 32-output design.
- **LUTs and registers scale down more than linearly with kernel count**
  at every width (e.g. core-only: 1,796 vs. 12,039 LUTs, a 6.7x reduction
  for a 2x reduction in kernel count) -- consistent with the 32-output
  design's own finding that the BN-fold/ReLU tail stage (not the raw
  dot-product IP) absorbs a disproportionate share of LUT/register cost,
  since that stage's Q.20 multiply-add-compare logic scales with kernel
  count while some fixed control overhead does not.
  Power drops accordingly (nearly half, e.g. 0.624 W vs. 1.334 W core-only).
- **Timing margin is equal or BETTER for the 16-output design.** WNS
  stays at +2.218 ns across ALL THREE 16-output widths tested (6-pixel
  core-only baseline, 128, 256) -- it never drops to +2.063 ns the way the
  32-output design's WNS did once the front end was added at realistic
  widths. This suggests the 32-output design's small timing degradation
  at wider images was tied to loading/fanout pressure from the
  DSP-saturated core, which the 16-output core's lighter DSP footprint
  does not reproduce.
- **The front end's own resource cost is comparable in absolute terms**
  (e.g. at width 128: 16,444 - 1,796 = 14,648 LUTs added by the front end
  for the 16-output design, vs. 30,120 - 12,039 = 18,081 LUTs for the
  32-output design) -- expected, since `window3x3_stream_3chan_flattened`'s
  line buffers are sized by `IMG_WIDTH`, not by kernel count, so this cost
  is largely independent of how many output channels the downstream core
  computes. The two numbers are not identical because total-design LUT
  counts also reflect global synthesis optimization/fanout effects across
  the whole netlist, not solely the front end's own logic in isolation.

## 7. Interpretation: does the 16-output subproblem leave DSP headroom?

**Yes, clearly.** At every image width tested (core alone, 128, 256), the
16-output folded Conv-BN-ReLU streaming design:

- Uses **405/740 (54.73%) DSP48E1 slices** -- **335 DSPs (45.27%) of
  real, measured headroom** remaining on the same Artix-7 200T part that
  the 32-output design saturates completely.
- **Meets 100 MHz timing with positive slack** (WNS = +2.218 ns) at every
  width tested, matching or beating the 32-output design's own margin.
- Uses well under 25% of the part's LUTs and under 8% of its registers
  even at the realistic 256-pixel-wide image, versus up to 35.3%/9.5% for
  the 32-output design.
- Was verified end-to-end (streaming front end + arithmetic core
  together, real UAVSAR pixel data in) by GHDL simulation for the first
  time in this repo's line of 32-output artifacts, at 256/256 matching
  outputs.

This makes the 16-output design a genuinely **comfortably-fitting**
subproblem on this part, in contrast to the 32-output design, which
fits exactly at 100% DSP utilization with no room for anything else on
the same target. It provides a cleaner FPGA-vs-GPU comparison POINT in
the sense that its resource usage is not already at the part's ceiling --
but this report does not perform or claim any GPU comparison.

## 8. Limitations

- GHDL verification uses the same 6x6 real-tile sub-block (16 valid
  windows) used throughout this repo -- not an exhaustive sweep of all
  possible INT8 input combinations, and not a full 256x256 tile.
- Real-tile input scale (`scale_x`) is inherited from one specific tile
  (`tile_16_42.tif`); a different tile's activation range would change
  `SCALE_FX`/`BIAS_FX` (though not the raw INT8 weights).
- Power estimates are Vivado's vectorless (Medium-confidence) estimate at
  `out_of_context` synthesis, not a place-and-route or board measurement.
- This is 16 of the first Conv2d layer's 32 output channels only -- not
  the full first layer, not the full U-Net.
- No board was used or is claimed to have been used anywhere in this
  report.
- The measured 405-DSP figure (vs. a naive 370-DSP linear estimate) was
  not further decomposed per-kernel in this task; it is reported as
  measured, not explained down to individual DSP48E1 mapping decisions.

## 9. Commands run

```bash
python3 scripts/generate_first_layer_16out_folded_arithmetic_core_pkg.py

bash hardware/vhdl_conv3x3/run_ghdl_16out_folded_arithmetic_core_streaming_front_end.sh

module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_200t.tcl

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl \
  -tclargs 128 hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width128_200t

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl \
  -tclargs 256 hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width256_200t
```
