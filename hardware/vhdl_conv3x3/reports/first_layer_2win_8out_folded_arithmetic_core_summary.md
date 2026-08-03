# First-Layer 2-Window x 8-Output Spatial-Parallelism Arithmetic Core: GHDL + Vivado Results

Date: August 3, 2026

This report covers a NEW hardware artifact that tests a DIFFERENT
resource trade than the existing 16-output comfortable subproblem
(commit `fe79e096`): instead of computing 16 output channels for ONE
spatial window per cycle, this core computes only 8 output channels but
for TWO independent spatial windows per cycle. Both designs compute the
SAME total number of output-channel results per cycle (16) -- the
question is whether the Artix-7 200T supports this reshaped 2-lane
architecture without hitting the 32-output design's 100%-DSP wall.

## Claim boundary (read this first)

- **Arithmetic-core-only, pre-extracted-windows scope.** Both spatial
  lanes consume ALREADY-FLATTENED 27-value windows. **No 2-pixel/cycle
  streaming front end was built for this core.** That is explicitly
  future work, not part of this task -- there is no line-buffer/windowing
  logic anywhere in this artifact.
- **NOT a full first layer.** Each lane computes only 8 of the first
  Conv2d layer's 32 output channels.
- **NOT full U-Net FPGA inference.**
- **NOT board-measured.** GHDL simulation + out-of-context Vivado
  synthesis estimates only.
- **Does NOT claim FPGA is faster than GPU** -- no GPU comparison is made
  in this report (see Section 9 of the accompanying interpretation for
  why this is not directly comparable to the 16-output GPU benchmark).
- Power figures are Vivado's vectorless, Medium-confidence synthesis-time
  estimate, not silicon-measured.
- **No existing file was modified.** The 16-output core/package and the
  32-output core/package are all reused read-only or left untouched;
  every new file introduced by this artifact uses a distinct
  `2win_8out`/`8out`-qualified name.

## 1. What was built

| File | Purpose |
|---|---|
| `scripts/generate_first_layer_8out_folded_arithmetic_core_pkg.py` | Slices the existing, already-verified 32-kernel merged Q.20 package to channels 0-7; cross-checks the slice against the existing per-kernel real-tile Q.20 packages |
| `hardware/vhdl_conv3x3/first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd` | 8-kernel Q.20 weight/scale/bias package (generated) |
| `hardware/vhdl_conv3x3/first_layer_2win_8out_folded_arithmetic_core.vhd` | NEW 2-window x 8-output spatial-parallelism arithmetic core |
| `hardware/vhdl_conv3x3/tb_first_layer_2win_8out_folded_arithmetic_core.vhd` | NEW self-checking, real-UAVSAR-activation GHDL testbench |
| `hardware/vhdl_conv3x3/run_ghdl_2win_8out_folded_arithmetic_core.sh` | GHDL run script |
| `hardware/vhdl_conv3x3/synth_first_layer_2win_8out_folded_arithmetic_core_200t.tcl` | Vivado synthesis script, `xc7a200tsbg484-1`, 100 MHz |
| `hardware/vhdl_conv3x3/vivado_reports_2win_8out_folded_arithmetic_core_200t/` | Utilization/timing/power text reports |
| `outputs/hardware_benchmarks/first_layer_2win_8out_spatial_parallelism/spatial_parallelism_comparison.csv` | Machine-readable comparison table (32-out, 16-out, 2win-8out) |
| `outputs/hardware_benchmarks/first_layer_2win_8out_spatial_parallelism/summary.md` | Short companion summary |

### What was reused, unmodified

- `conv3x3_dot_pipelined_dsp.vhd` -- the SAME DSP-steered 3x3 dot-product
  IP already used by the 16-output and 32-output cores, instantiated 48
  times (2 lanes x 8 kernels x 3 channels) -- the SAME total instance
  count as the 16-output core's 48 (16 kernels x 3 channels), just
  reorganized into 2 lanes.
- `real_tile_stimulus_pkg.vhd` -- the SAME real 6x6 UAVSAR-tile-derived
  pixel block (`tile_16_42.tif`, sub-block at row=125, col=125) already
  used by every other real-tile testbench in this repo.
- `first_conv_bn_relu_kernel{0..7}_real_tile_q20_pkg.vhd` -- the SAME 8
  (of 32) existing, independently-generated, already-GHDL-verified
  per-kernel real-tile Q.20 golden-vector packages already used by the
  direct-parallel single-kernel, 16-output streaming, and 32-output
  resource-shared designs.
- All weight/scale/bias constants in the new 8-kernel package are copied
  verbatim from the existing, already-verified 32-kernel merged package
  (`first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd`) -- **no new
  weight, scale, or bias value was invented anywhere in this task.** The
  generator script additionally cross-checked the slice against the
  independently-generated per-kernel packages before writing the file
  (Section 3).

**The existing 16-output core, its package, and the 32-output core and
its package were all left completely unmodified.**

## 2. Architecture

For each spatial lane L in {0, 1}, and each of 8 output kernels k
(identical per-kernel 6-cycle pipeline structure to the 16-output and
32-output cores: 3-cycle `conv3x3_dot_pipelined_dsp` raw dot product,
then registered sum/Q.20-scale/bias-ReLU stages), both lanes run fully
in parallel, sharing `clk`/`rst`/`valid_in` (both windows of a pair are
always presented and consumed in lockstep). 2 lanes x 8 kernels x 3
channels = 48 `conv3x3_dot_pipelined_dsp` instances total -- the SAME
total instance count as the 16-output core's 48, just reorganized from
"1 lane of 16 kernels" into "2 lanes of 8 kernels." Latency: 6 cycles
(valid_in to valid_out), identical to the 16-output and 32-output cores.

## 3. Consistency check (before any hardware was built)

`scripts/generate_first_layer_8out_folded_arithmetic_core_pkg.py` parsed
the existing, already-verified 32-kernel merged package, sliced out
channels 0-7, and cross-checked every sliced weight/`SCALE_FX`/`BIAS_FX`
value against the existing, independently-generated
`first_conv_bn_relu_kernel{0..7}_real_tile_q20_pkg.vhd` files:

```
OK: all 8 kernels' weights/SCALE_FX/BIAS_FX in the 32-kernel-package
slice are bit-for-bit identical to the existing, independently-generated
per-kernel real-tile Q.20 packages.
```

## 4. GHDL results

`bash hardware/vhdl_conv3x3/run_ghdl_2win_8out_folded_arithmetic_core.sh`
(VHDL-2008, GHDL 5.1.1):

```
PASS pair 1 (widx0=0 widx1=1) lane0_y0=0 lane1_y0=0
PASS pair 2 (widx0=2 widx1=3) lane0_y0=0 lane1_y0=1720531
PASS pair 3 (widx0=4 widx1=5) lane0_y0=88303 lane1_y0=0
PASS pair 4 (widx0=6 widx1=7) lane0_y0=0 lane1_y0=0
PASS pair 5 (widx0=8 widx1=9) lane0_y0=654700 lane1_y0=0
PASS pair 6 (widx0=10 widx1=11) lane0_y0=0 lane1_y0=0
PASS pair 7 (widx0=12 widx1=13) lane0_y0=2501662 lane1_y0=0
PASS pair 8 (widx0=14 widx1=15) lane0_y0=0 lane1_y0=0
=== All tb_first_layer_2win_8out_folded_arithmetic_core tests PASSED ===
  (8 pairs x 2 lanes x 8 kernels = 128 / 128 outputs match the SAME
  real-tile Q.20 golden vectors already used by the direct-parallel
  single-kernel, 16-output streaming, and 32-output resource-shared
  verifications)
```

**PASS count: 128/128 outputs match** (8 window pairs x 2 lanes x 8
kernels), exactly matching the task's example calculation. Windows were
pre-extracted directly from the real 6x6 UAVSAR-tile-derived block (the
same block used throughout this repo's real-tile verification work),
grouped into 8 consecutive row-major pairs, and driven 2 windows/cycle --
no streaming front end was used or built.

**Cross-architecture consistency:** lane 1's `y0=1720531` at window index
3 (pair 2) matches window 4 (1-based)'s `y0=1720531` result from the
16-output streaming testbench exactly (same real window, same kernel 0)
-- an independent tie-back confirming the 2-window x 8-output core
computes the identical Q.20 result as the 16-output and 32-output cores
for the channels/windows they share.

## 5. Vivado synthesis results (`xc7a200tsbg484-1`, 100 MHz)

Command:
```bash
module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_2win_8out_folded_arithmetic_core_200t.tcl
```

**`synth_design` completed with 0 errors, 0 critical warnings. All
user-specified timing constraints are met at 100 MHz.**

| Metric | 2-window x 8-output core (this run) |
|---|---:|
| LUTs | 1,888 (1.40%) |
| Registers | 1,254 (0.47%) |
| DSP48E1 | **406 / 740 (54.86%)** |
| Block RAM tiles | 0 (0.00%) |
| WNS (setup slack) | **+2.218 ns** |
| WHS (hold slack) | +0.262 ns |
| Total on-chip power (est.) | 0.630 W (dynamic 0.505 W + static 0.124 W) |
| Warnings | 2 (floorplanning-hierarchy notice for the flat generate-loop netlist; out-of-context clock-skew notice) |

Full text reports:
`hardware/vhdl_conv3x3/vivado_reports_2win_8out_folded_arithmetic_core_200t/
{first_layer_2win_8out_folded_arithmetic_core_utilization.txt,
..._timing_summary.txt, ..._power.txt}`.

**No DSP pre-fallback demand/overutilization warning appeared** -- like
the 16-output design, this core's DSP demand never approaches the part's
740-DSP ceiling.

## 6. Comparison against the 16-output 1-window and 32-output 1-window designs

Sources: `hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md`
(core-only row) and `hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md`.

| Design | Windows/cycle | Channels/window | Total channel-computations/cycle | LUTs | Registers | DSP48E1 | WNS (ns) | Power (W) | GHDL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32-out, 1 window | 1 | 32 | 32 | 12,039 (8.94%) | 6,564 (2.44%) | **740/740 (100.00%)** | +2.218 | 1.334 | 160/160 |
| 16-out, 1 window | 1 | 16 | 16 | 1,796 (1.33%) | 1,238 (0.46%) | 405/740 (54.73%) | +2.218 | 0.624 | 256/256 |
| **2-window x 8-out (this artifact)** | **2** | **8** | **16** | **1,888 (1.40%)** | **1,254 (0.47%)** | **406/740 (54.86%)** | **+2.218** | **0.630** | **128/128** |

Observations:

- **DSP usage is essentially identical to the 16-output design** (406 vs.
  405 -- a difference of exactly 1 DSP, i.e. 0.14 percentage points),
  confirming the prediction that reshaping the SAME total 16
  output-channel computations/cycle from "1 window x 16 channels" into "2
  windows x 8 channels" does not change the DSP budget: both designs
  instantiate 48 `conv3x3_dot_pipelined_dsp` copies, and Vivado maps them
  almost identically regardless of how they are grouped into
  lanes/kernels.
- **LUTs and registers are very slightly higher** (+92 LUTs, +5.1%; +16
  registers, +1.3%) than the 16-output design -- a small, plausible cost
  of the 2-lane array-indexed wiring/generate structure, not a
  qualitatively different resource profile.
- **Timing margin is IDENTICAL** (WNS = +2.218 ns, matching the 16-output
  AND the 32-output cores exactly) -- reshaping into 2 spatial lanes did
  not create any new critical path or timing risk on this target.
- **Both the 16-output and 2-window-x-8-output designs leave the SAME
  ~45% DSP headroom** (405/740 and 406/740 respectively) that the
  32-output design (740/740, zero headroom) does not have.

## 7. Throughput interpretation

Both the 16-output and 2-window-x-8-output cores compute the SAME total
16 output-channel results per cycle -- the difference is how that budget
is spatially organized:

| Design | Windows/cycle (theoretical, if fed every cycle) | Channels/window | Effective MACs/cycle |
|---|---:|---:|---:|
| 16-out, 1 window | 1 | 16 | 1 x 16 x 27 = 432 |
| 2-window x 8-out | 2 | 8 | 2 x 8 x 27 = 432 |

**At 100 MHz** (the SAME synthesis-supported clock target used
throughout this repo, contingent on this core actually meeting 100 MHz
timing, which Section 5 confirms):

- **16-output design**: 1 window/cycle x 100 MHz = **100,000,000
  windows/sec** (theoretical, arithmetic-core-only, assuming an ideal
  1-window/cycle input supply -- the 16-output design's own INTEGRATED
  STREAMING front end has already been synthesized and confirmed to
  supply this rate, per commit `fe79e096`).
- **2-window x 8-output design**: 2 windows/cycle x 100 MHz = **200,000,000
  windows/sec** (theoretical arithmetic-core throughput ONLY -- **no
  matching 2-pixel/cycle streaming front end has been built or
  synthesized for this core**, so this number describes the core's
  maximum throughput GIVEN an ideal, not-yet-built input supply, not a
  demonstrated system-level rate).
- **Effective MACs/sec, both designs**: `windows/sec x channels/window x
  27 MACs/output`:
  - 16-out: `100e6 x 16 x 27 = 43.2 GMAC/s`
  - 2-window x 8-out: `200e6 x 8 x 27 = 43.2 GMAC/s`
  - **Identical** -- confirming this is the SAME total arithmetic
    throughput, just reorganized from "fewer, wider windows" to "more,
    narrower-channel windows." The 2-window design does NOT do more total
    arithmetic per cycle; it changes WHICH axis (channel count vs.
    spatial window count) is parallelized for the same DSP budget.
- **32-output design (for context)**: `100e6 x 32 x 27 = 86.4 GMAC/s` --
  double the arithmetic throughput of either comfortable subproblem, but
  achieved by using 100% (740/740) of the part's DSPs with zero headroom,
  versus ~55% for either comfortable design.

## 8. Interpretation: does spatial parallelism avoid the DSP saturation wall?

**Yes, for 2 lanes.** The Artix-7 200T supports at least 2 spatial lanes
of this real trained first-layer subproblem (8 output channels each)
without approaching the 32-output design's 100%-DSP wall: DSP usage
(406/740, 54.86%) and timing margin (WNS = +2.218 ns) are essentially
identical to the single-lane 16-output design's own already-comfortable
result. This is a genuine, synthesis-confirmed data point supporting the
claim that output-channel parallelism can be traded for spatial-window
parallelism on this part, for this real-model subproblem, at this scale
(2 lanes) -- **it does NOT establish how many additional lanes could be
added before hitting a resource or timing wall; that was not tested
here.**

## 9. What this does and does not say about the GPU comparison

**This report does NOT compare against the 16-output GPU benchmark
(`outputs/hardware_benchmarks/first_layer_16out_gpu_vs_fpga/`).** That
benchmark measured GPU throughput for computing **16** output channels
per window; this artifact's arithmetic core computes only **8** output
channels per window (per lane). Directly comparing this core's raw
windows/sec against that benchmark's GPU windows/sec would silently
compare two DIFFERENT amounts of arithmetic work per window and
overstate or understate either side. **The real, fair comparison enabled
by this artifact is an FPGA-internal ARCHITECTURE TRADEOFF**:
output-channel parallelism (16-output, 1 window/cycle) vs. spatial-window
parallelism (2-window, 8-output/window) for the SAME total DSP budget and
the SAME total MAC/cycle -- not an FPGA-vs-GPU claim. A normalized
GPU comparison (e.g. GPU throughput for an 8-channel slice, or a
per-MAC GPU throughput figure applied to this core's 43.2 GMAC/s) was
NOT performed in this task and would need explicit normalization to be
meaningful.

## 10. Limitations

- GHDL verification uses the same 6x6 real-tile sub-block (16 valid
  windows, paired into 8) used throughout this repo -- not an exhaustive
  sweep of all possible INT8 input combinations, and not a full 256x256
  tile.
- **No streaming front end exists for this core.** The 200M windows/sec
  theoretical throughput figure (Section 7) assumes an ideal 2-window/cycle
  input supply that has not been built, synthesized, or verified. Building
  a matching 2-pixel/cycle line-buffer front end is explicitly future
  work.
- Real-tile input scale (`scale_x`) is inherited from one specific tile
  (`tile_16_42.tif`); a different tile's activation range would change
  `SCALE_FX`/`BIAS_FX` (though not the raw INT8 weights).
- Power estimates are Vivado's vectorless (Medium-confidence) estimate at
  `out_of_context` synthesis, not a place-and-route or board measurement.
- This is 8 of the first Conv2d layer's 32 output channels per lane only
  -- not the full first layer, not the full U-Net.
- No board was used or is claimed to have been used anywhere in this
  report.
- Only 2 spatial lanes were tested; whether 3+ lanes remain feasible on
  this part was not investigated.

## 11. Commands run

```bash
python3 scripts/generate_first_layer_8out_folded_arithmetic_core_pkg.py

bash hardware/vhdl_conv3x3/run_ghdl_2win_8out_folded_arithmetic_core.sh

module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"

vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_2win_8out_folded_arithmetic_core_200t.tcl
```
