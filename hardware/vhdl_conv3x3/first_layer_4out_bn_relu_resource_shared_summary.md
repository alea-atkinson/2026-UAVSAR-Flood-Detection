# Resource-Shared Folded Conv-BN-ReLU Prototype: Implementation Summary

Date: July 9, 2026

This document reports the implemented prototype for the design plan in
`resource_shared_first_layer_conv_bn_relu_plan.md`. **This is a
4-of-32-output prototype, not the complete 32-channel resource-shared
design** -- see Section headers below for exactly what was built and
measured versus what remains an estimate. No board testing, no measured
speedup, and no measured board power are claimed anywhere in this
document. The canonical 5x5x3 toy input is used throughout; this is not
a real UAVSAR inference example.

## 1. Purpose

The design plan (`resource_shared_first_layer_conv_bn_relu_plan.md`)
proposed a resource-shared (time-multiplexed) architecture to address
the DSP saturation observed in the direct-parallel 32-output first
Conv2d design (740/740 DSPs, 100% utilization,
`first_layer_32out_dsp_200t_summary.md`). This document implements and
measures the smallest serious prototype of that architecture: a
resource-shared, folded, Q.16 fixed-point Conv-BN-ReLU design covering 4
output kernels, to validate the core resource-sharing claim before
committing to a full 32-kernel build.

## 2. What was implemented

**4 of 32 output kernels (kernels 0-3)**, chosen as the smallest kernel
count that still exercises the complete resource-sharing pattern:
multiple kernels sharing both (a) one time-multiplexed dot-product engine
and (b) one time-multiplexed Q.16 fixed-point BN+ReLU unit. This is
**Deliverable B ("implement if feasible... a smaller validated
prototype")** from the task, not the full 32-kernel Deliverable B
target -- the design plan (Deliverable A) documents how to scale this
same architecture to 32 kernels without structural changes (Sections 12,
13, 18 of the plan).

## 3. Architecture chosen

Three levels, extending the existing, unmodified
`conv3x3_dot_time_mux.vhd` and the existing pattern established by
`conv3x3_3chan_8out_time_mux.vhd` / `stream_conv3x3_3chan_8out_time_mux.vhd`
(raw convolution only, 8 kernels, no BN/ReLU):

1. **`conv3x3_dot_time_mux`** (existing, unmodified) -- ONE 8x8 signed
   multiplier, time-shared over 9 taps per dot product.
2. **`conv3x3_3chan_4out_bn_relu_time_mux`** (new) -- reuses the ONE
   Level-1 instance sequentially across 4 kernels x 3 channels = 12
   (kernel, channel) steps for a single captured window. After each
   kernel's third channel completes, the raw INT32 sum is passed through
   a **second shared resource**: ONE Q.16 fixed-point multiply-add-ReLU
   unit (the same two-stage structure verified in
   `stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd`), reused once per
   kernel -- not once per output and not 4 in parallel.
3. **`stream_conv3x3_3chan_4out_bn_relu_time_mux`** (new) -- streaming
   wrapper: 3 unmodified `window3x3_stream` instances capture all 9
   valid windows of the 5x5 toy image (phase 1), then the single shared
   Level-2 engine processes the 9 buffered windows sequentially,
   non-overlapped (phase 2) -- directly extending
   `stream_conv3x3_3chan_8out_time_mux.vhd`'s two-phase pattern.

Total unique compute resources: **one 8x8 multiplier, one Q.16
fixed-point multiply-add-ReLU unit**, reused across all 4 kernels and all
9 windows.

## 4. Scheduling/output order

**Window-major, kernel-minor** (Section 11 of the design plan): each
`valid_out` pulse carries all 4 kernels' outputs for ONE window, before
advancing to the next window, in the same row-major window order
`window3x3_stream` already produces. This matches the existing 8-output
raw time-mux design's output order and the direct-parallel 32-output
design's output order, so results are directly, position-by-position
comparable.

## 5. Latency

**Measured directly by the testbench (not asserted as a precomputed
figure)**: **1,305 total clock cycles** for the full 5x5x3 toy image (25
capture cycles + 9 windows processed sequentially, averaging ~142 cycles
per window). This is close to the design plan's own estimate (Section
12/13: ~140 cycles/window, ~1,285 cycles total for a 4-kernel scale-down
of the 32-kernel formula) -- confirming the plan's latency-scaling
reasoning was directionally accurate.

For comparison: the existing raw-only 8-kernel time-mux design measured
2,421 cycles for the same image (`stream_time_mux_8out_summary.md`); the
direct-parallel design needs only a handful of cycles once its pipeline
is primed. The latency cost of resource sharing is real and large, as
expected.

## 6. Golden generation method

New script `scripts/generate_resource_shared_32out_bn_relu_vectors.py
--num-kernels 4` (generalizes the existing per-kernel BatchNorm-folding
and Q.16 fixed-point methodology from `generate_bn_relu_fixed_point_vectors.py`
to a full kernel *set* in one run): loads the same trained checkpoint,
extracts `enc1.block.0` weights and `enc1.block.1` BatchNorm parameters
for kernels 0-3, folds BN into each kernel, quantizes each kernel's
folded weights to INT8 (symmetric, per-output-channel), computes
`SCALE_FX`/`BIAS_FX` per kernel (Q.16), computes `raw_conv_int32`,
`product_fx`, `biased_fx`, `relu_fx` for all 9 window positions x 4
kernels (36 total), and writes both JSON/CSV/MD test vectors and a VHDL
package (`first_layer_4out_bn_relu_resource_shared_pkg.vhd`) with a
window-major/kernel-minor flat `EXPECTED_RELU_FX` array. Kernel 0's and
kernel 2's individual values were cross-checked and found bit-identical
to the earlier per-kernel scripts' published values (e.g. kernel 0
window 1: 1522809; kernel 2 window 1: 573008) -- confirming this new
generalized script reproduces the exact same numbers as the
already-verified per-kernel pipeline.

**0 of 36 outputs clamp to zero** for this specific toy input (same
limitation already documented for kernel 0/kernel 2 individually: the
toy patch's raw conv magnitudes are large enough that `biased_fx` stays
positive for all four kernels' folded biases here).

## 7. GHDL result

`bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared.sh`
(VHDL-2008, GHDL 5.1.1):

- Analyzed (dependency order): `window3x3_stream.vhd` (existing,
  unmodified), `conv3x3_dot_time_mux.vhd` (existing, unmodified),
  `first_layer_4out_bn_relu_resource_shared_pkg.vhd` (new),
  `conv3x3_3chan_4out_bn_relu_time_mux.vhd` (new),
  `stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd` (new),
  `tb_stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd` (new) -- all OK.
- Elaborated and simulated -- **all 36 outputs (9 windows x 4 kernels)
  matched the exact Q.16 fixed-point golden values**, e.g. window 1:
  `y0=1522809 y1=1988771 y2=573008 y3=288688` (matching the golden array's
  first 4 entries exactly).
- Final report: `=== All stream_conv3x3_3chan_4out_bn_relu_time_mux
  tests PASSED === (9 windows x 4 kernels = 36 / 36 outputs match Python
  Q.16 fixed-point golden vectors)`.

**Synthesis results below are interpreted as meaningful only because this
GHDL run passed all 36 checks.**

## 8. Regression results

All four required regressions re-run and pass, unmodified:

- `run_ghdl_kernel0_bn_relu_pipelined.sh`: PASS 9/9.
- `run_ghdl_kernel2_bn_relu_pipelined.sh`: PASS 9/9.
- `run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh`: PASS 9/9 (all 9 clamped, as before).
- `run_ghdl_32out_dsp.sh`: PASS 288/288.

No prior design's source file was modified by this task.

## 9. 200T synthesis result

`stream_conv3x3_3chan_4out_bn_relu_time_mux` on `xc7a200tsbg484-1`
(`synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_200t.tcl`):

| Resource | Used | Available | Util% |
|---|---:|---:|---:|
| LUTs | 1,500 | 134,600 | 1.11% |
| Registers | 3,180 | 269,200 | 1.18% |
| DSP48E1 | **2** | 740 | **0.27%** |
| Block RAM tiles | 0 | 365 | 0.00% |

- WNS (setup slack): **+0.597 ns** -- the 100 MHz / 10.000 ns clock
  constraint **is met**, though with much less margin than the
  direct-parallel design's +2.343 ns.
- WHS (hold slack): +0.262 ns.
- Total on-chip power (estimate): **0.148 W** (Dynamic 0.025 W, Static
  0.122 W).
- **`synth_design` completed successfully with 0 errors, 0 critical
  warnings, 32 non-critical warnings.** All 32 warnings are of the
  benign "unused sequential element ... will be removed" class:
  31 of them report specific unused bits of the deliberately
  generously-sized 48-bit `product_fx_r` register (bits 17-47 -- this
  toy input's numeric range never needs the full 48-bit headroom that
  was provisioned for consistency with kernel 0/kernel 2's own 48-bit
  choice, so Vivado trims the provably-constant upper bits during
  optimization), and 1 is the same benign
  `FSM_onehot_state_reg[2] unused` note already documented as benign
  elsewhere in this repo (`stream_time_mux_8out_summary.md`). None of
  these warnings indicate a correctness or timing problem.
- The weight LUT (12 folded INT8 kernels across `CH0/CH1/CH2_WEIGHT_LUT`)
  was mapped to **LUT-ROM/distributed logic, not BRAM** (0 Block RAM
  tiles used) -- small enough at this 4-kernel scale that Vivado did not
  choose BRAM.

## 10. Comparison to direct-parallel 32-output result

| Metric | Direct-parallel 32-out (measured) | **Resource-shared 4-out (measured, this task)** |
|---|---:|---:|
| GHDL | PASS 288/288 | PASS 36/36 |
| LUTs | 10,461 | **1,500** |
| Registers | 4,618 | **3,180** |
| DSP48E1 | **740/740 (100.00%)** | **2/740 (0.27%)** |
| BRAM | 0 | 0 |
| WNS | +2.343 ns | +0.597 ns |
| WHS | +0.262 ns | +0.262 ns |
| Total power (estimate) | 1.324 W | 0.148 W |
| Warnings | 1 (DSP overutilized) | 32 (benign, unused-bit trimming) |
| Kernel count | 32 (complete layer) | 4 |
| Latency (5x5x3 toy input) | ~4-5 cycles (pipelined) | 1,305 cycles (measured) |

**This is not a like-for-like comparison at equal kernel count** -- the
direct-parallel row covers all 32 kernels, the resource-shared row covers
4. The DSP result is nonetheless the central, clearly interpretable
finding: even accounting for the 8x-fewer-kernels difference, going from
740 DSPs (32 kernels, direct-parallel) to 2 DSPs (4 kernels,
resource-shared) is not proportional -- it is qualitative. Direct-parallel
DSP demand scales with kernel count (roughly 23-25 DSPs/kernel, per the
8/16/24/32-output DSP-aware series); resource-shared DSP demand does
**not** scale with kernel count at all in this architecture (the same 1-2
DSP48E1 slices would be reused whether this design covered 4 kernels or
32), because only ONE dot-product engine and ONE BN+ReLU unit exist,
cycled through sequentially.

## 11. Interpretation

- **A resource-shared first-layer architecture was designed to address
  the DSP saturation observed in direct-parallel 32-output mapping**
  (Sections 1-2, and the accompanying design plan).
- **The design trades latency for reduced hardware resource pressure**:
  1,305 cycles for 4 kernels x 9 windows here, versus a few cycles for
  32 kernels x 9 windows in the direct-parallel design -- a large,
  explicit tradeoff, not a free lunch.
- **The resource-shared prototype matched Python Q.16 fixed-point golden
  outputs for the tested toy input**: 36/36 exact matches in GHDL.
- **The resource-shared prototype met 100 MHz synthesis timing on
  Artix-7 200T** (WNS = +0.597 ns), though with substantially less
  margin than the direct-parallel design, and using only 2 DSP48E1
  slices (0.27%) versus the direct-parallel design's 740/740 (100%).
- This provides direct evidence, at reduced scale, that the core
  architectural claim of the design plan holds: DSP demand under this
  resource-sharing strategy does not grow with kernel count the way it
  does under direct-parallel mapping. It does **not** yet prove the full
  32-kernel design would synthesize cleanly, meet timing with the same
  margin, or scale the weight-LUT resource (BRAM vs. LUT-ROM) the same
  way -- those remain open questions for the next scaling step (see
  Limitations and Recommended next step).

## 12. Limitations

- **4 of 32 output kernels only.** This is not the complete first Conv2d
  layer, and does not claim to be. The design plan targets 32; this
  prototype validates the architecture at 1/8th that scale.
- **Toy input only, and specifically one that does not exercise the
  ReLU clamp branch** (0/36 outputs clamp for this input, same
  limitation as kernel 0/kernel 2's individual tests) -- functional
  correctness of the positive-passthrough path is verified here; the
  ReLU clamp branch is separately verified (for a different, single
  -kernel design) by `first_conv_bn_relu_relu_clamp_test_summary.md`,
  not re-verified in this resource-shared context.
- **Not a real UAVSAR inference example.** The canonical 5x5x3 toy
  patch, not a real tile.
- **Latency scaling to larger tiles or to 32 kernels is an estimate**
  (design plan Sections 12-13), not yet measured. The measured 1,305
  -cycle figure here is specific to 4 kernels and a 5x5 image.
- **Timing margin is much tighter** (+0.597 ns) than the direct-parallel
  design's (+2.343 ns) or the pipelined kernel0/kernel2 designs' (~+2.5
  ns) -- this specific 4-kernel implementation should not be assumed to
  retain positive WNS at a larger kernel count without re-synthesis;
  Vivado's synthesis-level estimate could tighten further as the design
  scales (e.g. wider muxes for a 32-entry weight LUT, more F7/F8 mux
  levels).
- **This does not prove the full 32-kernel resource-shared design will
  synthesize, meet timing, or use proportionally similar DSP counts** --
  only that the architecture is functionally correct and synthesizes
  cleanly (with a comfortable, if reduced, timing margin) at 4-kernel
  scale.
- No padding, pooling, downstream layers, or full model pipeline. The
  full U-Net does not fit and was not evaluated; end-to-end flood
  segmentation was not run on FPGA.
- No board testing, no measured speedup, no measured board power. All
  figures are Vivado `synth_design`/GHDL simulation results, never
  measurements from real hardware.

## 13. Recommended next step

Per the design plan's own recommended path (Section 18): scale this same
architecture's kernel count from 4 toward the full 32 (widening the
`NUM_KERNELS` generic-equivalent constants, the weight LUTs, and the step
counter -- no structural change to the FSM or the two shared engines),
and re-measure actual DSP/LUT/register/timing figures at that full
scale, to replace the design plan's Section 12-14 estimates with
measured results and to confirm whether the +0.597 ns timing margin
observed here survives the larger weight-LUT/mux width a 32-kernel
version would require.
