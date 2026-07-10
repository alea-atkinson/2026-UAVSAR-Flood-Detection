# Complete 32-Output Resource-Shared Folded Conv-BN-ReLU: Implementation Summary

Date: July 10, 2026

This document reports the COMPLETE (32-of-32-kernel) resource-shared
folded Conv-BN-ReLU implementation, scaled up from the smaller 4-kernel
prototype (preserved unmodified; see
`first_layer_4out_bn_relu_resource_shared_summary.md`). This is the
complete first Conv2d layer's output-channel scope, but still first-layer
convolution only -- not the full U-Net, not board-tested, no measured
speedup, and no measured board power. The canonical 5x5x3 toy input is
used throughout; this is not a real UAVSAR inference example.

## 1. Purpose

The 4-kernel resource-shared prototype gave strong, but scale-limited,
evidence that a time-multiplexed architecture avoids the DSP saturation
seen in the direct-parallel 32-output design (740/740 DSPs). This task
scales that same architecture -- unchanged FSM logic, unchanged shared
compute resources, only the kernel count widened via package constants
and mechanically-extended port lists -- to the complete 32-kernel first
Conv2d layer, to confirm whether the DSP-sharing result holds at full
scale rather than only at the smaller, easier-to-verify 4-kernel scope.

## 2. What was implemented

**All 32 of 32 output kernels** -- the complete first Conv2d layer.
Three levels, extending the verified 4-kernel design
(`conv3x3_3chan_4out_bn_relu_time_mux.vhd`,
`stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd`, both preserved
unmodified) with `NUM_KERNELS` widened from 4 to 32:

1. **`conv3x3_dot_time_mux`** (existing, unmodified) -- ONE 8x8 signed
   multiplier, time-shared over 9 taps per dot product.
2. **`conv3x3_3chan_32out_bn_relu_time_mux`** (new) -- reuses the ONE
   Level-1 instance sequentially across 32 kernels x 3 channels = 96
   (kernel, channel) steps for a single captured window, plus ONE shared
   Q.16 fixed-point multiply-add-ReLU unit, reused once per kernel (32
   times per window, not 32 in parallel). **No FSM logic changed** from
   the 4-kernel version -- every loop bound (`step` range, `kernel_idx`
   range, `y_regs` array size, the "last kernel" termination check) was
   already expressed in terms of the package's `NUM_KERNELS` constant,
   not a hardcoded "4"; only the port list (`y0..y31` instead of
   `y0..y3`) needed mechanical widening.
3. **`stream_conv3x3_3chan_32out_bn_relu_time_mux`** (new) -- streaming
   wrapper: 3 unmodified `window3x3_stream` instances capture all 9
   valid windows (phase 1), then the single shared Level-2 engine
   processes them sequentially (phase 2), producing all 32 kernels'
   outputs per window pulse.

Total unique compute resources: still **one 8x8 multiplier, one Q.16
fixed-point multiply-add-ReLU unit** -- unchanged from the 4-kernel
design, now reused across 32 kernels instead of 4.

## 3. Golden generation (existing script, no changes needed)

`scripts/generate_resource_shared_32out_bn_relu_vectors.py --num-kernels
32` -- **the exact same script used for the 4-kernel case, unmodified**,
since it was already written to scale via `--num-kernels`. Produces 288
outputs (9 windows x 32 kernels) in the same window-major, kernel-minor
order. Kernels 0's and 2's values were spot-checked and found
bit-identical to the earlier per-kernel and 4-kernel results (e.g. kernel
0 window 0: `relu_fx = 1522809`; kernel 2 window 0: `relu_fx = 573008`).

**148 / 288 outputs clamp to zero** for this toy input -- unlike the
4-kernel case (0/36 clamped), several of kernels 4-31 have a *positive*
folded bias (e.g. kernel 6: `b_folded = +0.0400`, kernel 13: `+0.0402`),
and combined with each kernel's own sign pattern across the 9 window
positions, many `biased_fx` values land at or below zero. This
incidentally gives the 32-kernel design broader functional coverage of
the ReLU clamp branch than the 4-kernel design did (though the dedicated
synthetic clamp test, `first_conv_bn_relu_relu_clamp_test_summary.md`,
remains the more deliberate, controlled test of that branch).

## 4. GHDL result

`bash hardware/vhdl_conv3x3/run_ghdl_32out_bn_relu_resource_shared.sh`
(VHDL-2008, GHDL 5.1.1):

- Analyzed (dependency order): `window3x3_stream.vhd` (existing,
  unmodified), `conv3x3_dot_time_mux.vhd` (existing, unmodified),
  `first_layer_32out_bn_relu_resource_shared_pkg.vhd` (new),
  `conv3x3_3chan_32out_bn_relu_time_mux.vhd` (new),
  `stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd` (new),
  `tb_stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd` (new) -- all OK.
- Elaborated and simulated -- **all 288 outputs (9 windows x 32 kernels)
  matched the exact Q.16 fixed-point golden values**, e.g. window 1:
  `y0=1522809 y1=1988771 y2=573008 y3=288688 y4=0 ... y31=0`.
- Final report: `=== All stream_conv3x3_3chan_32out_bn_relu_time_mux
  tests PASSED === (9 windows x 32 kernels = 288 / 288 outputs match
  Python Q.16 fixed-point golden vectors)`.
- **Measured latency: 10,125 total clock cycles** (25 capture + 9 windows
  averaging ~1,122 cycles each) -- very close to the design plan's own
  scaling formula (`35*N` cycles/window = `35*32 = 1,120`, Section 13 of
  `resource_shared_first_layer_conv_bn_relu_plan.md`), confirming the
  plan's latency estimate was accurate at full scale, not just
  directionally.

**Synthesis results below are interpreted as meaningful only because this
GHDL run passed all 288 checks.**

## 5. Regression results

All four required regressions re-run and pass, unmodified:

- `run_ghdl_kernel0_bn_relu_pipelined.sh`: PASS 9/9.
- `run_ghdl_kernel2_bn_relu_pipelined.sh`: PASS 9/9.
- `run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh`: PASS 9/9 (all 9 clamped, as before).
- `run_ghdl_32out_dsp.sh` (direct-parallel 32-output DSP-aware): PASS 288/288.

No prior design's source file was modified. The 4-kernel resource-shared
prototype's own source files, run script, synth script, and reports were
preserved (renamed to `..._4out_...` where needed) rather than
overwritten -- see Final Report.

## 6. 200T synthesis result

`stream_conv3x3_3chan_32out_bn_relu_time_mux` on `xc7a200tsbg484-1`
(`synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_200t.tcl`):

| Resource | Used | Available | Util% |
|---|---:|---:|---:|
| LUTs | 1,837 | 134,600 | 1.36% |
| Registers | 5,588 | 269,200 | 2.08% |
| DSP48E1 | **2** | 740 | **0.27%** |
| Block RAM tiles | 0 | 365 | 0.00% |

- WNS (setup slack): **+0.662 ns** -- meets the 100 MHz / 10.000 ns
  clock constraint, and slightly *better* margin than the 4-kernel
  design's +0.597 ns.
- WHS (hold slack): +0.262 ns.
- Total on-chip power (estimate): **0.165 W** (Dynamic 0.043 W, Static
  0.122 W).
- **`synth_design` completed successfully with 0 errors, 0 critical
  warnings, 32 non-critical warnings** -- the same benign class already
  documented for the 4-kernel design: 31 "unused bits of the
  generously-sized 48-bit `product_fx_r` register" notes, and 1 benign
  `FSM_onehot_state_reg[2] unused` note. No correctness or timing
  concern.
- 0 Block RAM tiles used -- the (now 32-entry) weight LUT was still
  mapped to LUT-ROM/distributed logic, not BRAM.

## 7. Comparison: 4-kernel vs. 32-kernel resource-shared, and vs. direct-parallel

| Metric | Resource-shared 4-out | **Resource-shared 32-out (this task)** | Direct-parallel 32-out |
|---|---:|---:|---:|
| GHDL | PASS 36/36 | **PASS 288/288** | PASS 288/288 |
| LUTs | 1,500 | **1,837** | 10,461 |
| Registers | 3,180 | **5,588** | 4,618 |
| DSP48E1 | 2/740 (0.27%) | **2/740 (0.27%)** | 740/740 (100.00%) |
| BRAM | 0 | 0 | 0 |
| WNS | +0.597 ns | **+0.662 ns** | +2.343 ns |
| WHS | +0.262 ns | +0.262 ns | +0.262 ns |
| Total power (estimate) | 0.148 W | **0.165 W** | 1.324 W |
| Warnings | 32 (benign) | 32 (benign) | 1 (DSP overutilized) |
| Kernel count | 4 | **32 (complete layer)** | 32 (complete layer) |
| Latency (5x5x3 toy input, measured) | 1,305 cycles | **10,125 cycles** | ~4-5 cycles (pipelined) |

**This is now a like-for-like comparison at equal kernel count** (32 vs.
32), unlike the earlier 4-vs-32 comparison. The result is unambiguous:

- **DSP48E1 usage did not change at all** going from 4 to 32 kernels (2
  in both cases) -- direct, measured confirmation that this
  architecture's DSP demand is independent of kernel count, exactly as
  the design plan predicted (Section 5: "Total unique compute resources
  ... regardless of kernel count").
- LUTs grew only modestly (1,500 -> 1,837, +22%) for an 8x increase in
  kernel count -- the larger weight LUT (32 vs. 4 entries per channel)
  and wider kernel-index muxing account for this, not a duplicated
  compute datapath.
- Registers grew more (3,180 -> 5,588, +76%), mostly attributable to the
  `y_regs`/`y_out_regs` output-holding arrays scaling from 4x48 bits to
  32x48 bits (28 x 48 = 1,344 additional bits alone) plus wider control
  sets, not a duplicated compute engine.
- **Timing margin held up, and slightly improved**, at full 32-kernel
  scale (+0.662 ns vs. +0.597 ns) -- addressing the open question flagged
  in the 4-kernel summary's own Limitations section ("this specific
  4-kernel implementation should not be assumed to retain positive WNS
  at a larger kernel count without re-synthesis"). It does; this is now
  a measured result, not an assumption.
- Compared to the direct-parallel 32-output design (same 32-kernel
  scope): DSP usage drops from 740/740 (100%) to 2/740 (0.27%), LUTs
  from 10,461 to 1,837 (-82%), power from 1.324 W to 0.165 W (-88%) --
  at the cost of latency growing from ~4-5 cycles to 10,125 cycles for
  the same 5x5x3 image.

## 8. Interpretation

- **This implements the complete 32-output first-layer folded
  Conv-BN-ReLU stage using the resource-shared architecture.**
- **The design trades latency for reduced hardware resource pressure**,
  confirmed at full scale: 10,125 cycles for the complete layer here,
  versus a handful of cycles in the direct-parallel design -- a large,
  explicit, now fully-measured tradeoff.
- **The resource-shared prototype matched Python Q.16 fixed-point golden
  outputs for the tested toy input**: 288/288 exact matches in GHDL,
  covering the complete 32-channel layer.
- **The resource-shared prototype met 100 MHz synthesis timing on
  Artix-7 200T** at full 32-kernel scale (WNS = +0.662 ns), using only 2
  DSP48E1 slices (0.27%) -- identical DSP usage to the 4-kernel
  prototype, and dramatically less than the direct-parallel design's
  740/740 (100%).
- This confirms, at the design's own target scale (all 32 kernels, the
  complete first Conv2d layer), the central architectural claim: DSP
  demand under this resource-sharing strategy does not grow with kernel
  count. This was previously demonstrated only at 4-kernel scale with an
  extrapolation to 32; it is now a direct, measured result at the actual
  target scale.

## 9. Limitations

- **First-layer convolution only.** This is the complete first Conv2d
  layer's 32 output channels, not the full U-Net, and does not claim
  otherwise.
- **Toy input only.** The canonical 5x5x3 toy patch, not a real UAVSAR
  tile -- this is not a real-world inference benchmark.
- **Latency for larger tiles remains an estimate** (design plan Section
  13's `S^2 + (S-2)^2 * 35*N` formula); the measured 10,125-cycle figure
  here is specific to the 5x5 toy image at N=32. No larger tile size was
  attempted.
- **No padding, pooling, downstream layers, or full model pipeline.**
  The full U-Net does not fit and was not evaluated; end-to-end flood
  segmentation was not run on FPGA.
- **No board testing, no measured speedup, no measured board power.**
  All figures are Vivado `synth_design`/GHDL simulation results, never
  measurements from real hardware.
- **This does not prove any latency, resource, or timing claim for a
  different sharing degree** (e.g. 2, 4, or 8 parallel resource-shared
  lanes instead of the single-engine extreme built here) -- that remains
  the design plan's own recommended next exploration (Section 18 of
  `resource_shared_first_layer_conv_bn_relu_plan.md`).
- Synthesis-level (out-of-context `synth_design`) result only, not
  placed-and-routed or board-tested.

## 10. Recommended next step

The design plan's Section 18 "not yet planned" item -- extending beyond
the toy input to realistic tile sizes -- is now the most natural next
step, since the complete 32-kernel single-engine architecture is fully
verified and its resource/timing behavior is now measured (not
estimated) at its target kernel count. Given the plan's own scaling
formula projects roughly 7.2e7 cycles (~0.72 s at 100 MHz) for a
realistic 256x256 tile at this same single-engine sharing degree, the
most productive next step is exploring the small-lane middle ground
(e.g. 2, 4, or 8 parallel resource-shared engines) to find a better
latency/resource tradeoff point before attempting any larger-tile
implementation.
