# Harder-Kernel Generalization Test: Kernel 2 Pipelined Folded Conv-BN-ReLU

Date: July 9, 2026

This document covers a second, harder-kernel VHDL Conv-BN-ReLU
proof-of-concept and its GHDL/synthesis results only. No board testing,
no measured speedup, and no measured board power are claimed anywhere in
this document.

## 1. Purpose

`hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_pipelined_summary.md`
showed that splitting kernel 0's single combinational fixed-point
multiply/add/ReLU stage into two registered pipeline stages recovered
100 MHz timing closure on Artix-7 200T (WNS -0.609 ns -> +2.461 ns). That
result was demonstrated for kernel 0's specific `SCALE_FX`/`BIAS_FX`
constants only. This task tests whether the same two-stage pipeline
structure also closes timing for a **harder** kernel -- one with more
extreme folded-weight/bias characteristics -- to give evidence about
whether the kernel0 timing fix generalizes, rather than being a kernel0
special case.

## 2. Why this kernel was selected

Per-channel metrics from the existing Python fidelity study
(`hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_fidelity/first_conv_bn_relu_per_channel_metrics.csv`):

| kernel | scale_w_folded | b_folded | max_abs_error | mae | pearson_correlation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00206228 | -0.014275 | 0.290454 | 0.005312 | 0.999998 |
| **2** | **0.00575681** | -0.011989 | **0.655282** | **0.013905** | 0.999853 |
| 11 | 0.00338138 | -0.000159 | 0.259978 | 0.004747 | 0.999641 |

**Kernel 2 was selected**, not kernel 11, because kernel 2 has:

- The **largest `scale_w_folded` of all 32 kernels** (0.00575681, ~2.79x
  kernel 0's 0.00206228, and ~1.70x kernel 11's) -- this directly and
  proportionally produces a larger `SCALE_FX` fixed-point constant, the
  clearest "more extreme fixed-point constant" criterion available.
- The **largest `max_abs_error` of all 32 kernels** (0.655282) in the
  Python fidelity study, confirming the same conclusion in
  `first_conv_bn_relu_fidelity_summary.md`: *"the two kernels with the
  largest folded-weight scale (kernels 2 and 11) are exactly the two
  kernels with the largest observed error... kernel 2 having the largest
  `scale_w_folded` (0.005757, the widest quantization step of any
  kernel)"*.

Kernel 11 has a lower Pearson correlation (0.999641 vs. kernel 2's
0.999853) and a smaller `b_folded` magnitude, but a smaller
`scale_w_folded` and smaller `max_abs_error` than kernel 2 -- kernel 2 is
the clearer "hardest" pick on the metric most directly relevant to this
task's fixed-point width/precision question (the quantization scale that
becomes `SCALE_FX`).

## 3. Generator changes

Rather than duplicating the kernel0 generator script by hand, a new,
generalized script was created:
`scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id <N>`. It
performs the exact same BatchNorm-folding formula, INT8 quantization
convention, toy-patch handling, and Q.16 fixed-point arithmetic as
`scripts/generate_kernel0_bn_relu_fixed_point_vectors.py` (that original
script is unchanged), generalized to accept any kernel index, and it adds
one new check the kernel0 script did not need: **a width check** that
compares the new kernel's `SCALE_FX` and max `|biased_fx|` against the
fixed VHDL widths already used by the kernel0 design (18-bit signed
`SCALE_FX_SIGNED`, 48-bit signed datapath), reporting a warning if a
kernel's constants would not fit -- rather than silently widening
anything. Run as:

```
python3 scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id 2
```

This produced `hardware/vhdl_conv3x3/first_conv_bn_relu_kernel2_pkg.vhd`
and test vectors under
`hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel2/`, without
touching any kernel0 file.

## 4. Fixed-point constants and vector summary

Same Q.16 format (16 fractional bits) as kernel 0 -- **no width change was
needed**; the generator's width check reported "OK" for kernel 2 (`SCALE_FX`
fits in 18 bits, `biased_fx` fits in 48 bits, same as kernel 0):

| Constant | Kernel 0 | Kernel 2 | Ratio |
|---|---:|---:|---:|
| `scale_bn` | 1.515678 | 3.753419 | 2.48x |
| `b_folded` | -0.014275 | -0.011989 | 0.84x |
| `scale_w_folded` | 0.00206228 | 0.00575681 | 2.79x |
| `SCALE_FX` (Q.16) | 135 | **377** | 2.79x |
| `BIAS_FX` (Q.16) | -936 | **-786** | 0.84x |
| max `\|biased_fx\|` | 3,891,249 | 1,966,400 | -- |
| bits needed for `biased_fx` | 23 | 22 | -- |

Expected outputs (9 valid positions, row-major; full detail in
`hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel2/kernel2_bn_relu_fixed_point_vectors.csv`):

| idx | raw_conv_int32 | product_fx | biased_fx | relu_fx (golden) |
|---|---:|---:|---:|---:|
| 0 | 1522 | 573794 | 573008 | 573008 |
| 1 | 1830 | 689910 | 689124 | 689124 |
| 2 | 2138 | 806026 | 805240 | 805240 |
| 3 | 3062 | 1154374 | 1153588 | 1153588 |
| 4 | 3370 | 1270490 | 1269704 | 1269704 |
| 5 | 3678 | 1386606 | 1385820 | 1385820 |
| 6 | 4602 | 1734954 | 1734168 | 1734168 |
| 7 | 4910 | 1851070 | 1850284 | 1850284 |
| 8 | 5218 | 1967186 | 1966400 | 1966400 |

(Values transcribed directly from
`hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel2/kernel2_bn_relu_fixed_point_vectors.csv`,
which matches `first_conv_bn_relu_kernel2_pkg.vhd`'s
`K2_BN_RELU_EXPECTED_FX` constant exactly.)

## 5. Whether ReLU clamping was exercised

**No.** As with kernel 0, all 9 `biased_fx` values for kernel 2 on this
toy patch are already strongly positive (the toy patch's raw conv
magnitudes dominate `BIAS_FX`), so `relu_fx == biased_fx` for every
position, and ReLU's negative-clamp branch is not numerically exercised
by these golden values. This is the same limitation already documented
for kernel 0 and is not specific to the kernel-2 test.

## 6. Pipeline structure

Identical two-stage structure to the kernel0 pipelined design, applied to
kernel 2's own folded INT8 weights and `SCALE_FX`/`BIAS_FX` constants:

- **Stage 1 (registered)**: `product_fx_r <= raw_y * SCALE_FX`
- **Stage 2 (registered)**: `biased_fx = product_fx_r + BIAS_FX`; ReLU
  compare-and-select; register final output.

New files: `hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel2_bn_relu_pipelined.vhd`
(structurally identical to the kernel0 pipelined design, only the package
import and weight-constant names differ) and its testbench.

## 7. Latency

**6 clock cycles total** (4 from the reused, unmodified
`stream_conv3x3_3chan_cell` + 2 for the pipelined multiply / bias-add+ReLU
stages) -- **identical latency** to the kernel0 pipelined design, since
latency depends only on pipeline structure, not on kernel constants.
Confirmed empirically: first output at main-loop `i=19`, same timing
pattern as kernel 0's pipelined testbench.

## 8. GHDL result

`bash hardware/vhdl_conv3x3/run_ghdl_kernel2_bn_relu_pipelined.sh`
(VHDL-2008, GHDL 5.1.1):

- Analyzed (dependency order): `window3x3_stream.vhd`,
  `conv3x3_dot_pipelined.vhd`, `stream_conv3x3_3chan_cell.vhd` (existing,
  unmodified), `first_conv_bn_relu_kernel2_pkg.vhd` (new),
  `stream_conv3x3_3chan_kernel2_bn_relu_pipelined.vhd` (new),
  `tb_stream_conv3x3_3chan_kernel2_bn_relu_pipelined.vhd` (new) -- all OK.
- Elaborated and simulated -- **all 9 valid output positions matched the
  exact Q.16 fixed-point golden values** from
  `first_conv_bn_relu_kernel2_pkg.vhd`.
- Final report line: `=== All
  stream_conv3x3_3chan_kernel2_bn_relu_pipelined tests PASSED === (9 / 9
  outputs match Python Q.16 fixed-point golden vectors)`.

**Synthesis results below are interpreted as meaningful only because this
GHDL run passed all 9 checks.**

## 9. Regression result

- `bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined.sh`
  (existing, unmodified kernel0 pipelined design and testbench):
  **PASS**, 9/9, unaffected.
- `bash hardware/vhdl_conv3x3/run_ghdl_32out_dsp.sh` (complete 32-output
  DSP-aware regression): **PASS**, 288/288, unaffected.

No prior design's source file was modified by this task.

## 10. 200T synthesis result

`stream_conv3x3_3chan_kernel2_bn_relu_pipelined` on `xc7a200tsbg484-1`
(`synth_stream_conv3x3_3chan_kernel2_bn_relu_pipelined_200t.tcl`):

| Resource | Used | Available | Util% |
|---|---:|---:|---:|
| LUTs | 1966 | 134,600 | 1.46% |
| Registers | 1386 | 269,200 | 0.51% |
| DSP48E1 | 0 | 740 | 0.00% |
| Block RAM tiles | 0 | 365 | 0.00% |

- WNS (setup slack): **+2.648 ns** -- the 100 MHz / 10.000 ns clock
  constraint **is met**.
- WHS (hold slack): +0.262 ns.
- Total on-chip power (estimate): **0.169 W** (Dynamic 0.047 W, Static
  0.122 W).
- **`synth_design` completed successfully with 0 warnings, 0 critical
  warnings, 0 errors** (20 informational messages only, same benign
  `HD.CLK_SRC` note as every other design in this repo).

## 11. Comparison to kernel0 pipelined result

| Metric | Kernel 0 pipelined | Kernel 2 pipelined (this task) |
|---|---:|---:|
| LUTs | 1914 | 1966 |
| Registers | 1372 | 1386 |
| DSP | 0 | 0 |
| BRAM | 0 | 0 |
| WNS (ns) | +2.461 | **+2.648** |
| WHS (ns) | +0.262 | +0.262 |
| Total power (W) | 0.168 | 0.169 |
| Latency (cycles) | 6 | 6 |
| GHDL result | 9/9 PASS | 9/9 PASS |

Resource usage is nearly identical (LUTs +52, Registers +14, both small
deltas plausibly from the different constant values feeding the
multiply/add logic, not a structural difference). **WNS is actually
slightly better for kernel 2** (+2.648 ns vs. +2.461 ns) despite kernel
2's larger `SCALE_FX` and being the "harder" kernel by the fidelity
study's own error metrics -- the larger quantization-error kernel did not
translate into a harder timing-closure problem for this two-stage
pipeline structure.

## 12. Interpretation

- **A harder first-layer kernel was tested with the same two-stage
  pipelined folded Conv-BN-ReLU structure** -- kernel 2, selected for
  having the largest `scale_w_folded` (2.79x kernel 0's) and the largest
  `max_abs_error` (0.655, vs. kernel 0's 0.290) of all 32 kernels in the
  existing Python fidelity study.
- **The selected kernel matched Python Q.16 fixed-point golden outputs in
  GHDL** -- all 9 positions, exact integer match.
- **The selected kernel did meet 100 MHz synthesis timing on Artix-7
  200T** (WNS = +2.648 ns, 0 failing endpoints), with resource usage and
  power essentially unchanged from kernel 0's pipelined result.
- **This gives evidence that the kernel0 timing fix generalizes** beyond
  kernel 0's specific constants: a kernel with a substantially larger
  `SCALE_FX` (377 vs. 135) and a documented larger quantization error in
  the float-fidelity study still closes timing cleanly with the same
  two-stage pipeline structure, with no width changes needed. This is
  evidence from **two kernels**, not proof that all 32 kernels will
  close timing -- kernel 2 was deliberately chosen as a stress test on
  the scale/error axis, not as an exhaustive sweep.

## 13. Limitations

- **Two kernels tested (0 and 2), not all 32.** This is evidence the
  timing fix generalizes beyond a single kernel, not proof it holds for
  every one of the 32 folded Conv-BN-ReLU channels.
- **Toy input only**, and as with kernel 0, this specific toy input does
  not exercise ReLU's negative-clamp branch for kernel 2 either (Section
  5) -- functional correctness of the multiply/add/rescale path is
  verified, but ReLU's zero-clamping behavior remains unexercised by
  either kernel's golden vectors.
- Kernel 2 was chosen specifically because it is an outlier on the
  scale/error axis (largest `scale_w_folded`, largest `max_abs_error`);
  this is a deliberate stress test of that specific dimension, not a
  representative "typical" kernel, and it says nothing about kernels
  that might stress other dimensions (e.g. unusually large `b_folded`
  magnitude, which kernel 2 does not have -- kernel 2's `b_folded` is
  actually slightly smaller in magnitude than kernel 0's).
- Synthesis-level (out-of-context `synth_design`) result only, not
  placed-and-routed or board-tested. WNS/WHS/power are Vivado estimates,
  not measurements.
- No padding, pooling, downstream layers, or full model pipeline. The
  full U-Net does not fit and was not evaluated; end-to-end flood
  segmentation was not run on FPGA.
- No board testing, no measured speedup, no measured board power.

## 14. Recommended next step

Since two kernels (0 and 2) with substantially different `SCALE_FX`
magnitudes both close timing with the same two-stage pipeline and nearly
identical resource/power cost, the most informative next step is to test
a kernel that stresses a *different* dimension than kernel 2 did --
specifically kernel 11, which has the second-largest `max_abs_error`
(0.260) but a distinctly different profile from kernel 2 (smaller
`scale_w_folded`, much smaller `b_folded` magnitude, and the lowest
Pearson correlation of the three kernels examined so far, 0.999641) --
before considering whether this two-stage pipeline structure could
reasonably be generalized (made kernel-selectable at the VHDL level)
rather than hand-duplicated per kernel.
