# Kernel 0 Folded Conv-BN-ReLU VHDL Proof-of-Concept

Date: July 9, 2026

This document covers a kernel-0-only VHDL hardware proof-of-concept and
its GHDL/synthesis results only. No board testing, no measured speedup,
and no measured board power are claimed anywhere in this document.

## 1. Purpose

`hardware/vhdl_conv3x3/first_conv_bn_relu_fidelity_summary.md` (a
Python-only numerical study, `scripts/analyze_first_conv_bn_relu_fidelity.py`)
showed that folding BatchNorm into the first Conv2d layer's weights/bias is
exact in float and remains close after INT8-style quantization, across all
32 output channels. That study explicitly stated: **no VHDL module in this
repo implements BatchNorm, bias, or ReLU** -- every VHDL prototype through
the completed 32-output first-layer scaling study
(`first_layer_32out_dsp_200t_summary.md`) still only computes the raw
valid-convolution datapath with `bias_int32 = 0`.

This task closes that gap with a small VHDL proof-of-concept: **Conv ->
folded bias/rescale -> ReLU**, for **kernel (output channel) 0 only**,
using the existing trained checkpoint and the existing Python
Conv-BN-ReLU fidelity study's folding formula as the numerical reference.
This is a proof-of-concept, not a new 32-channel scaling experiment.

## 2. Design scope

- **Kernel (output channel) 0 only**, not all 32 channels.
- Canonical 5x5x3 toy input, valid (no-padding) 3x3 output region -- same
  as every other VHDL prototype in this repo.
- Same trained checkpoint
  (`models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`)
  and the same `enc1.block.0` (Conv2d) / `enc1.block.1` (BatchNorm2d)
  layers already used by the Python fidelity study.
- No padding, pooling, downstream layers, or full model pipeline.
- No board testing, no measured speedup, no measured board power, no
  deployment claim.

## 3. Relationship to the Python first Conv-BN-ReLU fidelity study

This VHDL proof-of-concept reuses the **same BatchNorm folding formula**
as `scripts/analyze_first_conv_bn_relu_fidelity.py`:

```
scale_bn[0] = gamma[0] / sqrt(running_var[0] + eps)
w_folded[0] = w[0] * scale_bn[0]
b_folded[0] = beta[0] - running_mean[0] * scale_bn[0]
```

applied to kernel 0 only, from the same checkpoint tensors
(`enc1.block.0.weight`, `enc1.block.1.{weight,bias,running_mean,running_var}`,
`eps=1e-5`). It does **not** reuse that script's output directly, because
that script rescales the INT32 convolution result back to **float**
(`int32_accum * scale_x * scale_w_folded + b_folded`, in float64) before
comparing to PyTorch -- VHDL has no floating-point rescale step. A new,
separate script,
`scripts/generate_kernel0_bn_relu_fixed_point_vectors.py`, was written to
perform the entire rescale + bias + ReLU computation in genuine **fixed-point
integer** arithmetic instead, matching exactly what the VHDL computes (see
Section 4). This connects the prior raw convolution VHDL prototype
(`stream_conv3x3_3chan_cell.vhd`) to the folded Conv-BN-ReLU numerical
study, without claiming the two scripts produce identical numeric results
(they use different rescale arithmetic, by design).

**Sanity check performed**: the folded-and-quantized INT8 weights computed
here for kernel 0 turned out **bit-identical** to the *raw* (unfolded)
quantized INT8 weights already published in
`first_layer_kernels0_to31_pkg.vhd`'s `KERNEL0_CH*_W` constants. This is
expected, not a bug: symmetric per-channel INT8 quantization
(`scale = max(|w|)/127`) is invariant under multiplication by any positive
scalar, and `scale_bn[0] = 1.515678` is positive, so
`quantize(w_folded) == quantize(w)` for kernel 0. This also means kernel
0's raw INT32 convolution accumulation in this design is numerically
identical to the values already published for kernel 0 in every prior
multi-kernel package (`11287, 12749, ..., 28831` for the 9 toy-patch
positions).

## 4. Fixed-point arithmetic choice

**Q.16 format** (signed, 16 fractional bits): `real_value ~= fixed_value / 2^16`.

| Constant | Value | Definition |
|---|---:|---|
| `SCALE_FX` | 135 | `round(scale_x * scale_w_folded[0] * 2^16)`, unsigned magnitude |
| `BIAS_FX` | -936 | `round(b_folded[0] * 2^16)`, signed |

Where `scale_w_folded[0] = 0.00206228` (folded-weight INT8 quantization
scale) and `b_folded[0] = -0.014275` (folded bias). `scale_x = 1.0`
**exactly** for the toy patch -- a deliberate simplification, documented
in the generator script's docstring, that differs from the Python fidelity
study's general per-patch `scale_x = max(|x|)/127` formula. It is lossless
here because the toy patch's maximum magnitude (50, in channel 1) fits
inside INT8 range with zero rounding, and it keeps the raw INT32
convolution bit-identical to every other VHDL testbench's literal integer
pixel stimulus in this repo.

Datapath (exact integers, no floating point, no runtime shifts):

```
raw_conv_int32 = sum(x_int8 * w_folded_int8)      -- exact INT32, from the
                                                       reused stream_conv3x3_3chan_cell
product_fx      = raw_conv_int32 * SCALE_FX          -- exact, already Q.16
                                                       (raw_conv_int32 is Q.0, so no shift needed)
biased_fx        = product_fx + BIAS_FX                -- exact, Q.16 (same format, no shift)
relu_fx           = biased_fx if biased_fx > 0 else 0   -- exact, Q.16
```

No shift instructions appear anywhere in the VHDL datapath: an integer
(Q.0) times a Q.16 constant is automatically Q.16, and `BIAS_FX` is
generated directly in Q.16, so the add needs no rescaling either. The
`relu_fx` output is 48 bits signed (measured max `|biased_fx|` = 3,891,249,
needing >=23 bits; 48 bits is generous headroom for a small proof of
concept, not a tightly-optimized width).

**The VHDL testbench compares against the exact integer `relu_fx` golden
value, not a float-converted value** -- this avoids silently comparing
fixed-point VHDL output to float Python output.

## 5. Generated test vectors

`scripts/generate_kernel0_bn_relu_fixed_point_vectors.py` (new script) wrote:

- `hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_vectors.json`
- `hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_vectors.csv`
- `hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/kernel0_bn_relu_fixed_point_summary.md`
- `hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_pkg.vhd` (VHDL constants)

Each includes, for all 9 valid output positions: `raw_conv_int32` (exact
INT32 conv accumulation), `product_fx` (post-scale, Q.16), `biased_fx`
(post-bias, Q.16, pre-ReLU), and `relu_fx` (final, Q.16, the exact VHDL
golden value).

| idx | raw_conv_int32 | biased_fx | relu_fx (golden) |
|---|---:|---:|---:|
| 0 | 11287 | 1522809 | 1522809 |
| 1 | 12749 | 1720179 | 1720179 |
| 2 | 14211 | 1917549 | 1917549 |
| 3 | 18597 | 2509659 | 2509659 |
| 4 | 20059 | 2707029 | 2707029 |
| 5 | 21521 | 2904399 | 2904399 |
| 6 | 25907 | 3496509 | 3496509 |
| 7 | 27369 | 3693879 | 3693879 |
| 8 | 28831 | 3891249 | 3891249 |

**Note**: for this specific toy patch, every `biased_fx` value is already
strongly positive before ReLU (the toy patch's raw conv magnitudes are
large relative to `BIAS_FX = -936`), so ReLU's zero-clamping branch is
never actually exercised by these 9 golden values -- `relu_fx == biased_fx`
for all 9 positions. This is a real limitation of using the toy patch for
this specific kernel/bias combination (see Section 11).

## 6. GHDL result

`bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu.sh` (VHDL-2008, GHDL 5.1.1):

- Analyzed (dependency order): `window3x3_stream.vhd`,
  `conv3x3_dot_pipelined.vhd`, `stream_conv3x3_3chan_cell.vhd` (existing,
  **unmodified**), `first_conv_bn_relu_kernel0_pkg.vhd`,
  `stream_conv3x3_3chan_kernel0_bn_relu.vhd`,
  `tb_stream_conv3x3_3chan_kernel0_bn_relu.vhd` -- all OK.
- Elaborated and simulated `tb_stream_conv3x3_3chan_kernel0_bn_relu` --
  **all 9 valid output positions matched the exact Q.16 fixed-point
  golden values from Section 5.**
- Final report line: `=== All stream_conv3x3_3chan_kernel0_bn_relu tests
  PASSED === (9 / 9 outputs match Python Q.16 fixed-point golden vectors)`.
- 5-cycle total latency confirmed (4 cycles from the reused
  `stream_conv3x3_3chan_cell` + 1 cycle for the new BN+ReLU stage): first
  output appears one cycle later (`i=18`) than the 4-cycle raw-conv-only
  designs' first output (`i=17`).

**Synthesis results below are interpreted as meaningful only because this
GHDL run passed all 9 checks.**

## 7. Regression result

- `bash hardware/vhdl_conv3x3/run_ghdl_3chan_cell.sh` (existing, unmodified
  `stream_conv3x3_3chan_cell` testbench, the exact sub-component this new
  design reuses): **PASS**, 9/9 outputs, unaffected.
- `bash hardware/vhdl_conv3x3/run_ghdl_32out_dsp.sh` (complete 32-output
  DSP-aware regression, run since it completed in ~5 seconds -- fast
  enough to run in full rather than substituting the 8-output regression):
  **PASS**, 288/288 outputs, unaffected.

Neither prior design's source file was modified by this task.

## 8. 200T synthesis result

`stream_conv3x3_3chan_kernel0_bn_relu` on `xc7a200tsbg484-1`
(`synth_stream_conv3x3_3chan_kernel0_bn_relu_200t.tcl`):

| Resource | Used | Available | Util% |
|---|---:|---:|---:|
| LUTs | 1912 | 134,600 | 1.42% |
| Registers | 1330 | 269,200 | 0.49% |
| DSP48E1 | 0 | 740 | 0.00% |
| Block RAM tiles | 0 | 365 | 0.00% |

- WNS (setup slack): **-0.609 ns** -- the 100 MHz / 10.000 ns clock
  constraint is **not met** at this synthesis-level estimate (42 failing
  endpoints, TNS = -25.396 ns).
- WHS (hold slack): +0.262 ns (met).
- Total on-chip power (estimate): **0.167 W** (Dynamic 0.045 W, Static
  0.122 W).
- **`synth_design` completed successfully with 0 warnings, 0 critical
  warnings, 0 errors** (20 informational messages only) -- the negative
  WNS is a timing-closure result, not a synthesis warning or error.

## 9. Warnings/errors

None from `synth_design` itself. The only warning anywhere in the log is
the same benign `WARNING: [Timing 38-242] ... HD.CLK_SRC ...` informational
note about out-of-context clock-skew estimation that appears in every
design synthesized in this repo. **The negative WNS (-0.609 ns) is the
one substantive finding from this synthesis run** -- see Section 10.

## 10. Comparison to the kernel0 raw convolution design

`stream_conv3x3_3chan_cell` (kernel 0, raw, no BN/bias/ReLU) on the same
200T target, from the existing larger-target comparison
(`vivado_reports_larger_target/stream_conv3x3_3chan_cell_*`):

| Design | LUTs | Registers | DSP | WNS (ns) | Total power (W) | Warnings |
|---|---:|---:|---:|---:|---:|---|
| Raw conv (kernel 0, existing) | 2731 | 1324 | 0 | **+4.456** | 0.175 | 0 |
| **Folded Conv-BN-ReLU (kernel 0, this task)** | **1912** | **1330** | **0** | **-0.609** | **0.167** | **0** |

Both designs use 0 DSPs, since this design reuses `conv3x3_dot_pipelined`
(the LUT-only multiply variant), matching the raw design's own datapath --
not `conv3x3_dot_pipelined_dsp`. LUT count is actually **lower** in the
new design (1912 vs. 2731); this is a Vivado synthesis-boundary effect
(the raw design's own top-level `bias` port is driven externally by the
testbench in the standalone report, while here it is a compile-time
`ZERO32` constant folded directly by the synthesis tool) rather than
evidence the BN+ReLU addition reduced logic -- it should not be read as
"folded Conv-BN-ReLU is cheaper than raw conv."

**The key difference is timing, not area**: the raw conv-only design
comfortably meets the 100 MHz target (+4.456 ns of slack), while adding
this task's single-stage fixed-point BN+ReLU logic (a 32-bit x 18-bit
signed multiply, a 48-bit add, and a compare-and-select, all combined
into one un-pipelined combinational block before the final register)
turns that slack negative (-0.609 ns). This is a real, honestly-reported
timing-closure result for this specific implementation choice, not a
resource or synthesis failure.

## 11. Interpretation

- **A kernel0 folded Conv-BN-ReLU fixed-point VHDL proof-of-concept was
  created.**
- **The VHDL output matches the Python fixed-point golden reference for
  the 5x5x3 toy input** -- all 9 positions, exact integer match, confirmed
  by GHDL.
- **This connects the prior raw convolution VHDL prototype to the folded
  Conv-BN-ReLU numerical study**: the same folding formula and the same
  checkpoint tensors used by
  `scripts/analyze_first_conv_bn_relu_fidelity.py` now have a
  corresponding, GHDL-verified, genuinely fixed-point VHDL implementation
  for kernel 0, closing the gap that study's own limitations section
  explicitly flagged ("No VHDL BatchNorm/bias/ReLU implementation yet").
- The toy patch does not exercise ReLU's clamping branch for kernel 0
  (Section 5) -- functional correctness of the multiply/add/rescale path
  is verified, but the ReLU comparison's negative-result branch is not
  exercised by these specific 9 golden values.
- The synthesis result shows this specific single-stage BN+ReLU addition
  does not close timing at 100 MHz on the Artix-7 200T in this run
  (WNS = -0.609 ns), even though the raw convolution alone comfortably
  does (+4.456 ns). This indicates the added combinational path (wide
  fixed-point multiply + add + ReLU compare, all unpipelined) is the
  likely bottleneck, not DSP or LUT resource pressure (both are far under
  budget).

## 12. Limitations

- **Kernel 0 only.** Not all 32 folded Conv-BN-ReLU channels are
  implemented.
- **Toy input only**, and specifically one where ReLU's negative-clamp
  branch is not numerically exercised (Section 5) -- this proof-of-concept
  does not demonstrate ReLU actually zeroing a value in hardware.
- **`scale_x = 1.0` is exact for this specific toy patch only** (see
  Section 4); this is not the general per-patch activation-scale
  convention used for real UAVSAR tiles in the Python fidelity study.
- **Timing not closed at 100 MHz** in this synthesis run (WNS = -0.609 ns,
  42 failing endpoints) -- this is a synthesis-level, out-of-context
  estimate, not a placed-and-routed result, but it should not be
  overlooked or treated as a clean pass.
- No padding, pooling, downstream layers, or full model pipeline.
- No board testing, no measured speedup, no measured board power -- all
  figures are Vivado `synth_design` / `report_timing_summary` /
  `report_power` (vector-less, "Medium confidence") estimates.
- The full U-Net does not fit and was not evaluated; end-to-end flood
  segmentation was not run on FPGA.

## 13. Recommended next step

Pipeline the fixed-point BN+ReLU stage across two clock cycles (e.g.,
register the multiply result in one stage, then compute the bias-add and
ReLU compare in a second stage) instead of the current single combinational
block, and re-synthesize to check whether that alone recovers positive
WNS at 100 MHz on the Artix-7 200T -- this would isolate whether the
critical path is dominated by the wide multiply or by the combined
multiply+add+compare chain, before considering extending this proof of
concept to more than one output channel.
