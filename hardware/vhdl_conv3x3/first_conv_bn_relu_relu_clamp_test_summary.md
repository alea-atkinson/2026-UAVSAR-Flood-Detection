# Synthetic ReLU Clamp Test for the Folded Conv-BN-ReLU Fixed-Point Flow

Date: July 9, 2026

**This is a synthetic unit test, not a real UAVSAR inference example.**
It uses a deliberately constructed all-zero input patch to force the
ReLU clamp (negative) branch and verify it in VHDL. No board testing, no
measured speedup, and no measured board power are claimed anywhere in
this document.

## 1. Purpose

The kernel 0 and kernel 2 pipelined folded Conv-BN-ReLU proof-of-concepts
(`first_conv_bn_relu_kernel0_pipelined_summary.md`,
`first_conv_bn_relu_harder_kernel_pipelined_summary.md`) both matched
Python Q.16 fixed-point golden outputs in GHDL and met 100 MHz synthesis
timing on Artix-7 200T. However, both used the canonical 5x5x3 toy input,
and for both kernels all 9 `biased_fx` values happened to be positive --
so `relu_fx == biased_fx` for every output, and the ReLU clamp
(negative-branch) logic that exists in the VHDL was never numerically
exercised. This task adds a small, targeted test that forces at least one
negative `biased_fx` value and verifies the VHDL output clamps it to
zero.

## 2. Why this test was needed

Grep of both prior summaries confirms the gap explicitly: kernel 0's
summary states *"the toy patch does not exercise ReLU's clamping branch
for kernel 0... functional correctness of the multiply/add/rescale path
is verified, but the ReLU comparison's negative-result branch is not
exercised"*, and kernel 2's summary repeats the same limitation. The
`if biased_fx_comb > 0 then ... else (others => '0') end if` clamp logic
has existed in both pipelined designs since they were written, but no
GHDL run had ever actually taken the `else` branch. This test closes that
specific gap.

## 3. Synthetic input choice

**All-zero 5x5x3 input patch** (`pixel_c0 = pixel_c1 = pixel_c2 = 0` for
all 25 streamed pixel positions), chosen after inspecting
`raw_conv_int32`, `product_fx`, and `biased_fx` for the candidate simple
inputs listed in the task (all zeros, sign-flipped toy patch, constant
negative values, scaled negative pattern):

- An all-zero input makes `raw_conv_int32 = sum(0 * w) = 0` for **every**
  one of the 9 valid output positions, **regardless of the kernel's
  weights** -- this is the simplest, most deterministic, and most
  kernel-independent way to force a known `raw_conv_int32`.
- With `raw_conv_int32 = 0`, `product_fx = 0 * SCALE_FX = 0` and
  `biased_fx = 0 + BIAS_FX = BIAS_FX` exactly, for all 9 positions.
- `BIAS_FX` is **negative for every kernel generated so far** (kernel 0:
  -936, kernel 2: -786), so an all-zero input deterministically clamps
  **all 9** outputs to zero -- a stronger, not weaker, confirmation than
  forcing just one position negative, and one that required no manual
  search for a specific negative-forcing input pattern.
- This was confirmed by direct inspection (not assumed) via the
  generator script's console output and the written JSON/CSV before any
  VHDL was written (Section 6 below).

Sign-flipped or constant-negative patterns were considered but not
needed once the all-zero case was confirmed to reliably clamp every
position for a kernel with negative `BIAS_FX`.

## 4. Kernel selected

**Kernel 0**, for two reasons: (1) it is the kernel with the most
existing verified infrastructure (raw convolution sub-component, folded
weights, pipelined design) to build on with minimal new code, and (2) its
`BIAS_FX = -936` was already known and negative, so an all-zero input was
predicted (and then confirmed) to force all 9 outputs to clamp without
needing a search over kernels.

## 5. Fixed-point arithmetic/constants used

**Identical** to kernel 0's existing pipelined design -- no new
arithmetic, no width changes, no changes to the folded weights:

| Constant | Value |
|---|---:|
| Folded INT8 weights | same `K0_FOLDED_CH0/1/2_W` values as `first_conv_bn_relu_kernel0_pkg.vhd` |
| `SCALE_FX` (Q.16) | 135 |
| `BIAS_FX` (Q.16) | -936 |
| Output datapath width | 48-bit signed (same as kernel 0/kernel 2) |
| Pipeline structure | same two-stage: Stage 1 registers `product_fx`, Stage 2 computes `biased_fx` and applies ReLU |

**Only the input test patch changed** (all-zero instead of the canonical
toy patch) -- confirmed by direct comparison of
`first_conv_bn_relu_kernel0_relu_clamp_pkg.vhd`'s `K0_FOLDED_CH*_W`,
`SCALE_FX`, and `BIAS_FX` constants against
`first_conv_bn_relu_kernel0_pkg.vhd`'s, which are identical.

Generated via the existing generalized generator, extended with a new
optional `--input-mode` flag (default `toy_5x5`, unchanged from before):

```
python3 scripts/generate_bn_relu_fixed_point_vectors.py --kernel-id 0 --input-mode synthetic_relu_clamp
```

## 6. Number and positions of ReLU-clamped outputs

**9 / 9 outputs clamped to zero** (all valid output positions):
`raw_conv_int32 = 0`, `product_fx = 0`, `biased_fx = -936` for every
position `(row, col)` in `{(0,0), (0,1), (0,2), (1,0), (1,1), (1,2),
(2,0), (2,1), (2,2)}` -- confirmed directly in the generator script's
console output and written to
`hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0_relu_clamp/kernel0_relu_clamp_bn_relu_fixed_point_vectors.csv`
(`num_clamped_outputs: 9`, `clamped_positions_row_col`: all 9 pairs, in
the JSON).

## 7. GHDL result

`bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh`
(VHDL-2008, GHDL 5.1.1):

- Analyzed (dependency order): `window3x3_stream.vhd`,
  `conv3x3_dot_pipelined.vhd`, `stream_conv3x3_3chan_cell.vhd` (existing,
  unmodified), `first_conv_bn_relu_kernel0_relu_clamp_pkg.vhd` (new),
  `stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd` (new),
  `tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp.vhd`
  (new) -- all OK.
- Elaborated and simulated -- **all 9 valid output positions matched the
  exact Q.16 fixed-point golden value of 0**, and the testbench itself
  counted and reported the clamp total:
  `=== ReLU clamp count: 9 / 9 outputs clamped to zero ===`, followed by
  `=== All stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp
  tests PASSED === (9 / 9 outputs match Python Q.16 fixed-point golden
  vectors, ReLU clamp branch exercised)`.
- The testbench includes an explicit assertion
  (`assert clamped_count >= 1 ... severity failure`) that would have
  failed the run if the clamp branch were never taken -- this is not
  merely reported informationally, it is a hard pass/fail condition.

**This is the first GHDL run in this project's folded Conv-BN-ReLU work
to numerically exercise the ReLU clamp (negative) branch.**

## 8. Regression results

- `bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined.sh`
  (existing, unmodified kernel0 pipelined toy-input design): **PASS**,
  9/9, unaffected.
- `bash hardware/vhdl_conv3x3/run_ghdl_kernel2_bn_relu_pipelined.sh`
  (existing, unmodified kernel2 pipelined toy-input design): **PASS**,
  9/9, unaffected.
- `bash hardware/vhdl_conv3x3/run_ghdl_32out_dsp.sh` (complete 32-output
  DSP-aware regression): **PASS**, 288/288, unaffected.

No prior design's source file was modified by this task, except the
generalized generator script itself (extended with a new optional,
backward-compatible `--input-mode` flag; see Final Report for details).

## 9. 200T synthesis result

`stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp` on
`xc7a200tsbg484-1`
(`synth_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_relu_clamp_200t.tcl`):

| Resource | Used | Available | Util% |
|---|---:|---:|---:|
| LUTs | 1914 | 134,600 | 1.42% |
| Registers | 1372 | 269,200 | 0.51% |
| DSP48E1 | 0 | 740 | 0.00% |
| Block RAM tiles | 0 | 365 | 0.00% |

- WNS (setup slack): **+2.461 ns** -- the 100 MHz / 10.000 ns clock
  constraint **is met**.
- WHS (hold slack): +0.262 ns.
- Total on-chip power (estimate): **0.168 W** (Dynamic 0.046 W, Static
  0.122 W).
- **`synth_design` completed successfully with 0 warnings, 0 critical
  warnings, 0 errors.**

**These figures are identical, to the last digit, to the kernel0
pipelined design's own 200T synthesis result** -- expected, since the
RTL structure, folded weights, `SCALE_FX`, and `BIAS_FX` are all
byte-identical between the two designs; only the simulation-only
testbench's input stimulus differs, which has no effect on synthesis.
Per the task, synthesis was secondary here (the main goal was the
functional clamp test); this confirms the clamp-test top remains
timing-clean, as expected, rather than being a new synthesis result in
its own right.

## 10. Interpretation

- **A synthetic ReLU clamp test was added**, using an all-zero input
  patch chosen after directly inspecting `raw_conv_int32`, `product_fx`,
  and `biased_fx` to confirm it forces the clamp condition.
- **At least one negative pre-ReLU fixed-point output was clamped to
  zero** -- in fact all 9 valid output positions were, since `BIAS_FX`
  is negative for kernel 0 and an all-zero input makes `biased_fx =
  BIAS_FX` identically for every position.
- **The VHDL output matched Python Q.16 fixed-point golden outputs for
  the synthetic clamp case** -- exact integer match, GHDL-verified, with
  an explicit hard-failing assertion guarding against a silent pass.
- **This verifies the ReLU zeroing branch for the folded Conv-BN-ReLU
  proof-of-concept** -- specifically for kernel 0's pipelined two-stage
  structure. Combined with the earlier toy-input tests (which verified
  the positive-passthrough branch for kernels 0 and 2), both branches of
  the `if biased_fx_comb > 0 then ... else ... end if` logic have now
  been numerically exercised in GHDL for at least one kernel.

## 11. Limitations

- **This is a synthetic unit test, not a real UAVSAR inference example.**
  The all-zero input patch is a deliberate construction to force a known
  arithmetic condition, not derived from or representative of any real
  UAVSAR tile.
- **Kernel 0 only.** The clamp branch was verified for kernel 0's
  specific `BIAS_FX = -936`; kernel 2's `BIAS_FX = -786` is also negative
  (so an all-zero input would very likely clamp there too, by the same
  reasoning), but this was not separately built and tested in VHDL in
  this task.
- **All 9 positions clamp identically** in this test (since the input is
  uniformly zero everywhere), so this test does not exercise a *mixed*
  case where some positions clamp and others do not within the same
  streaming pass. The positive-passthrough behavior in a mixed scenario
  was not tested here; positive passthrough was verified separately (and
  exclusively) by the earlier all-positive toy-input tests.
- **Not all 32 folded Conv-BN-ReLU channels are implemented.**
- **No padding, pooling, downstream layers, or full model pipeline.**
  The full U-Net does not fit and was not evaluated; end-to-end flood
  segmentation was not run on FPGA.
- Synthesis-level (out-of-context `synth_design`) result only, not
  placed-and-routed or board-tested. WNS/WHS/power are Vivado estimates,
  not measurements. As noted in Section 9, the synthesis result here is
  identical to the already-established kernel0 pipelined result and was
  not expected to differ, since the RTL is unchanged from that design.
- **This does not prove all future kernels will clamp correctly** --
  only kernel 0, with this specific synthetic input, was tested.

## 12. Recommended next step

Build a **mixed** synthetic test (some output positions positive, others
negative within the same streaming pass) to confirm the pipeline
correctly switches between the pass-through and clamp branches
position-by-position in a single run, rather than uniformly taking one
branch for all 9 outputs as this test does -- this would be a stronger
confirmation that the two-stage pipeline's Stage-2 compare-and-select
logic is not somehow tied to a fixed decision across the whole streaming
pass.
