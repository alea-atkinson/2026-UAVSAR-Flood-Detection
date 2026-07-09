# Kernel 0 Folded Conv-BN-ReLU -- Pipelined Timing-Closure Fix

Date: July 9, 2026

This document covers a pipelined variant of the kernel-0-only VHDL
Conv-BN-ReLU proof-of-concept and its GHDL/synthesis results only. No
board testing, no measured speedup, and no measured board power are
claimed anywhere in this document.

## 1. Purpose

`hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_vhdl_summary.md`
reported that the unpipelined kernel-0 folded Conv-BN-ReLU design
(`stream_conv3x3_3chan_kernel0_bn_relu.vhd`) passed GHDL exactly (9/9
outputs matched the Python Q.16 fixed-point golden vectors) but did
**not** meet the 100 MHz timing constraint on Artix-7 200T synthesis
(WNS = -0.609 ns, 42 failing endpoints), even though the raw convolution
alone comfortably did (+4.456 ns). The suspected cause was the single
un-pipelined combinational stage combining a 32x18-bit signed multiply, a
48-bit add, and a ReLU compare-and-select. This task tests pipelining as
a timing-closure fix by splitting that one stage into two registered
stages.

## 2. What changed from the unpipelined version

**Only pipelining changed.** Per the task's explicit constraint, the
Python golden fixed-point arithmetic was **not** changed:

- Same `first_conv_bn_relu_kernel0_pkg.vhd` (existing, unmodified) --
  same `SCALE_FX = 135`, `BIAS_FX = -936`, same Q.16 format, same 9
  `K0_BN_RELU_EXPECTED_FX` golden values.
- Same `stream_conv3x3_3chan_cell.vhd` raw-convolution sub-component
  (existing, unmodified), fed the same folded INT8 kernel-0 weights.
- Same test vectors under
  `hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_kernel0/` (existing,
  unmodified).
- The **only** RTL change is splitting the unpipelined design's one
  combinational stage (multiply -> add -> ReLU compare, all before a
  single register) into two registered stages, in a **new** file
  (`stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd`) -- the existing
  unpipelined file was not modified.

## 3. Pipeline stage description

| Stage | Operation | Registered? |
|---|---|---|
| (raw conv, unchanged) | `stream_conv3x3_3chan_cell` -- 4-cycle latency, produces `raw_y` (INT32) | yes (internal to the reused sub-component) |
| **Stage 1 (new)** | `product_fx_r <= raw_y * SCALE_FX` (Q.16, exact, no shift -- `raw_y` is Q.0) | **yes** |
| **Stage 2 (new)** | `biased_fx = product_fx_r + BIAS_FX` (Q.16, no shift); `y_bn_relu_fx_r <= biased_fx if biased_fx > 0 else 0` (ReLU) | **yes** |

Stage 1 registers only the multiply result. Stage 2 computes the bias-add
and ReLU compare-and-select combinationally from the now-registered
`product_fx_r`, then registers the final result -- moving the bias-add
and ReLU compare off the same cycle as the wide multiply, exactly as
requested.

## 4. Latency difference

| Design | Latency (cycles) | First output (main-loop `i`) |
|---|---:|---:|
| Unpipelined | 5 (4 raw conv + 1 BN+ReLU) | 18 |
| **Pipelined** | **6 (4 raw conv + 2 BN+ReLU)** | **19** |

Exactly one additional cycle, as expected from adding one pipeline
register. Confirmed empirically in GHDL: every pipelined output appears
exactly 10 ns (one clock period) later than the corresponding unpipelined
output, with identical values (e.g. output 1: unpipelined at 216 ns,
pipelined at 226 ns, both `y_bn_relu_fx = 1522809`).

## 5. GHDL result

`bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu_pipelined.sh`
(VHDL-2008, GHDL 5.1.1):

- Analyzed (dependency order): `window3x3_stream.vhd`,
  `conv3x3_dot_pipelined.vhd`, `stream_conv3x3_3chan_cell.vhd` (existing,
  unmodified), `first_conv_bn_relu_kernel0_pkg.vhd` (existing,
  unmodified), `stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd`
  (new), `tb_stream_conv3x3_3chan_kernel0_bn_relu_pipelined.vhd` (new) --
  all OK.
- Elaborated and simulated -- **all 9 valid output positions matched the
  same exact Q.16 fixed-point golden values used by the unpipelined
  design's testbench**, bit-for-bit identical.
- Final report line: `=== All
  stream_conv3x3_3chan_kernel0_bn_relu_pipelined tests PASSED === (9 / 9
  outputs match Python Q.16 fixed-point golden vectors)`.
- Latency confirmed as 6 cycles (Section 4).

**Synthesis results below are interpreted as meaningful only because this
GHDL run passed all 9 checks with values identical to the unpipelined
design.**

## 6. Regression result

- `bash hardware/vhdl_conv3x3/run_ghdl_kernel0_bn_relu.sh` (existing,
  unmodified unpipelined design and testbench): **PASS**, 9/9, unaffected.
- `bash hardware/vhdl_conv3x3/run_ghdl_32out_dsp.sh` (complete 32-output
  DSP-aware regression): **PASS**, 288/288, unaffected.

Neither prior design's source file was modified by this task.

## 7. 200T synthesis result

`stream_conv3x3_3chan_kernel0_bn_relu_pipelined` on `xc7a200tsbg484-1`
(`synth_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_200t.tcl`):

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
  warnings, 0 errors** (20 informational messages only, same benign
  `HD.CLK_SRC` out-of-context timing note as every other design in this
  repo).

## 8. Comparison table: raw, unpipelined, pipelined

| Design | LUTs | Registers | DSP | WNS (ns) | Total power (W) | Warnings |
|---|---:|---:|---:|---:|---:|---|
| Raw conv (kernel 0, existing) | 2731 | 1324 | 0 | +4.456 | 0.175 | 0 |
| Unpipelined BN-ReLU (existing) | 1912 | 1330 | 0 | **-0.609** | 0.167 | 0 |
| **Pipelined BN-ReLU (this task)** | **1914** | **1372** | **0** | **+2.461** | **0.168** | **0** |

LUT count is essentially unchanged between unpipelined and pipelined
(1912 -> 1914, +2), consistent with the same arithmetic operations just
split across a register boundary rather than added or removed. Register
count increased by 42 (1330 -> 1372), consistent with the new Stage-1
`product_fx_r` register (48 bits) plus its `valid` bit needing separate
flip-flops that were not present before. Power is essentially unchanged
(0.167 W -> 0.168 W). DSP usage remains 0 in all three designs, since
this design (like the unpipelined one) reuses `conv3x3_dot_pipelined`,
the LUT-only multiply variant, not the DSP-aware one.

## 9. Whether timing recovered

**Yes.** WNS moved from -0.609 ns (unpipelined, failing, 42 failing
endpoints) to **+2.461 ns (pipelined, passing, 0 failing endpoints)** --
a swing of +3.070 ns. Splitting the single combinational
multiply-add-ReLU stage into two registered stages recovered positive
timing margin for this kernel-0 proof-of-concept at 100 MHz on the
Artix-7 200T, at the cost of one additional clock cycle of latency and a
small (42-register) increase in register count.

## 10. Interpretation

- **The pipelined kernel0 folded Conv-BN-ReLU design matches the same
  Python fixed-point golden outputs** as the unpipelined design --
  confirmed bit-for-bit identical across all 9 positions in GHDL.
- **Pipelining was tested as a timing-closure fix for the unpipelined
  fixed-point multiply/add/ReLU stage**, and it worked for this specific
  case: the added register boundary between the multiply and the
  bias-add+ReLU compare was sufficient to bring the critical path back
  under the 10 ns clock period.
- **WNS became positive (+2.461 ns), so pipelining recovered 100 MHz
  timing for this kernel0 proof-of-concept.** This is consistent with the
  unpipelined design's own suspected critical path (the wide
  multiply-then-add-then-compare chain) being the actual bottleneck, not
  DSP or LUT resource pressure -- both designs use 0 DSPs and nearly
  identical LUT counts, yet only the pipelined one closes timing.
- This result is specific to kernel 0's constants (`SCALE_FX=135`,
  `BIAS_FX=-936`) and this specific toy input; it does not establish that
  every kernel's folded scale/bias values would produce the same
  critical-path length or the same margin.

## 11. Limitations

- **Kernel 0 only.** Not all 32 folded Conv-BN-ReLU channels are
  implemented.
- **Toy input only**, and as with the unpipelined design, this specific
  toy input does not exercise ReLU's negative-clamp branch for kernel 0
  (all 9 golden values are positive both before and after ReLU).
- **One extra cycle of latency** (6 vs. 5) is the direct cost of this
  timing fix; no attempt was made to minimize that cost further (e.g. by
  balancing pipeline stages differently).
- Synthesis-level (out-of-context `synth_design`) result only, not
  placed-and-routed or board-tested. WNS/WHS/power are Vivado estimates,
  not measurements.
- No padding, pooling, downstream layers, or full model pipeline. The
  full U-Net does not fit and was not evaluated; end-to-end flood
  segmentation was not run on FPGA.
- No board testing, no measured speedup, no measured board power.

## 12. Recommended next step

Since pipelining recovered positive timing margin (+2.461 ns) for kernel
0's specific fixed-point constants, the next reasonable step is to check
whether this same two-stage pipelining approach generalizes: pick one or
two of the 32 kernels with substantially different `scale_w_folded` /
`b_folded` magnitudes (e.g. kernel 2 or kernel 11, flagged in the
Python fidelity study as having the largest folded-weight quantization
scale and the largest observed error) and confirm their fixed-point
constants also produce a positive-WNS synthesis with this same two-stage
structure, before considering extending the folded Conv-BN-ReLU
proof-of-concept to more than one output channel.
