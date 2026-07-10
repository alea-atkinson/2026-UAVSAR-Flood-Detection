# 2-Lane Resource-Shared Folded Conv-BN-ReLU: Implementation Summary

Date: July 10, 2026

This is the first multi-lane implementation of the resource-shared
first-layer folded Conv-BN-ReLU architecture, following the
recommendation in
`multi_lane_resource_shared_conv_bn_relu_design_memo.md` (Section 9/12).
Lane 0 (kernels 0-15) and lane 1 (kernels 16-31) each have their own
time-multiplexed dot-product engine and their own shared Q.16 BN+ReLU
unit, running IN PARALLEL on the same captured window, with a
synchronization barrier before each window's combined 32-value output is
emitted. This is a PROTOTYPE for the COMPLETE 32-output first Conv2d
layer, not full U-Net inference and not board-tested.

## 1. What was built

- `conv3x3_3chan_16out_bn_relu_time_mux.vhd` -- one lane engine. Generic
  `KERNEL_OFFSET` (0 or 16) selects which 16-kernel slice of the
  EXISTING, UNMODIFIED `first_layer_32out_bn_relu_resource_shared_pkg`'s
  weight/SCALE_FX/BIAS_FX LUTs this instance uses. Architecturally
  identical to `conv3x3_3chan_32out_bn_relu_time_mux.vhd` (same FSM:
  `S_IDLE/S_RUN/S_BN1/S_BN2`), just with a local kernel count of 16.
- `stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd` -- top-level
  streaming wrapper. Reuses the unmodified `window3x3_stream` capture
  phase; instantiates two lane engines (`KERNEL_OFFSET => 0` and
  `KERNEL_OFFSET => 16`) fed the SAME buffered window; a barrier in the
  phase-2 FSM latches each lane's outputs independently as they arrive
  and only advances to the next window once BOTH lanes have reported
  done.
- `tb_stream_conv3x3_3chan_32out_bn_relu_2lane_time_mux.vhd` -- checks
  all 288 outputs against the EXISTING, UNCHANGED `EXPECTED_RELU_FX`
  golden array from `first_layer_32out_bn_relu_resource_shared_pkg`. No
  new Python vector generation was needed, confirming the memo's Section
  4/7 prediction that this lane partitioning preserves window-major/
  kernel-minor output order exactly.
- `run_ghdl_32out_bn_relu_2lane_resource_shared.sh` and
  `synth_stream_conv3x3_3chan_32out_bn_relu_2lane_resource_shared_200t.tcl`
  -- regression and synthesis scripts, following the existing conventions
  in this directory exactly.

The existing single-lane 32-output design (all files under
`*_32out_bn_relu_time_mux*` without `2lane`) was **not modified** and
still reproduces its original measured 10,125-cycle result unchanged
(reconfirmed below).

## 2. GHDL result

**PASS 288/288** (9 windows x 32 kernels). All outputs match the
EXISTING Q.16 fixed-point golden vectors exactly, reused unchanged from
the 1-lane design's package.

**Measured latency (5x5x3 toy input): 5,085 cycles.**

## 3. Regression results

All five required regressions were rerun, unmodified, and all passed:

| Regression | Result |
|---|---|
| `run_ghdl_kernel0_bn_relu_pipelined.sh` | PASS |
| `run_ghdl_kernel2_bn_relu_pipelined.sh` | PASS |
| `run_ghdl_kernel0_bn_relu_pipelined_relu_clamp.sh` | PASS |
| `run_ghdl_32out_dsp.sh` (direct-parallel 32-output) | PASS (288/288) |
| `run_ghdl_32out_bn_relu_resource_shared.sh` (1-lane 32-output) | PASS (288/288), **10,125 cycles reconfirmed unchanged** |

## 4. Vivado synthesis result

Target `xc7a200tsbg484-1`, 100 MHz / 10.000 ns, out-of-context
`synth_design`. Synthesis finished with **0 errors, 0 critical
warnings, 64 warnings** (all benign -- unused `product_fx_r` bit
warnings from constant-range folding in each lane's BN stage, same
pattern as the 1-lane design, plus the standard out-of-context
`HD.CLK_SRC` note).

| Metric | Value |
|---|---:|
| Slice LUTs | 2,531 / 134,600 (1.88%) |
| Slice Registers | 5,878 / 269,200 (2.18%) |
| DSP48E1 | 4 / 740 (0.54%) |
| Block RAM Tile | 0 / 365 (0.00%) |
| WNS | +0.229 ns |
| WHS | +0.262 ns |
| Total on-chip power | 0.176 W |
| Dynamic power | 0.054 W |
| Static power | 0.122 W |
| Warnings / Critical warnings / Errors | 64 / 0 / 0 |

All user-specified timing constraints are met (positive WNS and WHS at
100 MHz).

## 5. Comparison table

| Design | LUTs | Registers | DSP48E1 | BRAM | WNS | Power | Latency (5x5x3 toy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct-parallel 32-output | 10,461 | 4,618 | 740/740 | 0 | +2.343 ns | 1.324 W | ~4-5 cycles |
| Resource-shared, 1 lane (32 kernels/lane) | 1,837 | 5,588 | 2/740 | 0 | +0.662 ns | 0.165 W | 10,125 cycles (measured) |
| **Resource-shared, 2 lanes (16 kernels/lane) -- this task** | **2,531** | **5,878** | **4/740** | **0** | **+0.229 ns** | **0.176 W** | **5,085 cycles (measured)** |
| Memo's 2-lane estimate (Section 5) | ~3,288 | ~8,424 | ~4 | -- | -- | -- | ~5,065 |

## 6. Comparison against the memo's 2-lane estimate

| Metric | Memo estimate | Measured | Delta |
|---|---:|---:|---:|
| LUTs | ~3,288 | 2,531 | **-757 (-23.0%)**, better than estimated |
| Registers | ~8,424 | 5,878 | **-2,546 (-30.2%)**, better than estimated |
| DSP48E1 | ~4 | 4 | **exact match** |
| Latency (cycles) | ~5,065 | 5,085 | **+20 (+0.4%)**, essentially exact |

The DSP and latency estimates (Section 5/6 of the memo) held almost
exactly. The LUT and, especially, register estimates were
**conservative (pessimistic)**: the memo's linear "fixed cost per lane +
per-kernel cost" model, fit from only two single-lane data points (4 and
32 kernels), overestimated real growth by roughly a quarter to a third.
A plausible explanation is that Vivado's synthesis optimizer shares some
logic across the two structurally-identical lane instances (e.g. common
subexpression elimination is more effective with two smaller,
near-identical modules present simultaneously than the naive per-instance
sum assumed) -- but the exact mechanism was not investigated further, per
scope. This is a genuinely useful, positive result for the case that
higher lane counts (4, 8) may also come in cheaper than the memo's linear
model projected, though that remains unconfirmed until those lane counts
are actually built and synthesized.

Against the two established baselines: the 2-lane design cuts latency
roughly **2.0x** versus the 1-lane design (10,125 -> 5,085 cycles) at the
cost of 2x the DSPs (2 -> 4, still 0.54% of the part) and a modest LUT
increase (1,837 -> 2,531, +37.8%). Register count, already above the
direct-parallel design's 4,618 at 1 lane (5,588), grew further to 5,878
at 2 lanes (+27.3% over direct-parallel) -- this widening gap versus
direct-parallel registers is a real, measured trend worth watching at
higher lane counts, though at 2 lanes it remains a modest excess, not a
resource-budget concern on this part (2.18% utilization).

## 7. Safe claims

**Safe now (measured):**
- "The 2-lane resource-shared design cuts measured latency on the 5x5x3
  toy input roughly in half versus the 1-lane design (10,125 -> 5,085
  cycles) while still using only 4 of 740 available DSP48E1 slices."
- "The 2-lane design's actual synthesized LUT and register counts came
  in lower than the design memo's linear extrapolation predicted,
  suggesting the memo's resource-scaling model was conservative."
- "The 2-lane design meets 100 MHz timing on `xc7a200tsbg484-1` with
  positive WNS (+0.229 ns) and WHS (+0.262 ns)."

**Not yet safe (would need further work):**
- Any claim about 4-lane or 8-lane actual resource usage -- still
  estimates until built and synthesized.
- Any claim about board-measured power, timing, or latency.
- Any claim about 256x256-tile or larger-image latency -- still a
  formula-based estimate, not measured.
- Any claim of a general "N-lane" resource-scaling law -- only 1-lane and
  2-lane are now measured data points; the true scaling curve beyond
  2 lanes remains unconfirmed.

## 8. Limitations

Same scope boundaries as every other design in this repository:
first-layer Conv-BN-ReLU only, not full U-Net, canonical 5x5x3 toy input
only (no larger tile has been run), not board-tested, no measured board
power, and this synthesis result is a Vivado out-of-context `synth_design`
report -- not a placed-and-routed or board-measured result.
