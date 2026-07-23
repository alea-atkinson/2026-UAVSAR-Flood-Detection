# UAVSAR Flood-Segmentation FPGA Feasibility: Hardware Results Summary

Compiled July 23, 2026. This document collects the major hardware
milestones from this side project into one place, pulling numbers only
from existing markdown reports and Vivado/GHDL summaries already
committed under `hardware/vhdl_conv3x3/` (and, for model-behavior
context, `outputs/hardware_fidelity/`). No numbers are invented here;
every figure below is traceable to a source file cited in its row or
paragraph.

## 1. Scope

This is an FPGA feasibility study for the trained UAVSAR
flood-segmentation model's **first learned Conv-BN-ReLU inference
stage only** (`enc1.block.0` Conv2d -> `enc1.block.1` BatchNorm2d ->
ReLU, the first 3-channel-input layer of the encoder). It is **not** a
full FPGA implementation of the U-Net, **not** a board deployment (every
result here is GHDL simulation and/or Vivado out-of-context synthesis
estimation, never a programmed physical device), and **not** a measured
hardware speedup or measured hardware power claim -- every latency,
resource, timing, and power figure quoted below is either an RTL
simulation cycle count or a Vivado synthesis-level estimate, not a
board measurement.

## 2. Main artifact chain

```
Trained PyTorch checkpoint
  (models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt)
        |
        v
Learned first-layer weights + BatchNorm parameters
  (enc1.block.0.weight, enc1.block.1.{weight,bias,running_mean,running_var})
        |
        v
Fixed-point / Python golden vectors
  (BatchNorm folded into Conv weights/bias; symmetric INT8 weight +
   activation quantization; Q.16/Q.20 fixed-point scale/bias constants;
   exact-integer golden outputs computed in Python)
        |
        v
VHDL simulation (GHDL)
  (toy 5x5x3 patch, then real UAVSAR-tile-derived patches; bit-exact
   integer comparison against the Python golden vectors)
        |
        v
Vivado synthesis / resource estimates
  (Artix-7 35T and 200T targets, out-of-context synth_design; LUT/
   register/DSP/BRAM utilization, timing summary, vector-less power
   estimate)
        |
        v
Segmentation-impact analysis
  (swap the fixed-point first-layer approximation into the otherwise
   unmodified float PyTorch U-Net; measure Dice/IoU/prediction-flip
   impact on real held-out UAVSAR tiles)
```

## 3. Key verification milestones

All "GHDL checks" and "max integer difference" figures below are exact
integer comparisons between VHDL simulation output and Python-computed
golden vectors -- every design listed passed 100% of its checks with a
max integer difference of exactly 0 (bit-exact).

| Milestone | Commit | Scope | GHDL checks | Max int diff | Main meaning |
|---|---|---|---|---:|---|
| One-kernel original verification | `358a50a7` | Kernel 0 only (1/32 channels), folded Conv-BN-ReLU, Q.16, synthetic 5x5x3 toy patch | 9/9 PASS | 0 | First genuine fixed-point (integer-only) Conv-BN-ReLU datapath, bit-exact vs. Python. This specific unpipelined synthesis did NOT close 100 MHz timing (WNS -0.609 ns), motivating the later pipelined redesign (see synthesis table, kernel0/kernel2 pipelined). |
| 8-output verification | `07faf24b` | Kernels 0-7 (8/32 channels), raw Conv2d (no BN/ReLU folding yet), toy patch | 72/72 PASS | 0 | Confirmed the window-sharing streaming architecture scales correctly to multiple kernels with sub-linear LUT/register growth, before DSP-aware or BN-folded variants were introduced. |
| 32-output direct-parallel first Conv2d verification | `af222b1a` | Complete 32/32 output channels, raw Conv2d (no BN/ReLU), DSP-aware, Artix-7 200T | 288/288 PASS | 0 | Complete first Conv2d layer synthesizes and meets 100 MHz timing on the larger Artix-7 200T, but fully saturates the part's DSP budget (740/740, 100%, 0 headroom). |
| 32-output resource-shared Conv-BN-ReLU verification | `3b682e43` | Complete 32/32 output channels, folded Conv-BN-ReLU, 1-lane time-multiplexed | 288/288 PASS | 0 | DSP demand made independent of kernel count (2 DSPs for all 32 channels) at the cost of ~2,000x more clock cycles (10,125 vs. ~4-5) for the same toy input -- the core resource/latency tradeoff of this architecture family. |
| 2-lane resource-shared Conv-BN-ReLU verification | `255b344d` | Complete 32/32 output channels, 2 parallel time-multiplexed lanes | 288/288 PASS | 0 | ~2x latency cut (10,125 -> 5,085 cycles) vs. the 1-lane design at 2x the DSP cost (2 -> 4), confirming multi-lane resource-sharing behaves as the design memo predicted. |
| Real-tile kernel0/kernel2 verification | `3fcac25f` | Kernels 0 and 2 (2/32 channels), REAL UAVSAR tile-derived 6x6 patch (not toy), Q.16 | 32/32 PASS | 0 | First confirmation that the existing folded Conv-BN-ReLU VHDL datapath is bit-exact on genuine sensor-derived data, not only synthetic patterns. Also surfaced that Q.16's `SCALE_FX` is very coarse for real per-tile activation scales. |
| Real-tile all-32-kernel Q.20 verification | `6f61b302` | All 32/32 output channels, SAME real tile/sub-block as above, Q.20 | 512/512 PASS | 0 | Extends real-data verification to the complete first Conv2d layer at the higher-precision Q.20 format recommended for real-tile deployment accuracy. |

## 4. Key synthesis / architecture tradeoffs

All figures are Vivado `synth_design` out-of-context estimates at
100 MHz / 10.000 ns. "Target" is the Artix-7 part used for that
synthesis run (35T = `xc7a35tcpg236-1`, 200T = `xc7a200tsbg484-1`).

| Design | Target | LUTs | Registers | DSPs | BRAM | WNS (ns) | Power (W) | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1-output parallel | 35T | 2,731 / 20,800 | 1,324 / 41,600 | 0 / 90 | 0 / 50 | +4.456 | 0.120 | Baseline raw-conv datapath; comfortable timing margin, no DSP/BRAM used. |
| 4-output parallel | 35T | 5,020 / 20,800 | 3,214 / 41,600 | 0 / 90 | 0 / 50 | +5.117 | 0.177 | Sub-linear LUT growth (1.84x LUTs for 4x kernels) from sharing window generators across kernels. |
| 8-output LUT-only | 35T | 9,374 / 20,800 | 5,704 / 41,600 | 0 / 90 | 0 / 50 | +5.105 | 0.271 | 45.07% LUT utilization at only 8/32 channels -- continuing this trend to 32 channels was judged infeasible on the 35T without an architecture change. |
| 8-output DSP-aware | 35T | 6,991 / 20,800 | 4,405 / 41,600 | 90 / 90 (100%, overutilized warning; 216 multiplies requested) | 0 / 50 | +2.371 | 0.346 | 25% LUT reduction vs. LUT-only, but exhausts the ENTIRE 35T DSP budget for just 8/32 channels -- DSP-based scaling of this architecture is infeasible past ~4 kernels on this part. |
| 32-output direct-parallel DSP-aware (200T) | 200T | 10,461 / 134,600 | 4,618 / 269,200 | 740 / 740 (100%) | 0 / 365 | +2.343 | 1.324 | Complete first Conv2d layer synthesizes and meets timing on the larger part, but fully saturates the DSP budget (0 headroom); cross-channel summation fully pushed into LUT/CARRY4 fallback. |
| 32-output single-lane resource-shared Conv-BN-ReLU | 200T | 1,837 / 134,600 | 5,588 / 269,200 | 2 / 740 (0.27%) | 0 / 365 | +0.662 | 0.165 | DSP demand independent of kernel count; trades ~2,000x latency for near-total DSP/LUT savings vs. the direct-parallel design. |
| 2-lane resource-shared Conv-BN-ReLU | 200T | 2,531 / 134,600 | 5,878 / 269,200 | 4 / 740 (0.54%) | 0 / 365 | +0.229 | 0.176 | ~2x latency reduction over 1-lane at modest extra resource cost; still under 1% DSP utilization. |
| Q.16 vs. Q.20 real-tile synthesis (kernel 0 & kernel 2) | 200T | 1,984-2,110 (+3.7-3.8% Q.16->Q.20) | 1,428-1,454 (+0.6-0.8%) | 0 / 740 (unchanged, both formats) | 0 / 365 (unchanged) | +2.39 to +2.78 (all pass) | 0.170-0.173 (+~1.2%) | Q.20's real-tile deployment-accuracy improvement over Q.16 costs only a few dozen extra LUTs and ~1% more power per kernel -- essentially free. |

Sources: `vivado_synthesis_summary.md`, `vivado_4out_synthesis_summary.md`,
`vivado_8out_synthesis_summary.md`, `vivado_8out_dsp_synthesis_summary.md`,
`first_layer_32out_dsp_200t_summary.md`,
`first_layer_32out_bn_relu_resource_shared_summary.md`,
`first_layer_32out_bn_relu_2lane_resource_shared_summary.md`,
`reports/real_tile_q16_vs_q20_synthesis_comparison_summary.md`.

## 5. Model-behavior connection

- **Real-tile activation fidelity** (`outputs/hardware_fidelity/first_layer_real_tile_activation_fidelity/activation_fidelity_summary.md`):
  across 32 real fp2 held-out tiles (66,064,384 activation values
  compared), the hardware-style fixed-point folded first-layer
  activations track the true PyTorch float activations closely: MAE
  0.003198, max absolute error 0.071938, P99 absolute error 0.021899,
  Pearson correlation 0.999951.
- **All-flight-path first-layer fixed-point segmentation impact**
  (`outputs/hardware_fidelity/first_layer_fixed_point_segmentation_impact_all_fps_summary.md`):
  replacing ONLY the first Conv-BN-ReLU stage with the hardware-style
  fixed-point approximation (rest of the U-Net unchanged, float) was
  tested across all 7 leave-one-flight-path-out held-out splits. The
  largest absolute Dice change was +0.003606 and the largest absolute
  IoU change was +0.003279 (both on fp1); the highest binary
  prediction-change rate was 0.5423% (fp1) and the mean across all 7
  paths was 0.2747%. Every path's network-surgery correctness check
  passed.
- **Threshold flip diagnostics**
  (`outputs/hardware_fidelity/first_layer_fixed_point_flip_diagnostics/flip_diagnostics_summary.md`):
  for the outlier tiles examined, 100% of pixels whose binary prediction
  flipped sit within 0.1 of the 0.5 decision threshold -- the flips are
  overwhelmingly low-confidence boundary movement on pixels the float
  model itself was already uncertain about, not confidently-decided
  predictions actually changing.
- **Q.20 precision recommendation**
  (`reports/real_tile_fixed_point_precision_sensitivity_summary.md`,
  `reports/real_tile_q16_vs_q20_synthesis_comparison_summary.md`):
  Q.20 gives a large real-tile accuracy improvement over Q.16 (roughly
  4-40x lower max-abs-difference vs. the true float reference for
  kernels 0 and 2), while Q.24 gives no further improvement -- Q.20 is
  the recommended fixed-point format, confirmed bit-exact in GHDL for
  all 32 output channels (512/512 PASS) and shown to cost negligibly
  more FPGA resources than Q.16.

## 6. Final safe claim

The first learned Conv-BN-ReLU stage of this trained UAVSAR
flood-segmentation model has a working, GHDL-verified fixed-point VHDL
implementation, confirmed bit-exact against Python golden vectors on
both synthetic toy patches and real UAVSAR tile-derived patches, across
all 32 output channels. Vivado synthesis estimates show this stage fits
comfortably on Artix-7 FPGA targets under a resource-shared architecture
(low-single-digit percent DSP/LUT utilization), while a fully parallel
architecture meets timing but saturates the DSP budget of the parts
tried. Replacing this one stage with its fixed-point approximation, with
the rest of the U-Net left as trained float, changes final segmentation
Dice/IoU by well under 1% across all seven held-out flight paths tested,
and the pixels that do change prediction are overwhelmingly low-confidence
boundary cases. These results support first-layer FPGA feasibility at the
simulation and synthesis-estimate level; they are not a measured hardware
speedup, board power figure, or full-model FPGA deployment claim.

## 7. Limitations

- **First-layer / first Conv-BN-ReLU focus only.** Every result in this
  summary concerns `enc1.block.0` -> `enc1.block.1` -> ReLU, the first
  stage of the encoder -- no other layer of the U-Net has a VHDL
  implementation.
- **No full FPGA U-Net.** Nothing here implements, synthesizes, or
  estimates the complete encoder/bottleneck/decoder U-Net on FPGA.
- **No board testing.** Every figure in this summary comes from GHDL
  RTL simulation or Vivado out-of-context `synth_design` /
  `report_timing_summary` / `report_power`, never from a programmed
  physical device.
- **No measured hardware speedup or measured hardware power.** All
  latency figures are RTL simulation cycle counts (not wall-clock
  measurements on hardware), and all power figures are Vivado's
  vector-less ("Medium confidence") static estimates.
- **Some FPGA timing/power values are Vivado estimates, not
  placed-and-routed or board results** -- every synthesis run in this
  summary used `synth_design -mode out_of_context`, which does not
  perform placement or routing.
- **The real-tile all-kernel verification uses one tile and one 6x6
  sub-block location** (`tile_16_42.tif`, row_offset=125,
  col_offset=125) -- a different tile or sub-block location was not
  tried across all 32 kernels.
