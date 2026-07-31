# First-Layer 32-Output Folded Arithmetic-Core Prototype: GHDL + Vivado Results

Date: July 31, 2026

This report covers a NEW, concrete hardware artifact built to test a real
architecture assumption used by
`scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py`: that a
speed-oriented, direct-parallel Artix-7 200T arithmetic core can consume
one pre-extracted, flattened 27-value 3x3x3 window per cycle and produce
all 32 folded Conv-BN-ReLU output channels per cycle (after a 6-cycle
pipeline fill), at 100 MHz. This is not a summary table of prior results --
it is a new, synthesizable VHDL design, GHDL-verified against Python
golden vectors and synthesized end-to-end on the real target part.

## Claim boundary (read this first)

- **Arithmetic-core prototype only.** No image streaming, no line
  buffers, no sliding-window generation (the caller is assumed to have
  already extracted the flattened window), no padding, no full U-Net.
- **No board testing.** GHDL simulation + Vivado synthesis estimates only
  -- no measured hardware speedup, no measured board power.
- Power figures below are Vivado's vectorless, **Medium-confidence**
  estimate, not silicon-measured.

## 1. What was built

| File | Purpose |
|---|---|
| `scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py` | Merges two existing, already-verified constant families into one consolidated Q.20 package; generates 5 golden test vectors |
| `hardware/vhdl_conv3x3/first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd` | Consolidated all-32-kernel Q.20 weight/scale/bias package (generated) |
| `hardware/vhdl_conv3x3/first_layer_32out_folded_arithmetic_core.vhd` | The new single-shot 32-output folded Conv-BN-ReLU arithmetic core |
| `hardware/vhdl_conv3x3/tb_first_layer_32out_folded_arithmetic_core_vectors_pkg.vhd` | Testbench stimulus/expected-output package (generated) |
| `hardware/vhdl_conv3x3/tb_first_layer_32out_folded_arithmetic_core.vhd` | Self-checking GHDL testbench |
| `hardware/vhdl_conv3x3/run_ghdl_32out_folded_arithmetic_core.sh` | GHDL run script |
| `hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_200t.tcl` | Vivado synthesis script, `xc7a200tsbg484-1`, 100 MHz |
| `hardware/vhdl_conv3x3/test_vectors/first_layer_32out_folded_arithmetic_core/*` | JSON/CSV/MD test-vector transparency artifacts (generated) |

**No existing, already-verified VHDL file was modified.** The new design
reuses, unmodified: `conv3x3_dot_pipelined_dsp.vhd` (the exact same
DSP-steered 3x3 dot-product IP already used by the already-synthesized
740/740-DSP 32-output direct-parallel raw-Conv2d design), and the raw
INT8 weights already in `first_layer_kernels0_to31_pkg.vhd` plus the Q.20
`SCALE_FX`/`BIAS_FX` constants already in the 32
`first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd` files (all
already GHDL-verified by the prior all-32-kernel Q.20 report). No new
weight, scale, or bias value was invented anywhere in this task.

## 2. Architecture

A **single-shot** (not streamed) module: the caller presents one
already-flattened 27-value window (9 pixels x 3 channels) on 27 named
`signed(7 downto 0)` ports every cycle; 6 cycles later, all 32 Q.20
fixed-point, ReLU-applied outputs appear on `y0..y31`. Internally, for
each of the 32 kernels: 3x `conv3x3_dot_pipelined_dsp` (existing,
unmodified, 3-cycle latency) compute the raw per-channel dot products;
Stage A (registered, cycle 4) sums the 3 channels; Stage B (registered,
cycle 5) applies the Q.20 `SCALE_FX_LUT(k)` multiply; Stage C (registered,
cycle 6) adds `BIAS_FX_LUT(k)` and applies ReLU. 96 dot-product instances
total (32 kernels x 3 channels), fully parallel, fully pipelined -- 1
window/cycle steady-state throughput.

## 3. Consistency checks (before any hardware was built)

The generator script asserted, and confirmed:
- All 32 x 3 = 96 raw INT8 weight arrays are bit-for-bit identical between
  `first_layer_kernels0_to31_pkg.vhd` (raw Conv2d weights, no BN) and the
  32 `first_conv_bn_relu_kernel{k}_real_tile_q20_pkg.vhd` files' folded
  weights -- expected, since a strictly positive per-channel BatchNorm
  scale cancels exactly out of symmetric max-abs INT8 quantization.
- A genuine 3x3x3 window sliced from the existing real-tile stimulus
  (`real_tile_stimulus_pkg.vhd`) produces, when run through this script's
  independently-computed datapath, **exactly** the same 32-kernel output
  vector as the existing, already GHDL-verified all-32-kernel Q.20
  report's expected value at output position (row=0, col=0).

## 4. GHDL results

`bash hardware/vhdl_conv3x3/run_ghdl_32out_folded_arithmetic_core.sh`
(VHDL-2008, GHDL 5.1.1):

```
PASS window 0 y0=0 y1=62052 y31=0
PASS window 1 y0=0 y1=0 y31=9895
PASS window 2 y0=2387236 y1=2057010 y31=0
PASS window 3 y0=0 y1=0 y31=883495
PASS window 4 y0=0 y1=0 y31=0
=== All first_layer_32out_folded_arithmetic_core tests PASSED ===
  (5 windows x 32 kernels = 160 / 160 outputs match Python golden vectors)
```

**160/160 outputs match.** Outputs for the 5 back-to-back windows
(presented one per clock) appeared at simulation times 96, 106, 116, 126,
136 ns -- exactly 60 ns (6 clock cycles at the 10 ns/cycle testbench
clock) after each respective input, and exactly 10 ns (1 cycle) apart
from each other -- **directly confirming, in simulation, both the assumed
6-cycle latency and the assumed 1-window/cycle steady-state throughput**,
not just asserting them in a comment.

## 5. Vivado synthesis results (`xc7a200tsbg484-1`, 100 MHz)

`vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_200t.tcl`
(Vivado 2025.2, `hpcl5-5.hslinux`):

| Resource | This design (folded, 32-out) | Existing raw-conv-only 32-out design | Delta |
|---|---:|---:|---:|
| LUTs | 12,039 (8.94%) | 10,461 (7.77%) | +1,578 |
| Registers | 6,564 (2.44%) | 4,618 (1.72%) | +1,946 |
| DSP48E1 | **740 / 740 (100.00%)** | 740 / 740 (100.00%) | +0 |
| Block RAM tiles | 0 (0.00%) | 0 (0.00%) | +0 |
| WNS (setup slack) | **+2.218 ns** | +2.343 ns | -0.125 ns |
| WHS (hold slack) | +0.262 ns | +0.262 ns | +0 |
| Total on-chip power (est.) | 1.334 W (dynamic 1.206 W + static 0.128 W) | 1.324 W (dynamic 1.196 W + static 0.128 W) | +0.010 W |
| Warnings | 2 (DSP overutilization demand estimate; out-of-context clock-skew notice) | 1 (DSP overutilization) | +1 (expected, out-of-context notice) |

**`synth_design` completed successfully. All user-specified timing
constraints are met at 100 MHz** (positive WNS and WHS). No utilization
figure exceeds 100%.

**DSP demand (pre-fallback) was reported as "Used = 1044, Available =
740"** -- the identical number already seen for the raw-conv-only design.
This means Vivado's default synthesis strategy did **not** attempt to map
the 32 new Q.20 BN-fold scale multiplies onto DSP48E1 slices at all (no
`use_dsp` attribute was applied to that stage, unlike the raw
dot-products); they were synthesized directly into LUT/carry fabric from
the start. The **entire cost of adding the folded BN + ReLU stage on top
of the already-DSP-saturated raw-conv datapath landed on LUTs and
registers** (+1,578 LUTs, +1,946 registers), not on additional DSPs --
and the design still closed timing at 100 MHz with only a small (0.125
ns) reduction in setup slack.

## 6. Does this support or weaken the benchmark's one-window-per-cycle assumption?

**It supports the assumption, with one important qualification.**

- **Supports:** the complete single-shot, 32-output, folded-BN-ReLU,
  1-window/cycle, 6-cycle-latency arithmetic core -- exactly the
  architecture the benchmark assumed -- synthesizes on the real Artix-7
  200T target, GHDL-verifies bit-exact against independently-computed
  Python golden vectors, and **meets 100 MHz timing with positive slack**.
  This is a materially stronger claim than the benchmark's original
  estimate, which only cycle-counted an assumed architecture without ever
  building or synthesizing the complete folded 32-output design.
- **Qualification (ties directly to the benchmark script's own "6b.
  Architecture sanity check" section):** the design lands at **exactly**
  the same 740/740 DSP ceiling as the raw-conv-only design, with zero DSP
  headroom remaining. The BN-fold stage's cost was absorbed entirely by
  extra LUT/register fabric, not extra DSPs, because there were no DSPs
  left to give it. This is direct, synthesis-confirmed evidence (not just
  the qualitative argument in the benchmark's Section 6b) that
  **multi-window-per-cycle unrolling is not achievable on this target
  with this same DSP-heavy mapping** -- the design is already fully using
  the part's DSP budget for ONE window per cycle. A faster design would
  need a different resource trade (more LUT-based multiplies, lower
  precision, or a larger FPGA), exactly as Section 6b already argued
  qualitatively -- this prototype now provides the quantitative synthesis
  evidence for that argument.

## 7. Limitations

- Golden test vectors are small by design (5 windows: 1 real, 4
  synthetic edge cases) -- not an exhaustive sweep of all possible INT8
  input combinations.
- Real-tile input scale (`scale_x`) is inherited from one specific tile
  (`tile_16_42.tif`); a different tile's activation range would change
  `SCALE_FX`/`BIAS_FX` (though not the raw INT8 weights).
- Power estimate is Vivado's vectorless (Medium-confidence) estimate at
  `out_of_context` synthesis, not a place-and-route or board measurement.
- This is one Conv2d+BatchNorm+ReLU stage only, not the full U-Net.
- No board was used or is claimed to have been used anywhere in this
  report.

## 8. Commands run

```bash
python3 scripts/generate_first_layer_32out_folded_arithmetic_core_vectors.py

bash hardware/vhdl_conv3x3/run_ghdl_32out_folded_arithmetic_core.sh

module load vivado/2025.2
export LD_LIBRARY_PATH="$HOME/.local/lib/vivado_compat:$LD_LIBRARY_PATH"
vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_32out_folded_arithmetic_core_200t.tcl
```
