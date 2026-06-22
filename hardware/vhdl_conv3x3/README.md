# VHDL Conv3×3 Prototype — U-Net FPGA Acceleration

**This is NOT a full U-Net implementation.**

This directory is a first FPGA/VHDL prototype of the core arithmetic operation
behind U-Net convolution layers: the signed multiply-accumulate (MAC) and the
3×3 kernel dot product.  It establishes the HDL foundation for a future Conv2d
accelerator targeting the Digilent Cmod A7-35T (Xilinx Artix-7 xc7a35tcpg236-1).

---

## Connection to Profiling Results

Hardware profiling (`scripts/profile_unet_inference_for_hardware.py`,
`scripts/benchmark_unet_pipeline_variants_for_hardware.py`) of the trained
U-Net checkpoint established:

| Metric | Value |
|---|---|
| Parameters | 7,763,041 |
| Conv MACs per 256×256 tile | ~12.07 GMACs |
| Dominant operation | 3×3 Conv2d inside DoubleConv blocks |
| GPU forward pass (bs=1) | ~1.2 ms/tile (~800 tiles/sec) |
| GPU forward pass (bs=4) | ~0.94 ms/tile (~1018 tiles/sec) |
| CPU forward pass | ~29 ms/tile (~34 tiles/sec) |
| Data loading/preprocessing bottleneck | ~8.0 ms/tile in the raw pipeline; reduced substantially with DataLoader workers/preloaded tensors |

The model is dominated by 3×3 Conv2d multiplications.  Every output activation
at every spatial position is the result of a 3×3 dot product like the one
implemented here, summed over all input channels.  An FPGA systolic array or
parallel MAC array targeting this operation could serve as a prototype for accelerating the convolution-heavy part of inference, especially for embedded or lower-power deployment scenarios.

---

## File Overview

```
hardware/vhdl_conv3x3/
├── mac_unit.vhd                    Clocked MAC: acc += a*b  (INT8 in, INT32 out)
├── conv3x3_dot.vhd                 Combinational 3×3 dot product + bias (INT8/INT32)
├── conv3x3_dot_pipelined.vhd       3-stage pipelined dot product (INT8/INT32, 3-cycle latency)
├── tb_conv3x3_dot.vhd              Self-checking testbench — combinational design
├── tb_conv3x3_dot_pipelined.vhd    Self-checking testbench — pipelined design (clocked)
├── run_ghdl.sh                     GHDL simulation — combinational testbench
├── run_ghdl_pipelined.sh           GHDL simulation — pipelined testbench
├── run_vivado_sim.tcl              Vivado xsim — combinational testbench
├── run_vivado_sim_pipelined.tcl    Vivado xsim — pipelined testbench
└── README.md                       This file
```

### `mac_unit.vhd`

Clocked signed multiply-accumulate unit.  Inputs `a` and `b` are INT8
(`signed(7 downto 0)`); the accumulator `acc_out` is INT32 (`signed(31 downto 0)`).

```
acc_out <= acc_out + resize(a * b, 32)   -- on rising_edge, when en='1'
```

Controls: `rst` (synchronous reset), `clear` (zero accumulator between windows),
`en` (load product this cycle).

One MAC unit computes one term of the kernel dot product per clock cycle.
Nine MAC units in parallel (or one unit pipelined over 9 cycles) cover a full
3×3 window.  Deeper pipelines add latency but allow higher clock frequencies.

### `conv3x3_dot.vhd`

Purely combinational 3×3 dot product.  Nine multipliers in parallel, results
sign-extended to 32 bits and summed with a 32-bit bias.  No registers, no
latches, no clock.

```
y = bias + Σ(p_i * w_i)  for i = 0..8
```

This is the spatial dot product for **one output channel at one pixel position**
in a **single-input-channel** Conv2d layer.  A full layer also loops over input
channels; a full spatial map also slides the window across H×W positions.

### `conv3x3_dot_pipelined.vhd`

Three-stage registered version of `conv3x3_dot.vhd`.  Same INT8 inputs and INT32
output, but computation is split across pipeline stages so each stage's critical
path is shorter — enabling higher clock frequencies on FPGA.

```
Stage 1 (Multiply)    : register 9 products p_i*w_i  →  9 × signed(15 downto 0)
Stage 2 (Partial-Sum) : register 3 row sums           →  3 × signed(31 downto 0)
Stage 3 (Final-Out)   : register y = bias + row sums  →  signed(31 downto 0)
```

Ports add `clk`, `rst` (synchronous), `valid_in`, and `valid_out`.
`valid_out` is asserted exactly **3 clock cycles** after `valid_in`.

### `tb_conv3x3_dot.vhd`

Self-checking testbench with three test cases (combinational — no clock needed):

| Test | Pixels | Weights | Bias | Expected |
|---|---|---|---|---|
| 1 | 1..9 | Sobel-like: [1,0,−1; 1,0,−1; 1,0,−1] | 0 | **−6** |
| 2 | all 1 | all 1 | 10 | **19** |
| 3 | all 10 | all −1 | 100 | **10** |

Failure prints `FAIL: ...` at `severity failure` (stops simulation).
Success prints `PASS: ...` and `=== All ... tests PASSED ===`.

### `tb_conv3x3_dot_pipelined.vhd`

Clocked self-checking testbench.  Drives a 100 MHz clock, applies the same three
test vectors, and waits for `valid_out` before checking `y`.  Uses a
`wait until rising_edge(clk) / exit when valid_out = '1'` loop so the check
works for any pipeline depth without hard-coding cycle counts.

---

## Why Pipelining Matters on FPGA

A purely combinational 3×3 dot product chains 9 multipliers and 8 adders into
a single combinational cone.  On Artix-7 at typical PVT corners, this chain
can limit Fmax to 100–150 MHz or below, depending on routing.

Splitting the computation into registered stages breaks the critical path:

| Stage | Operation | Critical-path reduction |
|---|---|---|
| 1 | Multiply only | 9 parallel DSP48E1 — fast, isolated |
| 2 | Partial sums (3 groups of 3) | Short adder tree per group |
| 3 | Final sum + bias (3 terms) | Minimal logic |

Each stage now has a shorter critical path, so the synthesizer can place and
route at a higher clock frequency.  **The cost is latency** (3 cycles instead
of 0), but latency does not reduce throughput in a streaming pipeline: once
primed, a new result emerges every clock cycle.

For the U-Net use case this means:
- Higher Fmax → more MACs/second per FPGA → faster inference per tile.
- The 3-cycle latency is negligible compared to the hundreds of cycles needed
  to process a full 256×256 feature map through one Conv2d layer.

---

## Simulation

### Option A — GHDL (open-source, no license required)

```bash
# Install
sudo apt install ghdl          # Ubuntu/Debian
sudo dnf install ghdl          # Fedora
brew install ghdl              # macOS

# Combinational testbench
cd hardware/vhdl_conv3x3
bash run_ghdl.sh
bash run_ghdl.sh --vcd && gtkwave tb_conv3x3_dot.vcd

# Pipelined testbench
bash run_ghdl_pipelined.sh
bash run_ghdl_pipelined.sh --vcd && gtkwave tb_conv3x3_dot_pipelined.vcd
```

### Option B — Vivado xsim (requires Xilinx Vivado ≥ 2020.1)

**Combinational design — via Tcl:**
```bash
cd hardware/vhdl_conv3x3
vivado -mode batch -source run_vivado_sim.tcl
```

**Pipelined design — via Tcl:**
```bash
cd hardware/vhdl_conv3x3
vivado -mode batch -source run_vivado_sim_pipelined.tcl
```

**Via xvhdl/xelab/xsim directly (after sourcing Vivado settings):**
```bash
source /opt/Xilinx/Vivado/2024.2/settings64.sh   # adjust path/version
cd hardware/vhdl_conv3x3

# Combinational
xvhdl --2008 mac_unit.vhd conv3x3_dot.vhd tb_conv3x3_dot.vhd
xelab -debug typical tb_conv3x3_dot -s tb_conv3x3_dot_sim
xsim  tb_conv3x3_dot_sim --runall

# Pipelined
xvhdl --2008 conv3x3_dot_pipelined.vhd tb_conv3x3_dot_pipelined.vhd
xelab -debug typical tb_conv3x3_dot_pipelined -s tb_pip_sim
xsim  tb_pip_sim --runall
```

Expected output — combinational:
```
PASS: conv3x3_dot test 1 — y = -6  (expected -6)
PASS: conv3x3_dot test 2 — y = 19  (expected 19)
PASS: conv3x3_dot test 3 — y = 10  (expected 10)
=== All conv3x3_dot tests PASSED ===
```

Expected output — pipelined:
```
PASS test 1 (pipelined): y = -6  (expected -6,  latency = 3 cycles)
PASS test 2 (pipelined): y = 19  (expected 19)
PASS test 3 (pipelined): y = 10  (expected 10)
=== All conv3x3_dot_pipelined tests PASSED ===
```

---

## What This Prototype Is and Is Not

| Aspect | This prototype | Full Conv2d accelerator |
|---|---|---|
| Kernel size | 3×3, fixed | 3×3 |
| Input channels | 1 (single dot product) | C_in (32–512 in U-Net) |
| Spatial sweep | Not implemented | H×W sliding window |
| Line buffer | Not implemented | Required for streaming |
| BatchNorm | Not implemented | Fold into Conv weights |
| Activation (ReLU) | Not implemented | Comparator + clamp |
| Quantization | INT8 shown in types | Needs calibration/training |
| Pipeline registers | **Implemented** (`conv3x3_dot_pipelined.vhd`, 3 stages) | Required for high frequency |
| Skip connections | Not implemented | Requires BRAM buffering |
| Full U-Net | No | Would need all of the above |

---

## Recommended Next Steps (in order)

1. ✅ **Pipeline registers** — implemented in `conv3x3_dot_pipelined.vhd`
   (multiply → partial-sum → bias add, 3 stages, 3-cycle latency).

2. **Add a line buffer** for streaming image windows.  A 3-row FIFO with
   a shift register of width 3 presents a new 3×3 window each clock cycle
   without re-reading from BRAM, eliminating the main memory bandwidth
   bottleneck.

3. **Sum over input channels**.  Extend `conv3x3_dot` with an outer loop
   (or parallel lanes) over C_in input channels.  Each lane has its own
   3×3 dot product; the channel accumulator adds them all.

4. **Fold BatchNorm into Conv weights** before synthesis.  At inference,
   BN parameters (γ, β, μ, σ) can be absorbed into the Conv2d weight
   matrix and bias:
   ```
   w_folded = w * γ / sqrt(σ² + ε)
   b_folded = (b - μ) * γ / sqrt(σ² + ε) + β
   ```
   This eliminates separate BN hardware entirely.

5. **INT8 quantization**.  Export the model to ONNX with INT8 weights and
   activations (torch.quantization or ONNX Runtime quantization).  The
   bit widths in `mac_unit.vhd` and `conv3x3_dot.vhd` already match INT8
   inference convention: INT8 × INT8 → INT32 accumulator.

6. **Synthesize on Cmod A7-35T** via Vivado.  The xc7a35tcpg236-1 has
   90 DSP48E1 slices.  Each DSP slice implements one 18×18 signed MAC in
   one clock cycle.  A single 3×3 kernel requires 9 MACs; 10 parallel
   3×3 engines fit in 90 DSPs.

7. **Evaluate hls4ml or Vitis-AI** for a higher-level synthesis path from
   the ONNX model directly to FPGA bitstream, skipping hand-written VHDL
   for the full network while retaining this prototype as a reference for
   the core arithmetic.

---

*Generated as part of the 2026 UAVSAR Flood Detection hardware acceleration study.*
*Profiling data: `outputs/hardware_profile/unet_pipeline_variant_benchmark.md`*
