# FPGA Scaling Estimate — 3×3 Streaming Convolution Prototype

**Scope:** First Conv2d layer of the UAVSAR U-Net: `Conv2d(3, 32, 3, padding=1)`.
**Device:** Digilent Cmod A7-35T — Artix-7 XC7A35T, 90 DSP slices, 20,800 LUT6.

---

## 1. Compute Demand

| Quantity | Value | Notes |
|---|---|---|
| Input channels (`C_in`) | 3 | UAVSAR: VV, VH, magnitude (3 channels) |
| Kernel size | 3×3 | |
| Weights per output channel | 27 | 3 channels × 9 positions |
| **MACs per output pixel (1 output ch)** | **27** | 3 × 3 × 3 |
| Output channels (`C_out = base_channels`) | 32 | from checkpoint `args.base_channels` |
| **MACs per output pixel (all 32 output ch)** | **864** | 27 × 32 |

For a 256×256 tile (valid convolution → 254×254 output):

| Scope | Total MACs per tile |
|---|---|
| 1 output channel | 27 × 64,516 = **1,741,932** |
| All 32 output channels | 864 × 64,516 = **55,741,824** |

---

## 2. DSP Utilization — Fully Parallel Implementation

Each INT8 × INT8 multiplication maps to one DSP48E1 slice on Artix-7.

| Configuration | DSPs required | Artix-7 35T budget | Utilization | Verdict |
|---|---|---|---|---|
| 1 output channel (prototype) | 27 | 90 | 30% | **Feasible** |
| All 32 output channels (fully parallel) | 864 | 90 | 960% | **Exceeds device** |

A fully parallel implementation of the complete first layer (864 DSPs) would
require approximately 9× the DSP count available on the 35T.
The maximum number of fully parallel output channels that fit is
**3** (floor(90 ÷ 27)).

> One output channel (27 DSPs, 30% utilization) is a feasible
> prototype on the Cmod A7-35T.  Scaling to all 32 output channels on this
> board requires time-multiplexing the MAC array over multiple clock cycles.

---

## 3. Weight Storage

| Scope | Count | Size (INT8, 1 byte each) |
|---|---|---|
| 1 output channel | 27 INT8 values | 27 B |
| All 32 output channels (first layer) | 864 INT8 values | 864 B (0.84 kB) |

The full first-layer weight tensor (864 bytes) fits easily in on-chip
BRAM (50 × 36 kB = 1800 kB available on 35T).  Weight loading from BRAM
to DSP inputs enables time-multiplexed channel reuse without off-chip memory.

---

## 4. Output Pixel Count

The trained model uses `padding=1`, preserving spatial dimensions.
The current VHDL prototype uses **valid convolution** (no padding).

| Mode | Output spatial size | Output pixels |
|---|---|---|
| Valid convolution (current VHDL) | 254×254 | 64,516 |
| Padded convolution (trained model) | 256×256 | 65,536 |
| Difference | 2px border per edge | 1,020 pixels (1.56%) |

For the prototype and testbenches, the 2-pixel border loss (one pixel per edge)
is acceptable.  A production implementation would pad the input stream with zeros
before the line-buffer stage to match the trained model's `padding=1`.

---

## 5. Throughput Model (One Output Channel)

Assuming one output pixel per clock at steady state (after the 4-clock pipeline
fill), and an Artix-7 35T fabric clock of 100 MHz (conservative; actual Fmax
to be determined by synthesis):

| Metric | Estimate |
|---|---|
| Clock rate (placeholder) | 100 MHz |
| Output pixels per second | 100 M pixels/s |
| Time to process one 254×254 output (1 channel) | ≈ 0.65 ms |
| Time to process all 32 channels (time-multiplexed) | ≈ 20.65 ms (no reuse overhead) |

> These are order-of-magnitude estimates only.  Actual Fmax, pipeline stall
> cycles, memory bandwidth, and control overhead are not accounted for and
> require synthesis and post-place-and-route timing analysis.

---

## 6. Realistic Next Steps

The current prototype validates arithmetic correctness for one output channel.
Reaching a deployable multi-channel accelerator requires:

| Step | Description |
|---|---|
| **Time-multiplexing** | Reuse the single `stream_conv3x3_3chan_cell` for all 32 output channels by cycling the weight ROM and accumulating into separate output buffers |
| **Channel reuse control** | Add a weight address counter and output-channel mux; one complete tile pass per output channel |
| **Partial parallelism** | Instantiate *N* cells (e.g., N = 3 to fit DSP budget) and stripe output channels across them |
| **BRAM line buffers** | Replace shift-register line buffers with BRAM-based ping-pong buffers to support larger image widths and reduce FF utilization |
| **BatchNorm folding** | Absorb `enc1.block.1` (BN running mean/var and affine γ/β) into the INT8 weight values and INT32 bias before synthesis; eliminates the BN layer from hardware |
| **Calibrated INT8 quantization** | Profile real UAVSAR tile pixel statistics to choose representative `scale_x` and per-channel `scale_w`; use post-training quantization calibration |
| **Padding support** | Insert a zero-padding stage before the line buffer to match `padding=1` and preserve 256×256 spatial dimensions |
| **Larger device** | If Artix-7 100T (240 DSPs), Zynq-7020 (220 DSPs + ARM), or Zynq UltraScale+ are available, fully parallel 32-channel first layer becomes feasible |

---

## 7. Summary Table

| Parameter | Value |
|---|---|
| Conv kernel | 3×3×3 = 27 weights per output channel |
| MACs per output pixel (1 channel) | 27 |
| MACs per output pixel (32 channels) | 864 |
| Fully parallel DSPs — 1 channel | 27 / 90 available (30%) ✓ |
| Fully parallel DSPs — 32 channels | 864 / 90 available (960%) ✗ |
| Max fully-parallel channels on 35T | 3 |
| Weight size (first layer, INT8) | 864 B = 0.84 kB |
| Output pixels (valid conv, 256×256) | 64,516  (254×254) |
| Output pixels (padded conv, as trained) | 65,536  (256×256) |
| VHDL pipeline latency | 4 clock cycles |

*Note: All DSP estimates assume one DSP48E1 per INT8×INT8 multiplier and fully
combinational reuse.  Actual synthesis may share multipliers across clock cycles
(retiming), reducing DSP count at the cost of clock rate.*

---

*Generated by `scripts/estimate_fpga_conv3x3_scaling.py`.*
