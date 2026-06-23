"""
estimate_fpga_conv3x3_scaling.py

Scaling estimates for the FPGA 3×3 streaming convolution prototype:
  - MACs, weights, DSP requirements for one output channel and all 32
  - Comparison against Artix-7 35T DSP budget
  - Output pixel count for valid vs padded convolution
  - Notes on time-multiplexing and next steps

Output: hardware/vhdl_conv3x3/fpga_scaling_estimate.md
"""

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_PATH  = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "fpga_scaling_estimate.md"

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Convolution kernel geometry
IN_CHANNELS   = 3
KERNEL_H      = 3
KERNEL_W      = 3
KERNEL_SIZE   = KERNEL_H * KERNEL_W          # 9 positions per channel

# Weights per output channel
WEIGHTS_PER_OUT_CHAN = IN_CHANNELS * KERNEL_SIZE       # 3 × 9 = 27

# MACs per output pixel, one output channel
MACS_PER_PIXEL_1CH  = WEIGHTS_PER_OUT_CHAN             # 27

# First U-Net layer parameters
BASE_CHANNELS = 32                                     # from checkpoint args
MACS_PER_PIXEL_FULL = MACS_PER_PIXEL_1CH * BASE_CHANNELS  # 864

# Fully parallel DSPs: one MAC → one DSP48E1
DSP_1CH    = MACS_PER_PIXEL_1CH                        # 27
DSP_FULL   = MACS_PER_PIXEL_FULL                       # 864

# Artix-7 35T available DSP slices (Xilinx UG574 / DS180)
ARTIX35_DSP    = 90
ARTIX35_LUT    = 20_800
ARTIX35_FF     = 41_600
ARTIX35_BRAM36 = 50

# Weight counts (INT8 = 1 byte each)
WEIGHTS_1CH_BYTES   = WEIGHTS_PER_OUT_CHAN             # 27 bytes
WEIGHTS_FULL_BYTES  = WEIGHTS_PER_OUT_CHAN * BASE_CHANNELS  # 864 bytes

# Output pixel count: valid convolution on 256×256 tile (no padding)
TILE_H, TILE_W  = 256, 256
VALID_H = TILE_H - KERNEL_H + 1   # 254
VALID_W = TILE_W - KERNEL_W + 1   # 254
VALID_PIXELS = VALID_H * VALID_W   # 64516

# Output pixel count: padded convolution (matches trained model padding=1)
PAD_PIXELS = TILE_H * TILE_W       # 65536

# -----------------------------------------------------------------------
# Derived fractions and comments
# -----------------------------------------------------------------------

dsp_fraction_1ch   = DSP_1CH  / ARTIX35_DSP * 100
dsp_fraction_full  = DSP_FULL / ARTIX35_DSP * 100
dsp_feasible_1ch   = DSP_1CH  <= ARTIX35_DSP
dsp_feasible_full  = DSP_FULL <= ARTIX35_DSP

# How many output channels fit in the DSP budget (fully parallel)
max_parallel_channels = ARTIX35_DSP // DSP_1CH        # floor(90/27) = 3

# -----------------------------------------------------------------------
# Print to console
# -----------------------------------------------------------------------

print(f"MACs per output pixel (1 output channel): {MACS_PER_PIXEL_1CH}")
print(f"MACs per output pixel (all {BASE_CHANNELS} output channels): {MACS_PER_PIXEL_FULL}")
print()
print(f"Fully parallel DSPs needed (1 output channel): {DSP_1CH}")
print(f"Fully parallel DSPs needed ({BASE_CHANNELS} output channels): {DSP_FULL}")
print(f"Artix-7 35T available DSPs:                  {ARTIX35_DSP}")
print()
print(f"1-channel cell uses {dsp_fraction_1ch:.0f}% of 35T DSP budget: "
      f"{'feasible' if dsp_feasible_1ch else 'EXCEEDS'}")
print(f"Full {BASE_CHANNELS}-channel first layer uses {dsp_fraction_full:.0f}% of 35T DSP budget: "
      f"{'feasible' if dsp_feasible_full else 'EXCEEDS BUDGET'}")
print(f"Maximum fully-parallel output channels on 35T: {max_parallel_channels}")
print()
print(f"First-layer weight count (1 output channel): {WEIGHTS_1CH_BYTES} INT8 values ({WEIGHTS_1CH_BYTES} bytes)")
print(f"First-layer weight count (all {BASE_CHANNELS} channels): {WEIGHTS_FULL_BYTES} INT8 values ({WEIGHTS_FULL_BYTES} bytes)")
print()
print(f"Output pixels — valid conv (256×256 tile, no padding): {VALID_H}×{VALID_W} = {VALID_PIXELS:,}")
print(f"Output pixels — padded conv (as in trained model):     {TILE_H}×{TILE_W} = {PAD_PIXELS:,}")
print(f"Valid-conv pixel deficit vs padded: {PAD_PIXELS - VALID_PIXELS} pixels ({(PAD_PIXELS - VALID_PIXELS)/PAD_PIXELS*100:.2f}%)")

# -----------------------------------------------------------------------
# Write markdown report
# -----------------------------------------------------------------------

md = f"""\
# FPGA Scaling Estimate — 3×3 Streaming Convolution Prototype

**Scope:** First Conv2d layer of the UAVSAR U-Net: `Conv2d(3, {BASE_CHANNELS}, 3, padding=1)`.
**Device:** Digilent Cmod A7-35T — Artix-7 XC7A35T, {ARTIX35_DSP} DSP slices, {ARTIX35_LUT:,} LUT6.

---

## 1. Compute Demand

| Quantity | Value | Notes |
|---|---|---|
| Input channels (`C_in`) | {IN_CHANNELS} | UAVSAR: VV, VH, magnitude (3 channels) |
| Kernel size | {KERNEL_H}×{KERNEL_W} | |
| Weights per output channel | {WEIGHTS_PER_OUT_CHAN} | {IN_CHANNELS} channels × {KERNEL_SIZE} positions |
| **MACs per output pixel (1 output ch)** | **{MACS_PER_PIXEL_1CH}** | {IN_CHANNELS} × {KERNEL_H} × {KERNEL_W} |
| Output channels (`C_out = base_channels`) | {BASE_CHANNELS} | from checkpoint `args.base_channels` |
| **MACs per output pixel (all {BASE_CHANNELS} output ch)** | **{MACS_PER_PIXEL_FULL}** | {MACS_PER_PIXEL_1CH} × {BASE_CHANNELS} |

For a 256×256 tile (valid convolution → {VALID_H}×{VALID_W} output):

| Scope | Total MACs per tile |
|---|---|
| 1 output channel | {MACS_PER_PIXEL_1CH} × {VALID_PIXELS:,} = **{MACS_PER_PIXEL_1CH * VALID_PIXELS:,}** |
| All {BASE_CHANNELS} output channels | {MACS_PER_PIXEL_FULL} × {VALID_PIXELS:,} = **{MACS_PER_PIXEL_FULL * VALID_PIXELS:,}** |

---

## 2. DSP Utilization — Fully Parallel Implementation

Each INT8 × INT8 multiplication maps to one DSP48E1 slice on Artix-7.

| Configuration | DSPs required | Artix-7 35T budget | Utilization | Verdict |
|---|---|---|---|---|
| 1 output channel (prototype) | {DSP_1CH} | {ARTIX35_DSP} | {dsp_fraction_1ch:.0f}% | **Feasible** |
| All {BASE_CHANNELS} output channels (fully parallel) | {DSP_FULL} | {ARTIX35_DSP} | {dsp_fraction_full:.0f}% | **Exceeds device** |

A fully parallel implementation of the complete first layer ({DSP_FULL} DSPs) would
require approximately {DSP_FULL // ARTIX35_DSP}× the DSP count available on the 35T.
The maximum number of fully parallel output channels that fit is
**{max_parallel_channels}** (floor({ARTIX35_DSP} ÷ {DSP_1CH})).

> One output channel ({DSP_1CH} DSPs, {dsp_fraction_1ch:.0f}% utilization) is a feasible
> prototype on the Cmod A7-35T.  Scaling to all {BASE_CHANNELS} output channels on this
> board requires time-multiplexing the MAC array over multiple clock cycles.

---

## 3. Weight Storage

| Scope | Count | Size (INT8, 1 byte each) |
|---|---|---|
| 1 output channel | {WEIGHTS_1CH_BYTES} INT8 values | {WEIGHTS_1CH_BYTES} B |
| All {BASE_CHANNELS} output channels (first layer) | {WEIGHTS_FULL_BYTES} INT8 values | {WEIGHTS_FULL_BYTES} B ({WEIGHTS_FULL_BYTES/1024:.2f} kB) |

The full first-layer weight tensor ({WEIGHTS_FULL_BYTES} bytes) fits easily in on-chip
BRAM ({ARTIX35_BRAM36} × 36 kB = {ARTIX35_BRAM36 * 36} kB available on 35T).  Weight loading from BRAM
to DSP inputs enables time-multiplexed channel reuse without off-chip memory.

---

## 4. Output Pixel Count

The trained model uses `padding=1`, preserving spatial dimensions.
The current VHDL prototype uses **valid convolution** (no padding).

| Mode | Output spatial size | Output pixels |
|---|---|---|
| Valid convolution (current VHDL) | {VALID_H}×{VALID_W} | {VALID_PIXELS:,} |
| Padded convolution (trained model) | {TILE_H}×{TILE_W} | {PAD_PIXELS:,} |
| Difference | {TILE_H - VALID_H}px border per edge | {PAD_PIXELS - VALID_PIXELS:,} pixels ({(PAD_PIXELS - VALID_PIXELS)/PAD_PIXELS*100:.2f}%) |

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
| Time to process one {VALID_H}×{VALID_W} output (1 channel) | ≈ {VALID_PIXELS/100e6*1000:.2f} ms |
| Time to process all {BASE_CHANNELS} channels (time-multiplexed) | ≈ {VALID_PIXELS * BASE_CHANNELS /100e6*1000:.2f} ms (no reuse overhead) |

> These are order-of-magnitude estimates only.  Actual Fmax, pipeline stall
> cycles, memory bandwidth, and control overhead are not accounted for and
> require synthesis and post-place-and-route timing analysis.

---

## 6. Realistic Next Steps

The current prototype validates arithmetic correctness for one output channel.
Reaching a deployable multi-channel accelerator requires:

| Step | Description |
|---|---|
| **Time-multiplexing** | Reuse the single `stream_conv3x3_3chan_cell` for all {BASE_CHANNELS} output channels by cycling the weight ROM and accumulating into separate output buffers |
| **Channel reuse control** | Add a weight address counter and output-channel mux; one complete tile pass per output channel |
| **Partial parallelism** | Instantiate *N* cells (e.g., N = {max_parallel_channels} to fit DSP budget) and stripe output channels across them |
| **BRAM line buffers** | Replace shift-register line buffers with BRAM-based ping-pong buffers to support larger image widths and reduce FF utilization |
| **BatchNorm folding** | Absorb `enc1.block.1` (BN running mean/var and affine γ/β) into the INT8 weight values and INT32 bias before synthesis; eliminates the BN layer from hardware |
| **Calibrated INT8 quantization** | Profile real UAVSAR tile pixel statistics to choose representative `scale_x` and per-channel `scale_w`; use post-training quantization calibration |
| **Padding support** | Insert a zero-padding stage before the line buffer to match `padding=1` and preserve {TILE_H}×{TILE_W} spatial dimensions |
| **Larger device** | If Artix-7 100T (240 DSPs), Zynq-7020 (220 DSPs + ARM), or Zynq UltraScale+ are available, fully parallel {BASE_CHANNELS}-channel first layer becomes feasible |

---

## 7. Summary Table

| Parameter | Value |
|---|---|
| Conv kernel | {IN_CHANNELS}×{KERNEL_H}×{KERNEL_W} = {WEIGHTS_PER_OUT_CHAN} weights per output channel |
| MACs per output pixel (1 channel) | {MACS_PER_PIXEL_1CH} |
| MACs per output pixel ({BASE_CHANNELS} channels) | {MACS_PER_PIXEL_FULL} |
| Fully parallel DSPs — 1 channel | {DSP_1CH} / {ARTIX35_DSP} available ({dsp_fraction_1ch:.0f}%) ✓ |
| Fully parallel DSPs — {BASE_CHANNELS} channels | {DSP_FULL} / {ARTIX35_DSP} available ({dsp_fraction_full:.0f}%) ✗ |
| Max fully-parallel channels on 35T | {max_parallel_channels} |
| Weight size (first layer, INT8) | {WEIGHTS_FULL_BYTES} B = {WEIGHTS_FULL_BYTES/1024:.2f} kB |
| Output pixels (valid conv, 256×256) | {VALID_PIXELS:,}  ({VALID_H}×{VALID_W}) |
| Output pixels (padded conv, as trained) | {PAD_PIXELS:,}  ({TILE_H}×{TILE_W}) |
| VHDL pipeline latency | 4 clock cycles |

*Note: All DSP estimates assume one DSP48E1 per INT8×INT8 multiplier and fully
combinational reuse.  Actual synthesis may share multipliers across clock cycles
(retiming), reducing DSP count at the cost of clock rate.*

---

*Generated by `scripts/estimate_fpga_conv3x3_scaling.py`.*
"""

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write(md)

print(f"\nWritten: {OUT_PATH.relative_to(REPO_ROOT)}")
