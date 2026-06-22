# U-Net Inference Profile — Hardware Acceleration Analysis

**Checkpoint**: `alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`
**Test CSV**: `heldout_fp2_test.csv`
**Device**: `cuda`
**Profiled**: 133 tiles (after 5-tile warmup)

---

## 1. Model Architecture

| Field | Value |
|---|---|
| Model type | U-Net (DoubleConv encoder-decoder + skip connections) |
| Input channels | 3 (SAR bands, per-tile normalized) |
| Base channels | 32 |
| Bottleneck channels | 512 |
| Parameters | 7,763,041 |
| Checkpoint | `alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt` |

**Layer stack**: enc1→enc2→enc3→enc4→bottleneck→dec4→dec3→dec2→dec1→out
Each `encN`/`decN` is a `DoubleConv` (two 3×3 Conv2d + BN + ReLU).
Downsampling via MaxPool2d(2); upsampling via ConvTranspose2d(kernel=2, stride=2).

---

## 2. Dataset / Inference Setup

| Field | Value |
|---|---|
| Test CSV | `heldout_fp2_test.csv` |
| Number of tiles | 138 |
| Input tensor shape | (1, 3, 256, 256) (B×C×H×W) |
| Threshold | 0.5 |
| Device | `cuda` |
| Batch size | 1 (per-tile timing) |
| DataLoader workers | 0 (synchronous, avoids timing noise) |

---

## 3. Timing Results

### Phase timing (per tile)

| Phase | Mean (ms) | Std (ms) | Min (ms) | P95 (ms) |
|---|---|---|---|---|
| Data loading | 7.153 | 0.810 | 5.914 | 8.923 |
| **Model forward** | **1.864** | 0.348 | 1.626 | 2.710 |
| Postprocessing | 0.081 | 0.018 | 0.068 | 0.123 |
| **Total pipeline** | **9.098** | — | — | — |

**Throughput**: 109.9 tiles/sec
**Latency**: 9.1 ms/tile
**Total inference time** (138 tiles): 1.26 s

---

## 4. Layer / Block Timing

Measured via PyTorch forward hooks with `torch.cuda.synchronize()` at each boundary.

| Block | Category | Mean (ms) | Std (ms) | P95 (ms) |
|---|---|---|---|---|
| `enc1` | encoder | 0.194 | 0.034 | 0.244 |
| `enc2` | encoder | 0.118 | 0.039 | 0.200 |
| `enc3` | encoder | 0.104 | 0.036 | 0.184 |
| `enc4` | encoder | 0.127 | 0.037 | 0.213 |
| `bottleneck` | bottleneck | 0.139 | 0.029 | 0.207 |
| `up4` | upsample | 0.052 | 0.014 | 0.077 |
| `dec4` | decoder | 0.165 | 0.019 | 0.203 |
| `up3` | upsample | 0.050 | 0.023 | 0.074 |
| `dec3` | decoder | 0.151 | 0.028 | 0.188 |
| `up2` | upsample | 0.053 | 0.022 | 0.070 |
| `dec2` | decoder | 0.149 | 0.017 | 0.177 |
| `up1` | upsample | 0.061 | 0.014 | 0.076 |
| `dec1` | decoder | 0.180 | 0.018 | 0.195 |
| `out` | final_conv | 0.032 | 0.021 | 0.056 |


### Block category totals

| Category | Mean total (ms) | % of timed layers |
|---|---|---|
| Encoder (enc1–enc4) | 0.542 | 34.5% |
| Bottleneck | 0.139 | 8.8% |
| Decoder (dec1–dec4) | 0.645 | 41.0% |
| Upsamplers (up1–up4) | 0.216 | 13.7% |
| Final conv (out) | 0.032 | 2.0% |

---

## 5. Compute Estimate

**Total Conv2d/ConvTranspose2d MACs per tile**: **12.07 GMACs**
*(Manual formula from Conv shapes; `thop` not installed.)*

| Block | MMACs | % of total |
|---|---|---|
| `enc1` | 660.6 | 5.5 |
| `enc2` | 906.0 | 7.5 |
| `enc3` | 906.0 | 7.5 |
| `enc4` | 906.0 | 7.5 |
| `bottleneck` | 906.0 | 7.5 |
| `dec4` | 1811.9 | 15.0 |
| `dec3` | 1811.9 | 15.0 |
| `dec2` | 1811.9 | 15.0 |
| `dec1` | 1811.9 | 15.0 |
| `out` | 2.1 | 0.0 |


---

## 6. Hardware-Relevant Bottleneck Notes

### What is the main bottleneck?

**Model forward pass** dominates. The bulk of compute time is in the
`DoubleConv` blocks (two 3×3 Conv2d + BatchNorm + ReLU each).
The bottleneck block (256→512→256 channels) is the single most
MAC-intensive stage.

- Data loading is **significant (>50% of forward time)** (3.84× forward time).
  Includes rasterio `.tif` read + per-tile percentile normalization.
- Postprocessing is **negligible (<5% of forward time)** (0.0436× forward time).

### Do convolution layers dominate?

**Yes.** All meaningful compute is Conv2d or ConvTranspose2d.
BatchNorm and ReLU are fast element-wise ops (negligible vs. convolution).
MaxPool2d is also cheap.

### Is data loading significant?

Yes — at 7.2 ms/tile it is significant (>50% of forward time). For on-device inference, consider pre-normalizing tiles.

### Is postprocessing negligible?

**Yes.** At 0.08 ms/tile, sigmoid + threshold is trivial.
It can be replaced with a LUT-based sigmoid approximation on FPGA.

### What should be accelerated first?

1. **3×3 Conv2d in DoubleConv blocks** — overwhelmingly dominant.
2. **2×2 ConvTranspose2d** upsampling — simple fixed-kernel, stride-2.
3. **BatchNorm folding** — eliminates BN layers before FPGA synthesis.

### What is the realistic next hardware step?

1. **BN folding**: absorb BatchNorm γ/β/μ/σ into Conv2d weight/bias (no inference cost).
2. **INT8 quantization**: `torch.quantization` or ONNX quantization to halve bandwidth.
3. **ONNX export** → Vitis-AI / hls4ml for FPGA synthesis flow.
4. **Tile streaming** on FPGA: 256×256 tiles with 512 channels at bottleneck
   require 128 MB on-chip per feature map — streaming
   or double-buffering needed to fit within BRAM.
5. **Target device**: mid-range Xilinx Zynq UltraScale+ (or Alveo U50) with
   sufficient DSP slices for 12.07 GMACs/tile at the required rate.

---

*Generated by `scripts/profile_unet_inference_for_hardware.py`*
*Checkpoint epoch: 18*
