# All-Flight-Path First-Layer Fixed-Point Segmentation Impact Summary

## Claim boundary

- This is a Python segmentation-impact study, not VHDL simulation.
- Only the first Conv-BN-ReLU stage is replaced with the hardware-style fixed-point approximation.
- The rest of the U-Net remains the original trained PyTorch float model.
- No board timing, board power, or full-FPGA U-Net deployment is claimed.

## Main result

Across all 7 leave-one-flight-path-out held-out test splits, all network-surgery checks passed: **True**.

The largest absolute Dice change across held-out paths was **0.003606**.
The largest absolute IoU change across held-out paths was **0.003279**.
The highest binary prediction-change rate across held-out paths was **0.5423%**.
The mean binary prediction-change rate across held-out paths was **0.2747%**.

Interpretation: replacing only the first Conv-BN-ReLU with the hardware-style fixed-point approximation caused very small segmentation-level changes across all tested held-out flight paths.

## Per-flight-path results

| Held-out FP | Tiles | Dice float | Dice fixed-point | Dice diff | IoU float | IoU fixed-point | IoU diff | % pixels changed | Surgery passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fp1 | 137 | 0.515101 | 0.518707 | +0.003606 | 0.346893 | 0.350172 | +0.003279 | 0.5423% | True |
| fp2 | 138 | 0.784236 | 0.783395 | -0.000842 | 0.645057 | 0.643918 | -0.001138 | 0.1580% | True |
| fp3 | 164 | 0.640997 | 0.641921 | +0.000924 | 0.471667 | 0.472669 | +0.001002 | 0.1817% | True |
| fp4 | 181 | 0.538858 | 0.538651 | -0.000207 | 0.368792 | 0.368599 | -0.000193 | 0.1750% | True |
| fp5 | 160 | 0.728153 | 0.729305 | +0.001152 | 0.572516 | 0.573942 | +0.001426 | 0.4852% | True |
| fp6 | 170 | 0.654987 | 0.654852 | -0.000136 | 0.486975 | 0.486825 | -0.000150 | 0.0893% | True |
| fp7 | 117 | 0.517130 | 0.519494 | +0.002364 | 0.348736 | 0.350890 | +0.002154 | 0.2914% | True |

## Source CSV

- `outputs/hardware_fidelity/first_layer_fixed_point_segmentation_impact_all_fps_summary.csv`
