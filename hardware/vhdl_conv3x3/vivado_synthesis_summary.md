# Vivado Synthesis Summary: 3-Channel Streaming 3x3 Convolution Cell

Date: June 29, 2026  
Tool: Vivado 2025.2  
Target part: xc7a35tcpg236-1  
Top module: `stream_conv3x3_3chan_cell`  
Synthesis mode: out-of-context  
Clock constraint: 10.000 ns, 100 MHz

## Result

Vivado synthesis completed successfully with:

- 0 errors
- 0 critical warnings
- 0 warnings

## Resource Utilization

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 2731 | 20800 | 13.13% |
| Slice Registers | 1324 | 41600 | 3.18% |
| DSPs | 0 | 90 | 0.00% |
| Block RAM Tiles | 0 | 50 | 0.00% |

## Timing

The design met the 100 MHz timing constraint.

| Timing metric | Value |
|---|---:|
| Clock period constraint | 10.000 ns |
| Worst setup slack | +4.456 ns |
| Worst hold slack | +0.262 ns |
| Worst pulse-width slack | +4.500 ns |
| Worst reported data path delay | 5.569 ns |

Because the design has positive setup and hold slack, the synthesized datapath meets timing at 100 MHz for the selected Artix-7 target.

## Power Estimate

| Power metric | Value |
|---|---:|
| Total on-chip power | 0.120 W |
| Dynamic power | 0.051 W |
| Device static power | 0.069 W |
| Junction temperature | 25.6 C |

## Notes

This synthesis run targets a proof-of-concept 3-channel streaming convolution cell, not a complete FPGA implementation of the full U-Net.

The synthesized module includes three streaming 3x3 window generators and three pipelined 3x3 dot-product datapaths, followed by channel accumulation and bias addition.

Vivado mapped the design to LUT/register logic and did not infer DSP or BRAM usage for this small constant-weight convolution cell. This is acceptable for the proof-of-concept result, but a larger scaled design may need explicit DSP-aware architecture choices.

The corresponding GHDL simulation verified the same 3-channel convolution datapath against Python-generated INT32 golden outputs using real quantized first-layer U-Net weights.
