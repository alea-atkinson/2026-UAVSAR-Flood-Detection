# GHDL Simulation Summary

This folder contains GHDL simulation results for the VHDL 3x3 convolution hardware prototype.

## Tool setup

- Simulator: GHDL 5.1.1
- VHDL standard: VHDL-2008
- Date run: June 29, 2026

## Summary of passing tests

| Testbench | Purpose | Result | Log file |
|---|---|---|---|
| `tb_stream_conv3x3_3chan_realweights.vhd` | Verifies the 3-channel streaming convolution cell using real first-layer weights from the trained Alea-tuned U-Net checkpoint | Passed: 9 / 9 outputs matched Python golden vectors | `ghdl_3chan_realweights_simulation_log.txt` |
| `tb_stream_conv3x3_3chan_cell.vhd` | Verifies the toy 3-channel streaming convolution cell with known channel sums and bias | Passed: 9 / 9 outputs matched expected value -2 | `ghdl_3chan_cell_simulation_log.txt` |
| `tb_stream_conv3x3_cell.vhd` | Verifies the single-channel streaming convolution path from pixel stream to 3x3 windows to pipelined dot product | Passed: 9 / 9 outputs matched expected value -6 | `ghdl_stream_cell_simulation_log.txt` |
| `tb_window3x3_stream.vhd` | Verifies the streaming 3x3 window generator on a 5x5 input image | Passed: 9 / 9 valid windows checked | `ghdl_window_simulation_log.txt` |
| `tb_conv3x3_dot_pipelined.vhd` | Verifies the pipelined 3x3 dot-product block | Passed all tests; latency confirmed as 3 cycles | `ghdl_pipelined_dot_simulation_log.txt` |
| `tb_conv3x3_dot.vhd` | Verifies the combinational 3x3 dot-product block | Passed all tests | `ghdl_comb_dot_simulation_log.txt` |

## Real trained-weight verification

The most important verification test is `tb_stream_conv3x3_3chan_realweights.vhd`.

It uses:

- checkpoint: `models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`
- tensor: `enc1.block.0.weight`
- output channel: 0
- quantization: symmetric INT8 weight quantization
- expected outputs: Python-generated INT32 golden vectors

Expected and observed outputs:

    11287, 12749, 14211
    18597, 20059, 21521
    25907, 27369, 28831

All 9 outputs matched the Python golden vectors.

## Notes

The VHDL currently verifies the valid 3x3 convolution datapath. The trained PyTorch U-Net uses padding=1 in the first convolution, so padding and border handling remain future steps for a fuller model-compatible implementation.

This simulation result supports the hardware section as an initial feasibility study/proof of concept, not a full U-Net FPGA deployment.
