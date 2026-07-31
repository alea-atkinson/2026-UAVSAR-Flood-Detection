# Vivado batch synthesis for first_layer_32out_folded_arithmetic_core --
# the single-shot (not streamed) 32-output folded Conv-BN-ReLU
# arithmetic-core prototype, targeting Artix-7 200T (xc7a200tsbg484-1),
# same clock target (100 MHz) as every other synthesis in this repo.
#
# ---- Why this run matters ---------------------------------------------
# This design reuses the SAME conv3x3_dot_pipelined_dsp raw-convolution IP
# (96 instances: 32 kernels x 3 channels) already used by
# stream_conv3x3_3chan_32out_cell_dsp, which ALONE already consumes
# 740/740 DSPs (100%) on this exact part (see
# first_layer_32out_dsp_200t_summary.md). This design ADDS 32 more Q.20
# scale multiplies (one per kernel, the BN-fold stage) on top of that
# already-saturated raw-conv datapath. This synthesis run is the concrete
# test of whether the benchmark's "one window per cycle, all 32 channels,
# folded BN+ReLU included, 100 MHz" assumption actually closes timing and
# fits on this part, or whether the added BN-fold multiplies push DSP/LUT
# pressure past what is achievable -- see
# hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md
# for the actual result and interpretation.
#
# Mirrors synth_stream_conv3x3_3chan_32out_cell_dsp_200t.tcl's structure
# exactly (same out-of-context synthesis style, same report set) so
# results are directly comparable.

set part_name "xc7a200tsbg484-1"
set top_name "first_layer_32out_folded_arithmetic_core"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_32out_folded_arithmetic_core_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_32out_folded_arithmetic_core.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
