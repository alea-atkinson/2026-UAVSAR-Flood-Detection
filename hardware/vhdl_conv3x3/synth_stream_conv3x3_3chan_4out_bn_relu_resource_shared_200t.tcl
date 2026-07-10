# Vivado batch synthesis for the RESOURCE-SHARED folded Conv-BN-ReLU
# prototype (stream_conv3x3_3chan_4out_bn_relu_time_mux).
#
# PRESERVED COPY: this is the frozen 4-of-32-kernel prototype's own
# synthesis script, kept under this "_4out_" name after the design was
# scaled to the complete 32-kernel version (see
# synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_200t.tcl and
# hardware/vhdl_conv3x3/first_layer_32out_bn_relu_resource_shared_summary.md).
# This script still targets the smaller, already-verified 4-kernel design
# covering kernels 0-3, reusing ONE dot-product engine
# (conv3x3_dot_time_mux) and ONE Q.16 fixed-point BN+ReLU unit across all
# 4 kernels and all 9 window positions of the canonical 5x5x3 toy input.
#
# Target: Artix-7 200T, xc7a200tsbg484-1 (synthesis target only -- NOT a
# claim of board possession/testing).
#
# Same clock (100 MHz / 10.000 ns) and out-of-context synthesis style as
# every other design in this directory. Reuses the existing, unmodified
# window3x3_stream.vhd and conv3x3_dot_time_mux.vhd, plus the new
# first_layer_4out_bn_relu_resource_shared_pkg.vhd,
# conv3x3_3chan_4out_bn_relu_time_mux.vhd, and
# stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd.

set part_name "xc7a200tsbg484-1"
set top_name "stream_conv3x3_3chan_4out_bn_relu_time_mux"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_4out_bn_relu_resource_shared_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_4out_bn_relu_resource_shared_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_3chan_4out_bn_relu_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_4out_bn_relu_time_mux.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
