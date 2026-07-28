# Vivado batch synthesis for the COMPLETE (32-of-32-kernel) RESOURCE-SHARED
# folded Conv-BN-ReLU design (stream_conv3x3_3chan_32out_bn_relu_time_mux).
#
# Target: Kintex-7, xc7k160tfbg484-1 -- see
# hardware/vhdl_conv3x3/reports/kintex7_synthesis_comparison_summary.md
# for why this part was selected. Synthesis target only -- NOT a claim of
# board possession/testing.
#
# This is the SAME, UNMODIFIED VHDL source already synthesized on Artix-7
# 200T in synth_stream_conv3x3_3chan_32out_bn_relu_resource_shared_200t.tcl
# -- only the target part differs. No new datapath was created for this
# comparison.
#
# Same clock (100 MHz / 10.000 ns) and out-of-context synthesis style as
# every other design in this directory. Reuses the existing, unmodified
# window3x3_stream.vhd, conv3x3_dot_time_mux.vhd,
# first_layer_32out_bn_relu_resource_shared_pkg.vhd,
# conv3x3_3chan_32out_bn_relu_time_mux.vhd, and
# stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd.

set part_name "xc7k160tfbg484-1"
set top_name "stream_conv3x3_3chan_32out_bn_relu_time_mux"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_32out_bn_relu_resource_shared_kintex7"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_32out_bn_relu_resource_shared_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_3chan_32out_bn_relu_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
