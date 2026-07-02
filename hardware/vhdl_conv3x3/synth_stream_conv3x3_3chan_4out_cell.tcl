# Vivado batch synthesis for 3-channel streaming 3x3 convolution cell,
# 4-output prototype (kernels 0-3 of enc1.block.0.weight).
# Target: Digilent Cmod A7-35T / Artix-7 XC7A35T-1CPG236C
#
# Mirrors synth_stream_conv3x3_3chan_cell.tcl (the 1-output flow) so
# resource/timing/power results are directly comparable.

set part_name "xc7a35tcpg236-1"
set top_name "stream_conv3x3_3chan_4out_cell"
set report_dir "hardware/vhdl_conv3x3/vivado_synth_reports"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_kernels0_to3_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_4out_cell.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/stream_conv3x3_3chan_4out_cell_utilization.txt
report_timing_summary -file $report_dir/stream_conv3x3_3chan_4out_cell_timing_summary.txt
report_power -file $report_dir/stream_conv3x3_3chan_4out_cell_power.txt

write_checkpoint -force $report_dir/stream_conv3x3_3chan_4out_cell_synth.dcp

puts "SYNTHESIS_COMPLETE: stream_conv3x3_3chan_4out_cell"
puts "Reports written to $report_dir"
