# Vivado batch synthesis for 3-channel streaming 3x3 convolution cell,
# DSP-aware 8-output variant (kernels 0-7 of enc1.block.0.weight).
# Target: Digilent Cmod A7-35T / Artix-7 XC7A35T-1CPG236C
#
# Mirrors synth_stream_conv3x3_3chan_8out_cell.tcl (the LUT-only 8-output
# flow) so resource/timing/power results are directly comparable. The only
# functional difference in the design under test is that
# conv3x3_dot_pipelined_dsp.vhd carries `use_dsp` synthesis attributes on
# its nine multiply-result signals, requesting DSP48E1 mapping instead of
# LUT fabric for each multiply.

set part_name "xc7a35tcpg236-1"
set top_name "stream_conv3x3_3chan_8out_cell_dsp"
set report_dir "hardware/vhdl_conv3x3/vivado_synth_reports"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_kernels0_to7_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_8out_cell_dsp.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/stream_conv3x3_3chan_8out_cell_dsp_utilization.txt
report_timing_summary -file $report_dir/stream_conv3x3_3chan_8out_cell_dsp_timing_summary.txt
report_power -file $report_dir/stream_conv3x3_3chan_8out_cell_dsp_power.txt

write_checkpoint -force $report_dir/stream_conv3x3_3chan_8out_cell_dsp_synth.dcp

puts "SYNTHESIS_COMPLETE: stream_conv3x3_3chan_8out_cell_dsp"
puts "Reports written to $report_dir"
