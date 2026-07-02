# Vivado batch synthesis for conv3x3_3chan_8out_time_mux -- a time-multiplexed
# 8-output SCHEDULER PROTOTYPE for ONE 3-channel 3x3 spatial window (reuses
# one conv3x3_dot_time_mux instance across 24 (kernel, channel) operations).
# Target: Digilent Cmod A7-35T / Artix-7 XC7A35T-1CPG236C
#
# Mirrors the naming/structure of the other synth_*.tcl scripts in this
# directory so resource/timing/power results are directly comparable
# against the fully parallel 8-output prototype and the standalone
# time-multiplexed dot product.

set part_name "xc7a35tcpg236-1"
set top_name "conv3x3_3chan_8out_time_mux"
set report_dir "hardware/vhdl_conv3x3/vivado_synth_reports"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_kernels0_to7_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_3chan_8out_time_mux.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/conv3x3_3chan_8out_time_mux_utilization.txt
report_timing_summary -file $report_dir/conv3x3_3chan_8out_time_mux_timing_summary.txt
report_power -file $report_dir/conv3x3_3chan_8out_time_mux_power.txt

write_checkpoint -force $report_dir/conv3x3_3chan_8out_time_mux_synth.dcp

puts "SYNTHESIS_COMPLETE: conv3x3_3chan_8out_time_mux"
puts "Reports written to $report_dir"
