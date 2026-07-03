# Vivado batch synthesis for stream_conv3x3_3chan_8out_time_mux -- a
# STREAMING, time-multiplexed 8-output PROTOTYPE (wraps the unmodified
# window3x3_stream generators and the unmodified one-window
# conv3x3_3chan_8out_time_mux scheduler in a simple two-phase
# capture-then-process FSM).
# Target: Digilent Cmod A7-35T / Artix-7 XC7A35T-1CPG236C
#
# Mirrors the naming/structure of the other synth_*.tcl scripts in this
# directory so resource/timing/power results are directly comparable
# against the fully parallel 8-output streaming cell and the one-window
# time-multiplexed scheduler.

set part_name "xc7a35tcpg236-1"
set top_name "stream_conv3x3_3chan_8out_time_mux"
set report_dir "hardware/vhdl_conv3x3/vivado_synth_reports"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_kernels0_to7_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_3chan_8out_time_mux.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_8out_time_mux.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/stream_conv3x3_3chan_8out_time_mux_utilization.txt
report_timing_summary -file $report_dir/stream_conv3x3_3chan_8out_time_mux_timing_summary.txt
report_power -file $report_dir/stream_conv3x3_3chan_8out_time_mux_power.txt

write_checkpoint -force $report_dir/stream_conv3x3_3chan_8out_time_mux_synth.dcp

puts "SYNTHESIS_COMPLETE: stream_conv3x3_3chan_8out_time_mux"
puts "Reports written to $report_dir"
