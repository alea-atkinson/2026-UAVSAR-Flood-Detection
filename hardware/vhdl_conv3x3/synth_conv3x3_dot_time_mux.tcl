# Vivado batch synthesis for conv3x3_dot_time_mux -- a single-lane,
# time-multiplexed 3x3 dot-product PROTOTYPE (one reusable MAC, 9 cycles
# per dot product).
# Target: Digilent Cmod A7-35T / Artix-7 XC7A35T-1CPG236C
#
# Mirrors the naming/structure of the other synth_*.tcl scripts in this
# directory so resource/timing/power results are directly comparable
# against the fully parallel 1/4/8-output prototypes.

set part_name "xc7a35tcpg236-1"
set top_name "conv3x3_dot_time_mux"
set report_dir "hardware/vhdl_conv3x3/vivado_synth_reports"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_time_mux.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/conv3x3_dot_time_mux_utilization.txt
report_timing_summary -file $report_dir/conv3x3_dot_time_mux_timing_summary.txt
report_power -file $report_dir/conv3x3_dot_time_mux_power.txt

write_checkpoint -force $report_dir/conv3x3_dot_time_mux_synth.dcp

puts "SYNTHESIS_COMPLETE: conv3x3_dot_time_mux"
puts "Reports written to $report_dir"
