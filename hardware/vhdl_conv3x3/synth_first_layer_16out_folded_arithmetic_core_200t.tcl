# Vivado batch synthesis for first_layer_16out_folded_arithmetic_core --
# the single-shot (not streamed) 16-output (of 32) folded Conv-BN-ReLU
# arithmetic-core artifact, targeting Artix-7 200T (xc7a200tsbg484-1),
# same clock target (100 MHz) as every other synthesis in this repo.
#
# ---- Why this run matters ---------------------------------------------
# synth_first_layer_32out_folded_arithmetic_core_200t.tcl already showed
# the ALL-32-output folded core saturates this part at 740/740 (100%)
# DSPs -- zero headroom. This design reuses the SAME
# conv3x3_dot_pipelined_dsp raw-convolution IP, but only 16 x 3 = 48
# instances (HALF of the 32-output core's 96). This run is the concrete
# test of whether that smaller, comfortably-fitting subproblem still
# closes 100 MHz timing and, unlike the 32-output core, leaves real DSP
# headroom on this part -- see
# hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md
# for the actual result and interpretation.
#
# Mirrors synth_first_layer_32out_folded_arithmetic_core_200t.tcl's
# structure exactly (same out-of-context synthesis style, same report
# set) so results are directly comparable.

set part_name "xc7a200tsbg484-1"
set top_name "first_layer_16out_folded_arithmetic_core"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_16out_folded_bn_relu_real_tile_q20_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_16out_folded_arithmetic_core.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
