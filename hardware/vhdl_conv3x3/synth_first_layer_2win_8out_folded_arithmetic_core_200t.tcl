# Vivado batch synthesis for first_layer_2win_8out_folded_arithmetic_core --
# the NEW spatial-parallelism (2-window x 8-output) folded Conv-BN-ReLU
# arithmetic-core artifact, targeting Artix-7 200T (xc7a200tsbg484-1),
# same clock target (100 MHz) as every other synthesis in this repo.
#
# ---- Why this run matters ---------------------------------------------
# first_layer_16out_folded_arithmetic_core.vhd (commit `fe79e096`) already
# showed 1 spatial window x 16 output channels uses 405/740 DSPs, leaving
# headroom. This design reshapes the SAME total per-cycle output-channel
# budget (16, either "1 window x 16 channels" or "2 windows x 8
# channels") into 2 independent spatial lanes of 8 channels each. This
# run is the concrete test of whether that reshaping still fits and still
# closes 100 MHz timing on the SAME part -- see
# hardware/vhdl_conv3x3/reports/first_layer_2win_8out_folded_arithmetic_core_summary.md
# for the actual result and interpretation.
#
# Mirrors synth_first_layer_16out_folded_arithmetic_core_200t.tcl's
# structure exactly (same out-of-context synthesis style, same report
# set) so results are directly comparable.

set part_name "xc7a200tsbg484-1"
set top_name "first_layer_2win_8out_folded_arithmetic_core"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_2win_8out_folded_arithmetic_core_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_8out_folded_bn_relu_real_tile_q20_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_2win_8out_folded_arithmetic_core.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
