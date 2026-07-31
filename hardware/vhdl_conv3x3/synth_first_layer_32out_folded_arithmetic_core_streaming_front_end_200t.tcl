# Vivado batch synthesis for
# first_layer_32out_folded_arithmetic_core_streaming_front_end -- the
# INTEGRATED SKELETON that chains the NEW 3-channel streaming front end
# (window3x3_stream_3chan_flattened.vhd, built from 3 existing, unmodified
# window3x3_stream instances) directly into the EXISTING, unmodified
# direct-parallel folded 32-output arithmetic core
# (first_layer_32out_folded_arithmetic_core.vhd), targeting Artix-7 200T
# (xc7a200tsbg484-1), same clock target (100 MHz) as every other synthesis
# in this repo.
#
# ---- Why this run matters ---------------------------------------------
# hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md
# already established that the arithmetic core ALONE synthesizes and
# closes 100 MHz timing at 740/740 DSPs (100% of this part's budget).
# Commit 77a71e35 then GHDL-verified (simulation only) that a real
# streaming front end can correctly feed that core. This run is the
# concrete test of whether ADDING that front end -- 3 more
# window3x3_stream instances' line-buffer/control logic on top of an
# already-fully-DSP-saturated core -- still fits and still closes timing
# on the SAME part, or whether the extra LUT/register cost of the line
# buffers pushes the combined design past what is achievable. See
# hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_streaming_front_end_synthesis_summary.md
# for the actual result and interpretation.
#
# Mirrors synth_first_layer_32out_folded_arithmetic_core_200t.tcl's
# structure exactly (same out-of-context synthesis style, same report
# set) so results are directly comparable.

set part_name "xc7a200tsbg484-1"
set top_name "first_layer_32out_folded_arithmetic_core_streaming_front_end"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_32out_folded_arithmetic_core_streaming_front_end_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream_3chan_flattened.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_32out_folded_arithmetic_core.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
