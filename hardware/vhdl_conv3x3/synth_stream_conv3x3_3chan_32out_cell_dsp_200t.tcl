# Vivado batch synthesis for 3-channel streaming 3x3 convolution cell,
# DSP-aware 32-output variant -- the COMPLETE first Conv2d layer
# (kernels 0-31 of enc1.block.0.weight, shape [32, 3, 3, 3]).
# Target: Artix-7 200T, xc7a200tsbg484-1 (larger synthesis target -- NOT a
# claim of board possession/testing; see
# hardware/vhdl_conv3x3/larger_fpga_target_comparison_summary.md for why
# this part was selected as the larger comparison target).
#
# Mirrors synth_stream_conv3x3_3chan_24out_cell_dsp_200t.tcl (same clock,
# same out-of-context synthesis style, same report set) so results are
# directly comparable to the existing 8-, 16-, and 24-output DSP-aware
# results on this target. The only functional difference in the design
# under test is the output-channel count (32 kernels vs. 24/16/8);
# conv3x3_dot_pipelined_dsp.vhd is unchanged.
#
# Based on the 24-output result (731/740 DSPs, 1 DSP-overutilization
# warning, CARRY4/LUT fallback evidence), this run is NOT expected to
# produce a clean, zero-warning fit -- it is a mixed DSP+LUT mapping
# experiment to see whether the complete 32-output layer synthesizes at
# all and whether timing is still met.

set part_name "xc7a200tsbg484-1"
set top_name "stream_conv3x3_3chan_32out_cell_dsp"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_32out_dsp_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_kernels0_to31_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_32out_cell_dsp.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
