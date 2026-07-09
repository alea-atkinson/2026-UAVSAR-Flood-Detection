# Vivado batch synthesis for the PIPELINED kernel-2 folded Conv-BN-ReLU
# proof-of-concept (stream_conv3x3_3chan_kernel2_bn_relu_pipelined) --
# a harder-kernel generalization test of the kernel0 pipelined
# timing-closure fix.
# Target: Artix-7 200T, xc7a200tsbg484-1 (synthesis target only -- NOT a
# claim of board possession/testing; see
# hardware/vhdl_conv3x3/larger_fpga_target_comparison_summary.md for why
# this part was selected as this project's larger comparison target).
#
# Same clock (100 MHz / 10.000 ns) and out-of-context synthesis style as
# every other design in this directory, and directly comparable to
# synth_stream_conv3x3_3chan_kernel0_bn_relu_pipelined_200t.tcl. This
# design reuses the existing, unmodified stream_conv3x3_3chan_cell.vhd /
# conv3x3_dot_pipelined.vhd / window3x3_stream.vhd, plus the new
# first_conv_bn_relu_kernel2_pkg.vhd and
# stream_conv3x3_3chan_kernel2_bn_relu_pipelined.vhd.

set part_name "xc7a200tsbg484-1"
set top_name "stream_conv3x3_3chan_kernel2_bn_relu_pipelined"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_kernel2_bn_relu_pipelined_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_cell.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_conv_bn_relu_kernel2_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel2_bn_relu_pipelined.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
