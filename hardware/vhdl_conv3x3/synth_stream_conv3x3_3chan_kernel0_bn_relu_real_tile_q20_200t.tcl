# Vivado batch synthesis for the REAL-TILE Q.20 kernel-0 folded
# Conv-BN-ReLU design (stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20).
# Target: Artix-7 200T, xc7a200tsbg484-1 -- the only larger-than-35T target
# this Vivado 2025.2 WebPACK installation exposes (confirmed today: 175
# Artix-7 parts, 0 Kintex-7/Virtex-7/UltraScale/UltraScale+ parts). Synthesis
# target only -- NOT a claim of board possession/testing.
#
# Structurally IDENTICAL to
# stream_conv3x3_3chan_kernel0_bn_relu_real_tile.vhd (Q.16) -- same
# stream_conv3x3_3chan_cell sub-component, same two-stage pipeline; only
# the constants package differs (first_conv_bn_relu_kernel0_real_tile_q20_pkg,
# Q.20 SCALE_FX/BIAS_FX, from
# hardware/vhdl_conv3x3/reports/real_tile_fixed_point_precision_sensitivity_summary.md's
# recommendation). This is the Q.20 half of the Q.16-vs-Q.20 synthesis
# comparison; see
# synth_stream_conv3x3_3chan_kernel0_bn_relu_real_tile_200t.tcl for the
# Q.16 half.

set part_name "xc7a200tsbg484-1"
set top_name "stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_kernel0_bn_relu_real_tile_q20_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_cell.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_real_tile_q20_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel0_bn_relu_real_tile_q20.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
