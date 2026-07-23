# Vivado batch synthesis for the REAL-TILE Q.16 kernel-2 folded
# Conv-BN-ReLU design (stream_conv3x3_3chan_kernel2_bn_relu_real_tile).
# Target: Artix-7 200T, xc7a200tsbg484-1 -- the only larger-than-35T target
# this Vivado 2025.2 WebPACK installation exposes (confirmed today: 175
# Artix-7 parts, 0 Kintex-7/Virtex-7/UltraScale/UltraScale+ parts). Synthesis
# target only -- NOT a claim of board possession/testing.
#
# Same clock (100 MHz / 10.000 ns) and out-of-context synthesis style as
# every other design in this directory. This design reuses the existing,
# UNMODIFIED stream_conv3x3_3chan_cell.vhd / conv3x3_dot_pipelined.vhd /
# window3x3_stream.vhd, plus the real-tile verification's
# first_conv_bn_relu_kernel2_real_tile_pkg.vhd (Q.16 SCALE_FX, computed
# from a genuine real per-tile activation scale) and
# stream_conv3x3_3chan_kernel2_bn_relu_real_tile.vhd.
#
# Companion to synth_stream_conv3x3_3chan_kernel2_bn_relu_real_tile_q20_200t.tcl
# (Q.20 variant) -- this pair is the Q.16-vs-Q.20 synthesis comparison.

set part_name "xc7a200tsbg484-1"
set top_name "stream_conv3x3_3chan_kernel2_bn_relu_real_tile"
set report_dir "hardware/vhdl_conv3x3/vivado_reports_kernel2_bn_relu_real_tile_200t"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_cell.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_conv_bn_relu_kernel2_real_tile_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel2_bn_relu_real_tile.vhd

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
