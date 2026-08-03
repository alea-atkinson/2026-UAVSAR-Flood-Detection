# synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl
#
# Parameterized Vivado batch synthesis for
# first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd (the
# NEW integrated 16-output-channel streaming front end + folded
# Conv-BN-ReLU arithmetic core), targeting Artix-7 200T
# (xc7a200tsbg484-1), 100 MHz, run at a CALLER-SPECIFIED IMG_WIDTH via
# synth_design's own `-generic` mechanism -- the SAME mechanism
# synth_first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_200t.tcl
# already used for the 32-output design, so results are directly
# comparable. No width-specific wrapper file is needed; the top-level
# entity already exposes IMG_WIDTH as a generic (default 6).
#
# ---- Why this run matters ---------------------------------------------
# The 32-output integrated skeleton, at realistic widths (128, 256),
# already showed the streaming front end's line-buffer LUT/register cost
# grows linearly with IMG_WIDTH while DSPs stay untouched -- but that
# design was already at 740/740 (100%) DSPs from the core alone, leaving
# no room to observe headroom. This run measures the SAME line-buffer cost
# added to a core that uses only 48 (not 96) DSP-steered dot products, to
# see whether the 16-output subproblem remains comfortably within budget
# (DSPs, LUTs, timing) at 128- and 256-pixel-wide images -- see
# hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md
# for the actual result and interpretation.
#
# Usage:
#   vivado -mode batch -source hardware/vhdl_conv3x3/synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl \
#     -tclargs <img_width> <report_dir>
#
# Example:
#   vivado -mode batch -source .../synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl \
#     -tclargs 128 hardware/vhdl_conv3x3/vivado_reports_16out_folded_arithmetic_core_streaming_front_end_width128_200t
#
# Reports written (report_dir/<top_name>_{utilization,timing_summary,power}.txt,
# plus a .dcp checkpoint -- gitignored, not committed):

if {$argc < 2} {
    puts "ERROR: synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl requires 2 -tclargs:"
    puts "  <img_width> <report_dir>"
    puts "Got argc=$argc argv={$argv}"
    exit 1
}

set img_width  [lindex $argv 0]
set report_dir [lindex $argv 1]

set part_name "xc7a200tsbg484-1"
set top_name  "first_layer_16out_folded_arithmetic_core_streaming_front_end"

puts "=== synth_first_layer_16out_folded_arithmetic_core_streaming_front_end_200t.tcl ==="
puts "  img_width  = $img_width"
puts "  part_name  = $part_name"
puts "  report_dir = $report_dir"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/window3x3_stream_3chan_flattened.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/conv3x3_dot_pipelined_dsp.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_16out_folded_bn_relu_real_tile_q20_pkg.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_16out_folded_arithmetic_core.vhd
read_vhdl -vhdl2008 hardware/vhdl_conv3x3/first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd

synth_design -top $top_name -part $part_name -mode out_of_context \
    -generic IMG_WIDTH=$img_width

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name IMG_WIDTH=$img_width ($part_name)"
puts "Reports written to $report_dir"
