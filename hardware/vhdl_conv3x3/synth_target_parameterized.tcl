# synth_target_parameterized.tcl
# Generic, reusable, TARGET-PARAMETERIZED Vivado batch synthesis script.
#
# This mirrors the exact synth_design/report/checkpoint steps used by every
# per-design synth_*.tcl script already in this directory (e.g.
# synth_stream_conv3x3_3chan_8out_cell.tcl), but takes the target FPGA
# part, top module name, report output directory, and the list of VHDL
# source files as command-line arguments -- so ONE script can synthesize
# any existing design against any target part without duplicating
# near-identical per-design TCL files.
#
# It does NOT define any new VHDL architecture -- it only reads and
# synthesizes the SAME existing, already-verified VHDL source files used by
# the original Artix-7 flows.
#
# Usage:
#   vivado -mode batch -source hardware/vhdl_conv3x3/synth_target_parameterized.tcl \
#     -tclargs <part_name> <top_name> <report_dir> <src_file1> [<src_file2> ...]
#
# Example (8-output parallel design, larger target):
#   vivado -mode batch -source hardware/vhdl_conv3x3/synth_target_parameterized.tcl \
#     -tclargs xc7a200tsbg484-1 stream_conv3x3_3chan_8out_cell \
#     hardware/vhdl_conv3x3/vivado_reports_larger_target \
#     hardware/vhdl_conv3x3/window3x3_stream.vhd \
#     hardware/vhdl_conv3x3/conv3x3_dot_pipelined.vhd \
#     hardware/vhdl_conv3x3/first_layer_kernels0_to7_pkg.vhd \
#     hardware/vhdl_conv3x3/stream_conv3x3_3chan_8out_cell.vhd
#
# Clock target: 100 MHz / 10.000 ns, matching every existing Artix-7 flow.
# Synthesis mode: out-of-context, matching every existing Artix-7 flow.
#
# Reports written (named from $top_name, so multiple designs in the same
# report_dir do not collide):
#   $report_dir/${top_name}_utilization.txt
#   $report_dir/${top_name}_timing_summary.txt
#   $report_dir/${top_name}_power.txt
#   $report_dir/${top_name}_synth.dcp

if {$argc < 4} {
    puts "ERROR: synth_target_parameterized.tcl requires at least 4 -tclargs:"
    puts "  <part_name> <top_name> <report_dir> <src_file1> \[<src_file2> ...\]"
    puts "Got argc=$argc argv={$argv}"
    exit 1
}

set part_name  [lindex $argv 0]
set top_name   [lindex $argv 1]
set report_dir [lindex $argv 2]
set src_files  [lrange $argv 3 end]

puts "=== synth_target_parameterized.tcl ==="
puts "  part_name  = $part_name"
puts "  top_name   = $top_name"
puts "  report_dir = $report_dir"
puts "  src_files  = $src_files"

file mkdir $report_dir

foreach f $src_files {
    puts "Reading: $f"
    read_vhdl -vhdl2008 $f
}

synth_design -top $top_name -part $part_name -mode out_of_context

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization    -file $report_dir/${top_name}_utilization.txt
report_timing_summary -file $report_dir/${top_name}_timing_summary.txt
report_power          -file $report_dir/${top_name}_power.txt

write_checkpoint -force $report_dir/${top_name}_synth.dcp

puts "SYNTHESIS_COMPLETE: $top_name ($part_name)"
puts "Reports written to $report_dir"
