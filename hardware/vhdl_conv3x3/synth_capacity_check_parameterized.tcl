# synth_capacity_check_parameterized.tcl
#
# Generic, reusable Vivado batch script that synthesizes the trivial
# capacity_check_dummy_top.vhd design against a given target part, purely
# to read that PART's own resource "Available" totals from
# report_utilization -- a reliable, Vivado-verified capacity check, not
# an assumption from the part name/family.
#
# This does NOT synthesize any project datapath. See
# hardware/vhdl_conv3x3/reports/free_tier_target_capacity_check.md for
# how this is used and interpreted.
#
# Usage:
#   vivado -mode batch -source hardware/vhdl_conv3x3/synth_capacity_check_parameterized.tcl \
#     -tclargs <part_name> <report_dir>
#
# Reports written:
#   $report_dir/capacity_check_utilization.txt

if {$argc < 2} {
    puts "ERROR: synth_capacity_check_parameterized.tcl requires 2 -tclargs:"
    puts "  <part_name> <report_dir>"
    puts "Got argc=$argc argv={$argv}"
    exit 1
}

set part_name  [lindex $argv 0]
set report_dir [lindex $argv 1]

puts "=== synth_capacity_check_parameterized.tcl ==="
puts "  part_name  = $part_name"
puts "  report_dir = $report_dir"

file mkdir $report_dir

read_vhdl -vhdl2008 hardware/vhdl_conv3x3/capacity_check_dummy_top.vhd

synth_design -top capacity_check_dummy_top -part $part_name -mode out_of_context

report_utilization -file $report_dir/capacity_check_utilization.txt

puts "CAPACITY_CHECK_COMPLETE: $part_name"
puts "Report written to $report_dir/capacity_check_utilization.txt"
