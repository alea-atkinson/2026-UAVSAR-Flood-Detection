# analyze_dsp_lut_fallback.tcl
# Analysis-only Vivado batch script: opens an EXISTING synthesized
# checkpoint (.dcp) and generates additional reports to characterize
# DSP48E1 vs. LUT-fabric resource mapping. Does NOT modify or re-run
# synthesis, does NOT change any RTL, and does NOT write a new checkpoint.
#
# Usage:
#   vivado -mode batch -source hardware/vhdl_conv3x3/analyze_dsp_lut_fallback.tcl \
#     -tclargs <checkpoint.dcp> <report_dir> <label>
#
# Reports written (all prefixed by $label):
#   ${label}_hierarchical_utilization.txt        (report_utilization -hierarchical)
#   ${label}_hierarchical_utilization_depth4.txt (report_utilization -hierarchical -hierarchical_depth 4)
#   ${label}_dsp_utilization.txt                 (report_dsp_utilization, if supported by this Vivado version)
#   ${label}_timing_summary.txt                  (report_timing_summary, re-derived from the opened checkpoint)
#   ${label}_dsp48e1_cell_paths.txt              (one hierarchical cell path per DSP48E1 instance)
#   ${label}_carry4_cell_paths.txt               (one hierarchical cell path per CARRY4 instance)
#   ${label}_primitive_counts.txt                (REF_NAME counts across the whole hierarchical netlist)

if {$argc < 3} {
    puts "ERROR: analyze_dsp_lut_fallback.tcl requires 3 -tclargs:"
    puts "  <checkpoint.dcp> <report_dir> <label>"
    puts "Got argc=$argc argv={$argv}"
    exit 1
}

set dcp_path   [lindex $argv 0]
set report_dir [lindex $argv 1]
set label      [lindex $argv 2]

puts "=== analyze_dsp_lut_fallback.tcl ==="
puts "  dcp_path   = $dcp_path"
puts "  report_dir = $report_dir"
puts "  label      = $label"

file mkdir $report_dir

open_checkpoint $dcp_path

puts "\n--- report_utilization -hierarchical ---"
report_utilization -hierarchical -file $report_dir/${label}_hierarchical_utilization.txt

puts "\n--- report_utilization -hierarchical -hierarchical_depth 4 ---"
report_utilization -hierarchical -hierarchical_depth 4 -file $report_dir/${label}_hierarchical_utilization_depth4.txt

puts "\n--- report_dsp_utilization (may not be supported) ---"
if {[catch {report_dsp_utilization -file $report_dir/${label}_dsp_utilization.txt} err]} {
    set fh [open $report_dir/${label}_dsp_utilization.txt w]
    puts $fh "report_dsp_utilization FAILED or is not a recognized command in this Vivado version."
    puts $fh "Error message: $err"
    close $fh
    puts "  report_dsp_utilization FAILED: $err"
} else {
    puts "  report_dsp_utilization succeeded"
}

puts "\n--- report_timing_summary (re-derived from opened checkpoint) ---"
report_timing_summary -file $report_dir/${label}_timing_summary.txt

puts "\n--- Listing DSP48E1 cell hierarchical paths ---"
set dsp_cells [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
set fh [open $report_dir/${label}_dsp48e1_cell_paths.txt w]
puts $fh "Total DSP48E1 cells: [llength $dsp_cells]"
foreach c $dsp_cells {
    puts $fh $c
}
close $fh
puts "  DSP48E1 cell count: [llength $dsp_cells]"

puts "\n--- Listing CARRY4 cell hierarchical paths ---"
set carry_cells [get_cells -hierarchical -filter {REF_NAME == CARRY4}]
set fh [open $report_dir/${label}_carry4_cell_paths.txt w]
puts $fh "Total CARRY4 cells: [llength $carry_cells]"
foreach c $carry_cells {
    puts $fh $c
}
close $fh
puts "  CARRY4 cell count: [llength $carry_cells]"

puts "\n--- Whole-netlist primitive (REF_NAME) counts ---"
set all_cells [get_cells -hierarchical]
array unset counts
foreach c $all_cells {
    set rn [get_property REF_NAME $c]
    if {![info exists counts($rn)]} {
        set counts($rn) 0
    }
    incr counts($rn)
}
set fh [open $report_dir/${label}_primitive_counts.txt w]
puts $fh "Total hierarchical cells: [llength $all_cells]"
foreach rn [lsort [array names counts]] {
    puts $fh "$rn $counts($rn)"
}
close $fh
puts "  Wrote primitive counts for [llength [array names counts]] distinct REF_NAMEs"

puts "\nANALYSIS_COMPLETE: $label"
