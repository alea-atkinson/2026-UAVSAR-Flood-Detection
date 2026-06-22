# run_vivado_sim_stream_cell.tcl
# Vivado batch-mode simulation for stream_conv3x3_cell.
#
# ---- Usage (batch mode) --------------------------------------------------
#   cd hardware/vhdl_conv3x3
#   vivado -mode batch -source run_vivado_sim_stream_cell.tcl
#
# ---- Usage (xvhdl/xelab/xsim directly) ----------------------------------
#   source /opt/Xilinx/Vivado/2024.2/settings64.sh
#   cd hardware/vhdl_conv3x3
#   xvhdl --2008 window3x3_stream.vhd conv3x3_dot_pipelined.vhd \
#               stream_conv3x3_cell.vhd tb_stream_conv3x3_cell.vhd
#   xelab -debug typical tb_stream_conv3x3_cell -s tb_cell_sim
#   xsim  tb_cell_sim --runall
#
# ---- Target FPGA ---------------------------------------------------------
#   Digilent Cmod A7-35T  →  Xilinx Artix-7  xc7a35tcpg236-1
#
# ---- Expected output -----------------------------------------------------
#   PASS output 1 (stream_conv3x3_cell): y = -6
#   ...
#   PASS output 9 (stream_conv3x3_cell): y = -6
#   === All stream_conv3x3_cell tests PASSED ===  (9 / 9 outputs, all y = -6)

set script_dir [file dirname [file normalize [info script]]]

puts ""
puts "=== stream_conv3x3_cell Vivado Simulation ==="
puts "    Source dir: ${script_dir}"
puts ""

puts "--- Creating in-memory project (Artix-7 xc7a35tcpg236-1) ---"
create_project -in_memory -part xc7a35tcpg236-1

# ---- Design sources (dependency order) ---------------------------------
puts "--- Adding design sources ---"
foreach src {window3x3_stream.vhd conv3x3_dot_pipelined.vhd stream_conv3x3_cell.vhd} {
    add_files -norecurse ${script_dir}/${src}
    set_property file_type {VHDL 2008} [get_files ${src}]
    puts "  OK  ${src}"
}

# ---- Testbench ----------------------------------------------------------
puts "--- Adding testbench ---"
add_files -fileset sim_1 -norecurse ${script_dir}/tb_stream_conv3x3_cell.vhd
set_property file_type              {VHDL 2008} [get_files tb_stream_conv3x3_cell.vhd]
set_property used_in_synthesis      false        [get_files tb_stream_conv3x3_cell.vhd]
set_property used_in_implementation false        [get_files tb_stream_conv3x3_cell.vhd]

set_property top     tb_stream_conv3x3_cell [get_filesets sim_1]
set_property top_lib xil_defaultlib         [get_filesets sim_1]

puts "--- Launching xsim ---"
launch_simulation

puts "--- Running all stimuli (25 pixels + 3-clock drain, ~500 ns) ---"
run all

puts ""
puts "=== Simulation complete ==="
puts "    Check the Tcl console or xsim log for PASS/FAIL messages."
puts ""

close_sim
