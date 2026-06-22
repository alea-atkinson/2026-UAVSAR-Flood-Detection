# run_vivado_sim_window.tcl
# Vivado batch-mode simulation for window3x3_stream.
#
# ---- Usage (batch mode) --------------------------------------------------
#   cd hardware/vhdl_conv3x3
#   vivado -mode batch -source run_vivado_sim_window.tcl
#
# ---- Usage (xvhdl/xelab/xsim directly) ----------------------------------
#   source /opt/Xilinx/Vivado/2024.2/settings64.sh
#   cd hardware/vhdl_conv3x3
#   xvhdl --2008 window3x3_stream.vhd tb_window3x3_stream.vhd
#   xelab -debug typical tb_window3x3_stream -s tb_win_sim
#   xsim  tb_win_sim --runall
#
# ---- Target FPGA ---------------------------------------------------------
#   Digilent Cmod A7-35T  →  Xilinx Artix-7  xc7a35tcpg236-1
#
# ---- Expected output -----------------------------------------------------
#   PASS window 1 (stream):  [1,2,3;  6,7,8;  11,12,13]
#   PASS window 2 (stream):  [2,3,4;  7,8,9;  12,13,14]
#   PASS window 3 (stream):  [3,4,5;  8,9,10;  13,14,15]
#   === All window3x3_stream tests PASSED ===  (9 valid windows total)

set script_dir [file dirname [file normalize [info script]]]

puts ""
puts "=== window3x3_stream Vivado Simulation ==="
puts "    Source dir: ${script_dir}"
puts ""

puts "--- Creating in-memory project (Artix-7 xc7a35tcpg236-1) ---"
create_project -in_memory -part xc7a35tcpg236-1

puts "--- Adding design source ---"
add_files -norecurse ${script_dir}/window3x3_stream.vhd
set_property file_type {VHDL 2008} [get_files window3x3_stream.vhd]

puts "--- Adding testbench ---"
add_files -fileset sim_1 -norecurse ${script_dir}/tb_window3x3_stream.vhd
set_property file_type              {VHDL 2008} [get_files tb_window3x3_stream.vhd]
set_property used_in_synthesis      false        [get_files tb_window3x3_stream.vhd]
set_property used_in_implementation false        [get_files tb_window3x3_stream.vhd]

set_property top     tb_window3x3_stream [get_filesets sim_1]
set_property top_lib xil_defaultlib      [get_filesets sim_1]

puts "--- Launching xsim ---"
launch_simulation

puts "--- Running all stimuli (~600 ns for 5x5 image) ---"
run all

puts ""
puts "=== Simulation complete ==="
puts "    Check the Tcl console or xsim log for PASS/FAIL messages."
puts ""

close_sim
