# run_vivado_sim_pipelined.tcl
# Vivado batch-mode simulation for the PIPELINED conv3x3_dot testbench.
#
# ---- Usage (Option A) — Vivado batch mode --------------------------------
#   cd hardware/vhdl_conv3x3
#   vivado -mode batch -source run_vivado_sim_pipelined.tcl
#
# ---- Usage (Option B) — xvhdl/xelab/xsim directly -----------------------
# After sourcing Vivado settings64.sh:
#   cd hardware/vhdl_conv3x3
#   xvhdl --2008 conv3x3_dot_pipelined.vhd tb_conv3x3_dot_pipelined.vhd
#   xelab -debug typical tb_conv3x3_dot_pipelined -s tb_pip_sim
#   xsim  tb_pip_sim --runall
#
# ---- Target FPGA ---------------------------------------------------------
#   Digilent Cmod A7-35T  →  Xilinx Artix-7  xc7a35tcpg236-1
#
# ---- Expected output -----------------------------------------------------
#   PASS test 1 (pipelined): y = -6  (expected -6,  latency = 3 cycles)
#   PASS test 2 (pipelined): y = 19  (expected 19)
#   PASS test 3 (pipelined): y = 10  (expected 10)
#   === All conv3x3_dot_pipelined tests PASSED ===

set script_dir [file dirname [file normalize [info script]]]

puts ""
puts "=== conv3x3_dot_pipelined Vivado Simulation ==="
puts "    Source dir: ${script_dir}"
puts ""

# ---- Create in-memory project ------------------------------------------
puts "--- Creating in-memory project (Artix-7 xc7a35tcpg236-1) ---"
create_project -in_memory -part xc7a35tcpg236-1

# ---- Add pipelined design source ---------------------------------------
puts "--- Adding design file ---"
add_files -norecurse ${script_dir}/conv3x3_dot_pipelined.vhd
set_property file_type {VHDL 2008} [get_files conv3x3_dot_pipelined.vhd]

# ---- Add pipelined testbench -------------------------------------------
puts "--- Adding testbench ---"
add_files -fileset sim_1 -norecurse ${script_dir}/tb_conv3x3_dot_pipelined.vhd
set_property file_type              {VHDL 2008} [get_files tb_conv3x3_dot_pipelined.vhd]
set_property used_in_synthesis      false        [get_files tb_conv3x3_dot_pipelined.vhd]
set_property used_in_implementation false        [get_files tb_conv3x3_dot_pipelined.vhd]

# ---- Configure simulation top ------------------------------------------
set_property top     tb_conv3x3_dot_pipelined [get_filesets sim_1]
set_property top_lib xil_defaultlib           [get_filesets sim_1]

# ---- Launch and run ----------------------------------------------------
puts "--- Launching xsim ---"
launch_simulation

puts "--- Running all stimuli (3 test cases, ~500 ns) ---"
run all

puts ""
puts "=== Simulation complete ==="
puts "    Check the Tcl console or xsim log for PASS/FAIL messages."
puts ""

close_sim
