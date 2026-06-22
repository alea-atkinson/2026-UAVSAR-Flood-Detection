# run_vivado_sim.tcl
# Vivado batch-mode simulation for the conv3x3_dot testbench.
#
# ---- Usage (Option A) — source from Vivado Tcl console ----------------
#   cd hardware/vhdl_conv3x3
#   vivado -mode batch -source run_vivado_sim.tcl
#
# ---- Usage (Option B) — xvhdl/xelab/xsim directly (no GUI needed) ----
# After sourcing Vivado settings64.sh, run these three commands:
#   cd hardware/vhdl_conv3x3
#   xvhdl --2008 mac_unit.vhd conv3x3_dot.vhd tb_conv3x3_dot.vhd
#   xelab -debug typical tb_conv3x3_dot -s tb_conv3x3_dot_sim
#   xsim  tb_conv3x3_dot_sim --runall
#
# ---- Target FPGA -------------------------------------------------------
#   Digilent Cmod A7-35T  ->  Xilinx Artix-7  xc7a35tcpg236-1
#   (part used only to anchor the in-memory project; not needed for sim)
#
# ---- Notes -------------------------------------------------------------
#   * This script creates an in-memory (non-file-system) Vivado project.
#   * No IP, no constraints, no block design — pure VHDL simulation only.
#   * The testbench is self-checking: "failure" severity stops the sim.
#   * VHDL-2008 is required (use of report with severity note, etc.)

set script_dir [file dirname [file normalize [info script]]]

puts ""
puts "=== conv3x3_dot Vivado Simulation ==="
puts "    Source dir: ${script_dir}"
puts ""

# ---- Create in-memory project ------------------------------------------
puts "--- Creating in-memory project (Artix-7 xc7a35tcpg236-1) ---"
create_project -in_memory -part xc7a35tcpg236-1

# ---- Add design sources ------------------------------------------------
puts "--- Adding design files ---"
add_files -norecurse [list \
    ${script_dir}/mac_unit.vhd \
    ${script_dir}/conv3x3_dot.vhd \
]
set_property file_type {VHDL 2008} [get_files mac_unit.vhd]
set_property file_type {VHDL 2008} [get_files conv3x3_dot.vhd]

# ---- Add simulation-only testbench -------------------------------------
puts "--- Adding testbench ---"
add_files -fileset sim_1 -norecurse ${script_dir}/tb_conv3x3_dot.vhd
set_property file_type    {VHDL 2008}     [get_files tb_conv3x3_dot.vhd]
set_property used_in_synthesis    false   [get_files tb_conv3x3_dot.vhd]
set_property used_in_implementation false [get_files tb_conv3x3_dot.vhd]

# ---- Configure simulation top ------------------------------------------
set_property top     tb_conv3x3_dot [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# ---- Launch xsim and run -----------------------------------------------
puts "--- Launching simulation ---"
launch_simulation

puts "--- Running all stimuli ---"
run all

puts ""
puts "=== Simulation complete ==="
puts "    Check the Tcl console or xsim log for PASS/FAIL messages."
puts "    A 'failure' severity assert means a test case failed."
puts ""

close_sim
