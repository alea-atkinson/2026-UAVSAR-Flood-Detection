# run_vivado_sim_3chan_realweights.tcl
# Vivado batch simulation for tb_stream_conv3x3_3chan_realweights.
#
# This testbench drives stream_conv3x3_3chan_cell with real INT8 weights from:
#   models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
#   tensor key: enc1.block.0.weight, output channel 0
#
# ---- Usage (batch mode) --------------------------------------------------
#   cd hardware/vhdl_conv3x3
#   vivado -mode batch -source run_vivado_sim_3chan_realweights.tcl
#
# ---- Usage (xvhdl/xelab/xsim directly) ----------------------------------
#   source /opt/Xilinx/Vivado/2024.2/settings64.sh
#   cd hardware/vhdl_conv3x3
#   xvhdl --2008 window3x3_stream.vhd conv3x3_dot_pipelined.vhd \
#               stream_conv3x3_3chan_cell.vhd tb_stream_conv3x3_3chan_realweights.vhd
#   xelab -debug typical tb_stream_conv3x3_3chan_realweights -s tb_rw_sim
#   xsim  tb_rw_sim --runall
#
# ---- Target FPGA ---------------------------------------------------------
#   Digilent Cmod A7-35T  →  Xilinx Artix-7  xc7a35tcpg236-1
#
# ---- Expected output -----------------------------------------------------
#   PASS output 1 (stream_conv3x3_3chan_realweights): y = 11287
#   ...
#   PASS output 9 (stream_conv3x3_3chan_realweights): y = 28831
#   === All stream_conv3x3_3chan_realweights tests PASSED ===

set script_dir [file dirname [file normalize [info script]]]

puts ""
puts "=== stream_conv3x3_3chan_realweights Vivado Simulation ==="
puts "    Source dir: ${script_dir}"
puts ""

puts "--- Creating in-memory project (Artix-7 xc7a35tcpg236-1) ---"
create_project -in_memory -part xc7a35tcpg236-1

puts "--- Adding design sources ---"
foreach src {window3x3_stream.vhd conv3x3_dot_pipelined.vhd stream_conv3x3_3chan_cell.vhd} {
    add_files -norecurse ${script_dir}/${src}
    set_property file_type {VHDL 2008} [get_files ${src}]
    puts "  OK  ${src}"
}

puts "--- Adding testbench ---"
add_files -fileset sim_1 -norecurse ${script_dir}/tb_stream_conv3x3_3chan_realweights.vhd
set_property file_type              {VHDL 2008} [get_files tb_stream_conv3x3_3chan_realweights.vhd]
set_property used_in_synthesis      false        [get_files tb_stream_conv3x3_3chan_realweights.vhd]
set_property used_in_implementation false        [get_files tb_stream_conv3x3_3chan_realweights.vhd]

set_property top     tb_stream_conv3x3_3chan_realweights [get_filesets sim_1]
set_property top_lib xil_defaultlib                      [get_filesets sim_1]

puts "--- Launching xsim ---"
launch_simulation

puts "--- Running all stimuli (25 pixels + 4-clock drain, ~600 ns) ---"
run all

puts ""
puts "=== Simulation complete ==="
puts "    Check the Tcl console or xsim log for PASS/FAIL messages."
puts ""

close_sim
