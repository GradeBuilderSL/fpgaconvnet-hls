# get parameters from environment variables
set project_path $::env(HLS_PRJ_PATH)
set fpga         $::env(HLS_FPGA_PART)

# open project
open_project ${project_path}

# add any data files
add_files -tb [ glob ${project_path}/data/*.dat ]

# open solution
open_solution "solution"

# set part (not persisted between vitis-run sessions)
set_part $fpga

# run co-simulation
cosim_design -rtl verilog -trace_level all

exit
