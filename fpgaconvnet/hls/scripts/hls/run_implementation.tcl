# get parameters from environment variables
set project_path $::env(HLS_PRJ_PATH)
set fpga         $::env(HLS_FPGA_PART)

# open project
open_project ${project_path}

# open solution
open_solution "solution"

# set part (not persisted between vitis-run sessions)
set_part $fpga

# run implementation
export_design -flow impl -rtl verilog -format ip_catalog

exit
