# get parameters from environment variables
set project_path $::env(HLS_PRJ_PATH)
set fpga         $::env(HLS_FPGA_PART)
set clk_period   $::env(HLS_CLK_PERIOD)

# open project
open_project ${project_path}

# open solution
open_solution "solution"

# set part and clock (not persisted between vitis-run sessions)
set_part $fpga
create_clock -period $clk_period -name default
config_dataflow -default_channel fifo -fifo_depth 2
config_dataflow -strict_mode warning

# run c-synthesis
csynth_design

exit
