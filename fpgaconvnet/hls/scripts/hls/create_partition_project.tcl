# get the root directory
variable fpgaconvnet_root [file dirname [file dirname [file dirname [file normalize [info script]]]]]

# get parameters from environment variables
set project_path $::env(HLS_PRJ_PATH)
set fpga         $::env(HLS_FPGA_PART)
set clk_period   $::env(HLS_CLK_PERIOD)

puts "project: ${project_path}"

# default cflags
set default_cflags "-std=c++17 -fexceptions -D__VITIS_HLS__ \
    -include ${fpgaconvnet_root}/hardware/hlslib_config.h \
    -I${project_path}/include -I${project_path}/data \
    -I${fpgaconvnet_root}/hardware -I${fpgaconvnet_root}/hardware/hlslib/include"

# create open project (reset to clear any old Vivado HLS format project)
open_project -reset ${project_path}

# set top function
set_top fpgaconvnet_ip

# add files to the project
add_files [ glob ${project_path}/src/*.cpp ] -cflags "${default_cflags}"

# add testbench file to the project
add_files -tb [ glob ${project_path}/tb/*.cpp ] -cflags "${default_cflags}"

# add any data files
add_files -tb [ glob ${project_path}/data/*.dat ]

# create the solution
open_solution -reset "solution"

# set FPGA part
set_part $fpga

# set clock period
create_clock -period $clk_period -name default

# increase fifo depth
config_dataflow -default_channel fifo -fifo_depth 2
config_dataflow -strict_mode warning

exit
