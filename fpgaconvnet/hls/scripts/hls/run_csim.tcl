# get project path from environment variable
set project_path $::env(HLS_PRJ_PATH)

# open project
open_project ${project_path}

# add any data files
add_files -tb [ glob ${project_path}/data/*.dat ]

# open solution
open_solution "solution"

# run c-simulation
csim_design

exit
