#!/bin/bash

grid_dim_list=(100 500 1000 5000 10000 50000)

display_versions_list=(1 2)

# uncomment and comment the other one for 2 process 
list_number_processes=(2 4 8 16 32 64 128 256)

# uncomment and comment the other one for 2 process 
#list_number_processes=(2 4 8 16 32 64 128 256 512)

# uncomment and comment the other one for 4 process 
#list_number_processes=(4 8 16 32 64 128 256 512 1024)

# uncomment and comment the other one for 1 process 
#list_number_processes=(8 16 32 64 128 256 512 1024)

#change when you pass another list of processes above
nodes=1

not_show_evolution=1

# path to file with the host list
host_list="host_list.txt"

bin="../bin/gol_mpi"

echo "$(pwd)"


for grid_dim in "${grid_dim_list[@]}"
do
    for num_process in "${list_number_process[@]}"
    do
        per_host=$(( num_process / nodes))
        for version in "${display_version_list[@]}"
        do

            per_host=$(( num_process / nodes))

            #mpiexec -hostfile $host_list -perhost $per_host -np $num_precess ./bin $grid_dim $grid_dim 10 $version $not_show_evolution
        
        done
    done
done

