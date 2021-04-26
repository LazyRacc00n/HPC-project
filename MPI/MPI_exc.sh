#!/bin/bash

#grid_dim_list=(100 500 1000 5000 10000)
grid_dim_list=(15000)

display_versions_list=(1 2)

not_show_evolution=1
# path to file with the host list
host_list="host_list.txt"

bin_exec="../bin/gol_mpi"


#Count number of nodes in the file host_list --> if the number is not available exit from the program
# to change number of nodes, change file host_list.txt
nodes="$(cat host_list.txt | wc -l)"


#determanine automatically the list of the total number of processes
case $nodes in
    
    1)
        list_number_processes=(2 4 8 16 32 64)
        ;;
    2)
        list_number_processes=(2 4 8 16 32 64 128)
        ;;
    4)  list_number_processes=(4 8 16 32 64 128 256)
        ;;

    8)  list_number_processes=(8 16 32 64 128 256 512)
        ;;

    *)  
        printf "\n\n"
        echo "NOT VALID NUMBER OF NODES!!"
        echo "Available number of nodes: 1 - 2 - 4 - 8"
        echo "Please chage the file host_list.txt"
        echo "Or check if after last hostname there is an endline"
        printf "\n\n"
        exit 1
esac



for grid_dim in "${grid_dim_list[@]}"
do
   printf "\n - #NODES: $nodes - GRID DIM: $grid_dim x $grid_dim \n"
    for num_process in "${list_number_processes[@]}"
    do
        
        per_host=$(( num_process / nodes))

        printf "\n perhost: $per_host num_procceses: $num_process\n"

        
        for version in "${display_versions_list[@]}"
        do  

            mpiexec -hostfile $host_list -perhost $per_host -np $num_process ./$bin_exec $grid_dim $grid_dim 10 $version $not_show_evolution
        
        done
    done
done

