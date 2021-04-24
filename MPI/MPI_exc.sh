#!/bin/bash

grid_dim_list=(100 500 1000 5000 10000 50000)

display_versions_list=(1 2)


#Count number of nodes in the file host_list --> if the number is not available exit from the program
nodes="$(cat host_list.txt | wc -l)"

case $nodes in
    
    1)
        list_number_processes=(2 4 8 16 32 64 128 256)
        ;;
    2)
        list_number_processes=(2 4 8 16 32 64 128 256 512)
        ;;
    4)  list_number_processes=(4 8 16 32 64 128 256 512 1024)
        ;;

    8)  list_number_processes=(8 16 32 64 128 256 512 1024)
        ;;

    *)  
        printf "\n\n"
        echo "NOT VALID NUMBER OF NODES!!"
        echo "Available number of nodes: 1 - 2 - 4 - 8"
        echo "Please chage the file host_list.txt"
        printf "\n\n"
        exit 1
esac


for i in "${list_number_processes[@]}"
do
    echo $i
done

not_show_evolution=1

# path to file with the host list
host_list="host_list.txt"

bin_exec="../bin/gol_mpi"

for grid_dim in "${grid_dim_list[@]}"
do
   
    for num_process in "${list_number_processes[@]}"
    do
        
        per_host=$(( num_process / nodes))
        
        for version in "${display_versions_list[@]}"
        do

            mpiexec -hostfile $host_list -perhost $per_host -np $num_precess ./$bin_exec $grid_dim $grid_dim 10 $version $not_show_evolution}
        
        done
    done
done

