#!/bin/bash

GAME_DIM=(100 500 1000 5000 10000 15000) 
TIME=10


executable_no_vect="bin/gol_serial_no_vect"

executable="bin/gol_serial"

for dim in "${GAME_DIM[@]}"
do

    $executable $dim $dim $TIME
    
    
done


for dim in "${GAME_DIM[@]}"
do

    $executable_no_vect $dim $dim $TIME

done
