#!/bin/bash

GAME_DIM=(100 500 1000 5000 10000 15000)
TIME=10
executable="bin/gol_serial"

for dim in "${GAME_DIM[@]}"
do

    $executable $dim $dim $TIME
    
done
