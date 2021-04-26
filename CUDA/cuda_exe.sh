#!/bin/bash

GAME_DIM=(100 500 1000 5000 10000 50000)
BLOCK_SIZE=(32 64 128 256 512 1024)
TIME=10
executable="./bin/gol_cuda"

for dim in "${GAME_DIM[@]}"
do
    n_rows=$dim
    n_cols=$dim

    for threads in "${BLOCK_SIZE[@]}"
    do
        ./$executable $n_rows $n_cols $TIME $threads
    done

done

