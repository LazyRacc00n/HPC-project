#!/bin/bash

GAME_DIM=(100 500 1000 5000 10000 15000)
#comment after execution
#GAME_DIM=(15000)
THREADS=(2 4 8 16 32 64 128 256)
TIME=10
executable="../bin/gol_omp"




for dim in "${GAME_DIM[@]}"
do
    n_rows=$dim
    n_cols=$dim

    for threads in "${THREADS[@]}"
    do
        ./$executable $n_rows $n_cols $TIME $threads
    done

done

