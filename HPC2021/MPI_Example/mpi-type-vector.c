/****************************************************************************
 *
 * mpi-type-vector.c - Simple demo of the MPI_Type_vector call
 *
 * Based on https://computing.llnl.gov/tutorials/mpi/#Derived_Data_Types
 *
 * Copyright (C) 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-type-vector.c -o mpi-type-vector
 *
 * Run with:
 * mpirun -n 4 ./mpi-type-vector
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 4

int main( int argc, char *argv[] )  
{
    int numtasks, rank, source=0, tag=1, i;

    float a[SIZE][SIZE] =
        {{ 1.0,  2.0,  3.0,  4.0},
         { 5.0,  6.0,  7.0,  8.0},
         { 9.0, 10.0, 11.0, 12.0},
         {13.0, 14.0, 15.0, 16.0}};
    float b[SIZE];
    
    MPI_Status stat;
    MPI_Datatype columntype;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    
    /* create contiguous derived data type */
    MPI_Type_vector(SIZE,       /* number of blocks                           */
		    1,	        /* n. of elements within each block           */
		    SIZE,       /* n. of elements between start of each block */
		    MPI_FLOAT,  /* type of elements of each block             */
		    &columntype /* newtype                                    */
		    );
    MPI_Type_commit(&columntype);

    if ( rank == 0 && numtasks == 1 ) {
        fprintf(stderr, "FATAL: you must run at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        /* The master sends the second column to all other processes */
        for (i=1; i<numtasks; i++) 
            MPI_Send(&a[0][i % numtasks], 1, columntype, i, tag, MPI_COMM_WORLD);
    } else {    
        /* All other tasks can read the columntype as a (conventional)
           array of SIZE elements of type MPI_FLOAT; in this case the
           elements are received and stored contiguously in b */
        MPI_Recv(b, SIZE, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &stat);
        printf("rank= %d received %3.1f %3.1f %3.1f %3.1f\n",
               rank, b[0], b[1], b[2], b[3]);
    }
    
    /* free datatype when done using it */
    MPI_Type_free(&columntype);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
