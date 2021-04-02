/****************************************************************************
 *
 * mpi-async.c - Simple point-to-point communication for MPI using
 * asynchronous primitives
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
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * mpicc -Wall mpi-async.c -o mpi-async
 *
 * Run with:
 * mpirun -n 2 mpi-async
 *
 * Process 0 sends an integer value to process 1
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void big_computation( void )
{
    printf("Some big computation...\n");
}

int main( int argc, char *argv[])
{
    int rank, size, buf;
    MPI_Status status;
    MPI_Request req;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    if ( size < 2 ) {
        fprintf(stderr, "FATAL: you must run at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (rank == 0) {
        buf = 123456;
        /* asynchronous send */
        MPI_Isend( &buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);
        big_computation();
        MPI_Wait(&req, &status);
        printf("Master terminates\n");
    }
    else if (rank == 1) {
        /* synchronous receive */
        MPI_Recv( &buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
        printf("Received %d\n", buf);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}



