/****************************************************************************
 *
 * mpi-point-to-point.c - Simple point-to-point communication demo for MPI
 *
 * Copyright (C) 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * mpicc mpi-point-to-point.c -o mpi-point-to-point
 *
 * Run with:
 * mpirun -n 2 mpi-point-to-point
 *
 * Process 0 sends an integer value to process 1
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[])
{
    int my_rank, comm_size, buf;
    MPI_Status status;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (0 == my_rank && comm_size < 2) {
        fprintf(stderr, "FATAL: you must run at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Process 0 sends, Process 1 receives */
    if (my_rank == 0) {
        buf = 123456;
        MPI_Send( &buf,         /* send buffer  */
                  1,            /* count        */
                  MPI_INT,      /* datatype     */
                  1,            /* destination  */
                  0,            /* tag          */
                  MPI_COMM_WORLD /* communicator */
                  );
    }
    else if (my_rank == 1) {
        MPI_Recv( &buf,         /* receive buffer */
                  1,            /* count        */
                  MPI_INT,      /* datatype     */
                  0,            /* source       */
                  0,            /* tag          */
                  MPI_COMM_WORLD, /* communicator */
                  &status       /* status       */
                  );
        printf( "Received %d\n", buf );
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
