/****************************************************************************
 *
 * mpi-get-count.c - Shows how the MPI_Get_Count function can beused
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
 * mpicc mpi-get-count.c -o mpi-get-count
 *
 * Run with:
 * mpirun -n 2 mpi-get-count
 *
 * Process 0 sends a random number of integers to process 1
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define BUFLEN 16

int main( int argc, char *argv[])
{
    int rank, buf[BUFLEN] = {0};
    int count, i;
    MPI_Status status;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    /* Process 0 sends, Process 1 receives */
    if (rank == 0) {
        /* Initialize the random number generator (otherwise you
           always get the same number of items, since the RNG is
           deterministic) */
        srand(time(NULL));
        /* Fills the buffer with a random number of integers */
        count = 1 + rand()%BUFLEN;
        for (i=0; i<count; i++) {
            buf[i] = i;
        }
        MPI_Send(buf, count, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Sent %d integers\n", count);
    }
    else if (rank == 1) {
        MPI_Recv(buf, BUFLEN, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
        MPI_Get_count(&status, MPI_INT, &count);
        printf( "Received %d integers: ", count );
        for (i=0; i<count; i++) {
            printf("%d ", buf[i]);
        }
        printf("\n");
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
