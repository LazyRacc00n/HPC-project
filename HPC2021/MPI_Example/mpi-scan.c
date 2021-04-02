/*****************************************************************************
 *
 * mpi-scan.c - MPI_Scan demo
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
 * --------------------------------------------------------------------------
 *
 * This solution uses the naive approach: node 0 (the master) collects
 * all partial results, and computes the final value without using the
 * reduction primitive.
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-scan.c -o mpi-scan
 *
 * Run with:
 * mpirun -n 4 ./mpi-scan
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int *local_x, *scan_x;
    int local_N = 3, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    local_x = (int *)malloc(local_N * sizeof(*local_x));
    for (i = 0; i < local_N; i++) {
	local_x[i] = i + my_rank * local_N;
    }

    scan_x = (int *)malloc(local_N * sizeof(*scan_x));

    MPI_Scan(local_x,		/* sendbuf      */
	     scan_x,		/* recvbuf      */
	     local_N,		/* count        */
	     MPI_INT,		/* datatype     */
	     MPI_SUM,		/* operator     */
	     MPI_COMM_WORLD     /* communicator */
             );

    for (i = 0; i < local_N; i++) {
	printf("rank=%d scan_x[%d]=%d\n", my_rank, i, scan_x[i]);
    }

    free(local_x);
    free(scan_x);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
