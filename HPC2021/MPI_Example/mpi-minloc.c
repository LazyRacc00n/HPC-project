/*****************************************************************************
 *
 * mpi-minloc.c - MPI_Reduction with MPI_MINLOC demo
 *
 * Copyright (C) 2019 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * This program shows how MPI_Reduction can be used with the
 * MPI_MINLOC operator (MPI_MAXLOC can be used in the same
 * way). MPI_MINLOC returns the global minimum and an index attached
 * to that minimum.
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-minloc.c -o mpi-minloc
 *
 * Run with:
 * mpirun -n 4 ./mpi-minloc
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    struct { /* This corresponds to the predefined type MPI_DOUBLE_INT */
        double val;
        int   idx;
    } in, out;

    int my_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    in.val = my_rank * 1.5;
    in.idx = 1 + my_rank;
    
    MPI_Reduce( &in,            /* sendbuf      */
                &out,           /* recvbuf      */
                1,              /* count        */
                MPI_DOUBLE_INT, /* datatype     */
                MPI_MAXLOC,     /* operator     */
                0,              /* root         */
                MPI_COMM_WORLD  /* communicator */
                );

    /* At this point, the answer resides on process root */

    printf("\n\nThe node %d send the value %f with idx %d\n\n", my_rank, in.val, in.idx);

    if (0 == my_rank) {
        /* read ranks out */
        printf("The maximum is %f with idx %d\n", out.val, out.idx);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
