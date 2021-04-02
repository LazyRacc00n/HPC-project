/****************************************************************************
 *
 * mpi-hello.c - Hello, world in MPI
 *
 * Copyright (C) 2016 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * mpicc -Wall mpi-hello.c -o mpi-hello
 *
 * Run with:
 * mpirun -n 4 ./mpi-hello
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>

int main( int argc, char *argv[] )
{
    int rank, size, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init( &argc, &argv );	/* no MPI calls before this line */
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Get_processor_name(hostname, &len);
    printf("Greetings from process %d of %d running on %s\n", rank, size, hostname);
    MPI_Finalize();		/* no MPI calls after this line */
    return 0;
}
