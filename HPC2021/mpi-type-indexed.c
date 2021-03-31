/****************************************************************************
 *
 * mpi-type-indexed.c - Simple demo of the MPI_Type_indexed call
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-type-indexed.c -o mpi-type-indexed
 *
 * Run with:
 * mpirun -n 4 ./mpi-type-indexed
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 16

void print_array( float* v, int n )
{
  int i;
  for (i=0; i<n; i++) {
    printf("%3.1f ", v[i]);
  }
  printf("\n");
}

void fill_array( float* v, float val, int n )
{
  int i;
  for (i=0; i<n; i++) {
    v[i] = val;
  }
}

int main( int argc, char *argv[] )  
{
  int numtasks, rank, source=0, tag=1, i;

    float a[SIZE] = 
        { 1.0,  2.0,  3.0,  4.0,
          5.0,  6.0,  7.0,  8.0,
          9.0, 10.0, 11.0, 12.0,
         13.0, 14.0, 15.0, 16.0};
    float b[SIZE];
    
    MPI_Status stat;
    MPI_Datatype idxtype;
    int array_of_blocklenghts[] = {1, 3, 4};
    int array_of_displacements[] = {2, 5, 12};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    
    /* create contiguous derived data type */
    MPI_Type_indexed(3,       
		     array_of_blocklenghts,
		     array_of_displacements,
		     MPI_FLOAT,
		     &idxtype
		     );
    MPI_Type_commit(&idxtype);

    if ( rank == 0 && numtasks == 1 ) {
        fprintf(stderr, "FATAL: you must run at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
      /* The master sends one element of idxtype to all other tasks */
      for (i=1; i<numtasks; i++) {
	MPI_Send(&a[0], 1, idxtype, 1, tag, MPI_COMM_WORLD);
	MPI_Send(&a[0], 1, idxtype, 1, tag, MPI_COMM_WORLD);
      }	
    } else 
      if (rank == 1) {    
	fill_array(b, -1, SIZE);
        MPI_Recv(b, 8, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &stat);
	printf("Rank %d received as MPI_FLOAT:\n", rank);
	print_array(b, SIZE);

	fill_array(b, -1, SIZE);
        MPI_Recv(b, 1, idxtype, source, tag, MPI_COMM_WORLD, &stat);
	printf("Rank %d received as idxtype:\n", rank);
	print_array(b, SIZE);
    }
    
    /* free datatype when done using it */
    MPI_Type_free(&idxtype);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
