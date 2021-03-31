/****************************************************************************
 *
 * mpi-vecsum.c - Parallel vector sum using MPI.
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
 * Parallel vector sum using MPI. This version works correctly for
 * every vector lengths, that are not required to be a multiple of the
 * communicator size.
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-vecsum2.c -o mpi-vecsum2
 *
 * Run with:
 * mpirun -n 4 ./mpi-vecsum2 [n]
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <mpi.h>

/*
 * Compute z[i] = x[i] + y[i], i=0, ... n-1
 */
void sum( double* x, double* y, double* z, int n )
{
    int i;
    for (i=0; i<n; i++) {
	z[i] = x[i] + y[i];
    }
}

int main( int argc, char* argv[] )
{
    double *x, *local_x, *y, *local_y, *z, *local_z;
    int n = 1000, local_n, i;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( 2 == argc ) {
	n = atoi(argv[1]);
    }
    
    local_n = n / comm_sz;

    x = y = z = NULL;

    if ( 0 == my_rank ) {
	/* The master allocates the vectors */
	x = (double*)malloc( n * sizeof(*x) );
	y = (double*)malloc( n * sizeof(*y) );
	z = (double*)malloc( n * sizeof(*z) );
	for ( i=0; i<n; i++ ) {
	    x[i] = i;
	    y[i] = n-1-i;
	}
    }

    /* All nodes (including the master) allocate the local vectors */
    local_x = (double*)malloc( local_n * sizeof(*local_x) );
    local_y = (double*)malloc( local_n * sizeof(*local_y) );
    local_z = (double*)malloc( local_n * sizeof(*local_z) );

    /* Scatter vector x */
    MPI_Scatter( x,		/* sendbuf */
		 local_n,	/* sendcount; how many elements to send to _each_ destination */
		 MPI_DOUBLE,	/* sent MPI_Datatype */
		 local_x,	/* recvbuf */
		 local_n,	/* recvcount (usually equal to sendcount) */
		 MPI_DOUBLE,	/* received MPI_Datatype */
		 0,		/* root */
		 MPI_COMM_WORLD /* communicator */
		 );

    /* Scatter vector y*/
    MPI_Scatter( y,		/* sendbuf */
		 local_n,	/* sendcount; how many elements to send to _each_ destination */
		 MPI_DOUBLE,	/* sent MPI_Datatype */
		 local_y,	/* recvbuf */
		 local_n,	/* recvcount (usually equal to sendcount) */
		 MPI_DOUBLE,	/* received MPI_Datatype */
		 0,		/* root */
		 MPI_COMM_WORLD /* communicator */
		 );

    /* All nodes compute the local result */
    sum( local_x, local_y, local_z, local_n );

    /* Gather results from all nodes */
    MPI_Gather( local_z,	/* sendbuf */
		local_n,	/* sendcount (how many elements each node sends */
		MPI_DOUBLE,	/* sendtype */
		z,		/* recvbuf */
		local_n,	/* recvcount (how many elements should be received from _each_ node */
		MPI_DOUBLE,	/* recvtype */
		0,		/* root (where to send) */
		MPI_COMM_WORLD	/* communicator */
		);

    /* The master takes care of the leftovers, if any */
    if ( (0 == my_rank) && (n % comm_sz) ) {
        const int skip = (n/comm_sz)*comm_sz;
        sum(x + skip, y + skip, z + skip, n % comm_sz);
    }

    /* Uncomment if you want process 0 to print the result */
#if 0
    if ( 0 == my_rank ) {
	for ( i=0; i<n; i++ ) {
	    printf("z[%d] = %f\n", i, z[i]);
	}
    }	
#endif

    /* The master checks the result */
    if ( 0 == my_rank ) {
        for (i=0; i<n; i++) {
            if ( fabs(z[i] - (n-1)) > 1e-6 ) {
                fprintf(stderr, "Test FAILED: z[%d]=%f, expected %f\n", i, z[i], (double)(n-1));
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        printf("Test OK\n");
    }

    free(x); /* If x == NULL, no operation is performed */
    free(y); 
    free(z);

    free(local_x); 
    free(local_y); 
    free(local_z);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
