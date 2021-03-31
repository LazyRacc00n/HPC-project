/****************************************************************************
 *
 * mpi-odd-even.c - Odd-even transposition sort in MPI
 *
 * Last updated in 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * The original copyright notice follows.
 *
 * --------------------------------------------------------------------------
 *
 * Copyright (c) 2000, 2013, Peter Pacheco and the University of San
 * Francisco. All rights reserved. Redistribution and use in source
 * and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * --------------------------------------------------------------------------
 *
 * This program requires that the vector length is an integer multiple
 * of the number of MPI processes. For a more general solution that
 * works with any vector length, see mpi-odd-even2.c
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-odd-even.c -o mpi-odd-even
 *
 * Run with:
 * mpirun -n 4 ./mpi-odd-even 
 *
 * mpi-odd-even takes the vector length n as an optional parameter.
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/**
 * Comparison function used by qsort. Return -1, 0 or 1 if *x is less
 * than, equal to, or greater than *y
 */
int compare( const void* x, const void* y )
{
    const double *vx = (const double*)x;
    const double *vy = (const double*)y;

    if ( *vx > *vy ) {
        return 1;
    } else {
        if ( *vx < *vy )
            return -1;
        else
            return 0;
    }
}

/**
 * Fill vector v with n random values drawn from the interval [0,1]
 */
void fill( double* v, int n )
{
    int i;
    for ( i=0; i<n; i++ ) {
        v[i] = rand() / (double)RAND_MAX;
    }
}

/**
 * Check whether the n vector array v is sorted according to the
 * comparison function compare()
 */
void check( const double* v, int n )
{
    int i;
    for (i=1; i<n; i++) {
        if ( compare( &v[i-1], &v[i] ) > 0 ) {
            fprintf(stderr, "FATAL: Check failed (v[%d]=%f, v[%d]=%f)\n",
                    i-1, v[i-1], i, v[i] );
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    printf("Check OK\n");
}

/**
 * Merge two sorted vectors |local_x| and |received_x|, with |local_n|
 * elements each, keeping the bottom half of the result in |local_x|.
 * |buffer| is used as temporary space, and must be allocated by the
 * caller with at least |local_n| elements of type double.
 */
void merge_low( double* local_x, double *received_x, double* buffer, int local_n )
{
    int i_l=0, i_r=0, i_b=0;
    while ( i_b < local_n ) {
        if ( compare( local_x + i_l, received_x + i_r ) < 0 ) {
            buffer[i_b] = local_x[i_l];
            i_l++;
        } else {
            buffer[i_b] = received_x[i_r];
            i_r++;
        }
        i_b++;
    }

    memcpy( local_x, buffer, local_n*sizeof(double) );
}

/**
 * Merge two sorted vectors |local_x| and |received_x|, with |local_n|
 * elements each, keeping the top half of the result in |local_x|.
 * |buffer| is used as temporary space, and must be allocated by the
 * caller with at least |local_n| elements of type double.
 */
void merge_high( double* local_x, double *received_x, double* buffer, int local_n )
{
    int i_l=local_n-1, i_r=local_n-1, i_b=local_n-1;
    while ( i_b >= 0 ) {
        if ( compare( local_x + i_l, received_x + i_r ) > 0 ) {
            buffer[i_b] = local_x[i_l];
            i_l--;
        } else {
            buffer[i_b] = received_x[i_r];
            i_r--;
        }
        i_b--;
    }

    memcpy( local_x, buffer, local_n*sizeof(double) );
}

/**
 * Performs a single exchange-compare step. Two adjacent nodes
 * exchange their data, merging with the local buffer. The node on the
 * left keeps the lower half of the merged vector, while the neighbor
 * on the right keeps the upper half.
 */
void do_sort_exchange( int phase,       /* number of this phase (0..comm_sz-1) */
                       double *local_x, /* block from the input array x assigned to this node */
                       double* received_x, /* block that will be received from the partner */
                       double *buffer,  /* temporary buffer used for merging */
                       int local_n,     /* length of local_x, received_x and buffer */
                       int my_rank,     /* my rank */
                       int even_partner, /* partner to use during even phases */
                       int odd_partner  /* parter to use during odd phases */
                       )
{
    /* If this is an even phase, my parther is even_partner; otherwise it is odd_partner */
    const int partner = (phase % 2 == 0 ? even_partner : odd_partner);

    if ( partner != MPI_PROC_NULL ) {
        MPI_Sendrecv(local_x,           /* sendbuf      */
                     local_n,           /* sendcount    */
                     MPI_DOUBLE,        /* datatype     */
                     partner,           /* dest         */
                     0,                 /* sendtag      */
                     received_x,        /* recvbuf      */
                     local_n,           /* recvcount    */
                     MPI_DOUBLE,        /* datatype     */
                     partner,           /* source       */
                     0,                 /* recvtag      */
                     MPI_COMM_WORLD,    /* comm         */
                     MPI_STATUS_IGNORE  /* status       */
                     );
        if ( my_rank < partner ) {
            merge_low( local_x, received_x, buffer, local_n );
        } else {
            merge_high( local_x, received_x, buffer, local_n );
        }
    }
}

int main( int argc, char* argv[] )
{
    double *x = NULL, *local_x, *received_x, *buffer;
    double tstart, tstop;
    int n = 100000, local_n, 
        phase,          /* compare-exchange phase */
        odd_partner,    /* neighbor to use during odd phase */
        even_partner    /* neighbor to use during even phase */
        ;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( 2 == argc ) {
        n = atoi(argv[1]);
    }

    /* MPI_Scatter requires that all blocks have the same size */
    if ( (0 == my_rank) && (n % comm_sz) ) {
        fprintf(stderr, "FATAL: the vector length (%d) must be multiple of %d\n", n, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    local_n = n / comm_sz;

    if ( 0 == my_rank ) {
        printf("Vector length: %d\n", n );
        printf("Number of MPI processes: %d\n", comm_sz );

        /* The master initializes the vector to be sorted */
        x = (double*)malloc( n * sizeof(*x) );
        tstart = MPI_Wtime();
        fill(x, n);
        tstop = MPI_Wtime();
        printf("Fill: %f\n", tstop - tstart );
    }

    /* All nodes initialize their local vectors */
    local_x = (double*)malloc( local_n * sizeof(*local_x) );
    received_x = (double*)malloc( local_n * sizeof(*received_x) );
    buffer = (double*)malloc( local_n * sizeof(*buffer) ); /* usato per la fase di merge */

    /* Find partners */
    if (my_rank % 2 == 0) {
        even_partner = (my_rank < comm_sz-1 ? my_rank + 1 : MPI_PROC_NULL );
        odd_partner = (my_rank > 0 ? my_rank - 1 : MPI_PROC_NULL );
    } else {
        even_partner = (my_rank > 0 ? my_rank - 1 : MPI_PROC_NULL );
        odd_partner = (my_rank < comm_sz-1 ? my_rank + 1 : MPI_PROC_NULL );
    }

    /* The root starts the timer */
    if ( 0 == my_rank ) {
        tstart = MPI_Wtime();
    }

    /* Scatter vector x */
    MPI_Scatter( x,             /* sendbuf */
                 local_n,       /* sendcount; how many elements to send to _each_ destination */
                 MPI_DOUBLE,    /* sent MPI_Datatype */
                 local_x,       /* recvbuf */
                 local_n,       /* recvcount (usually equal to sendcount) */
                 MPI_DOUBLE,    /* received MPI_Datatype */
                 0,             /* root */
                 MPI_COMM_WORLD /* communicator */
                 );

    /* sort local buffer */
    qsort( local_x, local_n, sizeof(*local_x), compare );

    /* phases of odd-even sort */
    for ( phase = 0; phase < comm_sz; phase++ ) {
        do_sort_exchange(phase, local_x, received_x, buffer, local_n, my_rank, even_partner, odd_partner);
    }

    /* Gather results from all nodes */
    MPI_Gather( local_x,        /* sendbuf */
                local_n,        /* sendcount (how many elements each node sends */
                MPI_DOUBLE,     /* sendtype */
                x,              /* recvbuf */
                local_n,        /* recvcount (how many elements should be received from _each_ node */
                MPI_DOUBLE,     /* recvtype */
                0,              /* root (where to send) */
                MPI_COMM_WORLD  /* communicator */
                );

    /* The root checks the sorted vector */
    if ( 0 == my_rank ) {
        tstop = MPI_Wtime();
        printf("Sort: %f\n", tstop - tstart );
        tstart = MPI_Wtime();
        check(x, n);
        tstop = MPI_Wtime();
        printf("Check: %f\n", tstop - tstart );
    }   

    free(x);
    free(local_x);
    free(received_x);
    free(buffer);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
