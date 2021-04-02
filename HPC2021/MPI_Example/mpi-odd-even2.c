/****************************************************************************
 *
 * mpi-odd-even2.c - Odd-even transposition sort in MPI
 *
 * Last updated in 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * This program is a heavily modified version of the MPI odd-even sort
 * implementation by Peter Pacheco. The original copyright notice follows.
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
 * This program implements a general version of odd-even sort.
 * Odd-even sort is described in G. Baudet and D. Stevenson, "Optimal
 * Sorting Algorithms for Parallel Computers," in IEEE Transactions on
 * Computers, vol. C-27, no. 1, pp. 84-87, Jan. 1978.
 * http://dx.doi.org/10.1109/TC.1978.1674957
 *
 * The general principle of this program is the same of
 * mpi-odd-even.c; however, this program handles the case where
 * processes have local vectors of different lenghts. The program
 * distributes the input vector in such a way that the lenghts of
 * subvectors do not differ by more than one. Furthermore, when
 * processes A and B exchange vectors of different lenghts p and q,
 * then process A keeps the lowest q elements while process B keeps
 * the highest p elements. In other words, after each exchange phase,
 * processes must exchange the vector lenghts as well.
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-odd-even2.c -o mpi-odd-even2
 *
 * Run with:
 * mpirun -n 4 ./mpi-odd-even
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int buf_len; /* the maximum length of any local buffer */

#ifndef max
int max(int a, int b)
{
    return (a > b ? a : b);
}
#endif

void print_array(int my_rank, double *v, int n)
{
    int i;
    printf("rank=%d v=[", my_rank);
    for (i=0; i<n; i++) {
        printf("%f", v[i]);
        if (i<n-1) printf(", ");
    }
    printf("]\n");
}

/**
 * Comparison function used by qsort. Returns -1, 0 or 1 if *x is less
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
 * Fill vector v with n random values drawn from the interval [0,1)
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
            printf("Check failed at element %d (v[%d]=%f, v[%d]=%f)\n",
                   i-1, i-1, v[i-1], i, v[i] );
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    printf("Check OK\n");
}

/**
 * Merge two sorted vectors |local_x| of length |local_n|, and
 * |received_x| of length |received_n|, keeping the lowest
 * |received_n| elements in |local_x|.  buffer is used as temporary
 * space and must be allocated by the called with at least
 * |received_n| elements.
 */
void merge_low( double *local_x, int local_n, double *received_x, int received_n, double* buffer )
{
    int idx_local = 0, idx_received = 0, idx_buf = 0;

    while ( idx_buf < received_n ) {
        if ( (idx_received >= received_n) || ((idx_local < local_n) && compare( local_x + idx_local, received_x + idx_received ) < 0) ) {
            buffer[idx_buf] = local_x[idx_local];
            idx_local++;
        } else {
            buffer[idx_buf] = received_x[idx_received];
            idx_received++;
        }
        idx_buf++;
    }

    memcpy( local_x, buffer, received_n*sizeof(double) );
}

/**
 * Merge two sorted vectors |local_x| of length |local_n|, and
 * |received_x| of length |received_n|, keeping the highest
 * |received_n| elements in |local_x|.  buffer is used as temporary
 * space and must be allocated by the called with at least
 * |received_n| elements.
 */
void merge_high( double* local_x, int local_n, double *received_x, int received_n, double* buffer )
{
    int idx_local = local_n-1, idx_received = received_n-1, idx_buf = received_n-1;
    while ( idx_buf >= 0 ) {
        if ( (idx_received < 0) || ((idx_local >= 0) && compare( local_x + idx_local, received_x + idx_received ) > 0) ) {
            buffer[idx_buf] = local_x[idx_local];
            idx_local--;
        } else {
            buffer[idx_buf] = received_x[idx_received];
            idx_received--;
        }
        idx_buf--;
    }

    memcpy( local_x, buffer, received_n*sizeof(double) );
}

/**
 * Performs a single exchange-compare step. Two adjacent nodes
 * exchange their data, merging with the local buffer. The node on the
 * left keeps the lower half of the merged vector, while the neighbor
 * on the right keeps the upper half. Returns the number of elements
 * received by the partner. If no exchange takes places, e.g., because
 * the partner is MPI_PROC_NULL, returns |local_n|.
 */
int do_sort_exchange( int phase,        /* number of this phase (0..comm_sz-1) */
                      double *local_x,  /* block from the input array x assigned to this node */
                      int local_n,      /* length of local_x */
                      double *received_x, /* block that will be received from the partner */
                      double *buffer,   /* temporary buffer used for merging */
                      int my_rank,      /* my rank */
                      int even_partner, /* partner to use during even phases */
                      int odd_partner   /* parter to use during odd phases */
                      )
{
    /* If this is an even phase, my parther is even_partner; otherwise it is odd_partner */
    const int partner = (phase % 2 == 0 ? even_partner : odd_partner);
    int received_n = local_n;

    if ( partner != MPI_PROC_NULL ) {
        MPI_Status status;

        MPI_Sendrecv(local_x,           /* sendbuf      */
                     local_n,           /* sendcount    */
                     MPI_DOUBLE,        /* datatype     */
                     partner,           /* dest         */
                     0,                 /* sendtag      */
                     received_x,        /* recvbuf      */
                     buf_len,           /* recvcount    */
                     MPI_DOUBLE,        /* datatype     */
                     partner,           /* source       */
                     0,                 /* recvtag      */
                     MPI_COMM_WORLD,    /* comm         */
                     &status            /* status       */
                     );
        /* How many elements did we receive from the partner? */
        MPI_Get_count(&status, MPI_DOUBLE, &received_n);
        if ( my_rank < partner ) {
            merge_low( local_x, local_n, received_x, received_n, buffer );
        } else {
            merge_high( local_x, local_n, received_x, received_n, buffer );
        }
    }
    return received_n;
}

int main( int argc, char* argv[] )
{
    double *x = NULL, *local_x, *received_x, *buffer;
    double tstart, tstop;
    int i, n = 100000, 
        local_n,        /* length of local chunk (and of temporary buffer) */
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

    /* Find partners */
    if (my_rank % 2 == 0) {
        even_partner = (my_rank < comm_sz-1 ? my_rank + 1 : MPI_PROC_NULL );
        odd_partner = (my_rank > 0 ? my_rank - 1 : MPI_PROC_NULL );
    } else {
        even_partner = (my_rank > 0 ? my_rank - 1 : MPI_PROC_NULL );
        odd_partner = (my_rank < comm_sz-1 ? my_rank + 1 : MPI_PROC_NULL );
    }

    /* Compute chunk size and displacements for use with
       scatterv/gatherv. Since this information is more or less
       required by all processes, it is computed by everyone. */
    int sendcounts[comm_sz];
    int displs[comm_sz];
    for (i=0; i<comm_sz; i++) {
        const int start_idx = n*i/comm_sz;
        const int end_idx = n*(i+1)/comm_sz;
        sendcounts[i] = end_idx - start_idx;
        displs[i] = start_idx;
    }

    buf_len = (n + comm_sz - 1) / comm_sz; /* n / comm_size, rounded up */
    local_n = sendcounts[my_rank];

    /* Local vectors allocated by all nodes */
    local_x = (double*)malloc( buf_len * sizeof(*local_x) );
    buffer = (double*)malloc( buf_len * sizeof(*buffer) );   
    received_x = (double*)malloc( buf_len * sizeof(*received_x) );

    /* The root starts the timer */
    if ( 0 == my_rank ) {
        tstart = MPI_Wtime();
    }

    /* Scatter input data  */
    MPI_Scatterv( x,             /* sendbuf */
                  sendcounts,    /* sendcounts */
                  displs,        /* displacements */
                  MPI_DOUBLE,    /* sent MPI_Datatype */
                  local_x,       /* recvbuf */
                  local_n,       /* recvcount */
                  MPI_DOUBLE,    /* received MPI_Datatype */
                  0,             /* root */
                  MPI_COMM_WORLD /* communicator */
                  );

    /* sort the local data */
    qsort( local_x, local_n, sizeof(*local_x), compare );

    /* phases of odd-even sort */
    for ( phase = 0; phase < comm_sz; phase++ ) {
        local_n = do_sort_exchange( phase, local_x, local_n, received_x, buffer, my_rank, even_partner, odd_partner );
    }

    /* Gather local_n so that the master knows the new values of sendcounts[] */
    MPI_Gather( &local_n,       /* sendbuf      */
                1,              /* sendcount    */
                MPI_INT,        /* datatype     */
                sendcounts,     /* recvbuf      */
                1,              /* recvcount    */
                MPI_INT,        /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD  /* communicator */
                );

    /* The master recomputes the displacements */
    if ( 0 == my_rank ) {
        displs[0] = 0;
        for (i=1; i<comm_sz; i++) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }

    /* The master gathers the local buffers from all nodes */
    MPI_Gatherv( local_x,       /* sendbuf */
                 local_n,       /* sendcount */
                 MPI_DOUBLE,    /* sendtype */
                 x,             /* recvbuf */
                 sendcounts,    /* recvcount (equal to sendcounts, in this case) */
                 displs,        /* displacements */
                 MPI_DOUBLE,    /* recvtype */
                 0,             /* root (where to send) */
                 MPI_COMM_WORLD /* communicator */
                 );

    /* The master checks the sorted vector */
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
