/*****************************************************************************
 * mpi-trap0.c - MPI implementation of the trapezoid rule; reworked
 * version of the code from http://www.cs.usfca.edu/~peter/ipp/.
 *
 * This solution uses the naive approach: node 0 (the master) collects
 * all partial results, and computes the final value without using the
 * reduction primitive.
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
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-trap0.c -o mpi-trap0
 *
 * Run with:
 * mpirun -n 4 ./mpi-trap0 
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * Function to be integrated
 */
double f( double x )
{
    return 4.0/(1.0 + x*x);
}

/*
 * Compute the area of function f(x) for x=[a, b] using the trapezoid
 * rule. The integration interval [a,b] is partitioned into n
 * subintervals of equal size.
 */
double trap( int my_rank, int comm_sz, double a, double b, int n )
{
    const double h = (b-a)/n;
    const int local_n_start = n * my_rank / comm_sz;
    const int local_n_end = n * (my_rank + 1) / comm_sz;
    double x = a + local_n_start * h;
    double my_result = 0.0;
    int i;
    
    for (i = local_n_start; i < local_n_end; i++) {
        my_result += h*(f(x) + f(x+h))/2.0;
        x += h;
    }
    return my_result;
}

int main( int argc, char* argv[] )
{
    double a = 0.0, b = 1.0, partial_result, result = 0.0;
    int n = 1000000, p;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* All nodes receive the command line parameters, therefore the
       master does not need to send them. */
    if ( 4 == argc ) {
        a = atof(argv[1]);
        b = atof(argv[2]);
        n = atoi(argv[3]);
    }

    /* All nodes (incl. the master) compute their local result */
    partial_result = trap( my_rank, comm_sz, a, b, n );

    if ( 0 != my_rank ) {
        /* all participants (other than the master) send their local
           result to the master */
        MPI_Send( &partial_result,      /* Buffer       */
                  1,                    /* Size         */
                  MPI_DOUBLE,           /* Type         */
                  0,                    /* dest         */
                  0,                    /* tag          */ 
                  MPI_COMM_WORLD        /* Communicator */
                  );
    } else {
        /* The master collects all partial results from other nodes */
        result = partial_result;
        for ( p=1; p<comm_sz; p++ ) {
            MPI_Recv( &partial_result,  /* Buffer       */
                      1,                /* Size         */
                      MPI_DOUBLE,       /* Type         */
                      p,                /* Sender (can be MPI_ANY_SENDER) */
                      0,                /* Tag (can be MPI_ANY_TAG) */
                      MPI_COMM_WORLD,   /* Communicator */
                      MPI_STATUS_IGNORE /* Status       */
                      );
            result += partial_result;
        }
        printf("Area: %f\n", result);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
