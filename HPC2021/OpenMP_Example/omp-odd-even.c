/****************************************************************************
 *
 * omp-odd-even.c - Odd-even transposition sort with OpenMP
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * OpenMP implementation of odd-even transposition sort.
 *
 * To compile:
 * gcc -fopenmp -std=c99 -Wall -pedantic omp-odd-even.c -o omp-odd-even -lgomp
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp-odd-even
 *
 ***************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* if *a > *b, swap them. Otherwise do nothing */
void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
    }
}

/* Fills vector v with a permutation of the integer values 0, .. n-1 */
void fill( int* v, int n )
{
    int i;
    int up = n-1, down = 0;
    for ( i=0; i<n; i++ ) {
	v[i] = ( i % 2 == 0 ? up-- : down++ );
    }
}

/* Odd-even transposition sort; this function uses two omp parallel
   for directives. */
void odd_even_sort_nopool( int* v, int n )
{
    int phase, i;
    for (phase = 0; phase < n; phase++) {
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
#pragma omp parallel for default(none) shared(n,v)
	    for (i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
#pragma omp parallel for default(none) shared(n,v)
	    for (i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
}

/* Same as above, but with a common pool of threads that are recycled
   in the omp for constructs */
void odd_even_sort_pool( int* v, int n )
{
    int phase, i;

#pragma omp parallel default(none) private(phase) shared(n,v)
    for (phase = 0; phase < n; phase++) {
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
#pragma omp for
	    for (i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
#pragma omp for
	    for (i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
}

void check( int* v, int n )
{
    int i;
    for (i=0; i<n-1; i++) {
	if ( v[i] != i ) {
	    printf("Check failed: v[%d]=%d, expected %d\n",
		   i, v[i], i );
	    abort();
	}
    }
    printf("Check ok!\n");
}

int main( int argc, char* argv[] )
{
    int n = 100000;
    int *v;
    int r;
    const int NREPS = 5;
    double tstart, tstop;

    if ( argc > 1 ) {
	n = atoi(argv[1]);
    }
    v = (int*)malloc(n*sizeof(v[0]));
    fill(v,n);
    printf("Without thread pool recycling: \t");
    tstart = hpc_gettime();
    for (r=0; r<NREPS; r++) {        
        odd_even_sort_nopool(v,n);
    }
    tstop = hpc_gettime();
    printf("Mean elapsed time %f\n", (tstop - tstart)/NREPS);
    check(v,n);
    fill(v,n);
    printf("With thread pool recycling: \t");
    tstart = hpc_gettime();
    for (r=0; r<NREPS; r++) {
        odd_even_sort_pool(v,n);
    }
    tstop = hpc_gettime();
    printf("Mean elapsed time %f\n", (tstop - tstart)/NREPS);
    check(v,n);
    free(v);
    return EXIT_SUCCESS;
}
