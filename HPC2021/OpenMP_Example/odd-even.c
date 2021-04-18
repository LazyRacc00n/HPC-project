/****************************************************************************
 *
 * odd-even.c - Serial implementation of the odd-even transposition sort.
 *
 * Last updated in 2019 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * gcc -std=c99 -Wall -pedantic odd-even.c -o odd-even
 *
 * Run with:
 * ./odd-even 1000
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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

/* Odd-even transposition sort */
void odd_even_sort( int* v, int n )
{
    int phase, i;
    double tstart = hpc_gettime();
    for (phase = 0; phase < n; phase++) {
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
	    for (i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
	    for (i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
    double tstop = hpc_gettime();
    printf("Sorting time %f\n", tstop - tstart);
}

void check( int* v, int n )
{
    int i;
    for (i=0; i<n-1; i++) {
	if ( v[i] != i ) {
	    fprintf(stderr, "Check failed: v[%d]=%d, expected %d\n",
		    i, v[i], i );
	    abort();
	}
    }
    printf("Check ok!\n");
}

int main( int argc, char* argv[] )
{
    int n = 50000;
    int* v;

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }
    v = (int*)malloc(n*sizeof(v[0])); assert(v);
    fill(v,n);
    odd_even_sort(v,n);
    check(v,n);
    free(v);
    return EXIT_SUCCESS;
}
