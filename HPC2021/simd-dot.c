/****************************************************************************
 *
 * simd-dot.c - Dot product using SSE instructions
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
 * gcc -Wall -msse simd-dot.c -o simd-dot
 *
 * Run with:
 * ./simd-dot <vector length>
 *
 * Example:
 * ./simd-dot 1024
 *
 ****************************************************************************/

#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>

/* Returns the dot product of n-element vectors x and y. */
float dot_serial( const float* x, const float* y, int n )
{
    float result = 0.0;
    int i;
    for ( i=0; i<n; i++ ) {
	result += x[i] * y[i];
    }
    return result;
}

/* Returns the dot product of n-element vectors x and y using SSE
   instructions. x and y are not required to be aligned,; furthermore,
   this function should work with any length n */
float dot_sse( const float* x, const float* y, int n )
{
    __m128 vx, vy, vtmp, vsum;
    int i;
    float result = 0.0, temp[4];

    vsum = _mm_setzero_ps(); /* set vsum[0..3] to zero */
    for ( i=0; i<n-3; i += 4 ) {
	vx = _mm_loadu_ps( &x[i] ); /* vx = [ x[i], x[i+1], x[i+2], x[i+3] ] */
	vy = _mm_loadu_ps( &y[i] ); /* vy = [ y[i], y[i+1], y[i+2], y[i+3] ] */
	vtmp = _mm_mul_ps( vx, vy ); /* vtmp = [ x[i]   * y[i], 
					         x[i+1] * y[i+1],
					         x[i+2] * y[i+2],
					         x[i+3] * y[i+3] ] */
	vsum = _mm_add_ps( vsum, vtmp ); /* accumulate */
    }
    /* SSE3 has a _mm_hadd_ps() for horizontal addition; previous
       versions of SSE have not, so we do the following */
    _mm_storeu_ps( temp, vsum );
    result = temp[0] + temp[1] + temp[2] + temp[3];
    /* Take care of any remaining element */
    for ( ; i<n; i++ ) {
	result += x[i] * y[i];
    }
    return result;
}

/* Fills vector v with n random values in [0, 1] */
void fill( float* v, int n )
{
    int i;
    for ( i=0; i<n; i++ ) {
	v[i] = rand() / (float)RAND_MAX;
    }
}

int main( int argc, char* argv[] )
{
    int n = 1024;
    float *x, *y, result_sse, result_serial;
    if ( argc == 2 ) 
	n = atoi(argv[1]);

    x = (float*)malloc( n * sizeof(float) );
    fill(x, n);
    y = (float*)malloc( n * sizeof(float) );
    fill(y, n);

    result_sse = dot_sse( x, y, n);
    result_serial = dot_serial( x, y, n );

    printf("SSE   : %f\n", result_sse);
    printf("Serial: %f\n", result_serial);

    return EXIT_SUCCESS;
}
