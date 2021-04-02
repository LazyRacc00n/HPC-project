/****************************************************************************
 *
 * reduction.c - Tree-structured reduction
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
 * Compile with:
 * gcc -std=c99 -Wall -pedantic reduction.c -o reduction
 *
 * Run with:
 * ./reduction
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Serial implementation of the sum-reduce operator */
float sum_reduce_serial( float* x, size_t n )
{
    float result = 0.0;
    int i;
    for (i=0; i<n; i++) {
        result += x[i];
    }
    return result;
}

/* Tree-structured implementation of the sum-reduce operator; this
   implementation works for any vector length n > 0 */
float sum_reduce( float* x, size_t n )
{
    int d, i;

    /* compute d, the largest power of two s.t. d < n */
    for (d=1; 2*d < n; d *= 2) 
        ;
    /* do reduction */
    for ( ; d>0; d /= 2) {
        for (i=0; i<d; i++) {
            if (i+d<n) x[i] += x[i+d];
        }
    }
    return x[0];
}

int main( void )
{
    float v[51], result;
    const int n = sizeof(v) / sizeof(v[0]);
    int i;
    for (i=0; i<n; i++) {
	v[i] = i+1;
    }
    result = sum_reduce(v, n);
    if ( fabs(result - n*(n+1)/2.0) > 1e-5 ) {
        fprintf(stderr, "Error: expected %f, got %f\n", n*(n+1)/2.0, result);
        return EXIT_FAILURE;
    }
    printf("Test ok\n");
    return EXIT_SUCCESS;
}
