/****************************************************************************
 *
 * vla-demo.c - Variable-length arrays demo
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
 *
 * gcc -std=c99 -Wall -Wpedantic vla-demo.c -o vla-demo
 *
 * Run with:
 *
 * ./vla-demo
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

void identity_mat( int *mat, size_t n )
{
    /* mat_m is a pointer to an array[n] of integers */
    int (*mat_m)[n] = (int (*)[n])mat;
    size_t i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            mat_m[i][j] = (i == j ? 1 : 0); /* un po' ridondante, ma comprensibile */
        }
    }
}

void print_mat( const int *mat, size_t n )
{
    /* mat_m is a pointer to an array[n] of (constant) integers */
    const int (*mat_m)[n] = (const int (*)[n])mat;
    size_t i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            printf("%d\t", mat_m[i][j]);
        }
        printf("\n");
    }
}

int main( void )
{
    const size_t n = 8;
    int *m = (int*)malloc(n*n*sizeof(int));
    identity_mat(m, n);
    print_mat(m, n);
    free(m);
    return EXIT_SUCCESS;
}
