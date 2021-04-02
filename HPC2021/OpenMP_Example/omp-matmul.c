/****************************************************************************
 *
 * omp-matmul.c - Dense matrix-matrix multiply
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
 * gcc -fopenmp omp-matmul.c -o omp-matmul
 *
 * Run with:
 * ./omp-matmul [n]
 *
 * (n = matrix size; default 1000)
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

/* Fills n x n square matrix m with random values */
void fill( double* m, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            m[i*n + j] = (double)rand() / RAND_MAX;
        }
    }
}

int min(int a, int b)
{
    return (a < b ? a : b);
}

/* compute r = p * q, where p, q, r are n x n matrices. The caller is
   responsible for allocating the memory for r */
void matmul( double *p, double* q, double *r, int n)
{
    int i, j, k;
#pragma omp parallel for collapse(2) private(k)
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = 0.0;
            for (k=0; k<n; k++) {
                r[i*n + j] += p[i*n + k] * q[k*n + j];
            }
        }
    }
}

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
  matrices. The caller is responsible for allocating the memory for
  r. This function allocates (and the frees) an additional n x n
  temporary matrix. */
void matmul_transpose( double *p, double* q, double *r, int n)
{
    int i, j, k;
    double *qT = (double*)malloc( n * n * sizeof(double) );

    /* transpose q, storing the result in qT */
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (i=0; i<n; i++) {
            for (j=0; j<n; j++) {
                qT[j*n + i] = q[i*n + j];
            }
        }    
        
        /* multiply p and qT row-wise */
#pragma omp for collapse(2) private(k)
        for (i=0; i<n; i++) {
            for (j=0; j<n; j++) {
                r[i*n + j] = 0.0;
                for (k=0; k<n; k++) {
                    r[i*n + j] += p[i*n + k] * qT[j*n + k];
                }
            }
        }
    }

    free(qT);
}

int main( int argc, char *argv[] )
{
    int n = 1000;
    double *p, *q, *r;
    double tstart, tstop;

    if ( argc > 2 ) {
        printf("Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    p = (double*)malloc( n * n * sizeof(double) );
    q = (double*)malloc( n * n * sizeof(double) );
    r = (double*)malloc( n * n * sizeof(double) );

    fill(p, n);
    fill(q, n);

    tstart = hpc_gettime();
    matmul(p, q, r, n);
    tstop = hpc_gettime();
    printf("Execution time %f\n", tstop - tstart);  

    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
