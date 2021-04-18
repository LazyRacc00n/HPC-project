/****************************************************************************
 *
 * omp-fibonacci-tasks.c - Compute Fibonacci numbers with OpenMP tasks
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
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-fibonacci-tasks.c -o omp-fibonacci-tasks
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp-fibonacci-tasks 10
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Compute the n-th Fibonacci number using OpenMP tasks.  This
   algorithm is based on the inefficient recursive version that
   performs O(2^n) calls. */
int fib( int n )
{
    int n1, n2;
    if (n < 2) {
        return 1;
    } else {
#pragma omp task shared(n1)
        n1 = fib(n-1);
#pragma omp task shared(n2)
        n2 = fib(n-2);
        /* Wait for the two tasks above to complete */
#pragma omp taskwait
        return n1 + n2;
    }
}

int main( int argc, char* argv[] )
{
    int n = 10, res;
    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }
    /* Create a thread pool */
#pragma omp parallel
    {
        /* Only the master invokes the recursive algorithms (otherwise
           all threads in the pool would start the recursion) */
#pragma omp master
        res = fib(n);
    }
    printf("fib(%d)=%d\n", n, res);
    return EXIT_SUCCESS;
}
