/****************************************************************************
 *
 * omp-demo2.c - "Hello world" with OpenMP
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
 * gcc -fopenmp omp-demo2.c -o omp-demo2
 *
 * Run with:
 * OMP_NUM_THREADS=8 ./omp-demo2
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void say_hello( void )
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", my_rank, thread_count);
}

int main( int argc, char* argv[] )
{
    int thr = 2;

    if ( argc == 2 )
	thr = atoi( argv[1] );

#pragma omp parallel num_threads(thr)
    say_hello();

    return 0;
}
