/****************************************************************************
 *
 * omp-nested.c - Nested parallelism with OpenMP
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
 * gcc -std=c99 -Wall -pedantic -fopenmp omp-nested.c -o omp-nested
 *
 * Run with:
 * OMP_NESTED=true ./omp-nested
 *
 ****************************************************************************/
#include <stdio.h>
#include <omp.h>

void greet(int level, int parent)
{
    printf("Level %d (parent=%d): greetings from thread %d of %d\n", 
           level, parent, omp_get_thread_num(), omp_get_num_threads());
}

int main( void )
{
    omp_set_num_threads(4);
#pragma omp parallel
    {
        greet(1, -1);
        int parent = omp_get_thread_num();
#pragma omp parallel
        {
            greet(2, parent);
            int parent = omp_get_thread_num();
#pragma omp parallel
            {
                greet(3, parent);
            }
        }
    }    
    return 0;
}
