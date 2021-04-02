/****************************************************************************
 *
 * omp-demo4.c - Demo with OpenMP
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
 * gcc -fopenmp omp-demo4.c -o omp-demo4 -lgomp
 *
 * Run with:
 * ./omp-demo4
 *
 ****************************************************************************/
#include <stdio.h>
#include <omp.h>

int main( int argc, char* argv[] )
{
    printf("Before parallel region: threads=%d, max_threads=%d\n",
           omp_get_num_threads(), omp_get_max_threads());
#pragma omp parallel
    {
        printf("Inside parallel region: threads=%d, max_threads=%d\n",
               omp_get_num_threads(), omp_get_max_threads());
    }

    return 0;
}
