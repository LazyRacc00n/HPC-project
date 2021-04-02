/****************************************************************************
 *
 * omp-reduction - Demo of reduction operators with OpenMP
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
 * ----------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -fopenmp omp-reduction.c -o omp-reduction
 *
 * Run with:
 * OMP_NUM_THREADS=1 ./omp-reduction
 * OMP_NUM_THREADS=2 ./omp-reduction
 * OMP_NUM_THREADS=4 ./omp-reduction
 *
 ****************************************************************************/

#include <stdio.h>

int main( void )
{
    int a = 2;
#pragma omp parallel reduction(*:a)
    {
	a += 2;
    }
    printf("%d\n",a);
    return 0;
}
