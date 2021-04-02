/****************************************************************************
 *
 * omp-lastprivate.c - Demo of the lastprivate clause
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
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-lastprivate.c -o omp-lastprivate
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp-lastprivate
 *
 ****************************************************************************/
#include <stdio.h>

int main( void )
{
    int tmp = 0, i;
#pragma omp parallel for firstprivate(tmp) lastprivate(tmp)
    for (i = 0; i < 1000; i++) {
        tmp = i;
    }
    printf("%d\n", tmp);
    return 0;
}
