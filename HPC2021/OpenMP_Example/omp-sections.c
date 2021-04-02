/****************************************************************************
 *
 * omp_sections.c - Demostration of the OpenMP "sections" work sharing directive
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
 * ----------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -fopenmp omp_sections.c -o omp_sections
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp_sections
 *
 *****************************************************************************/
#include <stdio.h>
#define N 1000

int main(void)
{
    int i;
    float a[N], b[N], c[N], d[N];
    /* Some initializations */
    for (i=0; i < N; i++) {
	a[i] = i * 1.5;
	b[i] = i + 22.35;
    }
#pragma omp parallel shared(a,b,c,d) private(i)
    {
#pragma omp sections nowait
	{
#pragma omp section
	    for (i=0; i < N; i++)
		c[i] = a[i] + b[i];
#pragma omp section
	    for (i=0; i < N; i++)
		d[i] = a[i] * b[i];
	}  /* end of sections (no barrier here) */
    }  /* end of parallel section (barrier here) */

    for ( i=0; i<N; i++) {
	printf("%f %f %f %f\n", a[i], b[i], c[i], d[i]);
    }

    return 0;
}
