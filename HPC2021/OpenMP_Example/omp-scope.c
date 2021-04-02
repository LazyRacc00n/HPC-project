/****************************************************************************
 *
 * omp-scoipe.c - Demonstration of the OpenMP "scope" clause.
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
 * gcc -std=c99 -Wall -pedantic -fopenmp omp-scope.c -o omp-scope
 *
 * Run with:
 * ./omp-scope
 *
 ****************************************************************************/
#include <stdio.h>

int main( void )
{
    int a=1, b=1, c=1, d=1;	
#pragma omp parallel num_threads(10) \
    private(a) shared(b) firstprivate(c)
    {	
	printf("Hello World!\n");
	a++;	
	b++;	
	c++;	
	d++;	
    }	
    printf("a=%d\n", a);
    printf("b=%d\n", b);
    printf("c=%d\n", c);
    printf("d=%d\n", d);
    return 0;
}
