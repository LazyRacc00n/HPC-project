/****************************************************************************
 *
 * mandelbrot.c - displays the Mandelbrot set using ASCII characters
 *
 * Copyright (C) 2017, 2020 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * This program computes and displays the Mandelbrot set using ASCII characters.
 * Since characters are displayed one at a time, 
 *
 * Compile with
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot-ordered -o omp-mandelbrot-ordered
 *
 * and run with:
 * OMP_NUM_THREADS=4 ./omp-mandelbrot-ordered
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

const int maxit = 500000; /* A very large value to make the program slower */
const int xsize = 78, ysize = 62;

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first n such that z_n > |bound|, or |maxit| if z_n is below
 * |bound| after |maxit| iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0, y = 0.0, xnew, ynew;
    int it;
    for ( it = 0; (it < maxit) && (x*x + y*y < 2*2); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0f*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

int main( int argc, char *argv[] )
{
    int x, y;
    float tstart, elapsed;
    const char charset[] = ".,c8M@jawrpogOQEPGJ";
    
    tstart = hpc_gettime();
    /* x, y are private by default, being used in "for" statements
       that are both parallelized by the collapse(2) directive */
#if __GNUC__ < 9
#pragma omp parallel for default(none) collapse(2) ordered
#else
#pragma omp parallel for default(none) shared(xsize,ysize,charset,maxit) collapse(2) ordered
#endif
    for ( y = 0; y < ysize; y++ ) {
        for ( x = 0; x < xsize; x++ ) {
            float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            float cy = 1 - 2.0 * (float)y / (ysize - 1);
            int v = iterate(cx, cy);
#pragma omp ordered 
            {
                char c = ' ';
                if (v < maxit) {
                    c = charset[v % (sizeof(charset)-1)];
                }
                putchar(c);
                if (x+1 == xsize) puts("|");
            }            
        }
    }
    elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    return EXIT_SUCCESS;
}
