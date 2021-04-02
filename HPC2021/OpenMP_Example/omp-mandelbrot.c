/****************************************************************************
 *
 * omp-mandelbrot.c - displays the Mandelbrot set
 *
 * Copyright (C) 2019, 2020 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * This program computes and display the Mandelbrot set. This program
 * requires the gfx library from
 * http://www.nd.edu/~dthain/courses/cse20211/fall2011/gfx (the
 * library should be already included in the archive containing this
 * source file)
 *
 * Compile with
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot.c gfx.c -o omp-mandelbrot -lX11
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp-mandelbrot
 *
 * If you enable the "runtime" scheduling clause, you can select the
 * scheduling type at runtime, e.g.,
 *
 * OMP_NUM_THREADS=4 OMP_SCHEDULE="static,64" ./omp-mandelbrot
 *
 * At the end, click the left mouse button to close the graphical window.
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include "gfx.h"

const int MAXIT = 10000;
/* const int XSIZE = 800, YSIZE = 600; */
const int XSIZE = 1024, YSIZE = 768;

typedef struct {
    int r, g, b;
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},    
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + (cx + i*cy);
 *
 * Returns the first n such that ||z_n|| > 2, or |MAXIT|
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/*
 * Draw a pixel at window coordinates (x, y) with the appropriate
 * color; (0,0) is the upper left corner of the window, y grows
 * downward.
 */
void drawpixel( int x, int y )
{
    const float cx = -2.5 + 3.5 * (float)(x) / (XSIZE - 1);
    const float cy = 1 - 2.0 * (float)(y) / (YSIZE - 1);
    const int v = iterate(cx, cy);
/* only one thread can access the display at any time */
#pragma omp critical 
    {
        if (v < MAXIT) {
            gfx_color( colors[v % NCOLORS].r,
                       colors[v % NCOLORS].g,
                       colors[v % NCOLORS].b );            
        } else {
            gfx_color( 0, 0, 0 );                        
        }
	gfx_point( x, y );
    }
}

int main( int argc, char *argv[] )
{
    int x, y;
    double tstart, elapsed;

    gfx_open(XSIZE, YSIZE, "Mandelbrot Set");
    tstart = hpc_gettime();
#if __GNUC__ < 9
#pragma omp parallel for default(none) private(x) schedule(runtime)
#else
#pragma omp parallel for default(none) shared(XSIZE,YSIZE) private(x) schedule(runtime)
#endif
    for ( y = 0; y < YSIZE; y++ ) {
	for ( x = 0; x < XSIZE; x++ ) {
	    drawpixel( x, y );
	}
    }
    elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    printf("Click to finish\n");
    gfx_wait();
    return EXIT_SUCCESS;
}
