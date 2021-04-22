/****************************************************************************
 *
 * game-of-life.c - Serial implementaiton of the Game of Life
 *
 * Copyright (C) 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last updated on 2019-10-01
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
 * gcc -std=c99 -Wall -Wpedantic game-of-life.c -o game-of-life
 *
 * Run with:
 * ./game-of-life 100
 *
 * To display the images
 * animate -delay 50 gol*.pbm
 *
 * To create a movie from the images:
 * ffmpeg -framerate 10 -i "gol%04d.pbm" -r 30 -vcodec mpeg4 gol.avi
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* for time() */

/* gen size (excluding ghost cells) */
#define SIZE 256

int cur = 0; /* index of current gen (must be 0 or 1) */
unsigned char gen[2][SIZE+2][SIZE+2];

/* some useful constants; starting and ending rows/columns of the domain */
const int ISTART = 1;
const int IEND   = SIZE; 
const int JSTART = 1;
const int JEND   = SIZE;

/*
     JSTART         JEND  
     |              |
     v              V
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\|
  +-+----------------+-+
  |\|                |\| <- ISTART
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\| <- IEND
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\| 
  +-+----------------+-+

 */

/* copy the sides of current gen to the ghost cells. This function
   uses the global variables cur and gen. gen[cur] is modified.*/
#if 1
void copy_sides( void )
{
    int i, j;
    /* copy top and bottom (one should better use memcpy() ) */
    for (j=JSTART; j<JEND+1; j++) {
        gen[cur][ISTART-1][j] = gen[cur][IEND  ][j];
        gen[cur][IEND+1  ][j] = gen[cur][ISTART][j];
    }
    /* copy left and right */
    for (i=ISTART; i<IEND+1; i++) {
        gen[cur][i][JSTART-1] = gen[cur][i][JEND  ];
        gen[cur][i][JEND+1  ] = gen[cur][i][JSTART];
    }
    /* copy corners */
    gen[cur][ISTART-1][JSTART-1] = gen[cur][IEND  ][JEND  ];
    gen[cur][ISTART-1][JEND+1  ] = gen[cur][IEND  ][JSTART];
    gen[cur][IEND+1  ][JSTART-1] = gen[cur][ISTART][JEND  ];
    gen[cur][IEND+1  ][JEND+1  ] = gen[cur][ISTART][JSTART];
}
#else
/* Another way to fill the ghost cells: change the ranges of the "for"
   cycles to copy entire rows and columns (including ghost cells). At
   the end, corners get filled with the correct values (draw an
   example to convince yourself). The interesting thing is that you
   can swap the two "for" cycles (i.e., first handle columns, then
   handle rows) and the final result is still correct. */
void copy_sides( )
{
    int i, j;
    /* Copy top and bottom (one can also use memcpy() ). We copy a
       whole row (including ghost cells). */
    for (j=0; j<JEND+2; j++) {
        gen[cur][ISTART-1][j] = gen[cur][IEND  ][j];
        gen[cur][IEND+1  ][j] = gen[cur][ISTART][j];
    }
    /* Copy left and right. We copy a whole column (including ghost
       cells). */
    for (i=0; i<IEND+2; i++) {
        gen[cur][i][JSTART-1] = gen[cur][i][JEND  ];
        gen[cur][i][JEND+1  ] = gen[cur][i][JSTART];
    }
    /* There is no need to fill the corners */
}
#endif

/* Compute the next gen given the current configuration; this
   function uses the global variables gen and cur; updates are
   written to the (1-cur) gen. */
void step( void )
{
    int i, j, next = 1 - cur;
    for (i=ISTART; i<IEND+1; i++) {
        for (j=JSTART; j<JEND+1; j++) {
            /* count live neighbors of cell (i,j) */
            int nbors = 
                gen[cur][i-1][j-1] + gen[cur][i-1][j] + gen[cur][i-1][j+1] + 
                gen[cur][i  ][j-1] +                     gen[cur][i  ][j+1] + 
                gen[cur][i+1][j-1] + gen[cur][i+1][j] + gen[cur][i+1][j+1];
 	    /* apply rules of the game of life to cell (i, j) */
            if ( gen[cur][i][j] && (nbors < 2 || nbors > 3)) {
                gen[next][i][j] = 0;
            } else {
                if ( !gen[cur][i][j] && (nbors == 3)) {
                    gen[next][i][j] = 1;
                } else {
                    gen[next][i][j] = gen[cur][i][j];
                }
            }
        }
    }
}

/* Initialize the current gen gen[cur] with alive cells with density
   p. This function uses the global variables cur and gen. gen[cur]
   is modified. */
void init( float p )
{
    int i, j;
    for (i=ISTART; i<IEND+1; i++) {
        for (j=JSTART; j<JEND+1; j++) {
            gen[cur][i][j] = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write gen[cur] to file fname in pbm (portable bitmap) format. This
   function uses the global variables cur and gen (neither is
   modified). */
void write_pbm( const char* fname )
{
    int i, j;
    FILE *f = fopen(fname, "w");
    if (!f) { 
        printf("Cannot open %s for writing\n", fname);
        abort();
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by game-of-life.c\n");
    fprintf(f, "%d %d\n", SIZE, SIZE);
    for (i=ISTART; i<IEND+1; i++) {
        for (j=JSTART; j<JEND+1; j++) {
            fprintf(f, "%d ", gen[cur][i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

#define BUFSIZE 128

int main( int argc, char* argv[] )
{
    int s, nsteps = 1000;
    char fname[BUFSIZE];

    srand(time(NULL)); /* init RNG */
    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc == 2 ) {
        nsteps = atoi(argv[1]);
    }
    cur = 0;
    init(0.3);
    for (s=0; s<nsteps; s++) {
        snprintf(fname, BUFSIZE, "gol%04d.pbm", s);
        write_pbm(fname);
        copy_sides();
        step();
        cur = 1 - cur;
    }
    return EXIT_SUCCESS;
}
