/****************************************************************************
 *
 * simd-vsum-intrinsics.c - Vector sum using SSE intrinsics
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 *
 * gcc -std=c99 -Wall -Wpedantic -mtune=native -lm simd-vsum-intrinsics.c -o simd-vsum-intrinsics
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <math.h>  /* for fabs() */

/* Return the sum of the n values stored in v,m using SIMD intrinsics.
   no particular alignment is required for v; n can be arbitrary */
float vsum_vector(float *v, int n)
{
    __m128 vv, vs;
    float ss = 0.0;
    int i;

    vs = _mm_setzero_ps();
    for (i=0; i<n-4+1; i += 4) {
        vv = _mm_loadu_ps( &v[i] );     /* load four floats into vv */
        vs = _mm_add_ps(vv, vs);        /* vs = vs + vv */
    }

    /* Horizontal sum */
    ss = vs[0] + vs[1] + vs[2] + vs[3];

    /* Take care of leftovers */
    for ( ; i<n; i++) {
        ss += v[i];
    }
    return ss;
}

float vsum_scalar(const float *v, int n)
{
    float s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += v[i];
    }
    return s;
}

void fill(float *v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = i%10;
    }
}

int main( int argc, char *argv[] )
{
    float *vec;
    int n = 1024;
    float vsum_s, vsum_v;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    /* The memory does not need to be aligned, since
       we are using the load unaligned intrinsic */
    vec = (float*)malloc(n*sizeof(*vec));
    fill(vec, n);
    vsum_s = vsum_scalar(vec, n);
    vsum_v = vsum_vector(vec, n);
    printf("Scalar sum=%f, vector sum=%f\n", vsum_s, vsum_v);
    if ( fabs(vsum_s - vsum_v) > 1e-5 ) {
        fprintf(stderr, "Test FAILED\n");
    } else {
        fprintf(stderr, "Test OK\n");
    }
    free(vec);
    return EXIT_SUCCESS;
}
