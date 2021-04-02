/****************************************************************************
 *
 * simd-vsum-vector.c - Vector sum using vector data type
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
 *
 * gcc -O2 -std=c99 -Wall -Wpedantic -march=native simd-vsum-vector.c -lm -o simd-vsum-vector
 *
 ****************************************************************************/

/* The posix_memalign() function is a POSIX extension; the function is
   defined in stdlib.h, but is visible only if _XOPEN_SOURCE is set to
   600. It is better to define this symbol _before_ including any
   system header, since stdlib.h might be included indirectly by some
   other header, and the functions it provides are "frozen" after the
   first include */
#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */

typedef float v4f __attribute__((vector_size(16)));
#define VLEN (sizeof(v4f)/sizeof(float))

float vsum_vector(const float *v, int n)
{
    v4f vs = {0.0f, 0.0f, 0.0f, 0.0f};
    float ss = 0.0;
    const v4f *vv = (v4f*)v;
    int i;
    for (i=0; i<n-VLEN+1; i += VLEN) {
        vs += *vv;
        vv++;
    }

    /* Horizontal sum */
    ss = vs[0] + vs[1] + vs[2] + vs[3];

    /* Take care of leftovers, if any */
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
        return 1;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    /* WARNING: On some platforms, the compiler seems to emit SIMD
       aligned loads in vsum_vector; those loads failif the vector is
       not properly aligned.  malloc() guarantees that the result is
       "properly aligned for any built-in type"; on some 32 bit
       machines this alignment seems to be 8, which is insufficient
       for SIMD aligned loads. To be on the safe side, we enforce
       proper alignment with posix_memalign(). */    
    posix_memalign((void **)&vec, __BIGGEST_ALIGNMENT__, n*sizeof(*vec));

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
    return 0;
}
