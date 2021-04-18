/****************************************************************************
 *
 * simd-vsum-auto : Vector sum; this program is used to test compiler
 * auto-vectorization.
 *
 * Copyright (C) 2019 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * gcc -march=native -O2 -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized simd-vsum-auto.c -o simd-vsum-auto
 *
 * and observe that the loop in vec_sum is not vectorized.
 *
 * To forse vectorization anyway, use -funsafe-math-optimizations
 *
 * gcc -funsafe-math-optimizations -march=native -O2 -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized simd-vsum-auto.c -o simd-vsum-auto
 *
 * To see the assembly output:
 *
 * gcc -funsafe-math-optimizations -march=native -c -Wa,-adhln -g -O2 -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized simd-vsum-auto.c > simd-vsum-auto.s
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

float vsum(float *v, int n)
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

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }
    vec = (float*)malloc(n+sizeof(*vec));
    fill(vec, n);
    printf("sum = %f\n", vsum(vec, n));
    free(vec);
    return EXIT_SUCCESS;
}
