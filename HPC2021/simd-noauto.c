/****************************************************************************
 *
 * simd-noauto : examples where automatic vectorization (could) fail
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
 ****************************************************************************/
#include <stdio.h>

#define SIZE 1000000
float vec[SIZE];

typedef float v4f __attribute__((vector_size(16)));
#define VLEN (sizeof(v4f)/sizeof(float))

float f(int n)
{
    int i;
    float x = 0.0;
    float s = 0.0;
    const float delta = 0.1;
    for (i=1; i<n; i++) {
        s += x*x;
        x += delta;
    }
    return s; /* unused */
}

void init(float *v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = (float)i / n;
    }
}

int main( void )
{
    init(vec, SIZE);
    printf("%f\n", f(SIZE));
    return 0;
}
