/****************************************************************************
 *
 * simd-hsum.c - horizontal sum with SSE2
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
#include <x86intrin.h>

void print_xmm(__m128i v)
{
    int* vv = (int*)&v;
    printf("v0=%d v1=%d v2=%d v3=%d\n", vv[0], vv[1], vv[2], vv[3]);
}

int main( void )
{
    __m128i v, vp;
    int r;

    v  = _mm_set_epi32(19,-1, 77, 34); /* [34|77|-1|19] */
    print_xmm(v);
    vp = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));   /*  11.11.01.01  */
    print_xmm(vp);
    v = _mm_add_epi32(v, vp);
    print_xmm(v);
    vp = _mm_shuffle_epi32(v, 0xaa);   /*  10.10.10.10  */
    print_xmm(vp);
    v = _mm_add_epi32(v, vp);
    print_xmm(v);
    r = _mm_cvtsi128_si32(v);      /* get v0        */
    printf("%d\n", r);
    return EXIT_SUCCESS;
}
