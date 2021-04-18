/****************************************************************************
 *
 * simd-permute.c - Demo of SSE permute operation
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

void print_simd( __m128i v )
{
    const int* vv = (int*)&v;
    printf("%d %d %d %d\n", vv[0], vv[1], vv[2], vv[3]);
}

int main( void )
{
    __m128i v, v1, v2, v3, v4;

    v = _mm_set_epi32(19, -1, 77, 34);
    print_simd(v);
    v1 = _mm_shuffle_epi32(v, 0xa0); /* 10.10.00.00 */
    print_simd(v1);
    v2 = _mm_add_epi32(v, v1);
    print_simd(v2);
    v3 = _mm_shuffle_epi32(v2, 0x55); /* 01.01.01.01 */
    print_simd(v3);
    v4 = _mm_add_epi32(v2, v3);
    print_simd(v4);
    return 0;
}
