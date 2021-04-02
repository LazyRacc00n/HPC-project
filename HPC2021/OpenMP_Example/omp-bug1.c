/****************************************************************************
 *
 * omp-bug1.c - Exhibits different behavior of GCC >= 9.x
 *
 * Works with GCC < 9.x, compilation error with GCC >= 9.x
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
 ****************************************************************************/
#include <stdio.h>
#include <omp.h>

int main( void )
{
    const int foo = 1;
#pragma omp parallel default(none)
    {
        int baz = 0;
        baz += foo;
        printf("Thread %d: baz=%d\n", omp_get_thread_num(), baz);
    }
    return 0;
}
