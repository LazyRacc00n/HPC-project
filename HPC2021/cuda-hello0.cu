/****************************************************************************
 *
 * cuda-hello0.cu - Hello world with CUDA (no device code)
 *
 * Based on the examples from in the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-hello0.cu -o cuda-hello0
 *
 * Run with:
 * ./cuda-hello0
 *
 ****************************************************************************/

#include <stdio.h>

int main( void )
{
    printf("Hello, world!\n");
    return 0;
}
