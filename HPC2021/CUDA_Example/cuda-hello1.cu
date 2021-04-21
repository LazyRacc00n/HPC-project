/****************************************************************************
 *
 * cuda-hello1.cu - Hello world with CUDA (with dummy device code)
 *
 * Based on the examples from the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-hello1.cu -o cuda-hello1
 *
 * Run with:
 * ./cuda-hello1
 *
 ****************************************************************************/

#include <stdio.h>

__global__ void mykernel( void ) { }

int main( void )
{
    int devices = 0; 

    cudaError_t err = cudaGetDeviceCount(&devices); 

    if (devices > 0 && err == cudaSuccess) 
    { 
        // Run CPU+GPU code
        printf("\n\nSI GPU!!\n\n");
    } 
    else
    {  
        printf("\n\nNO GPU!!\n\n");
        // Run CPU only code
    } 
    
    
    mykernel<<<1,1>>>( );
    printf("Hello, world!\n");
    return 0;
}
