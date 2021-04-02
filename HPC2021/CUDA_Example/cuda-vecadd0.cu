/****************************************************************************
 *
 * cuda-vecadd0.cu - Sum two integers with CUDA
 *
 * Based on the examples from the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-vecadd0.cu -o cuda-vecadd0
 *
 * Run with:
 * ./cuda-vecadd0
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

__global__ void add( int *a, int *b, int *c )
{
    *c = *a + *b;
}

int main( void ) 
{
    int a, b, c;	          /* host copies of a, b, c */ 
    int *d_a, *d_b, *d_c;	  /* device copies of a, b, c */
    const size_t size = sizeof(int);
    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    /* Setup input values */
    a = 2; b = 7;
    /* Copy inputs to device */
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    /* Launch add() kernel on GPU */
    add<<<1,1>>>(d_a, d_b, d_c);
    /* Copy result back to host */
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    /* check result */
    if ( c != a + b ) {
        fprintf(stderr, "Test FAILED: expected %d, got %d\n", a+b, c);
    } else {
        printf("Test OK\n");
    }
    /* Cleanup */
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return EXIT_SUCCESS;
}
