/****************************************************************************
 *
 * cuda-stencil1d-shared.cu - 1D stencil using __shared__ memory
 *
 * Based on the examples from the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-stencil1d-shared.cu -o cuda-stencil1d-shared
 *
 * Run with:
 * ./cuda-stencil1d-shared
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLKDIM 1024
#define RADIUS 3
#define N (BLKDIM*1024)

__global__ void stencil1d(int *in, int *out) 
{
    __shared__ int temp[BLKDIM + 2 * RADIUS];
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    const int lindex = threadIdx.x + RADIUS;
    int result = 0, offset;
    /* Read input elements into shared memory */
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + blockDim.x] = in[gindex + blockDim.x];
    }
    __syncthreads(); 
    /* Apply the stencil */
    for (offset = -RADIUS ; offset <= RADIUS ; offset++) {
        result += temp[lindex + offset];
    }
    /* Store the result */
    out[gindex] = result;
}

int main( void ) 
{
    int *h_in, *h_out;	  /* host copies of in and out */
    int *d_in, *d_out;	  /* device copies of in and out */
    int i;
    const size_t size = (N+2*RADIUS)*sizeof(int);

    assert( N % BLKDIM == 0 );

    /* Allocate space for device copies of d_in and d_out */
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    /* Allocate space for host copies of h_in and h_out */
    h_in = (int*)malloc(size);
    h_out = (int*)malloc(size);
    /* Set all elements of vector h_in to one */
    for (i=0; i<N+2*RADIUS; i++) {
        h_in[i] = 1;
    }
    /* Copy input to device */
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    /* Launch stencil1d() kernel on GPU */
    stencil1d<<<(N + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_in, d_out);
    /* Copy result back to host */
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    /* Check the result */
    for (i=RADIUS; i<N+RADIUS; i++) {
        if ( h_out[i] != 7 ) {
            fprintf(stderr, "Error at index %d: h_out[%d] == %d, expected 7\n", i, i, h_out[i]);
            return EXIT_FAILURE;
        }
    }
    printf("Test OK\n");
    /* Cleanup */
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return EXIT_SUCCESS;
}
