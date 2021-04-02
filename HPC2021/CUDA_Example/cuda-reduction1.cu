/****************************************************************************
 *
 * cuda-reduction1.cu - Reduction with CUDA
 *
 * This version uses multiple threads of the same block to compute the
 * local reduction. The array length n must be a multiple of BLKDIM,
 * and BLKDIM must be a power of two.
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
 * ---------------------------------------------------------------------------
 *
 * This program realizes a not-so-simple sum-reduction on the
 * GPU. Each thread block copies a portion of the array in shared
 * memory; then, all threads within the same block cooperate to
 * compute the sum of the local data by organizing the computation as
 * a tree. The final reduction is executed on the CPU.
 *
 * Compile with:
 * nvcc cuda-reduction1.cu -o cuda-reduction1
 *
 * Run with:
 * ./cuda-reduction1
 *
 ****************************************************************************/

#include <stdio.h>
#include <assert.h>

/* BLKDIM must be a power of two */
#define BLKDIM 1024
#define N_OF_BLOCKS 1024
/* N must be an integer multiple of BLKDIM */
#define N ((N_OF_BLOCKS)*(BLKDIM))

/* d_sums is an array of N_OF_BLOCKS integers that reside in device
   memory; therefore, there is no need to cudaMalloc'ate it */
__device__ int d_sums[N_OF_BLOCKS];
int h_sums[N_OF_BLOCKS];

/* This kernel copies a portion of the array a[] of length n into
   thread-local shared memory. All threads cooperate to compute the
   sum of the local data; at the end, thread 0 stores the computed
   value on the appropriate entry of d_sums[]. Different thread blocks
   access different elements of d_sums[], so no race condition is
   possible.

   This function requires that BLKDIM is a power of two and that n is
   a muliple of BLKDIM. */
__global__ void sum( int *a, int n )
{
    __shared__ int temp[BLKDIM];
    int lindex = threadIdx.x;
    int bindex = blockIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;

    temp[lindex] = a[gindex];

    /* wait for all threads to finish the copy operation */
    __syncthreads(); 

    /* All threads within the block cooperate to compute the local sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2; 
        /* threads must synchronize before performing the next
           reduction step */
        __syncthreads(); 
    }

    if ( 0 == lindex ) {
        /* Thread 0 of each block copies the local sum (temp[0]) into
           the appropriate element of d_sums */
        d_sums[bindex] = temp[0];
    }
}

int main( void ) 
{
    int *h_a;
    int *d_a;
    int i, s=0;

    assert( 0 == N % BLKDIM ); /* N must be a multiple of BLKDIM */
    assert( (BLKDIM & (BLKDIM-1) ) == 0 ); /* check if BLKDIM is a power of two using the "bit hack" from http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */

    /* Allocate space for device copies of d_a */
    cudaMalloc((void **)&d_a, N*sizeof(int));
    /* Allocate space for host copies of h_a */
    h_a = (int*)malloc(N * sizeof(int));
    /* Set all elements of vector h_a to 2, so that we know that the
       result of the sum must be 2*N */
    for (i=0; i<N; i++) {
        h_a[i] = 2;
    }
    /* Copy inputs to device */
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    /* Launch sum() kernel on the GPU */
    sum<<<N_OF_BLOCKS, BLKDIM>>>(d_a, N);
    /* Copy the d_sums[] array from device memory to host memory h_sums[] */
    cudaMemcpyFromSymbol(h_sums, d_sums, N_OF_BLOCKS*sizeof(int));
    /* Perform the final reduction on the CPU */
    s = 0;
    for (i=0; i<N_OF_BLOCKS; i++) {
        s += h_sums[i];
    }
    /* Check result */
    if ( s != 2*N ) {
        printf("Check FAILED: expected %d, got %d\n", 2*N, s);
    } else {
        printf("Check OK: sum = %d\n", s);
    }
    /* Cleanup */
    free(h_a);
    cudaFree(d_a);
    return EXIT_SUCCESS;
}
