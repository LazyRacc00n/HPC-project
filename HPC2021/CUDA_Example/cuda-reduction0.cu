/****************************************************************************
 *
 * cuda-reduction0.cu - Reduction with CUDA
 *
 * Basic version: only the first thread of each block does a partial
 * reduction.
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
 * This program realizes a simple sum-reduction on the GPU. This
 * implementation is not efficient: each thread block copies a portion
 * of the array in shared memory; thread 0 of each block computes a
 * partial sum of the local data. The final reduction must be
 * completed by the CPU.
 *
 * Compile with:
 * nvcc cuda-reduction0.cu -o cuda-reduction0
 *
 * Run with:
 * ./cuda-reduction0
 *
 ****************************************************************************/

#include <stdio.h>
#include <assert.h>

#define BLKDIM 1024
#define N_OF_BLOCKS 1024
/* N must be an integer multiple of BLKDIM */
#define N ((N_OF_BLOCKS)*(BLKDIM))

/* d_sums is an array of N_OF_BLOCKS integers that reside in device
   memory; therefore, there is no need to cudaMalloc'ate it */
__device__ int d_sums[N_OF_BLOCKS];
int h_sums[N_OF_BLOCKS];

/* This kernel copies a portion of array a[] of n elements into
   thread-local shared memory. Thread 0 computes the sum of the local
   data, and stores the computed value on the appropriate entry of
   d_sums[]. Different thread blocks access different elements of
   d_sums[], so no race condition is possible. Note that the use of
   shared memory does not provide any advantage here, but only serves
   as a placeholder for more advanced versions of this code. */
__global__ void sum( int *a, int n )
{
    __shared__ int temp[BLKDIM];
    int lindex = threadIdx.x; /* local idx (index of the thread within the block) */
    int bindex = blockIdx.x; /* block idx (index of the block within the grid) */
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; /* global idx (index of the array element handled by this thread) */

    temp[lindex] = a[gindex];
    __syncthreads(); /* wait for all threads to finish the copy operation */
    /* only thread 0 computes the local sum */
    if ( 0 == lindex ) {
        int i, my_sum = 0;
        for (i=0; i<blockDim.x; i++) {
            my_sum += temp[i];
        }
        d_sums[bindex] = my_sum;
    }
}

int main( void ) 
{
    int *h_a;
    int *d_a;
    int i, s=0;
    assert( 0 == N % BLKDIM );
    /* Allocate space for device copies of d_a */
    cudaMalloc((void **)&d_a, N*sizeof(int));
    /* Allocate space for host copy of the array */
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
    cudaMemcpyFromSymbol(h_sums, d_sums, N_OF_BLOCKS*sizeof(h_sums[0]));
    /* Perform the final reduction on the CPU */
    s = 0;
    for (i=0; i<N_OF_BLOCKS; i++) {
        s += h_sums[i];
    }
    /* Check result */
    if ( s != 2*N ) {
        printf("Check FAILED: Expected %d, got %d\n", 2*N, s);
    } else {
        printf("Check OK: sum = %d\n", s);
    }
    /* Cleanup */
    free(h_a);
    cudaFree(d_a);
    return EXIT_SUCCESS;
}
