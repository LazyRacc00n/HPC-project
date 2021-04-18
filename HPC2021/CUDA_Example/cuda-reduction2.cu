/****************************************************************************
 *
 * cuda-reduction2.cu - Reduction with CUDA
 *
 * This is a version of cuda-reduction1.cu that works for any array
 * length n; however, it still requires that BLKDIM is a power of two.
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
 * nvcc cuda-reduction2.cu -o cuda-reduction2
 *
 * Run with:
 * ./cuda-reduction2
 *
 ****************************************************************************/

#include <stdio.h>
#include <assert.h>

/* For this program to work, BLKDIM must be a power of two */
#define BLKDIM 1024

__global__ void sum( int *a, int n, int *sums )
{
    __shared__ int temp[BLKDIM];
    int lindex = threadIdx.x;
    int bindex = blockIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;

    if ( gindex < n ) {
        temp[lindex] = a[gindex];
    } else {
        /* Threads that are mapped outside the array a[] fill their
           corresponding entry of temp[] with 0, which is the neutral
           element of the reduction operator + */
        temp[lindex] = 0;
    }

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
        sums[bindex] = temp[0];
    }
}

void init( int *v, int n )
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = 2;
    }
}

int main( int argc, char *argv[] ) 
{
    int *h_a, *h_tmp;
    int *d_a, *d_tmp;
    int i, s = 0;
    int n = 1024*512;
    
    assert( (BLKDIM & (BLKDIM-1)) == 0 ); /* check if BLKDIM is a power of two using the "bit hack" from http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    const size_t size = n * sizeof(*h_a);
    const int n_of_blocks = (n + BLKDIM - 1)/BLKDIM;
    const size_t size_tmp = n_of_blocks * sizeof(*h_tmp);

    /* Allocate space for host copies of a[] and tmp[] */
    h_a = (int*)malloc(size);
    h_tmp = (int*)malloc(size_tmp);
    init(h_a, n);

    /* Allocate space for device copies of d_a */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_tmp, size_tmp);

    /* Copy inputs to device */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    /* Launch sum() kernel on the GPU */
    sum<<<n_of_blocks, BLKDIM>>>(d_a, n, d_tmp);

    /* Copy the d_tmp[] array from device memory to host memory h_tmp[] */
    cudaMemcpy(h_tmp, d_tmp, size_tmp, cudaMemcpyDeviceToHost);

    /* Perform the final reduction on the CPU */
    s = 0;
    for (i=0; i<n_of_blocks; i++) {
        s += h_tmp[i];
    }
    /* Check result */
    const int expected = 2*n;
    if ( s != expected ) {
        printf("Check FAILED: got %d, expected %d\n", s, expected);
    } else {
        printf("Check OK: sum = %d\n", s);
    }
    /* Cleanup */
    free(h_a); free(h_tmp);
    cudaFree(d_a); cudaFree(d_tmp);
    return EXIT_SUCCESS;
}
