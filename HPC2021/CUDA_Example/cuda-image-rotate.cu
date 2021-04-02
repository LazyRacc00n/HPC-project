/****************************************************************************
 *
 * cuda-image-rotate.c - Image rotation with CUDA
 *
 * Copyright (C) 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * This program rotates a PGM image 90 degrees clockwise using CUDA.
 * Two kernels are implemented: one that does not use shared memory,
 * and one that does. This program requires that the image is square
 * (i.e., width == height), and that the width/height is a multiple of
 * BLKDIM.
 *
 * Input is read from stdin; output is written to stdout.
 *
 * This program illustrates the importance of collapsing memory
 * accesses to maximize the memory bandwidth. The kernel that uses
 * shared memory is much faster since writes can be collapsed.
 *
 * Compile with:
 *
 * nvcc cuda-image-rotate.cu -o cuda-image-rotate
 *
 * Run with:
 *
 * ./cuda-image-rotate 1001 < cat-1344.pgm > out.pgm
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define BLKDIM 32

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Used by the PGM read/write routines */
    unsigned char *bmap; /* buffer of width*height bytes; each byte represents a pixel */
} img_t;

/**
 * Read a PGM file from |f|. This function is not very robust; it may
 * fail on valid PGM images.
 */
void read_pgm( FILE *f, img_t* img )
{
    char buf[2048];
    const size_t BUFSIZE = sizeof(buf);
    char *s; 
    int nread;

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; the code below does not work if
       there are leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray > 255 (%d)\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
    /* Get the binary data */
    img->bmap = (unsigned char*)malloc((img->width)*(img->height));
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write image |img| to |f|
 */
void write_pgm( FILE *f, const img_t* img )
{
    fprintf(f, "P5\n");
    fprintf(f, "# produced by cuda-image-rotate.cu\n");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the memory allocated by image |img|, and set the additional
 * info to invalid values.
 */
void free_pgm( img_t *img )
{
    img->width = img->height = img->maxgrey = -1;
    free(img->bmap);
    img->bmap = NULL;
}

/**
 * Rotate square image |orig| of size |n|x|n| 90 degrees clockwise;
 * the new image goes to |rotated|. This kernel works for any value of
 * |n|.
 */
__global__
void rotate( unsigned char *orig, unsigned char *rotated, int n )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x<n && y<n) {
        rotated[n*x + (n - 1 - y)] = orig[n*y + x];
    }
}

/**
 * Same as above, using shared memory. This kernel requires that |n|
 * is a multiple of BLKDIM.
 */
__global__
void rotate_shared( unsigned char *orig, unsigned char *rotated, int n )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ unsigned char buf[BLKDIM][BLKDIM];
    
    if (x<n && y<n) {
        buf[local_x][BLKDIM-1-local_y] = orig[n*y+x];
        __syncthreads();

        const int dest_x = (gridDim.y - 1 - blockIdx.y) * BLKDIM + threadIdx.x;
        const int dest_y = blockIdx.x * BLKDIM + threadIdx.y;
        rotated[n*dest_y + dest_x] = buf[local_y][local_x];
    }
}

int main( int argc, char* argv[] )
{
    img_t img;
    unsigned char *d_orig, *d_new, *tmp;
    int nrotations = 1001;
    
    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s nrotations < input > output\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        nrotations = atoi(argv[1]);
    }
    
    read_pgm(stdin, &img);

    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }

    const int n = img.width;
    
    if (n % BLKDIM != 0) {
        fprintf(stderr, "FATAL: width/height (%d) must be a multiple of %d\n", n, BLKDIM);
        return EXIT_FAILURE;
    }

    /* Allocate buffers on the device */
    const size_t size = n*n;

    /* The cudaSafeCall() and cudaCheckError() macros are defined in
       hpc.h and can be used to check the results of CUDA operations
       and abort immediately if an error occur. To disable these
       checks, #define NO_CUDA_CHECK_ERROR _before_ including hpc.h */
    cudaSafeCall( cudaMalloc((void **)&d_orig, size) );
    cudaSafeCall( cudaMalloc((void **)&d_new, size) );

    /* Copy image to the device */
    cudaSafeCall( cudaMemcpy(d_orig, img.bmap, size, cudaMemcpyHostToDevice) );

    /* Define block and grid sizes */
    const dim3 block(BLKDIM, BLKDIM);
    const dim3 grid((img.width + BLKDIM - 1)/BLKDIM, (img.height + BLKDIM - 1)/BLKDIM);

    fprintf(stderr, "\nPerforming %d rotations (img size %dx%d)\n\n", nrotations, img.width, img.height);
    
    double tstart = hpc_gettime();
    for (int i=0; i<nrotations; i++) {
        rotate<<< grid, block >>>( d_orig, d_new, n );
        cudaCheckError(); /* Check whether the kernel completed succesfully */
        tmp = d_orig; d_orig = d_new; d_new = tmp;
    }
    cudaDeviceSynchronize();
    const double elapsed_noshmem = hpc_gettime() - tstart;
    const double Mpixels = ((double)img.width * img.height * nrotations)/1.0e6;
    fprintf(stderr, "No shmem: elapsed time %f s = %.2f Mpixels/s\n", elapsed_noshmem,
            Mpixels/elapsed_noshmem);
    
    /* Copy again image to the device, so we are sure that both
       kernels work on the same input (we should not care, since the
       timing is the same). */
    cudaMemcpy(d_orig, img.bmap, size, cudaMemcpyHostToDevice);

    tstart = hpc_gettime();
    for (int i=0; i<nrotations; i++) {
        rotate_shared<<< grid, block >>>( d_orig, d_new, n );
        cudaCheckError();
        tmp = d_orig; d_orig = d_new; d_new = tmp;
    }
    cudaDeviceSynchronize();
    const double elapsed_shmem = hpc_gettime() - tstart;
    fprintf(stderr, "Shmem   : elapsed time %f s = %.2f Mpixels/s\n", elapsed_shmem,
            Mpixels/elapsed_shmem);

    /* Copy output to host */
    cudaMemcpy(img.bmap, d_orig, size, cudaMemcpyDeviceToHost);   
    write_pgm(stdout, &img);

    free_pgm(&img);
    cudaFree(d_orig);
    cudaFree(d_new);
    return EXIT_SUCCESS;
}
