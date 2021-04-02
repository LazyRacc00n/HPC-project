/****************************************************************************
 *
 * cuda-rotate.cu - Rotate N points using constant memory
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
 * Compile with:
 * nvcc cuda-rotate.cu -o cuda-rotate -lm
 *
 * Run with:
 * ./cuda-rotate
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PLANE_ANG 360
#define BLKDIM 1024

__constant__ float c_sin[PLANE_ANG];
__constant__ float c_cos[PLANE_ANG];

/* Rotate all points of coords (px[i], py[i]) through an angle |angle|
   counterclockwise around the origin. 0 <= |angle| <= 359 */
__global__ void rotate_kernel(float* px, float *py, int n, int angle)  
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    angle = angle % PLANE_ANG; /* ensures 0 <= angle <= 359 */
    if (index < n ) {
        /* compute coordinates (prx, pry) of the rotated point (px[i], py[i]) */
        float prx = px[index] * c_cos[angle] - py[index]*c_sin[angle];
        float pry = py[index] * c_cos[angle] + px[index]*c_sin[angle];
        px[index] = prx;
        py[index] = pry;
    }
}

int main(int argc, char* argv[])
{
    float sin_table[PLANE_ANG], cos_table[PLANE_ANG];
    float *h_px, *h_py; /* coordinates in host memory */
    float *d_px, *d_py; /* coordinates in device memory */
    int i, a;
    const size_t NPOINTS = 1024*1024;
    const int ANGLE = 72;

    /* pre-compute the table of sin and cos; note that sin() and cos()
       expect the angle to be in radians */
    for (a=0; a<PLANE_ANG; a++) {
        sin_table[a] = sin(a * M_PI / 180.0f );
        cos_table[a] = cos(a * M_PI / 180.0f );
    }

    /* Ship the pre-computed tables to constant memory */
    cudaMemcpyToSymbol(c_sin, sin_table, sizeof(sin_table));
    cudaMemcpyToSymbol(c_cos, cos_table, sizeof(cos_table));

    const size_t size = NPOINTS * sizeof(*h_px);

    /* Create NPOINTS instances of the point (1,0) */
    h_px = (float*)malloc( size );
    h_py = (float*)malloc( size );
    for (i=0; i<NPOINTS; i++) {
        h_px[i] = 1.0f;
        h_py[i] = 0.0f;
    }

    /* Copy the points to device memory */
    cudaMalloc((void**)&d_px, size);
    cudaMalloc((void**)&d_py, size);
    cudaMemcpy(d_px, h_px, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, size, cudaMemcpyHostToDevice);

    /* Rotate all points by 72 degrees counterclockwise on the GPU */
    rotate_kernel<<< (NPOINTS + BLKDIM-1)/BLKDIM, BLKDIM >>>(d_px, d_py, NPOINTS, ANGLE);

    /* Copy result back to host memory */
    cudaMemcpy(h_px, d_px, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_py, d_py, size, cudaMemcpyDeviceToHost);

    /* Check results */
    for (i=0; i<NPOINTS; i++) {
        if ( fabs(h_px[i] - cos_table[ANGLE]) > 1e-5 ||
             fabs(h_py[i] - sin_table[ANGLE]) > 1e-5 ) {
            fprintf(stderr, "Test failed: (h_px[%d], h_py[%d]) expected (%f, %f) but got (%f, %f)\n",
                    i, i, cos_table[ANGLE], sin_table[ANGLE], h_px[i], h_py[i]);
            break;
        }
    }
    if ( i == NPOINTS ) {
        printf("Check OK\n");
    }

    /* free memory */
    free(h_px);
    free(h_py);
    cudaFree(d_px);
    cudaFree(d_py);
    return EXIT_SUCCESS;
}


