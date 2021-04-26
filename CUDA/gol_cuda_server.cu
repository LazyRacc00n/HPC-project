
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>


#define ALIVE 1
#define DEAD 0

void free_gen(unsigned int *gen){

        free(gen);
}

void swap(unsigned int **old, unsigned int ** new_) {

    unsigned int *temp = *old;

    *old = *new_;
    *new_ = temp;

}

// Allocate a matrix so as to have elements contiguos in memory
unsigned int * allocate_empty_gen(int rows, int cols)
{

        //allocate memory for an array of pointers and then allocate memory for every row
        unsigned int *gen = (unsigned int *)malloc(rows*cols* sizeof(unsigned int));

        return gen;

}

void show(unsigned int *curr_gen, int nRows, int nCols) {

        int i,j;

        printf("\033[H");
        for (i = 0; i < nRows; i++) {
                for (j = 0; j < nCols; j++) printf(curr_gen[i* nCols + j] ? "\033[07m  \033[m" : "  ");
                printf("\033[E");
        }
        fflush(stdout);
        usleep(200000);
}


void printbig(unsigned int *curr_gen, int nRows, int nCols, int z) {

        int i,j;

        FILE *f;

        if(z == 0) f = fopen("glife_cuda.txt", "w" );
        else f = fopen("glife_cuda.txt", "a" );

        for (i = 0; i < nRows; i++) {
                for (j = 0; j < nCols; j++) fprintf (f,"%c", curr_gen[i* nCols + j] ? 'x' : ' ');
                fprintf(f,"\n");
        }

        //separate fisrt evolution from last
        if( z == 0)
                fprintf(f,"\n\n\n\n\n\n ******************************************************************************************** \n\n\n\n\n\n");

        fflush(f);
        fclose(f);
}

// compute the elapsed wall-clock time between two time intervals. in ms
double elapsed_wtime(struct timeval start, struct timeval end) {

    return (double)((end.tv_sec * 1000000 + end.tv_usec) -
                       (start.tv_sec * 1000000 + start.tv_usec))/1000;


}


void writeFile(char* fileName, int w, int h, int z, bool first, double time , int n_core){
    FILE *f;


    if(first)   f = fopen(fileName, "w" );
    else f = fopen(fileName, "a" );

    if(first) fprintf(f,"%d-%d-%d,\n",w , h, z);

    // write file
    fprintf(f,"%d,%f",n_core , time);

    fprintf(f,"\n");
        fflush(f);
        fclose(f);

}

__device__ int compute_neighbor(int i, int j, int nRows, int nCols){

        // Guarda come vengono gestiti i bordi nell'originale
        int x = i % nRows;
        int y = j % nCols;
        return  x * nCols + y;
}


//int tid = threadIdx.x + blockIdx.x * blockDim.x;
// number of threds: ( N + number_thred_per_block) / number_thred_per_block

//MEMORY COALESCED ACCESS --> improve performace taking per rows

/*A 2D matrix is stored as 1D in memory:
        - in row-major layout, the element(x,y) ca be adressed as x*width+ y
        - A gen is composed by block, each block is composed by threads. All threads in same block have same block index.
        - to esure that  the extra threads do not do any work --> if(row<width && col<width) { --> written in the kernel
                                                                                                                                then do work
                                                                                                                          }
*/

/*
* a cell is born, if it has exactly three neighbours
* a cell dies of loneliness, if it has less than two neighbours
* a cell dies of overcrowding, if it has more than three neighbours
* a cell survives to the next generation, if it does not die of loneliness
* or overcrowding
*/
__global__ void cuda_evolve(unsigned int *curr_gen, unsigned int *next_gen, int nRows, int nCols, int block_size){


        const int bx = blockIdx.x, by = blockIdx.y;
        const int tx = threadIdx.x, ty = threadIdx.y;

        const int i = by * blockDim.y + ty;
        const int j = bx * blockDim.x + tx;

        //to esure that  the extra threads do not do any work
        if( !( i < nRows && j < nCols) ) return;

        int nAliveNeig = 0;

        // index --> i * nCols + j

        //compute the neighbors indexes
        int top_left =    compute_neighbor(i-1, j-1, nRows, nCols);
        int left =        compute_neighbor(i, j-1, nRows, nCols);
        int bottom_left = compute_neighbor(i+1, j-1, nRows, nCols);
        int top =         compute_neighbor(i-1, j, nRows, nCols);
        int top_right =   compute_neighbor(i-1, j+1, nRows, nCols);
        int right =       compute_neighbor(i, j+1, nRows, nCols);
        int bottom_right= compute_neighbor(i+1, j+1, nRows, nCols);
        int bottom =      compute_neighbor(i+1, j, nRows, nCols);

        //calculate how many neighbors around 3x3 are alive
        nAliveNeig = curr_gen[top_left] + curr_gen[left] + curr_gen[bottom_left]
                     +  curr_gen[top] + curr_gen[top_right] + curr_gen[right]
                     + curr_gen[bottom_right] + curr_gen[bottom];

        // store computation in next_gen
        next_gen[ i * nCols + j] = ( nAliveNeig == 3 || (nAliveNeig == 2 && curr_gen[ i * nCols + j]));

}



void game(int nRows, int nCols, int timestep, int block_size ){

        int z, x, y;
        struct timeval start, end;
        double tot_time = 0.;

        // allocation in CPU and initialization
        unsigned int * curr_gen = allocate_empty_gen(nRows, nCols);
        unsigned int * next_gen = allocate_empty_gen(nRows, nCols);


        //srand(10);
        for (x = 0; x < nRows; x++) for (y = 0; y < nCols; y++) curr_gen[x * nCols + y] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;

        // allocation in GPU
        size_t gen_size = nRows * nCols * sizeof(unsigned int);

        unsigned int *cuda_curr_gen;
        unsigned int *cuda_next_gen;

        cudaMalloc((void ** ) &cuda_curr_gen, gen_size );
        cudaMalloc((void ** ) &cuda_next_gen, gen_size );

        // copy matrix from the host (CPU) to the device (GPU)
        cudaMemcpy(cuda_curr_gen, curr_gen, gen_size, cudaMemcpyHostToDevice);

        // make a 2D grid of threads, with  block_size threads in total.
        int grid_threads = (int) sqrt(block_size);
        dim3 n_threads(grid_threads, grid_threads);

        // how many blocks from the grid dim
        dim3 n_blocks;
        n_blocks.x = ( nCols + n_threads.x - 1)/n_threads.x;
        n_blocks.y = ( nRows + n_threads.y - 1)/n_threads.y;

        if( nCols > 1000 ) printbig(curr_gen, nRows, nCols, 0);

        for(z=0; z < timestep; z++){

                if(nCols <= 1000){
                        cudaMemcpy(curr_gen, cuda_curr_gen, gen_size, cudaMemcpyDeviceToHost);
                        show(curr_gen, nRows, nCols);
                }

                // get starting time at iteration z
                gettimeofday(&start, NULL);


                // Call Kernel on GPU
                cuda_evolve<<<n_blocks, n_threads>>>(cuda_curr_gen, cuda_next_gen, nRows, nCols, block_size);
                cudaDeviceSynchronize();

                //swap cur_gen and next_gen when all the threads are done
                swap(&cuda_curr_gen, &cuda_next_gen);


                // get ending time of iteration z
                gettimeofday(&end, NULL);

                // sum up the total time execution
                tot_time += (double) elapsed_wtime(start, end);

                if (nCols > 1000)
                        printf("Iteration %d is : %f ms\n", z, (double) elapsed_wtime(start, end));

        }


        if( nCols > 1000 ){
                cudaMemcpy(curr_gen, cuda_curr_gen, gen_size, cudaMemcpyDeviceToHost);
                printbig(curr_gen, nRows, nCols, z);
        }

        // Save time execution
        char *fileName = (char*)malloc(50 * sizeof(char));
        sprintf(fileName, "Results/CUDA-%d-%d-%d.txt", nCols, nRows, timestep);

        writeFile(fileName, nCols, nRows, timestep, (block_size==32), tot_time, block_size);
        free(fileName);

        //free GPU memory
        cudaFree(cuda_curr_gen);
        cudaFree(cuda_next_gen);

        //free CPU memory
        free_gen(curr_gen);
        free_gen(next_gen);

}




int main(int c, char **v) {
        int w = 0, h = 0, t = 0, block_size = 32;

        if (c > 1) w = atoi(v[1]);
        if (c > 2) h = atoi(v[2]);
        if (c > 3) t = atoi(v[3]);
        if (c > 4) block_size = atoi(v[4]);

        if (w <= 0) w = 30;
        if (h <= 0) h = 30;
        if (t <= 0) t = 100;
        if (block_size < 32) block_size = 32; // number of threads per block

        game(w, h, t, block_size);
}




















