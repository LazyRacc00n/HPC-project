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



/*
* a cell is born, if it has exactly three neighbours
* a cell dies of loneliness, if it has less than two neighbours
* a cell dies of overcrowding, if it has more than three neighbours
* a cell survives to the next generation, if it does not die of loneliness
* or overcrowding
*/
__global__ void cuda_evolve(unsigned int *curr_gen, unsigned int *next_gen, int nRows, int nCols, int block_size){

        const int game_size = nRows * nCols;

        const int bx = blockIdx.x;
        const int tx = threadIdx.x;

        const int idx = bx * blockDim.x + tx;

        //to esure that  the extra threads do not do any work
        if( idx >= game_size ) return;

        int nAliveNeig = 0;

        // the column x
        int x = idx % nCols;

        // the row y: the yth element in the flatten array
        int y = idx - x;

        //compute the neighbors indexes starting from x and y
        int xLeft = ( x + nCols-1) %nCols;
        int xRight = ( x + 1) %nCols;

        int yTop = (y + game_size - nCols) % game_size;
        int yBottom = (y + nCols) % game_size;

        nAliveNeig = curr_gen[ xLeft + yTop] + curr_gen[x + yTop] + curr_gen[xRight + yTop]
                     +  curr_gen[xLeft + y]  + curr_gen[xRight + y]
                     + curr_gen[xLeft + yBottom] + curr_gen[x + yBottom] + curr_gen[xRight + yBottom];

        // store computation in next_gen
        next_gen[ x + y] = ( nAliveNeig == 3 || (nAliveNeig == 2 && curr_gen[x + y]));

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
        dim3 n_threads(block_size);

        // how many blocks from the grid dim, distribute the game board evenly
        dim3 n_blocks( (int) (nRows * nCols + n_threads.x -1) / n_threads.x );


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
        sprintf(fileName, "Results/CUDA-%d-%d-%d.csv", nCols, nRows, timestep);

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
