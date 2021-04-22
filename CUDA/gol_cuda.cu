//WEWE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> //ONLY UNIX SYSTEM TODO: uncomment on the cluster

#define ALIVE 1
#define DEAD 0

void free_gen(unsigned int **gen){

	free(gen[0]);
	free(gen);

}

void swap(unsigned int ***old, unsigned int ***new) {
	
    unsigned int **temp = *old;

    *old = *new;
    *new = temp;

}



// Allocate a matrix so as to have elements contiguos in memory
unsigned int ** allocate_empty_gen(int rows, int cols)
{

	int i;
	//allocate memory for an array of pointers and then allocate memory for every row
	unsigned int *gen = (unsigned int *)malloc(rows*cols* sizeof(unsigned int));
	unsigned int **array = (unsigned int **)malloc(rows*sizeof(unsigned int*));
	for (i = 0; i < rows; i++)
		array[i] = &(gen[cols*i]);

	return array;
}

//TODO: TO MODIFY
void show(void *u, int w, int h) {
	int x,y;
	int (*univ)[w] = u;
	printf("\033[H");
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
	//usleep(200000);
}


//TODO: TO MODIFY
void printbig(void *u, int w, int h, int z) {
	int x,y;
	int (*univ)[w] = u;
	
	FILE *f;
	
	if(z == 0) f = fopen("glife.txt", "w" );
	else f = fopen("glife.txt", "a" );
	
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) fprintf (f,"%c", univ[y][x] ? 'x' : ' ');
		fprintf(f,"\n");
	}
	fprintf(f,"\n\n\n\n\n\n ******************************************************************************************** \n\n\n\n\n\n");
	fflush(f);
	fclose(f);
}


// compute neighbors in around 3x3
int compute_neighbor(int i, int j, int nRows, int nCols){

	// Guarda come vengono gestiti i bordi nell'originale	
	int x = (i + nRows) % nRows;
	int y = (j + nCols) % nCols;
	return  x * nCols + y;
}


//int tid = threadIdx.x + blockIdx.x * blockDim.x;
// number of threds: ( N + number_thred_per_block) / number_thred_per_block

//MEMORY COALESCED ACCESS --> improve performaze taking per rows

/*A 2D matrix is stored as 1D in memory:
	- in row-major layout, the element(x,y) ca be adressed as x*width+ y
	- A gen is composed by block, each block is composed by threads. All threads in same block have same block index.
	- to esure that  the extra threads do not do any work --> if(row<width && col<width) { --> written in the kernel
																then do work
															  }
*/
__global__ void cuda_evolve(unsigned int *curr_gen, unsigned int *next_gen, int nRows, int nCols, int block_size){

	const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    
	//TODO: capire se è da usaere blockDim.y o va bene 
	const int i = by * blockDim.y + ty; 
    const int j = bx * blockDim.x + tx;

	//to esure that  the extra threads do not do any work
	if( i < nRows && j < nCols) return;

		// Envolve computation
		// TODO: count how many neighbors are alive
		int nAliveNeig = 0;

		// index --> i * nCols + j
		
		//calculate the neighbors OH MADONNA GIà è SOLO PI§ disordinato xdxdxdxd
		int top_left =    compute_neighbor(i-1, j-1, nRows, nCols);
		int left = 		  compute_neighbor(i, j-1, nRows, nCols);
		int bottom_left = compute_neighbor(i+1, j-1, nRows, nCols);
		int top = 		  compute_neighbor(i-1, j, nRows, nCols);
		int top_right =   compute_neighbor(i-1, j+1, nRows, nCols);
		int right =       compute_neighbor(i, j+1, nRows, nCols);
		int bottom_right= compute_neighbor(i+1, j+1, nRows, nCols);
		int bottom =      compute_neighbor(i+1, j, nRows, nCols);

		//calculate how many neighbors around 3x3 are alive
		nAliveNeig = curr_gen[top_left] + curr_gen[left] + curr_gen[bottom_left] \
					+	curr_gen[top] + curr_gen[top_right] + curr_gen[right] 	\ 
					+ 	curr_gen[bottom_right] + curr_gen[bottom];
		
		// store computation in next_gen
		next_gen[ i * nCols + j] = ( nAliveNeig == 3 || (nAliveNeig == 2 && curr_gen[ i * nCols + j]));
		

}





void game(int nRows, int nCols, int timestep, int block_size ){

	int z, x, y;
	struct timeval start, end;
	double tot_time = 0.;

	//TODO: allocation in CPU and initialization
	unsigned int ** curr_gen = allocate_empty_gen(nRows, nCols);
	unsigned int ** next_gen = allocate_empty_gen(nRows, nCols); 
	
	//srand(10);
	for (x = 0; x < w; x++) for (y = 0; y < h; y++) curr_gen[y][x] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;

	//TODO: allocation in GPU
	size_t gen_size = nRows * nCols * sizeof(unsigned int);

	unsigned int * cuda_curr_gen;
	unsigned int *cuda_next_gen;

	cudaMalloc((void ** ) &cuda_curr_gen, gen_size );
	cudaMalloc((void ** ) &cuda_next_gen, gen_size );

	// copy matrix from the host (CPU) to the device (GPU)
	cudaMemcpy(cuda_curr_gen, curr_gen, gen_size, cudaMemcpyHostToDevice);
	//cudaMemset(cuda_next_gen, DEAD, gen_size); inutile secondo me TODO: vedere se eliminare sta riga

	//calculate how many block and how many thread per block
	
	dim3 block(block_size, block_size), dimGrid;
    dimGrid.x = ( nCols + block.x - 1)/block.x;
    dimGrid.y = ( nRows + block.y - 1)/block.y;
	



	for(z=0; z < timestep; z++){
			
			// get starting time at iteration z
			gettimeofday(&start, NULL);

			// Call Kernel on GPU
			cuda_envolve << dimGrid, block >> (cuda_curr_gen, cuda_next_gen, nRows, nCols);

			cudaDeviceSynchronize(); 

			//swap cur_gen and next_gen
			swap(&cuda_curr_gen, &cuda_next_gen);

			// get ending time of iteration z
			gettimeofday(&end, NULL);
		
			// sum up the total time execution
			tot_time += (double) elapsed_wtime(start, end);
			
		
	}

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

