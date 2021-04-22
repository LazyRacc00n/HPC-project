//WEWE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> //ONLY UNIX SYSTEM TODO: uncomment on the cluster

#define ALIVE 1
#define DEAD 0


// Allocate a matrix so as to have elements contiguos in memory
unsigned int ** allocate_empty_grid(int rows, int cols)
{

	int i;
	//allocate memory for an array of pointers and then allocate memory for every row
	unsigned int *grid = (unsigned int *)malloc(rows*cols* sizeof(unsigned int));
	unsigned int **array = (unsigned int **)malloc(rows*sizeof(unsigned int*));
	for (i = 0; i < rows; i++)
		array[i] = &(grid[cols*i]);

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

	//TODO: to verify PERCHé + nRows???????????????????????????????????????????????????????????
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
	- A grid is composed by block, each block is composed by threads. All threads in same block have same block index.
	- to esure that  the extra threads do not do any work --> if(row<width && col<width) { --> written in the kernel
																then do work
															  }
*/
__global__ void cuda_evolve(unsigned int *curr_grid, unsigned int *next_grid, int nRows, int nCols){

	const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    

	// TODO: QUEL BLOCK_DIM in matmul è la dimensione della matrice o il numero di thread?
	// da verificare ma penso sia la seconda.
	const int i = bx * nCols + tx;
    const int j = by * nRows + ty;

	if( i < nRows && j < nCols){

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
		nAliveNeig = curr_grid[top_left] + curr_grid[left] + curr_grid[bottom_left] \
					+	curr_grid[top] + curr_grid[top_right] + curr_grid[right] 	\ 
					+ 	curr_grid[bottom_right] + curr_grid[bottom];
		
		// store computation in next_grid
		next_grid[ i * nCols + j] = ( nAliveNeig == 3 || (nAliveNeig == 2 && curr_grid[ i * nCols + j]));
		
	}


	

}


void game(int nRows, int nCols, int timestep ){

	int t=0;
	struct timeval start, end;
	double tot_time = 0.;

	//TODO: allocation in CPU
	unsigned int **curr_grid = allocate_empty_grid(nRows, nCols) , **next_grid = allocate_empty_grid(nRows, nCols); 

	//TODO: allocation in GPU
	size_t grid_size = nRows * nCols * sizeof(unsigned int);

	unsigned int * cuda_curr_grid, cuda_next_grid;

	//TODO: calculate how many block and how many thread per block
	
	//dim3 block(nRows, nRows);
    //dim3 grid((N+BLKDIM-1)/BLKDIM, (N+BLKDIM-1)/BLKDIM);
	
	//TODO: curr grid initialization ( possibility to do it also with cuda? )

	//TODO: copy in from HOST to DEVICE --> cudaMemcpy( dest, src, cudaMemcpyHostToDevice)


	for(t=0; t < timestep; t++){
			
			//TODO: MISSING STUFF
			// cuda_envolve << nThreadPerBlock, nBlock >> ()

			//TODO: cudaDeviceSynchronize()

			//TODO: swap cur_grid and next_grid

			
		
	}






	//TODO: free GPU memory
	//TODO: free CPU memory
}


 

/* OLD GAME
 
void game(int w, int h, int t) {
	int x,y,z;
	unsigned univ[h][w];
	//struct timeval start, end; 
	
	//initialization
	//srand(10);
	for (x = 0; x < w; x++) for (y = 0; y < h; y++) univ[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;
	
	if (x > 1000) printbig(univ, w, h,0);
	
	for(z = 0; z < t;z++) {
		if (x <= 1000) show(univ, w, h);
		//else gettimeofday(&start, NULL);
		
		evolve(univ, w, h);
		if (x > 1000) {
			gettimeofday(&end, NULL);
		    printf("Iteration %d is : %ld ms\n", z,
		       ((end.tv_sec * 1000000 + end.tv_usec) - 
		       (start.tv_sec * 1000000 + start.tv_usec))/1000 );
		}
	}
	if (x > 1000) printbig(univ, w, h,1);
}
 */
 
 
int main(int c, char **v) {
	int w = 0, h = 0, t = 0;
	if (c > 1) w = atoi(v[1]);
	if (c > 2) h = atoi(v[2]);
	if (c > 3) t = atoi(v[3]);
	if (w <= 0) w = 30;
	if (h <= 0) h = 30;
	if (t <= 0) t = 100;


	
	//game(w, h, t);
}

