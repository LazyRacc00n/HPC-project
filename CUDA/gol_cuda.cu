//WEWE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> //ONLY UNIX SYSTEM TODO: uncomment on the cluster

#define ALIVE 1
#define DEAD 0







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



//int tid = threadIdx.x + blockIdx.x * blockDim.x;
// number of threds: ( N + number_thred_per_block) / number_thred_per_block

//MEMORY COALESCED ACCESS --> improve performaze taking per rows

/*A 2D matrix is stored as 1D in memory:
	- in row-major layout, the element(x,y) ca be adressed as x*width + y
	- A grid is composed by block, each block is composed by threads. All threads in same block have same block index.
	- to esure that  the extra threads do not do any work --> if(row<width && col<width) { --> written in the kernel
																then do work
															  }
*/
__global__ void cuda_evolve(unsigned int *curr_grid, unsigned int *next_grid, int nRows, int nCols){

	const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    
    const int i = by * nRows + ty;
    const int j = bx * nCols + tx;

	if( row < nRows && col < nCols){

		// Envolve computation
		// TODO: count how many neighbors are alive

		
		// TODO: store computation in next_grid
	
	
	}


	//TODO: swap cur and next grid

}


void game(int nRows, int nCols, int timestep ){

	int t=0;
	unsigned int *curr_grid, *next_grid;

	//TODO: allocation in CPU

	//TODO: allocation in GPU
	
	//TODO: curr grid initialization ( possibility to do it also with cuda? )

	//TODO: copy in from HOST to DEVICE


	for(t=0; t < timestep; t++){
			
			//TODO: MISSING STUFF
			// cuda_envolve << nThreadPerBlock, nBlock >> ()
		
	}






	//TODO: free memory GPU
	//TODO:
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

