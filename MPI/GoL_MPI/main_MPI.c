/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#include "gameOfLife_MPI.h"



//------------------------------------ OLD CODE---------------------------------------------------------------------------------------

void show(void *u, int w, int h){
	int x, y;
	int(*univ)[w] = u;
	printf("\033[H");
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
			printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
	usleep(200000);
}

void printbig(void *u, int w, int h, int z)
{
	int x, y;
	int(*univ)[w] = u;

	FILE *f;

	if (z == 0)
		f = fopen("glife.txt", "w");
	else
		f = fopen("glife.txt", "a");

	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
			fprintf(f, "%c", univ[y][x] ? 'x' : ' ');
		fprintf(f, "\n");
	}
	fprintf(f, "\n\n\n\n\n\n ******************************************************************************************** \n\n\n\n\n\n");
	fflush(f);
	fclose(f);
}

void evolve(void *u, int w, int h)
{
	unsigned(*univ)[w] = u;
	unsigned new[h][w];
	int x, y, x1, y1;

	for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
		{
			int n = 0;
			for (y1 = y - 1; y1 <= y + 1; y1++)
				for (x1 = x - 1; x1 <= x + 1; x1++)
					if (univ[(y1 + h) % h][(x1 + w) % w])
						n++;
			if (univ[y][x])
				n--;
			new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
			/*
		 * a cell is born, if it has exactly three neighbours 
		 * a cell dies of loneliness, if it has less than two neighbours 
		 * a cell dies of overcrowding, if it has more than three neighbours 
		 * a cell survives to the next generation, if it does not die of loneliness 
		 * or overcrowding 
		 */
		}
	for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
			univ[y][x] = new[y][x];
}

void game(int w, int h, int t)
{
	int x, y, z;
	unsigned univ[h][w];
	struct timeval start, end;

	//initialization
	for (x = 0; x < w; x++)
		for (y = 0; y < h; y++)
			univ[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;

	if (x > 1000)
		printbig(univ, w, h, 0);

	for (z = 0; z < t; z++)
	{
		if (x <= 1000)
			show(univ, w, h);
		else
			gettimeofday(&start, NULL);

		evolve(univ, w, h);
		if (x > 1000)
		{
			gettimeofday(&end, NULL);
			printf("Iteration %d is : %ld ms\n", z,
				   ((end.tv_sec * 1000000 + end.tv_usec) -
					(start.tv_sec * 1000000 + start.tv_usec)) /
					   1000);
		}
	}
	if (x > 1000)
		printbig(univ, w, h, 1);
}



//------------------------------------ NEW CODE    ---------------------------------------------------------------------------------------



//function to allocate a block in a node and initialize the field of the struct it
void init_and_allocate_block(struct grid_block *gridBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour){
	
	int i;
	
	// initialize field of the struct that represent the block
	gridBlock -> numRows_ghost = nRows_with_ghost;
	gridBlock -> numCols_ghost = nCols_with_Ghost;
	gridBlock -> upper_neighbour = upper_neighbour;
	gridBlock -> lower_neighbour = lower_neighbour;
	
	
	//allocate memory for an array of pointers and then allocate memory for every row
	gridBlock->block = (unsigned int **)malloc( gridBlock->numRows_ghost * sizeof(unsigned int *));
	for(i=0; i < gridBlock->numRows_ghost; i++)
		gridBlock->block[i] = (unsigned int *) malloc(gridBlock->numCols_ghost * sizeof(unsigned int));
}


//function for initialize blocks
void init_grid_block(struct grid_block *gridBlock)
{
	int i, j;

	for (i = 1; i < gridBlock -> numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock-> numCols_ghost - 1; j++)
			gridBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}


//TODO: evolution of game of a block, and manage neighbours
void evolve_block(void *b, int nRows_local_with_ghost, int nCols_with_ghost, int time)
{
	int(*block)[nCols_with_ghost] = b;

	int t;

	for( t=0; t < time; t++ ){

		break;

	}

	return;

}

void game_block()
{

	//TODO: to be implemented yet
	return;
}

//TODO: Ghost Rows: In order to compute the envolve we need to send the first row (ghost) to the upper neighbor and the last
//TODO: row to the lower neighbour. (Try to use dataype to send and check if it improve performance)
//TODO: Ghost Columns: Copy fisrt column to the last ghost columns

// obtain the upper neighbour
int get_upper_neighbour(int size, int rank)
{

	return (rank == 0) ? size - 1 : rank - 1;
}

int get_lower_neighbour(int size, int rank)
{

	return (rank == size - 1) ? 0 : rank + 1;
}

int main(int argc, char **argv){

	int rank, size, err, MPI_root = 0;

	err = MPI_Init(&argc, &argv);

	if (err != 0){

		printf("\nError in MPI initialization!\n");
		MPI_Abort(MPI_COMM_WORLD, err);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int nCols = 0, nRows = 0, time = 0;
	if (argc > 1)
		nCols = atoi(argv[1]);
	if (argc > 2)
		nRows = atoi(argv[2]);
	if (argc > 3)
		time = atoi(argv[3]);
	if (nCols <= 0)
		nCols = 30;
	if (nRows <= 0)
		nRows = 30;
	if (time <= 0)
		time = 100;

	//send number of columns and number of rows to each process

	// send rows
	MPI_Bcast(&nRows, 1, MPI_INT, MPI_root, MPI_COMM_WORLD);
	MPI_Bcast(&nCols, 1, MPI_INT, MPI_root, MPI_COMM_WORLD);
	MPI_Bcast(&time, 1, MPI_INT, MPI_root, MPI_COMM_WORLD);

	//Each process compute the size of its chunks
	int n_rows_local = nRows / size;
	//if the division has remains are added to the last process;
	if (rank == size - 1)
		n_rows_local += nRows % size;

	// ghost colums are which that communicate with neighbours. Are a sort of
	// recever buffer
	int n_rows_local_with_ghost = n_rows_local + 2;
	int n_cols_with_ghost = nCols + 2;

	//printf("\nRank: %d - Rows local: %d - Cols: %d\n", rank, n_rows_local, nCols);

	int upper_neighbour = get_upper_neighbour(size, rank);
	int lower_neighbour = get_lower_neighbour(size, rank);

	//printf("\nRank: %d --> Upper Neighbour: %d\n",rank, upper_neighbour);
	//printf("\nRank: %d --> Lower Neighbour: %d\n",rank, lower_neighbour);

	
	struct grid_block blockGrid;
	struct grid_block nextBlockGrid;

	init_and_allocate_block(&blockGrid, n_rows_local_with_ghost, n_cols_with_ghost, upper_neighbour, lower_neighbour);

	// each node initialiaze own block randomly
	init_grid_block(&blockGrid);


	// print a block to test
	int i, j;
	if (rank == 1)
	{
		
		printf("\n\nBLOCKS DIMS RANK %d WITH GHOST: %d x %d \n\n", rank, n_rows_local_with_ghost, n_cols_with_ghost );
		for (i = 1; i < n_rows_local_with_ghost - 1; i++)
		{
			for (j = 1; j < n_cols_with_ghost - 1; j++)
				printf("%d ", blockGrid.block[i][j]);

			printf("\n");
		}
	}
	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}
