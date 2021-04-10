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


//function to allocate a block in a node and initialize the field of the struct it
void init_and_allocate_block(struct grid_block *gridBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour)
{

	int i;

	// initialize field of the struct that represent the block assigned to a node
	gridBlock->numRows_ghost = nRows_with_ghost;
	gridBlock->numCols_ghost = nCols_with_Ghost;
	gridBlock->upper_neighbour = upper_neighbour;
	gridBlock->lower_neighbour = lower_neighbour;

	//allocate memory for an array of pointers and then allocate memory for every row
	gridBlock->block = (unsigned int **)malloc(gridBlock->numRows_ghost * sizeof(unsigned int *));
	for (i = 0; i < gridBlock->numRows_ghost; i++)
		gridBlock->block[i] = (unsigned int *)malloc(gridBlock->numCols_ghost * sizeof(unsigned int));
}

//function for initialize blocks
void init_grid_block(struct grid_block *gridBlock)
{	
	srand(25);

	int i, j;

	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
			gridBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}

//TODO: evolution of game of a block, and manage neighbours
void evolve_block(struct grid_block *gridBlock,  int time)
{	
	int i,j,t;
	//TODO: see difference of performance in seding using derived datatype and without
	// create a derived datatyper to send a row
	MPI_Datatype row_block_type;
	MPI_Type_contiguous( gridBlock->numCols_ghost , MPI_UNSIGNED , &row_block_type);
	MPI_Type_commit(&row_block_type);

	MPI_Status stat;


	for (t = 0; t < time; t++){

		// send first row of the block to the upper neighbour
		MPI_Send( &gridBlock->block[1][0], 1 , row_block_type , gridBlock->upper_neighbour, 0 , MPI_COMM_WORLD);
		// send last row of the block to the lower neighbour
		MPI_Send( &gridBlock->block[gridBlock->numRows_ghost - 2 ][0], 1 , row_block_type , gridBlock->lower_neighbour, 0 , MPI_COMM_WORLD);

		// receive from below using  buffer the ghost row as receiver
		MPI_Recv( &gridBlock->block[gridBlock->numRows_ghost - 1 ][0], gridBlock->numCols_ghost, MPI_UNSIGNED, gridBlock->lower_neighbour , 0 , MPI_COMM_WORLD, &stat);
		
		// receive from top using  the ghost row as receiver buffer
		MPI_Recv( &gridBlock->block[0][0], gridBlock->numCols_ghost, MPI_UNSIGNED, gridBlock->upper_neighbour , 0 , MPI_COMM_WORLD, &stat);


		//ghost colums: 
		// 		-copy last column to the fisrt column
		// 		-copy the fisrt column to the last last column

		for (i = 0; i < gridBlock->numRows_ghost; i++){
			gridBlock->block[i][0] = gridBlock->block[i][gridBlock->numCols_ghost-2];
			gridBlock->block[i][gridBlock->numCols_ghost - 1] = gridBlock->block[i][1];
		}


		//Update to current grid to the next grid
		

		
	}
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

int main(int argc, char **argv)
{

	int rank, size, err, MPI_root = 0;

	err = MPI_Init(&argc, &argv);

	if (err != 0)
	{

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

	evolve_block(&blockGrid, 5);
	

	int i, j;
	
	// print a block to test
	
	if (rank == 1)
	{

		printf("\n\nBLOCKS DIMS RANK %d WITH GHOST: %d x %d \n\n", rank, n_rows_local_with_ghost, n_cols_with_ghost);
		for (i = 0; i < n_rows_local_with_ghost; i++)
		{
			for (j = 0; j < n_cols_with_ghost; j++)
				printf("%d ", blockGrid.block[i][j]);

			printf("\n");
		}
	}
	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}
