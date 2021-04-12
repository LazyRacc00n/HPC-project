/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 *
 
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "gameOfLife_MPI.h"


void allocate_empty_grid(unsigned int ***grid, int rows, int cols){

	int i,j;
	//allocate memory for an array of pointers and then allocate memory for every row
	*grid = (unsigned int **)malloc(rows * sizeof(unsigned int *));
	for (i = 0; i <  rows; i++)
		(*grid)[i] = (unsigned int *)malloc(cols * sizeof(unsigned int));
			//for(j=0; j < cols; )
}


//function to allocate a block in a node and initialize the field of the struct it
void init_and_allocate_block(struct grid_block *gridBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour, int rank, int size)
{

	int i;

	// initialize field of the struct that represent the block assigned to a node
	gridBlock->numRows_ghost = nRows_with_ghost;
	gridBlock->numCols_ghost = nCols_with_Ghost;
	gridBlock->upper_neighbour = upper_neighbour;
	gridBlock->lower_neighbour = lower_neighbour;
	gridBlock->rank = rank;
	gridBlock->mpi_size = size;


	//allocate memory for an array of pointers and then allocate memory for every row
	gridBlock->block = (unsigned int **)malloc(gridBlock->numRows_ghost * sizeof(unsigned int *));
	for (i = 0; i < gridBlock->numRows_ghost; i++)
		gridBlock->block[i] = (unsigned int *)malloc(gridBlock->numCols_ghost * sizeof(unsigned int));
}

//function for initialize blocks
void init_grid_block(struct grid_block *gridBlock)
{	
	srand(time(NULL));
	int i, j;

	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
			gridBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}



// Ghost Rows: In order to compute the envolve we need to send the first row (ghost) to the upper neighbor and the last
// row to the lower neighbour. (Try to use dataype to send and check if it improve performance)
// Ghost Columns: Copy fisrt column to the last ghost columns

//TODO: evolution of game of a block, and manage neighbours
void evolve_block(struct grid_block *gridBlock, unsigned int **next_gridBlock, int nRows, int nCols){

	int i,j,t, x, y;
	//TODO: see difference of performance in seding using derived datatype and without
	// create a derived datatyper to send a row
	MPI_Datatype row_block_type, row_block_without_ghost;
	
	//TODO: Try then if improve performance allocating this two type out of the time cycle
	MPI_Type_contiguous( gridBlock->numCols_ghost , MPI_UNSIGNED , &row_block_type);
	MPI_Type_contiguous( gridBlock->numCols_ghost-2 , MPI_UNSIGNED , &row_block_without_ghost);
	
	MPI_Type_commit(&row_block_type);
	MPI_Type_commit(&row_block_without_ghost);

	MPI_Status stat;

	
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


	//TODO: Display current grid:
		
		//send data to the root, if I'm not the root
		if( gridBlock->rank != MPI_root){
			//send all rows
			for( i=1; i < gridBlock->numRows_ghost - 1; i++ )
				MPI_Send(&gridBlock->block[i][1], 1 , row_block_without_ghost , MPI_root , 0 , MPI_COMM_WORLD);
		}
		
		
		if(gridBlock->rank == MPI_root){//if I'm the root: print and receive

			//print current grid
			for( i=1; i < gridBlock->numRows_ghost - 1; i++ ){
				for( j=1; j < gridBlock->numCols_ghost - 1; j++ )
					printf("%d ", gridBlock->block[i][j] );
				printf("\n");
			}

			int src, rec_idx;
			//Receive form other nodes ( excluding the root, 0 )
			for(src=1; src < gridBlock-> mpi_size; src++ ){

				// I need know how much rows the root must receive, are different for some node

				// TODO: Try a better solution: every node kowns the dimentions of the blocks of the other nodes

				//For now, I can compute the number of rows of each node

				int nRows_rec = nRows / gridBlock -> mpi_size;

				if( src == gridBlock -> mpi_size - 1) nRows_rec += nRows % gridBlock -> mpi_size;

				int buffer[nCols];

				for( rec_idx = 0; rec_idx < nRows_rec ; rec_idx++)
					MPI_Recv( &buffer[0], nCols, MPI_UNSIGNED , src , 0 , MPI_COMM_WORLD, &stat);
			}
		
		}

	

	//Update to current grid to the next grid
	for (i = 1; i < gridBlock->numRows_ghost - 1; i++){
		for (j = 1; j < gridBlock->numRows_ghost - 1; j++){
				
			int alive_neighbours = 0;

			for ( x = i-1 ; x <= i+1; x++)
				for( y = j-1; y <= j+1; y++)
					if( (i != x || j != y) && gridBlock->block[i][j] == ALIVE ) ++alive_neighbours;
				
				
			if( alive_neighbours < 2 ) next_gridBlock[i][j] = DEAD;

			if( gridBlock->block[i][j] == ALIVE && (alive_neighbours == 2 || alive_neighbours == 3)) next_gridBlock[i][j] = ALIVE;

			if( alive_neighbours > 3) next_gridBlock[i][j] = DEAD;

			if( gridBlock->block[i][j] == DEAD && (alive_neighbours == 3)) next_gridBlock[i][j] = ALIVE;

		}
	}
		

	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numRows_ghost - 1; j++)
			gridBlock->block[i][j] = next_gridBlock[i][j];
	
	MPI_Type_free (&row_block_type);
	MPI_Type_free (&row_block_without_ghost);

}


// call envolve and diaplay the evolution
void game_block(struct grid_block *gridBlock, int time, int nRows, int nCols)
{	
	int i,j;
	//allocate the next grid used to compute the evolution of the next time step
	unsigned int **next_gridBlock;

	//Random Initialization of the grid assigned to each node
	init_grid_block(gridBlock);
	allocate_empty_grid(&next_gridBlock, gridBlock->numRows_ghost, gridBlock->numCols_ghost);
	
	
	for (int t = 0; t < time; t++){

			evolve_block(gridBlock, next_gridBlock, nRows, nCols);
		}
	
}



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

	int rank, size, err;

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
	

	init_and_allocate_block(&blockGrid, n_rows_local_with_ghost, n_cols_with_ghost, upper_neighbour, lower_neighbour, rank, size);
	

	game_block(&blockGrid, time, nRows, nCols);
	

	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}
