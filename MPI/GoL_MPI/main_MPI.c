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

void allocate_empty_grid(unsigned int ***grid, int rows, int cols)
{

	int i, j;
	//allocate memory for an array of pointers and then allocate memory for every row
	*grid = (unsigned int **)malloc(rows * sizeof(unsigned int *));
	for (i = 0; i < rows; i++)
		(*grid)[i] = (unsigned int *)malloc(cols * sizeof(unsigned int));
}




void print_block(struct grid_block* gridBlock) {
    int x, y;

    // \033[H -> move cursor to top-left corner;
    // \033[J -> clear the console.
    printf("\033[H\033[J");
    for (x = 1; x < gridBlock->numRows_ghost - 1 ; x++) {
        for (y = 1; y < gridBlock->numCols_ghost - 1; y++)
            printf(gridBlock->block[x][y] == ALIVE ? "\033[07m  \033[m" : "  ");

        printf("\033[E");
    
	}
}



void print_received_row(int buffer[], int numCols){

	int x;
	printf("\033[H\033[J");
	for (x = 0; x < gridBlock->numRows_ghost; x++)
	
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
	gridBlock->remained_rows = ( nRows_with_ghost - 2 ) % size;


	//allocate memory for an array of pointers and then allocate memory for every row
	gridBlock->block = (unsigned int **)malloc(gridBlock->numRows_ghost * sizeof(unsigned int *));
	for (i = 0; i < gridBlock->numRows_ghost; i++)
		gridBlock->block[i] = (unsigned int *)malloc(gridBlock->numCols_ghost * sizeof(unsigned int));
}

//function for initialize blocks
void init_grid_block(struct grid_block *gridBlock)
{
	//srand(time(NULL));
	
	int i, j;

	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
			gridBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}

void display(struct grid_block *gridBlock, int nRows, int nCols, MPI_Datatype row_block_without_ghost, int t)
{

	int i, j;
	MPI_Status stat;

	//send data to the root, if I'm not the root
	if (gridBlock->rank != MPI_root)
	{
		//send all rows
		for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
			MPI_Send(&gridBlock->block[i][1], nCols, MPI_INT, MPI_root, 0, MPI_COMM_WORLD);
	}
	else{ 
		
		//if I'm the root: print and receive

		//printf("\n-------------------------------------Time Step - %d ------------------------------------------\n\n", t );
		//print current grid
		
	
		
		/*
		for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		{
			for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
				printf("%d ", gridBlock->block[i][j]);
			printf("\n");
		}
		*/

		print_block(gridBlock);

		int src, rec_idx, i_buf;

		//Receive form other nodes ( excluding the root, 0 )
		for (src = 1; src < gridBlock->mpi_size; src++){


			// I need know how much rows the root must receive, are different for some node
			//For now, I can compute the number of rows of each node

			int nRows_rec = nRows / gridBlock->mpi_size;

			if (src == gridBlock->mpi_size - 1)
				nRows_rec += nRows % gridBlock->mpi_size;

			int buffer[nCols];

			for (rec_idx = 0; rec_idx < nRows_rec; rec_idx++){
				
				MPI_Recv(&buffer[0], nCols, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);
				/*
				for (i_buf = 0; i_buf < nCols; i_buf++)
					printf("%d ", buffer[i_buf]);
				printf("\n");
				*/
			}
		}

		printf("\n\n");

	}
}


void print_buffer(struct grid_block *gridBlock, unsigned int *buffer) {
    
	int x, y;

    for (x = 1; x < gridBlock->numRows_ghost - 1; x++) {
        for (y = 1; y < gridBlock->numCols_ghost - 1; y++)
            printf(*((buffer + x*gridBlock->numCols_ghost) + y) == ALIVE ? "\033[07m  \033[m" : "  ");

        printf("\033[E");
    }
}


void display_version2(struct grid_block *gridBlock, int nRows, int nCols, MPI_Datatype row_block_without_ghost, int t)
{

	int i, j, src;
	MPI_Status stat;

	int rowLocal = gridBlock->numRows_ghost - 2;
	int colsLocal = gridBlock->numCols_ghost - 2;

	int rowBuf = rowLocal + gridBlock->remained_rows;

	if( gridBlock -> rank == 10){

		//print root chunk
		print_block(gridBlock);
		
		/*
		unsigned int buffer[rowBuf][colsLocal];

		// receive blocks from the other nodes
		for(src = 1; src < gridBlock->mpi_size; src++){
			MPI_Recv(&buffer[0][0], rowBuf * colsLocal, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);
			print_buffer(gridBlock, &buffer[0][0]);
			printf("");
		}
		*/
		fflush(stdout);
		
	}/*else
		MPI_Send(&gridBlock->block[1][1], rowBuf * colsLocal, MPI_INT, MPI_root, 0, MPI_COMM_WORLD);

	*/

//usleep(150000);
}





// Ghost Rows: In order to compute the envolve we need to send the first row (ghost) to the upper neighbor and the last
// row to the lower neighbour. (Try to use dataype to send and check if it improve performance)
// Ghost Columns: Copy fisrt column to the last ghost columns

//TODO: evolution of game of a block, and manage neighbours
void evolve_block(struct grid_block *gridBlock, unsigned int **next_gridBlock, int nRows, int nCols, MPI_Datatype row_block_type)
{

	int i, j, t, x, y;

	MPI_Status stat;

	// send first row of the block to the upper neighbour
	MPI_Send(&gridBlock->block[1][0], 1, row_block_type, gridBlock->upper_neighbour, 0, MPI_COMM_WORLD);

	// send last row of the block to the lower neighbour
	MPI_Send(&gridBlock->block[gridBlock->numRows_ghost - 2][0], 1, row_block_type, gridBlock->lower_neighbour, 0, MPI_COMM_WORLD);

	// receive from below using  buffer the ghost row as receiver
	MPI_Recv(&gridBlock->block[gridBlock->numRows_ghost - 1][0], gridBlock->numCols_ghost, MPI_INT, gridBlock->lower_neighbour, 0, MPI_COMM_WORLD, &stat);

	// receive from top using  the ghost row as receiver buffer
	MPI_Recv(&gridBlock->block[0][0], gridBlock->numCols_ghost, MPI_INT, gridBlock->upper_neighbour, 0, MPI_COMM_WORLD, &stat);




	//ghost colums:
	// 		-copy last column to the fisrt column
	// 		-copy the fisrt column to the last last column

	for (i = 0; i < gridBlock->numRows_ghost; i++)
	{
		gridBlock->block[i][0] = gridBlock->block[i][gridBlock->numCols_ghost - 2];
		gridBlock->block[i][gridBlock->numCols_ghost - 1] = gridBlock->block[i][1];
	}


/*
	TEST COMMUNICATION NEIGHBOURS:

	if( gridBlock->rank==2){

		printf("\n NODE rank %d: \n", gridBlock->rank);

		for (i = 0; i < gridBlock->numRows_ghost; i++){
		
			for (j = 0; j < gridBlock->numCols_ghost; j++)
				printf("%d ", gridBlock->block[i][j]);
		
			printf("\n");
		}

	}

*/

	//Update to current grid to the next grid
	for (i = 1; i < gridBlock->numRows_ghost - 1; i++){

		for (j = 1; j < gridBlock->numCols_ghost - 1; j++){

			int alive_neighbours = 0;

			for (x = i - 1; x <= i + 1; x++)
				for (y = j - 1; y <= j + 1; y++)
					if ((i != x || j != y) && gridBlock->block[x][y]) alive_neighbours++;


			//printf("\nAlive Neighbours: %d \n", alive_neighbours);
			
			if (gridBlock->block[i][j] && alive_neighbours < 2)
				next_gridBlock[i][j] = DEAD;

			if (gridBlock->block[i][j] && (alive_neighbours == 2 || alive_neighbours == 3))
				next_gridBlock[i][j] = ALIVE;

			if (alive_neighbours > 3)
				next_gridBlock[i][j] = DEAD;

			if ( !gridBlock->block[i][j] && (alive_neighbours == 3))
				next_gridBlock[i][j] = ALIVE;
		}
	}



	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
			gridBlock->block[i][j] = next_gridBlock[i][j];
}

// call envolve and diaplay the evolution
void game_block(struct grid_block *gridBlock, int time, int nRows, int nCols)
{
	int i, j, t;
	//allocate the next grid used to compute the evolution of the next time step
	unsigned int **next_gridBlock;

	//TODO: see difference of performance in seding using derived datatype and without
	// create a derived datatype to send a row
	MPI_Datatype row_block_type, row_block_without_ghost;

	// for the envolve
	MPI_Type_contiguous(gridBlock->numCols_ghost, MPI_INT, &row_block_type);

	// for the display
	MPI_Type_contiguous(gridBlock->numCols_ghost - 2, MPI_INT, &row_block_without_ghost);
	MPI_Type_commit(&row_block_type);
	MPI_Type_commit(&row_block_without_ghost);

	//Random Initialization of the grid assigned to each node
	init_grid_block(gridBlock);
	allocate_empty_grid(&next_gridBlock, gridBlock->numRows_ghost, gridBlock->numCols_ghost);

	for (t = 0; t < time; t++)
	{

		display(gridBlock, nRows, nCols, row_block_without_ghost, t);
		evolve_block(gridBlock, next_gridBlock, nRows, nCols, row_block_type);
	}

	// free the derived datatype
	MPI_Type_free(&row_block_type);
	MPI_Type_free(&row_block_without_ghost);
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

	struct grid_block blockGrid;

	init_and_allocate_block(&blockGrid, n_rows_local_with_ghost, n_cols_with_ghost, upper_neighbour, lower_neighbour, rank, size);

	game_block(&blockGrid, time, nRows, nCols);

	//MPI_Barrier(MPI_COMM_WORLD);
	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}
