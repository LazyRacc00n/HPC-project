/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 * 
*/

#include "gameOfLife_MPI.h"

// Allocate a matrix so as to have elements contiguos in memory
unsigned int **allocate_empty_grid(int rows, int cols)
{

	int i;
	//allocate memory for an array of pointers and then allocate memory for every row
	unsigned int *grid = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));
	unsigned int **array = (unsigned int **)malloc(rows * sizeof(unsigned int *));
	for (i = 0; i < rows; i++)
		array[i] = &(grid[cols * i]);

	return array;
}

void free_grid(unsigned int **grid)
{

	free(grid[0]);
	free(grid);
}

void print_block(struct grid_block *gridBlock)
{
	int x, y;

	// \033[H -> move cursor to top-left corner;
	// \033[J -> clear the console.
	printf("\033[H\033[J");
	for (x = 1; x < gridBlock->numRows_ghost - 1; x++)
	{
		for (y = 1; y < gridBlock->numCols_ghost - 1; y++)
			printf(gridBlock->block[x][y] ? "\033[07m  \033[m" : "  ");

		printf("\033[E");
	}
}

void print_received_row(int buffer[], int numCols)
{

	int x;

	for (x = 0; x < numCols; x++)
		printf(buffer[x] == ALIVE ? "\033[07m  \033[m" : "  ");
	printf("\033[E");
}

void print_received_row_big(int buffer[], int numCols, char filename[])
{

	int x;
	FILE *f;
	f = fopen(filename, "a");
	for (x = 0; x < numCols; x++)
		fprintf(f, "%c", buffer[x] ? 'x' : ' ');
	fprintf(f, "\n");
	fclose(f);
}

//function to allocate a block in a node and initialize the field of the struct it
void init_and_allocate_block(struct grid_block *gridBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour, int rank, int size, int time)
{

	int i;

	// initialize field of the struct that represent the block assigned to a node
	gridBlock->numRows_ghost = nRows_with_ghost;
	gridBlock->numCols_ghost = nCols_with_Ghost;
	gridBlock->upper_neighbour = upper_neighbour;
	gridBlock->lower_neighbour = lower_neighbour;
	gridBlock->rank = rank;
	gridBlock->mpi_size = size;
	gridBlock->time_step = time;

	//allocate memory for an array of pointers and then allocate memory for every row
	gridBlock->block = allocate_empty_grid(gridBlock->numRows_ghost, gridBlock->numCols_ghost);
}

//function for initialize blocks
void init_grid_block(struct grid_block *gridBlock)
{
	int i, j;

	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
			gridBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}

void printbig_block(struct grid_block *gridBlock, int t, char filename[])
{

	int x, y;

	FILE *f;

	if (t == 0)
		f = fopen(filename, "w");
	else
		f = fopen(filename, "a");

	// separate 1 time step from last
	if (t == gridBlock->time_step - 1)
		fprintf(f, "\n\n\n\n\n\n ***************************************************************************************************************************************** \n\n\n\n\n\n");

	for (x = 1; x < gridBlock->numRows_ghost - 1; x++)
	{
		for (y = 1; y < gridBlock->numCols_ghost - 1; y++)
			fprintf(f, "%c", gridBlock->block[x][y] ? 'x' : ' ');

		fprintf(f, "\n");
	}

	fflush(f);
	fclose(f);
}

void print_buffer_V2(int rows, int cols, unsigned int buffer[rows][cols])
{
	int i, j;

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
			printf(buffer[i][j] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
}

void printbig_buffer_V2(int rows, int cols, unsigned int buffer[rows][cols], char filename[])
{

	int i, j;
	FILE *f;

	f = fopen(filename, "a");

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
			fprintf(f, "%c", buffer[i][j] ? 'x' : ' ');
		fprintf(f, "\n");
	}

	fflush(f);
	fclose(f);
}

// VERSION 2 OF DISPLAY, WITH MPI_TYPE_VECTOR
void display_v2(struct grid_block *gridBlock, int nRows, int nCols, MPI_Datatype block_type, int t)
{

	int i, j;
	MPI_Status stat;
	char filename[] = "glife_MPI_v2.txt";

	//send data to the root, if I'm not the root
	if (gridBlock->rank != MPI_root)
	{
		int numRows = gridBlock->numRows_ghost - 2;
		int numCols = gridBlock->numCols_ghost - 2;

		unsigned int buff[numRows][numCols];

		MPI_Send(&(gridBlock->block[1][1]), 1, block_type, MPI_root, 0, MPI_COMM_WORLD);
	}
	else
	{
		//if I'm the root: print and receive the blocks of the other nodes
		//print_buffer(gridBlock->block)

		if (nCols > 1000 && (t == 0 || t == gridBlock->time_step - 1))
			printbig_block(gridBlock, t, filename);
		else if (nCols <= 1000)
			print_block(gridBlock);

		int src, rec_idx, i_buf, j_buf;

		//Receive form other nodes ( excluding the root, 0 )
		for (src = 1; src < gridBlock->mpi_size; src++)
		{
			// I need know how much rows the root must receive, are different for some node
			//For now, I can compute the number of rows of each node

			int nRows_received = nRows / gridBlock->mpi_size;
			if (src == gridBlock->mpi_size - 1)
				nRows_received += nRows % gridBlock->mpi_size;

			unsigned int buffer[nRows_received][nCols];

			MPI_Recv(&(buffer[0][0]), (nRows_received) * (nCols), MPI_UNSIGNED, src, 0, MPI_COMM_WORLD, &stat);

			if (nCols > 1000 && (t == 0 || t == gridBlock->time_step - 1))
				printbig_buffer_V2(nRows_received, nCols, buffer, filename);
			else if (nCols <= 1000)
				print_buffer_V2(nRows_received, nCols, buffer);
		}
	}

	if (nCols <= 1000)
	{
		fflush(stdout);
		usleep(200000);
	}
}

// VERSION 1 OF DISPLAY, EACH NODE SEND ROW BY ROW UNSING MPI_TYPE_COMNTIGUOS
void display_v1(struct grid_block *gridBlock, int nRows, int nCols, MPI_Datatype row_block_without_ghost, int t)
{

	int i, j;
	MPI_Status stat;
	char filename[] = "glife_MPI_v1.txt";

	//send data to the root, if I'm not the root
	if (gridBlock->rank != MPI_root)
	{
		//send all rows
		for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
			MPI_Send(&gridBlock->block[i][1], 1, row_block_without_ghost, MPI_root, 0, MPI_COMM_WORLD);
	}
	else
	{

		//if I'm the root: print and receive
		if ((nCols > 1000) && (t == 0 || t == gridBlock->time_step - 1))
			printbig_block(gridBlock, t, filename);
		else if (nCols <= 1000)
			print_block(gridBlock);

		int src, rec_idx, i_buf;

		//Receive form other nodes ( excluding the root, 0 )
		for (src = 1; src < gridBlock->mpi_size; src++)
		{
			// I need know how much rows the root must receive, are different for some node
			//For now, I can compute the number of rows of each node

			int nRows_rec = nRows / gridBlock->mpi_size;

			if (src == gridBlock->mpi_size - 1)
				nRows_rec += nRows % gridBlock->mpi_size;

			int buffer[nCols];

			for (rec_idx = 0; rec_idx < nRows_rec; rec_idx++)
			{

				MPI_Recv(&buffer[0], nCols, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);

				if ((nCols > 1000) && (t == 0 || t == gridBlock->time_step - 1))
					print_received_row_big(buffer, nCols, filename);
				else if (nCols <= 1000)
					print_received_row(buffer, nCols);
			}
		}
	}

	if (nCols <= 1000)
	{

		fflush(stdout);
		usleep(200000);
	}
}

void print_buffer(struct grid_block *gridBlock, unsigned int *buffer)
{

	int x, y;

	for (x = 1; x < gridBlock->numRows_ghost - 1; x++)
	{
		for (y = 1; y < gridBlock->numCols_ghost - 1; y++)
			printf(*((buffer + x * gridBlock->numCols_ghost) + y) == ALIVE ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
}

// Ghost Rows: In order to compute the envolve we need to send the first row (ghost) to the upper neighbor and the last
// row to the lower neighbour. (Try to use dataype to send and check if it improve performance)
// Ghost Columns: Copy fisrt column to the last ghost columns

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

	//Update to current grid to the next grid
	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
	{

		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
		{

			int alive_neighbours = 0;

			for (x = i - 1; x <= i + 1; x++)
				for (y = j - 1; y <= j + 1; y++)
					if ((i != x || j != y) && gridBlock->block[x][y])
						alive_neighbours++;

			if (gridBlock->block[i][j] && alive_neighbours < 2)
				next_gridBlock[i][j] = DEAD;

			if (gridBlock->block[i][j] && (alive_neighbours == 2 || alive_neighbours == 3))
				next_gridBlock[i][j] = ALIVE;

			if (alive_neighbours > 3)
				next_gridBlock[i][j] = DEAD;

			if (!gridBlock->block[i][j] && (alive_neighbours == 3))
				next_gridBlock[i][j] = ALIVE;
		}
	}

	for (i = 1; i < gridBlock->numRows_ghost - 1; i++)
		for (j = 1; j < gridBlock->numCols_ghost - 1; j++)
			gridBlock->block[i][j] = next_gridBlock[i][j];
}

// call envolve and diaplay the evolution
void game(struct grid_block *gridBlock, int time, int nRows, int nCols, int version)
{
	int i, j, t;
	//allocate the next grid used to compute the evolution of the next time step
	unsigned int **next_gridBlock;
	struct timeval start, end;
	double exec_time = 0.;

	// create a derived datatype to send a row
	MPI_Datatype row_block_type, row_block_without_ghost, block_type;

	// for the envolve
	MPI_Type_contiguous(gridBlock->numCols_ghost, MPI_UNSIGNED, &row_block_type);

	//to send a block, thanks to use of MPI derived datatype
	MPI_Type_vector(gridBlock->numRows_ghost - 2, gridBlock->numCols_ghost - 2, gridBlock->numCols_ghost, MPI_UNSIGNED, &block_type);

	// for the display
	MPI_Type_contiguous(nCols, MPI_UNSIGNED, &row_block_without_ghost);

	MPI_Type_commit(&row_block_type);
	MPI_Type_commit(&row_block_without_ghost);
	MPI_Type_commit(&block_type);

	//Random Initialization of the grid assigned to each node
	init_grid_block(gridBlock);

	next_gridBlock = allocate_empty_grid(gridBlock->numRows_ghost, gridBlock->numCols_ghost);

	//synchronize all the nodes to start the time
	MPI_Barrier(MPI_COMM_WORLD);
	if (gridBlock->rank == 0)
		gettimeofday(&start, NULL);

	for (t = 0; t < time; t++)
	{

		evolve_block(gridBlock, next_gridBlock, nRows, nCols, row_block_type);

		if (version == 1)
		{
			display_v1(gridBlock, nRows, nCols, row_block_without_ghost, t);
		}
		else
		{

			display_v2(gridBlock, nRows, nCols, block_type, t);
		}
	}

	//synchronize all the nodes to end the time
	MPI_Barrier(MPI_COMM_WORLD);
	if (gridBlock->rank == 0)
	{
		gettimeofday(&end, NULL);
		exec_time = (double)elapsed_wtime(start, end);
		char *fileName = (char *)malloc(50 * sizeof(char));
		sprintf(fileName, "MPI_Experiments/Exp01-MPI-%d-%d-%d.csv", nCols, nRows, time);
		writeFile(fileName, gridBlock->mpi_size == 2, exec_time, gridBlock->mpi_size);
	}

	// free the derived datatype
	MPI_Type_free(&row_block_type);
	MPI_Type_free(&row_block_without_ghost);
	MPI_Type_free(&block_type);

	//free grids
	free_grid(gridBlock->block);
	free_grid(next_gridBlock);
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

	//Parse Arguments
	int nCols = 0, nRows = 0, time = 0, version = 0;
	if (argc > 1)
		nCols = atoi(argv[1]);

	if (argc > 2)
		nRows = atoi(argv[2]);

	if (argc > 3)
		time = atoi(argv[3]);

	if (argc > 4)
		version = atoi(argv[4]);

	if (nCols <= 0)
		nCols = 30;

	if (nRows <= 0)
		nRows = 30;

	if (time <= 0)
		time = 100;

	if (version <= 0 || version > 2)
	{

		printf("\n\nVersion not exists ! It will be executed with the version 1!\n\n");
		version = 1;
	}

	err = MPI_Init(&argc, &argv);

	if (err != 0)
	{

		printf("\nError in MPI initialization!\n");
		MPI_Abort(MPI_COMM_WORLD, err);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//send number of columns and number of rows to each process
	MPI_Bcast(&nRows, 1, MPI_INT, MPI_root, MPI_COMM_WORLD);
	MPI_Bcast(&nCols, 1, MPI_INT, MPI_root, MPI_COMM_WORLD);

	MPI_Bcast(&time, 1, MPI_INT, MPI_root, MPI_COMM_WORLD);

	//Each process compute the size of its chunks
	int n_rows_local = nRows / size;
	//if the division has remains are added to the last process;
	if (rank == size - 1)
		n_rows_local += nRows % size;

	// ghost colums are which that communicate with neighbours.
	// recever buffer
	int n_rows_local_with_ghost = n_rows_local + 2;
	int n_cols_with_ghost = nCols + 2;

	int upper_neighbour = get_upper_neighbour(size, rank);
	int lower_neighbour = get_lower_neighbour(size, rank);

	struct grid_block blockGrid;

	init_and_allocate_block(&blockGrid, n_rows_local_with_ghost, n_cols_with_ghost, upper_neighbour, lower_neighbour, rank, size, time);

	game(&blockGrid, time, nRows, nCols, version);

	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}
