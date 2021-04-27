/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 * 
*/

#include "mpi_utils.h"


void swap(unsigned int ***old, unsigned int ***new) {
    unsigned int **temp = *old;

    *old = *new;
    *new = temp;
}



// VERSION 2 OF DISPLAY, WITH MPI_TYPE_VECTOR
void display_v2(struct gen_block *genBlock, int nRows, int nCols, MPI_Datatype block_type, int t, bool exec_time)
{

	int i, j;
	MPI_Status stat;
	char filename[] = "MPI/glife_MPI_v2.txt";

	//send data to the root, if I'm not the root
	if (genBlock->rank != MPI_root)
	{
		int numRows = genBlock->numRows_ghost - 2;
		int numCols = genBlock->numCols;

		unsigned int buff[numRows][numCols];

		MPI_Send(&(genBlock->block[1][1]), 1, block_type, MPI_root, 0, MPI_COMM_WORLD);
	}
	else
	{
		//if I'm the root: print and receive the blocks of the other nodes
		//print_buffer(genBlock->block)

		if (!exec_time)
		{
			if (nCols > 1000 && (t == 0 || t == genBlock->time_step - 1))
				printbig_block(genBlock, t, filename);
			else if (nCols <= 1000)
				print_block(genBlock);
		}
		int src, rec_idx, i_buf, j_buf;

		//Receive form other nodes ( excluding the root, 0 )
		for (src = 1; src < genBlock->mpi_size; src++)
		{
			// I need know how much rows the root must receive, are different for some node
			//For now, I can compute the number of rows of each node

			int nRows_received = nRows / genBlock->mpi_size;
			if (src == genBlock->mpi_size - 1)
				nRows_received += nRows % genBlock->mpi_size;

			unsigned int buffer[nRows_received][nCols];

			MPI_Recv(&(buffer[0][0]), (nRows_received) * (nCols), MPI_UNSIGNED, src, 0, MPI_COMM_WORLD, &stat);

			if (!exec_time)
			{
				if (nCols > 1000 && (t == 0 || t == genBlock->time_step - 1))
					printbig_buffer_V2(nRows_received, nCols, buffer, filename);
				else if (nCols <= 1000)
					print_buffer_V2(nRows_received, nCols, buffer);
			}
		}
	}

	if (nCols <= 1000)
	{
		fflush(stdout);
		usleep(200000);
	}
}

// VERSION 1 OF DISPLAY, EACH NODE SEND ROW BY ROW UNSING MPI_TYPE_COMNTIGUOS
// if exec_time id true is not printed the evolution in order to take the execution time of the sending blocks to the root
void display_v1(struct gen_block *genBlock, int nRows, int nCols, MPI_Datatype row_block_without_ghost, int t, bool exec_time)
{

	int i, j;
	MPI_Status stat;
	char filename[] = "MPI/glife_MPI_v1.txt";

	double partial_time = 0., start, end;

	//send data to the root, if I'm not the root
	if (genBlock->rank != MPI_root)
	{
		//send all rows
		for (i = 1; i < genBlock->numRows_ghost - 1; i++)
			MPI_Send(&genBlock->block[i][1], 1, row_block_without_ghost, MPI_root, 0, MPI_COMM_WORLD);
	}
	else
	{
		//if I'm the root: print and receive

		if (!exec_time)
		{

			if ((nCols > 1000) && (t == 0 || t == genBlock->time_step - 1))
				printbig_block(genBlock, t, filename);
			else if (nCols <= 1000)
				print_block(genBlock);
		}

		//exec_time = (double)elapsed_wtime(start, end);
		int src, rec_idx, i_buf;

		//Receive form other nodes ( excluding the root, 0 )
		for (src = 1; src < genBlock->mpi_size; src++)
		{

			// I need know how much rows the root must receive, are different for some node
			//For now, I can compute the number of rows of each node

			int nRows_rec = nRows / genBlock->mpi_size;

			if (src == genBlock->mpi_size - 1)
				nRows_rec += nRows % genBlock->mpi_size;

			int buffer[nCols];

			for (rec_idx = 0; rec_idx < nRows_rec; rec_idx++)
			{

				MPI_Recv(&buffer[0], nCols, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);

				if (!exec_time)
				{

					if ((nCols > 1000) && (t == 0 || t == genBlock->time_step - 1))
						print_received_row_big(buffer, nCols, filename);
					else if (nCols <= 1000)
						print_received_row(buffer, nCols);
				}
			}
		}
	}

	if (nCols <= 1000)
	{

		fflush(stdout);
		usleep(200000);
	}
}


// Ghost Rows: In order to compute the envolve we need to send the first row to the upper neighbor and the last
// row to the lower neighbour, thanks to the use of the top and bottom ghost rows.

void evolve_block(struct gen_block *genBlock, unsigned int** next_block, int nRows, int nCols, MPI_Datatype row_block_type)
{

	int i, j, t, x, y;

	MPI_Status stat;

	// send first row of the block to the upper neighbour
	MPI_Send(&genBlock->block[1][0], 1, row_block_type, genBlock->upper_neighbour, 0, MPI_COMM_WORLD);

	// send last row of the block to the lower neighbour
	MPI_Send(&genBlock->block[genBlock->numRows_ghost - 2][0], 1, row_block_type, genBlock->lower_neighbour, 0, MPI_COMM_WORLD);

	// receive from below using  buffer the ghost row as receiver
	MPI_Recv(&genBlock->block[genBlock->numRows_ghost - 1][0], genBlock->numCols, MPI_INT, genBlock->lower_neighbour, 0, MPI_COMM_WORLD, &stat);

	// receive from top using  the ghost row as receiver buffer
	MPI_Recv(&genBlock->block[0][0], genBlock->numCols, MPI_INT, genBlock->upper_neighbour, 0, MPI_COMM_WORLD, &stat);

	int rows = genBlock->numRows_ghost - 1;
	int cols = genBlock->numCols;
	//Update to current gen to the next gen
	for (i = 1; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{

			int alive_neighbours = 0;

			for (x = i - 1; x <= i + 1; x++)
				for (y = j - 1; y <= j + 1; y++)
					if ((i != x || j != y) && genBlock->block[x][ (y + nCols) % nCols ] )
						alive_neighbours++;

			next_block[i][j] = (alive_neighbours == 3 || (alive_neighbours == 2 && genBlock->block[i][j]));
		
		}
	}
	
	
}




// call envolve and diaplay the evolution
void game(struct gen_block *genBlock, int time, int nRows, int nCols, int version, bool exec_time, int num_nodes)
{
	int i, j, t;

	struct timeval start, end;
	double partial_time = 0., tot_time = 0., send_time = 0.;
	
	//allocate the next gen used to compute the evolution of the next time step
	unsigned int **next_genBlock;
	next_genBlock= allocate_empty_gen(genBlock->numRows_ghost, nCols);
	// create a derived datatype to send a row
	MPI_Datatype row_block_type, row_block_without_ghost, block_type;

	// for the envolve
	MPI_Type_contiguous(genBlock->numCols, MPI_UNSIGNED, &row_block_type);

	//to send a block, thanks to use of MPI derived datatype
	MPI_Type_vector(genBlock->numRows_ghost - 2, genBlock->numCols, genBlock->numCols, MPI_UNSIGNED, &block_type);

	// for the display
	MPI_Type_contiguous(nCols, MPI_UNSIGNED, &row_block_without_ghost);

	MPI_Type_commit(&row_block_type);
	MPI_Type_commit(&row_block_without_ghost);
	MPI_Type_commit(&block_type);


	for (t = 0; t < time; t++)
	{
		if (genBlock->rank == 0)
			gettimeofday(&start, NULL);

		evolve_block(genBlock, next_genBlock, nRows, nCols, row_block_type);
		swap(&genBlock->block, &next_genBlock);
		
		if (version == 1)
			display_v1(genBlock, nRows, nCols, row_block_without_ghost, t, exec_time);
		else
			display_v2(genBlock, nRows, nCols, block_type, t, exec_time);

		//synchronize all the nodes to end the time
		if (genBlock->rank == 0)
		{
			gettimeofday(&end, NULL);
			partial_time = (double)elapsed_wtime(start, end);
			tot_time += partial_time;
		}
	}
	
	
	if (genBlock->rank == 0)
	{

		char *fileName = (char *)malloc(200 * sizeof(char));
		char folder_name[300] =  "MPI_Results/";

		get_experiment_filename(version, num_nodes, folder_name);
		sprintf(fileName, folder_name, nCols, nRows, time);
		
		writeFile(fileName, nCols, nRows, time, (genBlock->mpi_size == 2 ||( num_nodes==4 && genBlock->mpi_size == 4 )|| (num_nodes == 8 && genBlock->mpi_size == 8) ) , tot_time, genBlock->mpi_size);
	}

	// free the derived datatype
	MPI_Type_free(&row_block_type);
	MPI_Type_free(&row_block_without_ghost);
	MPI_Type_free(&block_type);

	//free gens
	free_gen(genBlock->block);
	free_gen(next_genBlock);
}




int main(int argc, char **argv)
{

	int rank, size, err;
	bool exec_time;
	//int num_nodes = count_nodes("host_list.txt");

	//Parse Arguments
	int nCols = 0, nRows = 0, time = 0, version = 0, num_nodes=1;
	if (argc > 1)
		nCols = atoi(argv[1]);

	if (argc > 2)
		nRows = atoi(argv[2]);

	if (argc > 3)
		time = atoi(argv[3]);

	if (argc > 4)
		version = atoi(argv[4]);
	
	if (argc > 5)
		exec_time = (bool)atoi(argv[5]);

	if (argc > 5)
		num_nodes = atoi(argv[6]);

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

	// Adding ghost rows and that allow communicate with neighbors.
	int n_rows_local_with_ghost = n_rows_local + 2;

	int upper_neighbour = get_upper_neighbour(size, rank);
	int lower_neighbour = get_lower_neighbour(size, rank);

	struct gen_block blockgen;

	init_and_allocate_block(&blockgen, n_rows_local_with_ghost, nCols, upper_neighbour, lower_neighbour, rank, size, time);

	//MPI_Barrier(MPI_COMM_WORLD);
	game(&blockgen, time, nRows, nCols, version, exec_time, num_nodes);

	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}