/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 * 
*/

#include "gameOfLife_MPI.h"

// Allocate a matrix so as to have elements contiguos in memory
unsigned int **allocate_empty_gen(int rows, int cols)
{

	int i;
	//allocate memory for an array of pointers and then allocate memory for every row
	unsigned int *gen = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));
	unsigned int **array = (unsigned int **)malloc(rows * sizeof(unsigned int *));
	for (i = 0; i < rows; i++)
		array[i] = &(gen[cols * i]);

	return array;
}

void free_gen(unsigned int **gen)
{

	free(gen[0]);
	free(gen);
}


void print_block(struct gen_block *genBlock)
{
	int x, y;

	// \033[H -> move cursor to top-left corner;
	// \033[J -> clear the console.
	printf("\033[H\033[J");
	for (x = 1; x < genBlock->numRows_ghost - 1; x++)
	{
		for (y = 1; y < genBlock->numCols_ghost - 1; y++)
			printf(genBlock->block[x][y] ? "\033[07m  \033[m" : "  ");

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
void init_and_allocate_block(struct gen_block *genBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour, int rank, int size, int time)
{

	int i;

	// initialize field of the struct that represent the block assigned to a node
	genBlock->numRows_ghost = nRows_with_ghost;
	genBlock->numCols_ghost = nCols_with_Ghost;
	genBlock->upper_neighbour = upper_neighbour;
	genBlock->lower_neighbour = lower_neighbour;
	genBlock->rank = rank;
	genBlock->mpi_size = size;
	genBlock->time_step = time;

	//allocate memory for an array of pointers and then allocate memory for every row
	genBlock->block = allocate_empty_gen(genBlock->numRows_ghost, genBlock->numCols_ghost);
}

//function for initialize blocks
void init_gen_block(struct gen_block *genBlock)
{
	int i, j;

	for (i = 1; i < genBlock->numRows_ghost - 1; i++)
		for (j = 1; j < genBlock->numCols_ghost - 1; j++)
			genBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}

void printbig_block(struct gen_block *genBlock, int t, char filename[])
{

	int x, y;

	FILE *f;

	if (t == 0)
		f = fopen(filename, "w");
	else
		f = fopen(filename, "a");

	// separate 1 time step from last
	if (t == genBlock->time_step - 1)
		fprintf(f, "\n\n\n\n\n\n ***************************************************************************************************************************************** \n\n\n\n\n\n");

	for (x = 1; x < genBlock->numRows_ghost - 1; x++)
	{
		for (y = 1; y < genBlock->numCols_ghost - 1; y++)
			fprintf(f, "%c", genBlock->block[x][y] ? 'x' : ' ');

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
void display_v2(struct gen_block *genBlock, int nRows, int nCols, MPI_Datatype block_type, int t, bool exec_time)
{

	int i, j;
	MPI_Status stat;
	char filename[] = "MPI/glife_MPI_v2.txt";

	//send data to the root, if I'm not the root
	if (genBlock->rank != MPI_root)
	{
		int numRows = genBlock->numRows_ghost - 2;
		int numCols = genBlock->numCols_ghost - 2;

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

void print_buffer(struct gen_block *genBlock, unsigned int *buffer)
{

	int x, y;

	for (x = 1; x < genBlock->numRows_ghost - 1; x++)
	{
		for (y = 1; y < genBlock->numCols_ghost - 1; y++)
			printf(*((buffer + x * genBlock->numCols_ghost) + y) == ALIVE ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
}

// Ghost Rows: In order to compute the envolve we need to send the first row to the upper neighbor and the last
// row to the lower neighbour, thanks to the use of the top and bottom ghost rows.

void evolve_block(struct gen_block *genBlock, unsigned int **next_genBlock, int nRows, int nCols, MPI_Datatype row_block_type)
{

	int i, j, t, x, y;

	MPI_Status stat;

	// send first row of the block to the upper neighbour
	MPI_Send(&genBlock->block[1][0], 1, row_block_type, genBlock->upper_neighbour, 0, MPI_COMM_WORLD);

	// send last row of the block to the lower neighbour
	MPI_Send(&genBlock->block[genBlock->numRows_ghost - 2][0], 1, row_block_type, genBlock->lower_neighbour, 0, MPI_COMM_WORLD);

	// receive from below using  buffer the ghost row as receiver
	MPI_Recv(&genBlock->block[genBlock->numRows_ghost - 1][0], genBlock->numCols_ghost, MPI_INT, genBlock->lower_neighbour, 0, MPI_COMM_WORLD, &stat);

	// receive from top using  the ghost row as receiver buffer
	MPI_Recv(&genBlock->block[0][0], genBlock->numCols_ghost, MPI_INT, genBlock->upper_neighbour, 0, MPI_COMM_WORLD, &stat);

	//ghost colums:
	// 		-copy last column to the fisrt column
	// 		-copy the fisrt column to the last last column

	for (i = 0; i < genBlock->numRows_ghost; i++)
	{
		genBlock->block[i][0] = genBlock->block[i][genBlock->numCols_ghost - 2];
		genBlock->block[i][genBlock->numCols_ghost - 1] = genBlock->block[i][1];
	}

	//Update to current gen to the next gen
	for (i = 1; i < genBlock->numRows_ghost - 1; i++)
	{

		for (j = 1; j < genBlock->numCols_ghost - 1; j++)
		{

			int alive_neighbours = 0;

			for (x = i - 1; x <= i + 1; x++)
				for (y = j - 1; y <= j + 1; y++)
					if ((i != x || j != y) && genBlock->block[x][y])
						alive_neighbours++;

			if (genBlock->block[i][j] && alive_neighbours < 2)
				next_genBlock[i][j] = DEAD;

			if (genBlock->block[i][j] && (alive_neighbours == 2 || alive_neighbours == 3))
				next_genBlock[i][j] = ALIVE;

			if (alive_neighbours > 3)
				next_genBlock[i][j] = DEAD;

			if (!genBlock->block[i][j] && (alive_neighbours == 3))
				next_genBlock[i][j] = ALIVE;
		}
	}

	
	for (i = 1; i < genBlock->numRows_ghost - 1; i++)
		for (j = 1; j < genBlock->numCols_ghost - 1; j++)
			genBlock->block[i][j] = next_genBlock[i][j];
	
}

char *my_itoa(int num, char *str)
{
    if(str == NULL)
		return NULL;
        
    sprintf(str, "%d", num);
    return str;
}


void get_experiment_filename(int version, int num_nodes, char* folder_name){

	struct stat st = {0};
	char exp_folder[20];
	
	(version == 2) ? strcpy(exp_folder,"experiment_V2") : strcpy(exp_folder,"experiment_V1");
	
	
	char results_filename[200];
	(version == 2) ? strcpy(results_filename,"/Exp-MPI-%d-%d-%d_V2.csv") : strcpy(results_filename,"/Exp-MPI-%d-%d-%d_V1.csv");
	char str_num_node[5];

	strcat(folder_name, exp_folder);
	if (stat(folder_name, &st) == -1) mkdir(folder_name,  0700);
	
	strcat(folder_name,"/Node-");
	my_itoa(num_nodes, str_num_node);
	
	strcat(folder_name, str_num_node);
			
	if (stat(folder_name, &st) == -1) mkdir(folder_name,  0700);

	strcat(folder_name, results_filename);
}

// call envolve and diaplay the evolution
void game(struct gen_block *genBlock, int time, int nRows, int nCols, int version, bool exec_time, int num_nodes)
{
	int i, j, t;
	//allocate the next gen used to compute the evolution of the next time step
	unsigned int **next_genBlock;
	struct timeval start, end;
	double partial_time = 0., tot_time = 0., send_time = 0.;
	
	//Random Initialization of the gen assigned to each node
	init_gen_block(genBlock);

	// create a derived datatype to send a row
	MPI_Datatype row_block_type, row_block_without_ghost, block_type;

	// for the envolve
	MPI_Type_contiguous(genBlock->numCols_ghost, MPI_UNSIGNED, &row_block_type);

	//to send a block, thanks to use of MPI derived datatype
	MPI_Type_vector(genBlock->numRows_ghost - 2, genBlock->numCols_ghost - 2, genBlock->numCols_ghost, MPI_UNSIGNED, &block_type);

	// for the display
	MPI_Type_contiguous(nCols, MPI_UNSIGNED, &row_block_without_ghost);

	MPI_Type_commit(&row_block_type);
	MPI_Type_commit(&row_block_without_ghost);
	MPI_Type_commit(&block_type);

	

	next_genBlock = allocate_empty_gen(genBlock->numRows_ghost, genBlock->numCols_ghost);

	for (t = 0; t < time; t++)
	{
		if (genBlock->rank == 0)
			gettimeofday(&start, NULL);

		evolve_block(genBlock, next_genBlock, nRows, nCols, row_block_type);

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
		
		writeFile(fileName, genBlock->mpi_size == 2, tot_time, genBlock->mpi_size);
	}

	// free the derived datatype
	MPI_Type_free(&row_block_type);
	MPI_Type_free(&row_block_without_ghost);
	MPI_Type_free(&block_type);

	//free gens
	free_gen(genBlock->block);
	free_gen(next_genBlock);
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


int count_nodes(char filename[])
{

	int ch = 0;
	int count = 0;
	FILE *file;

	if ((file = fopen(filename, "r")) == NULL)
		return -1;

	while (!feof(file))
	{
		ch = fgetc(file);
		if (ch == '\n') count++;
		
	}

	fclose(file);

	return count;
}






int main(int argc, char **argv)
{

	int rank, size, err;
	bool exec_time;
	int num_nodes = count_nodes("host_list.txt");

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

	if (argc > 5)
		exec_time = (bool)atoi(argv[5]);

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
	int n_cols_with_ghost = nCols + 2;

	int upper_neighbour = get_upper_neighbour(size, rank);
	int lower_neighbour = get_lower_neighbour(size, rank);

	struct gen_block blockgen;

	init_and_allocate_block(&blockgen, n_rows_local_with_ghost, n_cols_with_ghost, upper_neighbour, lower_neighbour, rank, size, time);

	MPI_Barrier(MPI_COMM_WORLD);
	game(&blockgen, time, nRows, nCols, version, exec_time, num_nodes);

	//-----------------------------------------------------------------------------------------------

	err = MPI_Finalize();

	return EXIT_SUCCESS;
}