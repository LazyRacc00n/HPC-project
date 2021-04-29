#include "mpi.h"
#include "../utils.h"

#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALIVE 1
#define DEAD 0
#define MPI_root 0

//structure that represent a block assigned to a process
struct gen_block
{

    int numRows_ghost; // number of rows of the local block + ghost rows
    int numCols;       // number of columns of the local block + ghost columns

    int upper_neighbour; // rank of upper nodes
    int lower_neighbour; // rank of lower nodes

    int rank;     // rank of the node
    int mpi_size; // total size of the communicator ( MPI_COMM_WORLD )

    unsigned int **block; //matrix that represent the local gen

    int time_step; // total number of timestep
};

// convert int to string
char *my_itoa(int num, char *str);

// build file name and folder in which store the experiments
void get_experiment_filename(int version, int num_nodes, char *folder_name);

// get rank upper neighbor
int get_upper_neighbour(int size, int rank);

// get rank lower neighbor
int get_lower_neighbour(int size, int rank);

// print buffer version 1
void print_buffer(struct gen_block *genBlock, unsigned int *buffer);

// print buffer version 2
void print_buffer_V2(int rows, int cols, unsigned int buffer[rows][cols]);

//print buffer big version 2
void printbig_buffer_V2(int rows, int cols, unsigned int buffer[rows][cols], char filename[]);

// print local grid big
void printbig_block(struct gen_block *genBlock, int t, char filename[]);

//print local grid
void print_block(struct gen_block *genBlock);

//version 1
void print_received_row(int buffer[], int numCols);
//Version 1
void print_received_row_big(int buffer[], int numCols, char filename[]);

// initialization of the local grid
void init_gen_block(struct gen_block *genBlock);

//allocation grids and struct field initialization
void init_and_allocate_block(struct gen_block *genBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour, int rank, int size, int time);

//allocate memory for a 2D vector with elements contiguos in memory
unsigned int **allocate_empty_gen(int rows, int cols);

// Free memory 2D array
void free_gen(unsigned int **gen);

// swap currend grid with the
void swap(unsigned int ***old, unsigned int ***new);