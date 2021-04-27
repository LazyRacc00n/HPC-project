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
struct gen_block {

    int numRows_ghost; // number of rows of the local block + ghost rows
    int numCols; // number of columns of the local block + ghost columns

    int upper_neighbour; // rank of upper nodes
    int lower_neighbour; // rank of lower nodes

    int rank; // rank of the node
    int mpi_size; // total size of the communicator ( MPI_COMM_WORLD )
    
    unsigned int **block; //matrix that represent the local gen

    int time_step; // total number of timestep
    
};


char *my_itoa(int num, char *str);

int get_upper_neighbour(int size, int rank);

int get_lower_neighbour(int size, int rank);

int count_nodes(char filename[]);

void get_experiment_filename(int version, int num_nodes, char* folder_name);

void print_buffer(struct gen_block *genBlock, unsigned int *buffer);
void print_buffer_V2(int rows, int cols, unsigned int buffer[rows][cols]);
void printbig_buffer_V2(int rows, int cols, unsigned int buffer[rows][cols], char filename[]);
void printbig_block(struct gen_block *genBlock, int t, char filename[]);
void print_block(struct gen_block *genBlock);
void print_received_row(int buffer[], int numCols);
void print_received_row_big(int buffer[], int numCols, char filename[]);

void init_gen_block(struct gen_block *genBlock);
void init_and_allocate_block(struct gen_block *genBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour, int rank, int size, int time);

unsigned int **allocate_empty_gen(int rows, int cols);

void free_gen(unsigned int **gen);