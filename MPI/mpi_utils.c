#include "mpi_utils.h"

char *my_itoa(int num, char *str)
{
    if (str == NULL)
        return NULL;

    sprintf(str, "%d", num);
    return str;
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
        if (ch == '\n')
            count++;
    }

    fclose(file);

    return count;
}

void get_experiment_filename(int version, int num_nodes, char *folder_name)
{

    struct stat st = {0};
    char exp_folder[20];

    (version == 2) ? strcpy(exp_folder, "experiment_V2") : strcpy(exp_folder, "experiment_V1");

    char results_filename[200];
    (version == 2) ? strcpy(results_filename, "/Exp-MPI-%d-%d-%d_V2.csv") : strcpy(results_filename, "/Exp-MPI-%d-%d-%d_V1.csv");
    char str_num_node[5];

    strcat(folder_name, exp_folder);
    if (stat(folder_name, &st) == -1)
        mkdir(folder_name, 0700);

    strcat(folder_name, "/Node-");
    my_itoa(num_nodes, str_num_node);

    strcat(folder_name, str_num_node);

    if (stat(folder_name, &st) == -1)
        mkdir(folder_name, 0700);

    strcat(folder_name, results_filename);
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

void print_buffer(struct gen_block *genBlock, unsigned int *buffer)
{

    int x, y;

    for (x = 1; x < genBlock->numRows_ghost - 1; x++)
    {
        for (y = 1; y < genBlock->numCols - 1; y++)
            printf(*((buffer + x * genBlock->numCols) + y) == ALIVE ? "\033[07m  \033[m" : "  ");
        printf("\033[E");
    }
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
        for (y = 0; y < genBlock->numCols; y++)
            fprintf(f, "%c", genBlock->block[x][y] ? 'x' : ' ');

        fprintf(f, "\n");
    }

    fflush(f);
    fclose(f);
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

void print_block(struct gen_block *genBlock)
{
    int x, y;

    // \033[H -> move cursor to top-left corner;
    // \033[J -> clear the console.
    printf("\033[H\033[J");
    for (x = 1; x < genBlock->numRows_ghost - 1; x++)
    {
        for (y = 0; y < genBlock->numCols; y++)
            printf(genBlock->block[x][y] ? "\033[07m  \033[m" : "  ");

        printf("\033[E");
    }
}

//function for initialize blocks
void init_gen_block(struct gen_block *genBlock)
{
    int i, j;

    for (i = 1; i < genBlock->numRows_ghost - 1; i++)
        for (j = 0; j < genBlock->numCols; j++)
            genBlock->block[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;
}

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

//function to allocate a block in a node and initialize the field of the struct it
void init_and_allocate_block(struct gen_block *genBlock, int nRows_with_ghost, int nCols_with_Ghost, int upper_neighbour, int lower_neighbour, int rank, int size, int time)
{

    int i, j;

    // initialize field of the struct that represent the block assigned to a node
    genBlock->numRows_ghost = nRows_with_ghost;
    genBlock->numCols = nCols_with_Ghost;
    genBlock->upper_neighbour = upper_neighbour;
    genBlock->lower_neighbour = lower_neighbour;
    genBlock->rank = rank;
    genBlock->mpi_size = size;
    genBlock->time_step = time;

    //allocate memory for an array of pointers and then allocate memory for every row
    genBlock->block = allocate_empty_gen(genBlock->numRows_ghost, genBlock->numCols);
    genBlock->next_genBlock = allocate_empty_gen(genBlock->numRows_ghost, genBlock->numCols);

    //Random Initialization of the gen assigned to each node
    init_gen_block(genBlock);

    for (i = 0; i < genBlock->numRows_ghost; i++)
        for (j = 0; j < genBlock->numCols; j++)
            genBlock->next_genBlock[i][j] = DEAD;
}
