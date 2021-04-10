#include "mpi.h"

#define ALIVE 1
#define DEAD 0

struct grid_block{

    int numRows_ghost;
    int numCols_ghost;

    int upper_neighbour;
    int lower_neighbour;

    unsigned int **block;
    
};