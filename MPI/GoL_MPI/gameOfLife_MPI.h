#include "mpi.h"

#define ALIVE 1
#define DEAD 0
#define MPI_root 0


//structure that represent a block assigned to a node
struct grid_block{

    int numRows_ghost;
    int numCols_ghost;

    int upper_neighbour;
    int lower_neighbour;

    int rank;
    int mpi_size;

    unsigned int **block;
    
};