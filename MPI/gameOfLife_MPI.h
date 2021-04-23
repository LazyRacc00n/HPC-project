#include "mpi.h"
#include "../../utils.h"


#define ALIVE 1
#define DEAD 0
#define MPI_root 0


//structure that represent a block assigned to a node
struct gen_block{

    int numRows_ghost; // number of rows of the local block + ghost rows
    int numCols_ghost; // number of columns of the local block + ghost columns

    int upper_neighbour; // rank of upper nodes
    int lower_neighbour; // rank of lower nodes

    int rank; // rank of the node
    int mpi_size; // total size of the communicator ( MPI_COMM_WORLD )
    
    unsigned int **block; //matrix that represent the local gen

    int time_step;
    
};