#define ALIVE 1
#define DEAD 0

struct block{

    int numRows_ghost;
    int numCols_ghost;

    int upper_neighbours;
    int lower_neighbours;

    unsigned int **block;
    
}