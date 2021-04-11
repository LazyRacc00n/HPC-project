#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ALIVE 1
#define DEAD 0


void allocate_empty_grid(unsigned int ***grid, int rows, int cols){

	int i,j;
	//allocate memory for an array of pointers and then allocate memory for every row
	*grid = (unsigned int **)malloc(rows * sizeof(unsigned int *));
	for (i = 0; i <  rows; i++)
		(*grid)[i] = (unsigned int *)malloc(cols * sizeof(unsigned int));
			//for(j=0; j < cols; )
}

void fill(unsigned int*** a) {
    int i, j;
    *a = malloc(2 * sizeof **a);
    for (i = 0; i < 2; i++) {
        (*a)[i] = malloc(2 * sizeof *(*a)[i]);
    }

    for (i = 0; i <2; i++) {
        for (j = 0; j < 2; j++) {
            (*a)[i][j] = 9;
        }
    }
}

int main(int argc, char **argv){

    unsigned int **grid, **next_grid;

  
    allocate_empty_grid(&grid, 12,12);
    allocate_empty_grid(&next_grid, 12,12);

    for (int i = 0; i < 11; i++)
		for (int j = 1; j < 11; j++)
			grid[i][j] = rand() < RAND_MAX / 10 ? ALIVE : DEAD;

    for(int i=1; i < 11; i++){
        for (int j = 1; j < 11; j++)
            printf("%d ", grid[i][j]);
        printf("\n");
    }

    for (int i = 1; i <  11; i++){
			for (int j = 1; j < 11; j++){
				
				int alive_neighbours = 0;

				for ( int x = i-1 ; x <= i+1; ++x){
					for( int y = j-1; y <= j+1; ++y)
						if( (i != x || j != y) && grid[i][j] == ALIVE ) ++alive_neighbours;
                    
                }
				

                


				if( alive_neighbours < 2 ) next_grid[i][j] = DEAD;

				if( grid[i][j] == ALIVE && (alive_neighbours == 2 || alive_neighbours == 3)) next_grid[i][j] = ALIVE;

				if( alive_neighbours > 3) next_grid[i][j] = DEAD;

				if( grid[i][j] == DEAD && (alive_neighbours == 3)) next_grid[i][j] = ALIVE;

			}
	}

    return 0;
    
        

}