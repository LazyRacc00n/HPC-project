
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h> // boolean type


void show(unsigned int **univ, int w, int h) {
	int x,y;

	printf("\033[H");
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
	usleep(200000);
}



void swap(unsigned int ***old, unsigned int ***new) {
	
    unsigned int **temp = *old;

    *old = *new;
    *new = temp;

}

// Allocate a matrix so as to have elements contiguos in memory
unsigned int ** allocate_empty_grid(int rows, int cols)
{

	int i;
	//allocate memory for an array of pointers and then allocate memory for every row
	unsigned int *grid = (unsigned int *)malloc(rows*cols* sizeof(unsigned int));
	unsigned int **array = (unsigned int **)malloc(rows*sizeof(unsigned int*));
	for (i = 0; i < rows; i++)
		array[i] = &(grid[cols*i]);

	return array;
}

void free_grid(unsigned int **grid){

	free(grid[0]);
	free(grid);

}


void funzioneInutile(unsigned int **univ, unsigned int **univ_prime, int w, int h){

    /*unsigned int ** univ = *u;
    unsigned int ** univ_prime = *v;*/

    univ_prime[h-1][w-1] = 99999999;
    
}

int main(){
    int x, y;
    int h=100;
    int w = 100;
    unsigned int **univ = allocate_empty_grid(h, w);
	unsigned int **univ_prime = allocate_empty_grid(h, w);

    for (x = 0; x < w; x++) for (y = 0; y < h; y++) univ[y][x] = 1 ;
    for (x = 0; x < w; x++) for (y = 0; y < h; y++) univ_prime[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;

    
    funzioneInutile(univ, univ_prime, w, h);
    swap(&univ, &univ_prime);

    //show(univ, w, h);
    //show(univ_prime, w, h);
    printf("%d" , univ[h-1][w-1]);

    free_grid(univ);
    free_grid(univ_prime);
}