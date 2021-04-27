/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h> // boolean type



// compute the elapsed wall-clock time between two time intervals. in ms
double elapsed_wtime(struct timeval start, struct timeval end) {

    return (double)((end.tv_sec * 1000000 + end.tv_usec) - 
		       (start.tv_sec * 1000000 + start.tv_usec))/1000;

   
}


void writeFile(char* fileName , int w, int h, int z, bool first, double time , int n_core){
    FILE *f;


    if(first)   f = fopen(fileName, "w" );
    else f = fopen(fileName, "a" ); 

	if(first) fprintf(f,"%d-%d-%d,\n",w , h, z);

    // write file
    fprintf(f,"%d,%f",n_core , time);

    fprintf(f,"\n");
	fflush(f);
	fclose(f);
    
}

void printbig(unsigned int **univ, int w, int h, int z) {
	int x,y;
	
	FILE *f;
	
	if(z == 0) f = fopen("glife.txt", "w" );
	else f = fopen("glife.txt", "a" );
	
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) fprintf (f,"%c", univ[y][x] ? 'x' : ' ');
		fprintf(f,"\n");
	}
	fprintf(f,"\n\n\n\n\n\n ******************************************************************************************** \n\n\n\n\n\n");
	fflush(f);
	fclose(f);
}

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
unsigned int ** allocate_empty_gen(int rows, int cols)
{

	int i;
	//allocate memory for an array of pointers and then allocate memory for every row
	unsigned int *gen = (unsigned int *)malloc(rows*cols* sizeof(unsigned int));
	unsigned int **array = (unsigned int **)malloc(rows*sizeof(unsigned int*));
	for (i = 0; i < rows; i++)
		array[i] = &(gen[cols*i]);

	return array;
}

void free_gen(unsigned int **gen){

	free(gen[0]);
	free(gen);

}

/*
* a cell is born, if it has exactly three neighbours 
* a cell dies of loneliness, if it has less than two neighbours 
* a cell dies of overcrowding, if it has more than three neighbours 
* a cell survives to the next generation, if it does not die of loneliness 
* or overcrowding 
*/
void evolve(unsigned int **univ, unsigned int **new, int w, int h) {
	
		int x,y,x1,y1,n;
        
		for ( y = 0; y < h; y++) 
        	for ( x = 0; x < w; x++) {
		    	n = 0;

				// look at the 3x3 neighbourhood
		    	for (y1 = y - 1; y1 <= y + 1; y1++)
			    	for (x1 = x - 1; x1 <= x + 1; x1++)

						// skip the current cell [y, x]
				    	if ((y != y1 || x != x1) && univ[(y1 + h) % h][(x1 + w) % w]) n++;

		    new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
		
	    }
}

void game(int w, int h, int t) {
	int x,y,z;
	unsigned int **current_gen = allocate_empty_gen(h, w);
	unsigned int **next_gen = allocate_empty_gen(h, w);

	struct timeval start, end;
	double tot_time = 0.;

	//initialization
	//srand(10);
	for (x = 0; x < w; x++) for (y = 0; y < h; y++) current_gen[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;
	
	if (x > 1000) printbig(current_gen, w, h,0);
	
	for(z = 0; z < t;z++) {

		if (x <= 1000) show(current_gen, w, h);
		
		// get starting time at iteration z
		gettimeofday(&start, NULL);
		
		// lets evolve the current generation
		evolve(current_gen, next_gen, w, h);
		// the next generation is now the current and used in the next iteration
		swap(&current_gen, &next_gen); 

		// get ending time of iteration z
		gettimeofday(&end, NULL);
		
		// sum up the total time execution
		tot_time += (double) elapsed_wtime(start, end);
		
		
		if (x > 1000) 
			printf("Iteration %d is : %f ms\n", z, (double) elapsed_wtime(start, end));
		
	}
	
	if (x > 1000) printbig(current_gen, w, h,1);


    // Allocates storage
	char *fileName = (char*)malloc(50 * sizeof(char));
	sprintf(fileName, "Serial-%d-%d-%d.csv", w, h, t);

	writeFile(fileName, w, h, t, true, tot_time, 0);
	free(fileName);


	free_gen(current_gen);
	free_gen(next_gen);


}
 
int main(int c, char **v) {
	int w = 0, h = 0, t = 0;
	if (c > 1) w = atoi(v[1]);
	if (c > 2) h = atoi(v[2]);
	if (c > 3) t = atoi(v[3]);
	if (w <= 0) w = 30;
	if (h <= 0) h = 30;
	if (t <= 0) t = 100;
	game(w, h, t);
}

