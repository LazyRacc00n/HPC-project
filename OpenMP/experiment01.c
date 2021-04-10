/*
 * The Game of Life
 *
 * https://www.geeksforgeeks.org/conways-game-life-python-implementation/
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

 
void show(void *u, int w, int h) {
	int x,y;
	int (*univ)[w] = u;
	printf("\033[H");
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
	usleep(200000);
}



void printbig(void *u, int w, int h, int z) {
	int x,y;
	int (*univ)[w] = u;
	
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



void evolve(void *u, int w, int h) {
	unsigned (*univ)[w] = u;
	unsigned new[h][w];
	int x,y,x1,y1;
 
	for (y = 0; y < h; y++) 
        for (x = 0; x < w; x++) {
		    int n = 0;
		    for (y1 = y - 1; y1 <= y + 1; y1++)
			    for (x1 = x - 1; x1 <= x + 1; x1++)
				    if (univ[(y1 + h) % h][(x1 + w) % w]) n++;
		    if (univ[y][x]) n--;
		    new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
		/*
		 * a cell is born, if it has exactly three neighbours 
		 * a cell dies of loneliness, if it has less than two neighbours 
		 * a cell dies of overcrowding, if it has more than three neighbours 
		 * a cell survives to the next generation, if it does not die of loneliness 
		 * or overcrowding 
		 */
	    }
	    for (y = 0; y < h; y++) for (x = 0; x < w; x++) univ[y][x] = new[y][x];
}
 
 
 
void game(int w, int h, int t, int threads) {
	int x,y,z;
	unsigned univ[h][w];
	struct timeval start, end;
	double tot_time = 0.;

	//initialization
	for (x = 0; x < w; x++) for (y = 0; y < h; y++) univ[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;
	
	if (x > 1000) printbig(univ, w, h,0);
	
	for(z = 0; z < t;z++) {
		if (x <= 1000) show(univ, w, h);
		
		// get starting time at iteration z
		gettimeofday(&start, NULL);
		
		// lets evolve the current generation
		evolve(univ, w, h);

		// get ending time of iteration z
		gettimeofday(&end, NULL);
		tot_time += elapsed_wtime(start, end);
		
		if (x > 1000) {
			
		    printf("Iteration %d is : %ld ms\n", z,
		       ((end.tv_sec * 1000000 + end.tv_usec) - 
		       (start.tv_sec * 1000000 + start.tv_usec))/1000 );
		}
	}
	if (x > 1000) printbig(univ, w, h,1);

    // Allocates storage
	char *fileName = (char*)malloc(50 * sizeof(char));
	sprintf(fileName, "Serial-%d-%d-%d.txt", w, h, t);

	writeFile(fileName, true, tot_time, threads);



    
}
 
 
 
int main(int c, char **v) {
    
	int w = 0, h = 0, t = 0, threads = 1;
    // first parameter WIDTH
	if (c > 1) w = atoi(v[1]);

    // second parameter HEIGTH
	if (c > 2) h = atoi(v[2]);

    // third parameter TIME
	if (c > 3) t = atoi(v[3]);

    // fourth parameter Number of threads/core
    if (c > 4) threads = atoi(v[4]);

	if (w <= 0) w = 30;
	if (h <= 0) h = 30;
	if (t <= 0) t = 100;
	game(w, h, t, threads);
}
