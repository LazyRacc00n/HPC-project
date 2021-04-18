/*
 * 
 *
 * Functions to save files about the time execution
 *
 */


#include "utils.h"
#include <stdbool.h>

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



// compute the elapsed wall-clock time between two time intervals. in ms
double elapsed_wtime(struct timeval start, struct timeval end) {

    return (double)((end.tv_sec * 1000000 + end.tv_usec) - 
		       (start.tv_sec * 1000000 + start.tv_usec))/1000;

   
}


void writeFile(char* fileName, bool first, double time , int n_core){
    FILE *f;


    if(first)   f = fopen(fileName, "w" );
    else f = fopen(fileName, "a" ); 

    // write file
    fprintf(f,"%d,%f",n_core , time);

    fprintf(f,"\n");
	fflush(f);
	fclose(f);
    
}

