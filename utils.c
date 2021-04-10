/*
 * 
 *
 * Functions to save files about the time execution
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

// compute the elapsed wall-clock time between two time intervals. in ms
double elapsed_wtime(struct timeval start, struct timeval end) {
    return (double) ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
}


void writeFile(char* fileName, bool first, double time , int n_core){
    FILE *f;


    if(first)   f = fopen(fileName, "w" );
    else f = fopen(fileName, "a" ); 

    // write file
    fprintf(f,"%d,%ld",n_core , time);

    fprintf(f,"\n");
	fflush(f);
	fclose(f);
    
}

