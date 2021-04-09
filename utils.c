/*
 * 
 *
 * Functions to save files about the time execution
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


void writeFile(char fileName[], ){
    FILE *f;

    f = fopen(fileName, "w" );

    // write file
    fprintf(f,"");

    fprintf(f,"\n");
	fflush(f);
	fclose(f);
}