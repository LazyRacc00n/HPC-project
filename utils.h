#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h> // boolean type

double elapsed_wtime(struct timeval , struct timeval );
void show(void *u, int w, int h);
void printbig(void *u, int w, int h, int z);
void writeFile(char* fileName, bool first, double time , int n_core);