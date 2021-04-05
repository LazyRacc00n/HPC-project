/****************************************************************************
 *
 * mpi-type-struct.c - Simple demo of the MPI_Type_struct function
 *
 * Based on https://computing.llnl.gov/tutorials/mpi/#Derived_Data_Types
 *
 * Copyright (C) 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-type-struct.c -o mpi-type-struct
 *
 * Run with:
 * mpirun -n 4 ./mpi-type-struct
 *
 ****************************************************************************/
#include <mpi.h>
#include <stdio.h>

#define NELEM 25

int main(int argc, char *argv[])  
{
    int my_rank, comm_sz, i;
    
    typedef struct {
        float x, y, z;
        float velocity;
        int  n, type;
    } particle_t;

    particle_t   particles[NELEM];
    MPI_Datatype particletype, oldtypes[2] = {MPI_FLOAT, MPI_INT};
    int          blklens[2] = {4, 2};    
    MPI_Aint     displs[2], lb, extent;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);    
    displs[0] = 0;
    
    displs[1] = blklens[0] * extent;
    printf("\n\ndispl[1]: %d\n\n", displs[1]);
    /* define structured type and commit it */
    MPI_Type_struct(2,                  /* count                        */
                    blklens,            /* array of blocklen            */
                    displs,             /* array of displacements       */
                    oldtypes,           /* array of types               */
                    &particletype);

    MPI_Type_commit(&particletype);
    
    /* task 0 initializes the particle array and then sends it to each task */
    if (0 == my_rank) {
        for (i=0; i<NELEM; i++) {
            particles[i].x = i * 1.0;
            particles[i].y = i * -1.0;
            particles[i].z = i * 1.0; 
            particles[i].velocity = 0.25;
            particles[i].n = i;
            particles[i].type = i % 2; 
        }
    }
 
    MPI_Bcast(particles,        /* buffer       */
              NELEM,            /* count        */
              particletype,     /* datatype     */
              0,                /* source       */
              MPI_COMM_WORLD    /* comm         */
              );
    
    printf("rank= %d   %3.2f %3.2f %3.2f %3.2f %d %d\n", my_rank,
           particles[3].x, particles[3].y, particles[3].z, 
           particles[3].velocity,
           particles[3].n, particles[3].type);
    
    /* free datatype when done using it */
    MPI_Type_free(&particletype);
    MPI_Finalize();
    return 0;
}
