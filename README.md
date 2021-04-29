# HIGH PERFOMANCE COMPUTING PROJECT

Parallelization of the Game of Life  in openMP, MPI and CUDA


# Compilation and Execution

## With Makefile and Script

</br>

#### Compilation:
</br>
In the folder <i>Project</i> there is the makefile that compiles the OpenMP, MPI, the sequential version repectively in the folder <i>OpenMP</i>, <i> MPI</i> and <i>Project</i>. So, is possible to compile these file with the <i>make</i> command from bash. All the bin files will be located in the folder <i>bin</i>.
</br></br>

#### Execution:
</br>
In each folder <i> MPI</i>,<i>OpenMP</i> and <i> Project </i> there are the scripts bash that execute the program with the parameters mentioned in the <i> report </i>. The MPI script doesn't shows the evolution also for the small dimensions ( number of columns < 1000 ). If you want see the evolution on the terminal for small dimentions, change in the MPI script, the flag <i>not_show_evolution</i> from 1 to 0.</br>
To execute with the script, in the folder in which is located the script:</br></br>

```
bash name_of_the_script.sh
```
</br>

## Without Makefile and Script
</br>
In the folder in which are contained the files, <i>Project</i> for the serial, <i>MPI</i>, <i>OpenMp</i> and <i>CUDA</i> for the other.</br></br>
<b>Common execution parameters: </b></br>

- <i> nRows, nCols </i> are respectively the number of rows and columns of the Game of Life grid. </br> 
- <i> timesteps</i> are the number of time step with which execute the game.
</br></br>


## Serial:
</br>

### Compilation:
</br>

##### No Vectorization:

```
icc -O0 glife_sequential.c -o gol_sequntial
```

##### With Vectorization:

```
icc -O3 -ipo -xHost glife_sequential.c -o gol_sequntial
```

### Execution:

```
./gol_sequential nRows nCols timesteps
```


## OpenMP:
</br>

### Compilation:

```
icc experiment03.c -qopenmp -o gol_omp
```

### Execution:

```
./gol_omp nRows nCols timesteps number_of_threads
```

## MPI:
</br>

#### Compilation:

```
mpiicc main_MPI.c ../utils.c mpi_utils.c -o gol_mpi
```

#### Execution:
</br>
<b> MPI execution parameters:</b> </br></br>

- host_list_n.txt is file that contains the list of the host used, and n must be replace with the number of nodes ( 1, 2, 4, 8). These files are located in the folder <i>MPI/hostfile</i>, and there is a file for each nuber of node.

- 	n_process_per_host  is the number of process that each node must executes.
-	tot_procesess is the total number of processes.
-	Version ( 1 or 2) is the display version used. The experiments are made with the version 2.
-	show_result is a flag that indicates if show or not the evolution of the game in the terminal for nCols lower than 1000. 0 to see the evolution on the terminal, 1 otherwise.
-	nNodes is the number of nodes of the cluster used, and must be the same of the hostfile choosen (1, 2, 4, 8).



```
mpiexec -hostfile host_list_n.txt -perhost n_process_per_host -np tot_procesesses ./gol_mpi nRows nCols timesteps version show_result nNodes
```
</br>


## CUDA:
</br>

#### Compilation:
```
nvcc  gol_cuda.cu -o gol_cuda
```
#### Execution:

In this case, the parameter <i>number_of_threads</i> must be a multiple of 32, and the max is 1024.

```
./gol_cuda nRows nCols number_of_threads
```