##################################################################################
# Makefile for all versions (serial, vectorized, OMP, MPI, CUDA) of Game of Life #
##################################################################################

# Intel C compiler 
CC = icc 

# Intel C compiler, with MPI support
MPICC = mpiicc

# OMP flag 
OMP_FLAGS = -qopenmp

# Vectorization flags
VEC_FLAGS = -O3 -ipo -xHost

# OpenMP files
OMP_FILES = experiment01.c experiment02.c

omp: $(OMP_FILES) utils.c
	$(CC) $(OMP_FLAGS) -o OpenMP/$(OMP_FILES) 