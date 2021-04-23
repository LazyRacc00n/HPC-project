
#------------------------------------COMPILER--------------------------------

# Intel C compiler 
ICC = icc 

# Intel C compiler, with MPI support
MPICC = mpiicc

# cuda compiler
NVCC = nvcc

# OMP flag 
OMP_FLAGS = -qopenmp

#----------------------FLAGS-------------------------------
# Vectorization flags
VEC_FLAGS = -O3 -ipo -xHost

# OpenMP files
OMP_FILES = experiment03.c

omp: $(OMP_FILES) utils.c
	$(ICC) $(OMP_FLAGS) -o OpenMP/$(OMP_FILES) 