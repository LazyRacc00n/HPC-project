
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

#-------------------DIRECTORY------------------

OPENMP_DIR = OpenMP
CUDA_DIR = CUDA 
MPI_DIR = MPI

#-------------------FILES------------------
# OpenMP files
OMP_FILE = experiment03.c
MPI_FILE = main_MPI.c
CUDA_FILE = gol_cuda_serve.cu
UTILS_FILE = utils.c
SERIAL_FILE = glife.c

#bin location after compilation
BIN = bin

#---------------------------------- COMPILATION -----------------------------------------


all: bin_dir omp mpi cuda serial

bin_dir:
	mkdir -p $(BIN)

omp: $(OPENMP_DIR)/$(OMP_FILE) $(UTILS_FILE)
	$(ICC) $(OMP_FLAGS) -o BIN_DIR/gol_omp

mpi: $(MPI_DIR)/$(MPI_FILE) $(UTILS_FILE)
	$(MPI) -o $(BIN_DIR)/gol_mpi

cuda: $(CUDA_DIR)/$(CUDA_FILE) $(UTILS_FILE)
	$(NVCC) -o $(BIN_DIR)/$(OMP_FILES)

serial: $(SERIAL_FILE) $(UTILS_FILE)
	$(ICC) -o $(BIN_DIR)/$(OMP_FILES)
