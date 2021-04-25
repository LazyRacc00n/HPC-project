
#------------------------------------COMPILER--------------------------------

# Intel C compiler 
ICC = icc 

# Intel C compiler, with MPI support
MPICC = mpiicc


# OMP flag 
OMP_FLAGS = -qopenmp

#----------------------FLAGS-------------------------------
# Vectorization flags
VEC_FLAGS = -O3 -ipo -xHost

#-------------------DIRECTORY------------------

OPENMP_DIR = OpenMP
MPI_DIR = MPI

#-------------------FILES------------------
OMP_FILE = experiment03.c
MPI_FILE = main_MPI.c
MPI_UTILS_FILE = mpi_utils.c
UTILS_FILE = utils.c
SERIAL_FILE = glife_sequential.c

#bin location after compilation
BIN = bin

#---------------------------------- COMPILATION -----------------------------------------


all: bin_dir serial omp mpi 

bin_dir:
	mkdir -p $(BIN)

omp: $(OPENMP_DIR)/$(OMP_FILE) 
	$(ICC) $(OPENMP_DIR)/$(OMP_FILE) $(OMP_FLAGS) -o $(BIN)/gol_omp 

mpi: $(MPI_DIR)/$(MPI_FILE)
	$(MPICC) $(MPI_DIR)/$(MPI_FILE) $(UTILS_FILE) $(MPI_DIR)/$(MPI_UTILS_FILE) -o $(BIN)/gol_mpi 

serial: $(SERIAL_FILE) 
	$(ICC) $(SERIAL_FILE) -o $(BIN)/$gol_serial 




