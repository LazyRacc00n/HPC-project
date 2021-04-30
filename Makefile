
#------------------------------------COMPILER--------------------------------

# Intel C compiler 
ICC = icc 

# MPI Intel C compiler
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

VECT_FLAG = -DVECT

#---------------------------------- COMPILATION -----------------------------------------


all: bin_dir serial_vect omp mpi serial_no_vect

bin_dir:
	mkdir -p $(BIN)

omp: $(OPENMP_DIR)/$(OMP_FILE) 
	$(ICC) $(OPENMP_DIR)/$(OMP_FILE) $(OMP_FLAGS) -o $(BIN)/gol_omp 

mpi: $(MPI_DIR)/$(MPI_FILE)
	$(MPICC) $(MPI_DIR)/$(MPI_FILE) $(UTILS_FILE) $(MPI_DIR)/$(MPI_UTILS_FILE) -o $(BIN)/gol_mpi 

serial_vect: $(SERIAL_FILE) 
	$(ICC) $(VECT_FLAG)  $(SERIAL_FILE) -o $(BIN)/gol_serial 

serial_no_vect: $(SERIAL_FILE)
	$(ICC) -O0 $(SERIAL_FILE) -o $(BIN)/gol_serial_no_vect



