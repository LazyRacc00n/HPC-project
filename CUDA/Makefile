# cuda compiler
NVCC = nvcc
# cuda file
CUDA_FILE = gol_cuda_server.cu

BIN = bin

EXECUTABLE = gol_cuda

all: bin_dir

bin_dir:
	mkdir -p $(BIN)

cuda: $(CUDA_FILE) 
	$(NVCC) $(CUDA_FILE) -o $(BIN)/$(EXECUTABLE) 