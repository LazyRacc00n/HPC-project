EXE_OMP:=$(basename $(filter-out omp-bug1.c omp-bug2.c, $(wildcard omp-*.c)))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_SERIAL:=trap odd-even prefix-sum matmul-plain matmul-transpose game-of-life reduction vla-demo
EXE_SIMD:=$(basename $(wildcard simd-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_CUDA)
DISTFILES:=$(wildcard *.c *.h *.cu *.sh *.ods) Makefile README cat-1344.pgm
DISTNAME:=HPC2021
CFLAGS+=-std=c99 -Wall -Wpedantic
NVCC?=nvcc
MPICC?=mpicc
VCFLAGS+=--gpu-architecture compute_30	

ALL: mpi openmp serial simd cuda

dist: $(DISTNAME).zip

$(DISTNAME).zip: $(DISTFILES)
	rm -r -f $(DISTNAME)
	mkdir $(DISTNAME)
	cd $(DISTNAME)
	ln $(DISTFILES) $(DISTNAME)
	zip -r $(DISTNAME).zip $(DISTNAME)

gfx.o: gfx.c gfx.h

omp-mandelbrot: omp-mandelbrot.o gfx.o
omp-mandelbrot: LDLIBS+=-lX11

matmul-plain: LDLIBS+=-lrt	
matmul-plain: matmul.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

matmul-transpose: CFLAGS+=-DTRANSPOSE
matmul-transpose: matmul.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

$(EXE_OMP): CFLAGS+=-fopenmp 
$(EXE_OMP): LDLIBS+=-lrt
openmp: $(EXE_OMP)

$(EXE_MPI): CC=$(MPICC)
mpi: $(EXE_MPI)

mpi-vecsum mpi-vecsum2: LDLIBS+=-lm

serial: $(EXE_SERIAL)

$(EXE_SIMD): CFLAGS+=-march=native
simd: $(EXE_SIMD)

simd-vsum-vector: CFLAGS+=-lm

simd-vsum-intrinsics: CFLAGS+=-lm

simd-vsum-auto: CFLAGS+=-lm -O2 -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized

simd-vsum-auto.s: simd-vsum-auto.c
	gcc -S -c -fverbose-asm -g -O2 -march=native -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized $< -o $@

simd-vsum-auto.lst: simd-vsum-auto.s
	as -alhnd $< > $@

cuda: $(EXE_CUDA)

cuda-rotate cuda-matmul: NVCFLAGS+=-lm

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

clean:
	\rm -f $(EXE) *.o *~ *.pbm *.s *.lst

distclean: clean
	\rm -r -f $(DISTNAME) $(DISTNAME).zip
