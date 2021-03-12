# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_70 -code=sm_70
NVCCFLAGS = -O3 -arch=compute_70 -code=sm_70
LIBS =

TARGETS = serial gpu autograder

all:	$(TARGETS)

serial: serial.o uncommon.o
	$(CC) -o $@ $(LIBS) serial.o uncommon.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

serial.o: serial.cpp uncommon.h
	$(CC) -c $(CFLAGS) serial.cpp
autograder.o: autograder.cu common.h
	$(CC) -c $(CFLAGS) autograder.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu
uncommon.o: uncommon.cpp uncommon.h
	$(CC) -c $(CFLAGS) uncommon.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
