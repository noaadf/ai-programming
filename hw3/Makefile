CXX = g++
NVCC = /usr/local/cuda-12.8/bin/nvcc
CXXFLAGS = -std=c++17 -Wall
NVCCFLAGS = -std=c++17 -arch=sm_80

SRC_CPP = main.cpp tensor.cpp activation.cpp
SRC_CU  = layers.cu

OBJ = $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o) gemm.o

main: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -lcublas -lcurand -o main $(OBJ)

gemm.o: gemm.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) main
