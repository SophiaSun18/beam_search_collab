CXX ?= g++
NVCC ?= nvcc

CXXFLAGS ?= -Wall -Wextra -Ofast -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions -DEIGEN_DONT_PARALLELIZE $(EXTRA_CXXFLAGS)
ARCH ?= sm_89
CXXSTD ?= c++20
NVFLAGS ?= -std=$(CXXSTD) -O3 -arch=$(ARCH) $(EXTRA_NVFLAGS)
PTXAS_FLAGS ?= -Xptxas -v
INCLUDES := -I.

BIN_DIR := bin
CPU_TARGETS := distance_cpu replay_cpu
GPU_TARGETS := distance_gpu replay_gpu

$(shell mkdir -p $(BIN_DIR))

.PHONY: all cpu gpu clean

all: cpu gpu

cpu: $(CPU_TARGETS)

gpu: $(GPU_TARGETS)

distance_cpu: src/distance_cpu.cpp include/*.h include/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
	mv $@ $(BIN_DIR)/

replay_cpu: src/replay_cpu.cpp include/*.h include/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
	mv $@ $(BIN_DIR)/

distance_gpu: distance_gpu.cu include/*.h include/*.hpp include/*.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $<
	mv $@ $(BIN_DIR)/

replay_gpu: replay_gpu.cu include/*.h include/*.hpp include/*.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $<
	mv $@ $(BIN_DIR)/

clean:
	rm -f $(CPU_TARGETS) $(GPU_TARGETS) $(addprefix $(BIN_DIR)/,$(CPU_TARGETS) $(GPU_TARGETS)) *.o
