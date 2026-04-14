# Compiler for CPU
CXX = g++
CXXFLAGS = -Wall -Wextra -Ofast -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions  -DEIGEN_DONT_PARALLELIZE -mavx512f

# Compiler for GPU
NVCC      := nvcc
ARCH      = sm_89
CXXSTD    := c++20
NVFLAGS   := -std=$(CXXSTD) -O3 -arch=$(ARCH)
PTXAS_FLAGS := -Xptxas -v
INCLUDES  := -I.
HEADERS_DIR := include/

# Directories
SRC_DIR = src
BIN_DIR = bin
$(shell mkdir -p $(BIN_DIR))   # ensure bin exists

# CPU source files
SRC_BASE = $(SRC_DIR)/beam_cpu_base.cpp
SRC_PATHW = $(SRC_DIR)/beam_cpu_pathw.cpp
SRC_RABITQ = $(SRC_DIR)/beam_cpu_rabitq.cpp
SRC_PREPROCESS_RABITQ = $(SRC_DIR)/preprocess_rabitq.cpp

# CPU executables
CPU_TARGETS = cpu_base cpu_pathw cpu_rabitq preprocess_rabitq

# GPU source files
GPU_SOURCES = main.cu beam_gpu_base.cu beam_gpu_rabitq.cu
GPU_HEADERS = $(HEADERS_DIR)index.hpp $(HEADERS_DIR)qg.hpp $(HEADERS_DIR)utils.h $(HEADERS_DIR)search_bitonic.cuh $(HEADERS_DIR)search_rabitq.cuh
GPU_TARGET = gpu_search
GPU_MICROBENCH_TARGET = gpu_microbenchmark

.PHONY: all cpu gpu clean

# --------------------------------------------
# Default target: build all CPU and GPU binaries
# --------------------------------------------
all: cpu gpu

# --------------------------------------------
# CPU build
# --------------------------------------------
cpu: $(CPU_TARGETS)

cpu_base: $(SRC_BASE)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

cpu_pathw: $(SRC_PATHW)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

cpu_rabitq: $(SRC_RABITQ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

preprocess_rabitq: $(SRC_PREPROCESS_RABITQ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

# --------------------------------------------
# GPU build
# --------------------------------------------
gpu: $(GPU_TARGET) $(GPU_MICROBENCH_TARGET)

$(GPU_TARGET): $(GPU_SOURCES) $(GPU_HEADERS)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $(GPU_SOURCES)
	mv $@ $(BIN_DIR)/

$(GPU_MICROBENCH_TARGET): gpu_microbenchmark.cu $(GPU_HEADERS)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $<
	mv $@ $(BIN_DIR)/

# --------------------------------------------
# Clean all binaries
# --------------------------------------------
clean:
	rm -f $(CPU_TARGETS) $(BIN_DIR)/cpu_base $(BIN_DIR)/cpu_pathw \
	      $(BIN_DIR)/$(GPU_TARGET) $(BIN_DIR)/$(GPU_MICROBENCH_TARGET) *.o
