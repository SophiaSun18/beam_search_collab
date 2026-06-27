# Compiler for CPU
CXX = g++
CXXFLAGS = -Wall -Wextra -Ofast -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions  -DEIGEN_DONT_PARALLELIZE -mavx512f $(EXTRA_CXXFLAGS)

# Compiler for GPU
NVCC      := nvcc
ARCH      = sm_89
CXXSTD    := c++20
NVFLAGS   := -std=$(CXXSTD) -O3 -arch=$(ARCH) $(EXTRA_NVFLAGS)
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
SRC_RABITQ_PRUNED = $(SRC_DIR)/beam_cpu_rabitq_pruned.cpp
SRC_PREPROCESS_RABITQ = $(SRC_DIR)/preprocess_rabitq.cpp
SRC_PREPARE_PATHW_SIGNBIT = $(SRC_DIR)/prepare_pathw_signbit.cpp

# CPU executables
CPU_TARGETS = cpu_base cpu_pathw cpu_rabitq cpu_rabitq_pruned preprocess_rabitq prepare_pathw_signbit

# GPU source files
GPU_HEADERS = $(HEADERS_DIR)index.hpp $(HEADERS_DIR)qg.hpp $(HEADERS_DIR)utils.h $(HEADERS_DIR)utils.cuh $(HEADERS_DIR)distance.cuh
GPU_MODE_TARGETS = gpu_base gpu_pathw gpu_rabitq gpu_rabitq_pruned
GPU_BASE_FLAGS = -DGPU_SEARCH_MODE=1
GPU_PATHW_FLAGS = -DGPU_SEARCH_MODE=2
GPU_RABITQ_FLAGS = -DGPU_SEARCH_MODE=3
GPU_RABITQ_PRUNED_FLAGS = -DGPU_SEARCH_MODE=4

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

cpu_rabitq_pruned: $(SRC_RABITQ_PRUNED)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

preprocess_rabitq: $(SRC_PREPROCESS_RABITQ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

prepare_pathw_signbit: $(SRC_PREPARE_PATHW_SIGNBIT)
	$(CXX) $(CXXFLAGS) $^ -o $@
	mv $@ $(BIN_DIR)/

# --------------------------------------------
# GPU build
# --------------------------------------------
gpu: $(GPU_MODE_TARGETS)

gpu_base: main.cu beam_gpu_base.cu $(GPU_HEADERS) $(HEADERS_DIR)search_bitonic.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) $(GPU_BASE_FLAGS) -o $@ main.cu beam_gpu_base.cu
	mv $@ $(BIN_DIR)/

gpu_pathw: main.cu beam_gpu_pathw.cu $(GPU_HEADERS) $(HEADERS_DIR)search_pathw.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) $(GPU_PATHW_FLAGS) -o $@ main.cu beam_gpu_pathw.cu
	mv $@ $(BIN_DIR)/

gpu_rabitq: main.cu beam_gpu_rabitq.cu $(GPU_HEADERS) $(HEADERS_DIR)search_rabitq.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) $(GPU_RABITQ_FLAGS) -o $@ main.cu beam_gpu_rabitq.cu
	mv $@ $(BIN_DIR)/

gpu_rabitq_pruned: main.cu beam_gpu_rabitq_pruned.cu $(GPU_HEADERS) $(HEADERS_DIR)search_rabitq_pruned.cuh $(HEADERS_DIR)search_rabitq_pruned_spec.cuh $(HEADERS_DIR)search_rabitq_pruned_spec_adaptive.cuh $(HEADERS_DIR)search_rabitq_pruned_spec_adaptive_config.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) $(GPU_RABITQ_PRUNED_FLAGS) -o $@ main.cu beam_gpu_rabitq_pruned.cu
	mv $@ $(BIN_DIR)/

# --------------------------------------------
# Clean all binaries
# --------------------------------------------
clean:
	rm -f $(CPU_TARGETS) $(addprefix $(BIN_DIR)/,$(CPU_TARGETS)) \
	      $(GPU_MODE_TARGETS) $(addprefix $(BIN_DIR)/,$(GPU_MODE_TARGETS)) \
	      gpu_search $(BIN_DIR)/gpu_search *.o
