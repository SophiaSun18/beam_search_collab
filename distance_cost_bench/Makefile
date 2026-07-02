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
TRACE_CPU_TARGETS := trace_cpu_base trace_cpu_rabitq
TRACE_GPU_TARGET := trace_gpu
TRACE_GPU_SOURCES := trace_gpu_main.cu trace_gpu_base.cu trace_gpu_rabitq.cu
TRACE_GPU_FLAGS ?= -DGPU_STEPEFF_ONLY -DGPU_ACCESS_TRACE=1

$(shell mkdir -p $(BIN_DIR))

.PHONY: all cpu gpu trace_cpu clean

all: cpu gpu

cpu: $(CPU_TARGETS)

gpu: $(GPU_TARGETS)

trace_cpu: $(TRACE_CPU_TARGETS)

distance_cpu: src/distance_cpu.cpp include/*.h include/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
	mv $@ $(BIN_DIR)/

replay_cpu: src/replay_cpu.cpp include/*.h include/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
	mv $@ $(BIN_DIR)/

trace_cpu_base: src/trace_cpu_base.cpp include/*.h include/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
	mv $@ $(BIN_DIR)/

trace_cpu_rabitq: src/trace_cpu_rabitq.cpp include/*.h include/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
	mv $@ $(BIN_DIR)/

distance_gpu: distance_gpu.cu include/*.h include/*.hpp include/*.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $<
	mv $@ $(BIN_DIR)/

replay_gpu: replay_gpu.cu include/*.h include/*.hpp include/*.cuh
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $<
	mv $@ $(BIN_DIR)/

$(TRACE_GPU_TARGET): $(TRACE_GPU_SOURCES) include/*.h include/*.hpp include/*.cuh
	$(NVCC) $(NVFLAGS) $(TRACE_GPU_FLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ $(TRACE_GPU_SOURCES)
	mv $@ $(BIN_DIR)/

clean:
	rm -f $(CPU_TARGETS) $(GPU_TARGETS) $(TRACE_CPU_TARGETS) $(TRACE_GPU_TARGET) \
	      $(addprefix $(BIN_DIR)/,$(CPU_TARGETS) $(GPU_TARGETS) $(TRACE_CPU_TARGETS) $(TRACE_GPU_TARGET)) *.o
