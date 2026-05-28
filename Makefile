NVCC ?= nvcc
CUDA_ARCH ?= sm_89
BUILD_DIR ?= build

NVFLAGS ?= --use_fast_math -std=c++17 -O3 -gencode arch=compute_89,code=$(CUDA_ARCH) -cudart static -Xptxas -v

INCLUDES := -Iinclude
GPU_SOURCES := main.cu beam_gpu_base.cu beam_gpu_rabitq_bitplane_navl2_prune_es_spec_auto.cu

.PHONY: all gpu clean

all: gpu

gpu:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o $(BUILD_DIR)/beam_search_gpu $(GPU_SOURCES)

clean:
	rm -rf $(BUILD_DIR)
