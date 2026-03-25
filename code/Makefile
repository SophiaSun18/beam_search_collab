NVCC = nvcc
NVFLAGS = -std=c++17 -O3
NVFLAGS += -gencode arch=compute_70,code=sm_70    # V100
NVFLAGS += -gencode arch=compute_75,code=sm_75    # RTX 2080
NVFLAGS += -gencode arch=compute_80,code=sm_80    # A100
NVFLAGS += -gencode arch=compute_86,code=sm_86    # RTX 3090
NVFLAGS += -gencode arch=compute_89,code=sm_89    # RTX 4090

# NVFLAGS = -std=c++17 -O3 -arch=sm_86

INCLUDES = -I.

BUILD_DIR ?= ../build
TARGET_NAME = beam_search_gpu
TARGET = $(BUILD_DIR)/$(TARGET_NAME)
DEBUG_TARGET = $(BUILD_DIR)/$(TARGET_NAME)_debug

SOURCES = main.cu beam_gpu_base.cu
HEADERS = index.hpp search_bitonic.cuh utils.cuh $(wildcard search_bitonic/*.cuh) $(wildcard ../tools/*.hpp)

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o $@ main.cu beam_gpu_base.cu

sm86: $(TARGET)

sm80: $(TARGET)

sm75: $(TARGET)

sm89: $(TARGET)

debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -std=c++17 -g -G -arch=sm_86 $(INCLUDES) -o $@ main.cu beam_gpu_base.cu

clean:
	rm -f $(TARGET) $(DEBUG_TARGET)

test_siftsmall:
	$(TARGET) ../anns/data/siftsmall/siftsmall_base.fvecs \
	            ../anns/data/siftsmall/siftsmall_query.fvecs \
	            ../anns/data/siftsmall/siftsmall_groundtruth.ivecs \
	            ../anns/data/siftsmall/graph-d32 100

test_sift1m:
	$(TARGET) ../anns/data/sift/sift_base.fvecs \
	            ../anns/data/sift/sift_query.fvecs \
	            ../anns/data/sift/sift_groundtruth.ivecs \
	            ../anns/data/sift/graph-d32 100

# ==================== Telerun Server Tests ====================
# Data path on telerun server: /home/cxh/telerun/executor/data/
# Usage: make test_siftsmall_telerun or make test_sift1m_telerun

TELERUN_DATA = /home/cxh/telerun/executor/data

test_siftsmall_telerun:
	$(TARGET) $(TELERUN_DATA)/siftsmall/siftsmall_base.fvecs \
	            $(TELERUN_DATA)/siftsmall/siftsmall_query.fvecs \
	            $(TELERUN_DATA)/siftsmall/siftsmall_groundtruth.ivecs \
	            $(TELERUN_DATA)/siftsmall/graph-d32 100

test_sift1m_telerun:
	$(TARGET) $(TELERUN_DATA)/sift/sift_base.fvecs \
	            $(TELERUN_DATA)/sift/sift_query.fvecs \
	            $(TELERUN_DATA)/sift/sift_groundtruth.ivecs \
	            $(TELERUN_DATA)/sift/graph-d32 100

# RTX 4000 Ada (same as RTX 4090)
rtx4000: sm89

# Quadro RTX 4000 (Turing)
rtx4000_turing: sm75

.PHONY: all clean debug test_siftsmall test_sift1m test_siftsmall_telerun test_sift1m_telerun sm80 sm86 sm75 sm89 rtx4000 rtx4000_turing
