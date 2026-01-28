NVCC = nvcc
NVFLAGS = -std=c++17 -O3
NVFLAGS += -gencode arch=compute_70,code=sm_70    # V100
NVFLAGS += -gencode arch=compute_75,code=sm_75    # RTX 2080
NVFLAGS += -gencode arch=compute_80,code=sm_80    # A100
NVFLAGS += -gencode arch=compute_86,code=sm_86    # RTX 3090
NVFLAGS += -gencode arch=compute_89,code=sm_89    # RTX 4090

# NVFLAGS = -std=c++17 -O3 -arch=sm_86

INCLUDES = -I.

TARGET = beam_search_gpu

SOURCES = main.cu beam_gpu_base.cu
HEADERS = index.hpp search_bitonic.cuh utils.cuh

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o $@ main.cu beam_gpu_base.cu

sm86: $(SOURCES) $(HEADERS)
	$(NVCC) -std=c++17 -O3 -arch=sm_86 $(INCLUDES) -o $(TARGET) main.cu beam_gpu_base.cu

sm80: $(SOURCES) $(HEADERS)
	$(NVCC) -std=c++17 -O3 -arch=sm_80 $(INCLUDES) -o $(TARGET) main.cu beam_gpu_base.cu

sm75: $(SOURCES) $(HEADERS)
	$(NVCC) -std=c++17 -O3 -arch=sm_75 $(INCLUDES) -o $(TARGET) main.cu beam_gpu_base.cu

sm89: $(SOURCES) $(HEADERS)
	$(NVCC) -std=c++17 -O3 -arch=sm_89 $(INCLUDES) -o $(TARGET) main.cu beam_gpu_base.cu

debug: $(SOURCES) $(HEADERS)
	$(NVCC) -std=c++17 -g -G -arch=sm_86 $(INCLUDES) -o $(TARGET)_debug main.cu beam_gpu_base.cu

clean:
	rm -f $(TARGET) $(TARGET)_debug

test_siftsmall:
	./$(TARGET) ../anns/data/siftsmall/siftsmall_base.fvecs \
	            ../anns/data/siftsmall/siftsmall_query.fvecs \
	            ../anns/data/siftsmall/siftsmall_groundtruth.ivecs \
	            ../anns/data/siftsmall/graph-d32 100

test_sift1m:
	./$(TARGET) ../anns/data/sift/sift_base.fvecs \
	            ../anns/data/sift/sift_query.fvecs \
	            ../anns/data/sift/sift_groundtruth.ivecs \
	            ../anns/data/sift/graph-d32 100

# RTX 4000 Ada (same as RTX 4090)
rtx4000: sm89

# Quadro RTX 4000 (Turing)
rtx4000_turing: sm75

.PHONY: all clean debug test_siftsmall test_sift1m sm80 sm86 sm75 sm89 rtx4000 rtx4000_turing
