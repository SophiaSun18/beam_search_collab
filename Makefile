NVCC      := nvcc
TARGET    := beam_search_gpu.out

ARCH      = sm_89
CXXSTD    := c++20

NVFLAGS   := -std=$(CXXSTD) -O3 -arch=$(ARCH)
INCLUDES  := -I.

SOURCES   := main.cu beam_gpu_base.cu
HEADERS   := index.hpp utils.cuh search_bitonic.cuh

.PHONY: all debug clean test_siftsmall test_sift1m

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -o $@ $(SOURCES)

debug:
	$(NVCC) -std=$(CXXSTD) -g -G -arch=$(ARCH) $(INCLUDES) \
	        -o $(TARGET)_debug $(SOURCES)

clean:
	rm -f $(TARGET) $(TARGET)_debug

SIFTSMALL_DIR := data/siftsmall
SIFT1M_DIR    := data/sift
K 			  := 100

test_siftsmall: $(TARGET)
	./$(TARGET) $(SIFTSMALL_DIR)/siftsmall_base.fvecs \
	            $(SIFTSMALL_DIR)/siftsmall_query.fvecs \
	            $(SIFTSMALL_DIR)/siftsmall_groundtruth.ivecs \
	            $(SIFTSMALL_DIR)/graph-d32 $(K)

test_sift1m: $(TARGET)
	./$(TARGET) $(SIFT1M_DIR)/sift_base.fvecs \
	            $(SIFT1M_DIR)/sift_query.fvecs \
	            $(SIFT1M_DIR)/sift_groundtruth.ivecs \
	            $(SIFT1M_DIR)/graph-d32 $(K)
