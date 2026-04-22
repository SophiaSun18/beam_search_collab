/// @brief GPU microbenchmarks for performance-critical search functions
/// @detail Benchmarks warp_l2_distance, estimated_distance_gpu, \
/// tensor-core MMA estimation, topk_by_bitonic_sort, \
/// hashtable_insert, and pick_expanders individually. \
/// Launches many blocks to saturate the GPU, reports min/avg/max per-call ns.

#define QG_BQUERY 6
#include "include/search_tensor_rabitq.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int BDIM   = 128;
static constexpr int BPAD   = 128;
static constexpr int NREP   = 10000;
static constexpr int NCAND  = 32;
static constexpr int BWORDS = BPAD / 64;
static constexpr int NBENCH = 6;

/// @brief benchmark warp_l2_distance latency under full GPU load
/// @param qry query vector [BDIM]
/// @param dat data vectors [NCAND * BDIM]
/// @param out per-block output: out[blockIdx.x] = total ns
__global__ void benchL2(const float* __restrict__ qry,
                        const float* __restrict__ dat,
                        gpu_long_t* out) {
    int tid = threadIdx.x;

    __shared__ float sq[BDIM];
    for (int i = tid; i < BDIM; i += blockDim.x)
        sq[i] = qry[i];
    __syncthreads();

    float acc = 0;
    for (int i = 0; i < 100; i++)
        acc += warp_l2_distance(BDIM, sq, dat + (i % NCAND) * BDIM);
    __syncthreads();

    gpu_long_t t0 = gpu_globaltimer();
    for (int i = 0; i < NREP; i++)
        acc += warp_l2_distance(BDIM, sq, dat + (i % NCAND) * BDIM);
    __syncthreads();
    gpu_long_t t1 = gpu_globaltimer();

    if (tid == 0) {
        out[blockIdx.x] = t1 - t0;
        if (acc == -999.0f) out[blockIdx.x] = 0;
    }
}

/// @brief benchmark estimated_distance_gpu (RaBitQ) under full GPU load
/// @param codes packed binary codes [NCAND * BWORDS]
/// @param out per-block output: out[blockIdx.x] = total ns
__global__ void benchRq(const uint64_t* __restrict__ codes,
                        gpu_long_t* out) {
    int tid = threadIdx.x;

    __shared__ uint8_t qbuf[BPAD];
    __shared__ uint64_t cbuf[NCAND * BWORDS];
    __shared__ QueryFactors qf;

    if (tid == 0) {
        qf.quantized_query = qbuf;
        qf.sum_q  = BPAD * 8;
        qf.width  = 0.1f;
        qf.low_val = -1.0f;
    }
    for (int i = tid; i < BPAD; i += blockDim.x)
        qbuf[i] = static_cast<uint8_t>(i % 16);
    for (int i = tid; i < NCAND * BWORDS; i += blockDim.x)
        cbuf[i] = codes[i];
    __syncthreads();

    float acc = 0;
    for (int i = 0; i < 100; i++)
        acc += estimated_distance_gpu(qf, cbuf + (i % NCAND) * BWORDS,
                                      1.0f, 0.5f, 0.3f, 10.0f, BPAD);
    __syncthreads();

    gpu_long_t t0 = gpu_globaltimer();
    for (int i = 0; i < NREP; i++)
        acc += estimated_distance_gpu(qf, cbuf + (i % NCAND) * BWORDS,
                                      1.0f, 0.5f, 0.3f, 10.0f, BPAD);
    __syncthreads();
    gpu_long_t t1 = gpu_globaltimer();

    if (tid == 0) {
        out[blockIdx.x] = t1 - t0;
        if (acc == -999.0f) out[blockIdx.x] = 0;
    }
}

/// @brief benchmark tensor-core MMA tile estimation under full GPU load
/// @param codes packed binary codes [NCAND * BWORDS] for unpacking
/// @param out per-block output: out[blockIdx.x] = total ns
__global__ void benchTc(const uint64_t* __restrict__ codes,
                        gpu_long_t* out) {
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;

    __shared__ uint8_t qbuf[BPAD];
    __shared__ QueryFactors qf;
    __shared__ INDEX_T cidx[CANDIDATE_BUFFER_SIZE];
    __shared__ uint64_t cbuf[NCAND * BWORDS];
    __shared__ int8_t dcodes[TENSOR_RABITQ_BATCH * BPAD];
    __shared__ int8_t qmat[BPAD * TENSOR_RABITQ_TILE];
    __shared__ int mout[TENSOR_RABITQ_BATCH * TENSOR_RABITQ_TILE];

    if (tid == 0) {
        qf.quantized_query = qbuf;
        qf.sum_q  = BPAD * 8;
        qf.width  = 0.1f;
        qf.low_val = -1.0f;
    }
    for (int i = tid; i < BPAD; i += blockDim.x)
        qbuf[i] = static_cast<uint8_t>(i % 16);
    for (int i = tid; i < NCAND * BWORDS; i += blockDim.x)
        cbuf[i] = codes[i];
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x)
        cidx[i] = (i < NCAND) ? i : MAX_INDEX;
    for (int i = tid; i < TENSOR_RABITQ_BATCH * TENSOR_RABITQ_TILE; i += blockDim.x)
        mout[i] = 0;
    __syncthreads();

    prepare_tensor_query_matrix_gpu(qf, BPAD, qmat);
    __syncthreads();

    for (int i = 0; i < 100; i++) {
        unpack_candidate_tile_gpu(cidx, cbuf, BWORDS, BPAD, 0, dcodes);
        __syncthreads();
        if (wid == 0)
            estimate_distance_tensor_tile_gpu(dcodes, qmat, BPAD, mout);
        __syncthreads();
    }

    gpu_long_t t0 = gpu_globaltimer();
    for (int i = 0; i < NREP; i++) {
        unpack_candidate_tile_gpu(cidx, cbuf, BWORDS, BPAD, 0, dcodes);
        __syncthreads();
        if (wid == 0)
            estimate_distance_tensor_tile_gpu(dcodes, qmat, BPAD, mout);
        __syncthreads();
    }
    gpu_long_t t1 = gpu_globaltimer();

    if (tid == 0) {
        out[blockIdx.x] = t1 - t0;
    }
}

/// @brief benchmark topk_by_bitonic_sort under full GPU load
/// @param seed random distances for candidate reinit [CANDIDATE_BUFFER_SIZE]
/// @param out per-block output: out[blockIdx.x] = total ns
__global__ void benchSt(const float* __restrict__ seed,
                        gpu_long_t* out) {
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;

    __shared__ INDEX_T idx[RESULT_BUFFER_SIZE];
    __shared__ DISTANCE_T dst[RESULT_BUFFER_SIZE];

    for (int i = tid; i < RESULT_BUFFER_SIZE; i += blockDim.x) {
        idx[i] = i;
        dst[i] = seed[i % CANDIDATE_BUFFER_SIZE];
    }
    __syncthreads();

    topk_128_by_bitonic_sort<2, INTERNAL_TOPK / WARP_SIZE>(
        idx, dst, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, true);
    __syncthreads();

    gpu_long_t t0 = gpu_globaltimer();
    for (int i = 0; i < NREP; i++) {
        if (wid == 0) {
            for (int j = lid; j < CANDIDATE_BUFFER_SIZE; j += WARP_SIZE) {
                idx[INTERNAL_TOPK + j] = INTERNAL_TOPK + j + i;
                dst[INTERNAL_TOPK + j] = seed[(j + i) % CANDIDATE_BUFFER_SIZE];
            }
        }
        __syncthreads();
        topk_128_by_bitonic_sort<2, INTERNAL_TOPK / WARP_SIZE>(
            idx, dst, RESULT_BUFFER_SIZE,
            CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, false);
        __syncthreads();
    }
    gpu_long_t t1 = gpu_globaltimer();

    if (tid == 0) {
        out[blockIdx.x] = t1 - t0;
    }
}

/// @brief benchmark hashtable_insert under full GPU load
/// @param out per-block output: out[blockIdx.x] = total ns
__global__ void benchHt(gpu_long_t* out) {
    int tid = threadIdx.x;

    extern __shared__ char smem[];
    INDEX_T* ht = reinterpret_cast<INDEX_T*>(smem);

    hashtable_init(ht, BITLEN);
    __syncthreads();

    for (int i = 0; i < 100; i++) {
        if (i % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(ht, BITLEN);
            __syncthreads();
        }
        hashtable_insert(ht, BITLEN, tid + i * blockDim.x);
        __syncthreads();
    }

    hashtable_init(ht, BITLEN);
    __syncthreads();

    gpu_long_t t0 = gpu_globaltimer();
    for (int i = 0; i < NREP; i++) {
        if (i % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(ht, BITLEN);
            __syncthreads();
        }
        hashtable_insert(ht, BITLEN, tid + i * blockDim.x);
        __syncthreads();
    }
    gpu_long_t t1 = gpu_globaltimer();

    if (tid == 0) {
        out[blockIdx.x] = t1 - t0;
    }
}

/// @brief benchmark pick_expanders under full GPU load
/// @param out per-block output: out[blockIdx.x] = total ns
__global__ void benchPe(gpu_long_t* out) {
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;

    __shared__ INDEX_T topk[INTERNAL_TOPK];
    __shared__ INDEX_T plist[SEARCH_WIDTH];

    for (int i = tid; i < INTERNAL_TOPK; i += blockDim.x)
        topk[i] = i;
    __syncthreads();

    if (wid == 0) {
        for (int i = 0; i < 100; i++) {
            pick_expanders(SEARCH_WIDTH, plist, INTERNAL_TOPK, topk);
            for (int j = tid; j < INTERNAL_TOPK; j += WARP_SIZE)
                topk[j] &= 0x7fffffff;
        }
    }
    __syncthreads();

    for (int i = tid; i < INTERNAL_TOPK; i += blockDim.x)
        topk[i] = i;
    __syncthreads();

    gpu_long_t t0 = gpu_globaltimer();
    if (wid == 0) {
        for (int i = 0; i < NREP; i++) {
            pick_expanders(SEARCH_WIDTH, plist, INTERNAL_TOPK, topk);
            for (int j = tid; j < INTERNAL_TOPK; j += WARP_SIZE)
                topk[j] &= 0x7fffffff;
        }
    }
    __syncthreads();
    gpu_long_t t1 = gpu_globaltimer();

    if (tid == 0) {
        out[blockIdx.x] = t1 - t0;
    }
}

/// @brief fill float vector with pseudo-random values in [0, 1]
/// @param v target vector
/// @param n number of elements to generate
void fillRnd(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; i++)
        v[i] = static_cast<float>(rand()) / RAND_MAX;
}

/// @brief compute min/avg/max from per-block timings, print one table row
/// @param name function name for display
/// @param buf per-block ns values
/// @param nblk number of blocks
void report(const char* name, const gpu_long_t* buf, int nblk) {
    gpu_long_t mn = buf[0], mx = buf[0];
    double sum = 0;
    for (int i = 0; i < nblk; i++) {
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
        sum += static_cast<double>(buf[i]);
    }
    double avg = sum / nblk;
    printf("%-32s %8.1f %8.1f %8.1f %8d\n",
           name, mn / (double)NREP, avg / NREP, mx / (double)NREP, nblk);
}

int main(int argc, char** argv) {
    srand(42);

    cudaDeviceProp prop;
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    int nsm = prop.multiProcessorCount;

    // default: 4 blocks per SM to saturate GPU; override via argv[1]
    int scale = 4;
    if (argc > 1) scale = atoi(argv[1]);
    int nblk = nsm * scale;

    printf("Device: %s  (SMs: %d)\n", prop.name, nsm);
    printf("Config: dim=%d, reps=%d, block=%d, grid=%d (%d blocks/SM)\n\n",
           BDIM, NREP, BLOCK_SIZE, nblk, scale);

    // synthetic data
    std::vector<float> hqry(BDIM);
    std::vector<float> hdat(NCAND * BDIM);
    std::vector<float> hseed(CANDIDATE_BUFFER_SIZE);
    fillRnd(hqry, BDIM);
    fillRnd(hdat, NCAND * BDIM);
    fillRnd(hseed, CANDIDATE_BUFFER_SIZE);

    std::vector<uint64_t> hcodes(NCAND * BWORDS);
    for (auto& c : hcodes)
        c = static_cast<uint64_t>(rand())
          | (static_cast<uint64_t>(rand()) << 32);

    // device allocations
    float *dqry, *ddat, *dseed;
    uint64_t *dcodes;
    gpu_long_t *dout;

    size_t qsz = BDIM * sizeof(float);
    size_t dsz = NCAND * BDIM * sizeof(float);
    size_t ssz = CANDIDATE_BUFFER_SIZE * sizeof(float);
    size_t csz = NCAND * BWORDS * sizeof(uint64_t);
    size_t osz = static_cast<size_t>(nblk) * sizeof(gpu_long_t);

    CUDA_SAFE_CALL(cudaMalloc(&dqry,   qsz));
    CUDA_SAFE_CALL(cudaMalloc(&ddat,   dsz));
    CUDA_SAFE_CALL(cudaMalloc(&dseed,  ssz));
    CUDA_SAFE_CALL(cudaMalloc(&dcodes, csz));
    CUDA_SAFE_CALL(cudaMalloc(&dout,   osz));

    CUDA_SAFE_CALL(cudaMemcpy(dqry,   hqry.data(),   qsz, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ddat,   hdat.data(),   dsz, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dseed,  hseed.data(),  ssz, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dcodes, hcodes.data(), csz, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::vector<gpu_long_t> hout(nblk);
    uint32_t htsz = hashtable_getsize(BITLEN) * sizeof(INDEX_T);

    printf("%-32s %8s %8s %8s %8s\n",
           "Function", "min", "avg", "max", "blocks");
    printf("%-32s %8s %8s %8s %8s\n",
           "--------------------------------",
           "--------", "--------", "--------", "--------");

    // benchL2
    CUDA_SAFE_CALL(cudaMemset(dout, 0, osz));
    benchL2<<<nblk, BLOCK_SIZE>>>(dqry, ddat, dout);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(), dout, osz, cudaMemcpyDeviceToHost));
    report("warp_l2_distance", hout.data(), nblk);

    // benchRq
    CUDA_SAFE_CALL(cudaMemset(dout, 0, osz));
    benchRq<<<nblk, BLOCK_SIZE>>>(dcodes, dout);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(), dout, osz, cudaMemcpyDeviceToHost));
    report("estimated_distance_gpu", hout.data(), nblk);

    // benchTc
    CUDA_SAFE_CALL(cudaMemset(dout, 0, osz));
    benchTc<<<nblk, BLOCK_SIZE>>>(dcodes, dout);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(), dout, osz, cudaMemcpyDeviceToHost));
    report("tensor_mma_tile (unpack+mma)", hout.data(), nblk);

    // benchSt
    CUDA_SAFE_CALL(cudaMemset(dout, 0, osz));
    benchSt<<<nblk, BLOCK_SIZE>>>(dseed, dout);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(), dout, osz, cudaMemcpyDeviceToHost));
    report("topk_bitonic_sort", hout.data(), nblk);

    // benchHt
    CUDA_SAFE_CALL(cudaMemset(dout, 0, osz));
    benchHt<<<nblk, BLOCK_SIZE, htsz>>>(dout);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(), dout, osz, cudaMemcpyDeviceToHost));
    report("hashtable_insert", hout.data(), nblk);

    // benchPe
    CUDA_SAFE_CALL(cudaMemset(dout, 0, osz));
    benchPe<<<nblk, BLOCK_SIZE>>>(dout);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(), dout, osz, cudaMemcpyDeviceToHost));
    report("pick_expanders", hout.data(), nblk);

    CUDA_SAFE_CALL(cudaFree(dqry));
    CUDA_SAFE_CALL(cudaFree(ddat));
    CUDA_SAFE_CALL(cudaFree(dseed));
    CUDA_SAFE_CALL(cudaFree(dcodes));
    CUDA_SAFE_CALL(cudaFree(dout));

    return 0;
}
