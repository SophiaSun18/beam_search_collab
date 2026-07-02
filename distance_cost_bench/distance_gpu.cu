// Part 1 microbench (GPU): per-distance cost, mirroring the REAL search form.
//   - rabitq: a full block (8 warps) scans all `max_degree` neighbors of an
//     "expander" with codes/factors staged into shared memory, exactly as in
//     search_rabitq.cuh; warps split neighbors (i = warp_id; i += WARPS_PER_BLOCK).
//   - exact L2: a full block scans `max_degree` rows with warp_distance, one
//     warp per row (mirrors the exact candidate / result-pool fill path).
// Each iteration uses a different expander from a pool of NEXP, to avoid an
// unrealistic all-cache-hit. Per-op ns = block_time / (iters * max_degree).
#define QG_BQUERY 6
#include "include/utils.cuh"
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef MB_DIM
#define MB_DIM 128
#endif
#ifndef MB_PADDED
#define MB_PADDED 128
#endif
#define MB_DEG    32
#define MB_NEXP   256
#define MB_ITERS  4000
#define MB_CODE_BATCHES ((MB_DEG + WARP_SIZE - 1) / WARP_SIZE)
#define MB_CODE_BYTES   (MB_CODE_BATCHES * MB_PADDED * 4)
#define MB_CODE_WORDS   (MB_CODE_BYTES / 8)

static_assert((MB_CODE_BYTES % 8) == 0, "RaBitQ staged code bytes must be uint64 aligned");

template <typename T>
static __device__ inline void mb_staged_copy_async_begin_or_sync(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        __pipeline_memcpy_async(dst + i, src + i, sizeof(T));
    }
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
#endif
}

static __device__ inline void mb_staged_copy_async_finish_or_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

__global__ void benchRq(const float* __restrict__ q, const float* __restrict__ signs,
                         const uint64_t* __restrict__ pool_codes,
                         const float* __restrict__ pool_factors,
                         gpu_long_t* out) {
    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane_id = tid % WARP_SIZE;
    extern __shared__ char smem[];
    float*  rotated   = (float*)smem;
    uint8_t* quant    = (uint8_t*)(rotated + MB_PADDED);
    uint8_t* lut      = quant + MB_PADDED;
    uint64_t* st_code_words = reinterpret_cast<uint64_t*>(lut + (MB_PADDED << 2));
    uint8_t* st_codes = reinterpret_cast<uint8_t*>(st_code_words);
    float*   st_fac   = reinterpret_cast<float*>(st_code_words + MB_CODE_WORDS);
    __shared__ QueryFactors qf;
    if (tid == 0) { qf.rotated_query=rotated; qf.quantized_query=quant; qf.lut=lut;
                    qf.low_val=FLT_MAX; qf.high_val=-FLT_MAX; qf.width=0; qf.sum_q=0; }
    __syncthreads();
    query_prepare_gpu(q, qf, signs, MB_DIM, MB_PADDED);
    pack_lut_gpu(MB_PADDED, qf.quantized_query, qf.lut);
    __syncthreads();

    const int bpn = MB_PADDED >> 3;
    float acc = 0;
    long long t0 = clock64();
    for (int it = 0; it < MB_ITERS; it++) {
        const int e = it % MB_NEXP;
        const uint64_t* ecodes = pool_codes + (size_t)e * MB_CODE_WORDS;
        const float*   efac   = pool_factors + (size_t)e * 3 * MB_DEG;
        mb_staged_copy_async_begin_or_sync(st_code_words, ecodes, MB_CODE_WORDS);
        mb_staged_copy_async_begin_or_sync(st_fac, efac, 3 * MB_DEG);
        mb_staged_copy_async_finish_or_sync();
        __syncthreads();
        for (int i = warp_id; i < MB_DEG; i += WARPS_PER_BLOCK) {
            const int batch = i / WARP_SIZE, nidx = i & (WARP_SIZE - 1);
            const uint8_t* blk = st_codes + batch * MB_PADDED * 4;
            float d = scan_one_neighbor_lanes_gpu<32>(qf, blk, nidx,
                st_fac + i, st_fac + MB_DEG + i, st_fac + 2*MB_DEG + i,
                1.0f, MB_PADDED, bpn);
            if (lane_id == 0) acc += d;
        }
        __syncthreads();
    }
    long long t1 = clock64();
    if (tid == 0) out[blockIdx.x] = (acc == -1.0f) ? 0 : (gpu_long_t)(t1 - t0);
}

__global__ void benchL2(const float* __restrict__ q,
                         const float* __restrict__ pool_rows,
                         gpu_long_t* out) {
    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE;
    float acc = 0;
    long long t0 = clock64();
    for (int it = 0; it < MB_ITERS; it++) {
        const size_t base = (size_t)(it % MB_NEXP) * MB_DEG;
        for (int i = warp_id; i < MB_DEG; i += WARPS_PER_BLOCK) {
            acc += warp_distance(MB_DIM, q, pool_rows + (base + i) * MB_DIM, false);
        }
    }
    long long t1 = clock64();
    if (tid == 0) out[blockIdx.x] = (acc == -1.0f) ? 0 : (gpu_long_t)(t1 - t0);
}

static void report(const char* tag, const std::vector<gpu_long_t>& v, double ns_per_cycle) {
    gpu_long_t mn=v[0], mx=v[0]; double sum=0;
    for (auto x:v){ if(x<mn)mn=x; if(x>mx)mx=x; sum+=(double)x; }
    const double ops = (double)MB_ITERS * MB_DEG;
    printf("GPU,%s,min_ns=%.4f,avg_ns=%.4f,max_ns=%.4f\n", tag,
           mn/ops*ns_per_cycle, (sum/v.size())/ops*ns_per_cycle, mx/ops*ns_per_cycle);
}

int main(int argc, char** argv) {
    int scale = 4; if (argc > 1) scale = atoi(argv[1]);
    cudaDeviceProp prop; int dev; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    int nblk = prop.multiProcessorCount * scale;
    double ns_per_cycle = 1.0e6 / (double)prop.clockRate;
    printf("Device: %s SMs=%d clock=%.3f GHz grid=%d (real-form block scan)\n",
           prop.name, prop.multiProcessorCount, prop.clockRate/1.0e6, nblk);

    srand(42);
    std::vector<float> q(MB_DIM), signs(MB_PADDED);
    for (auto& x:q) x=(float)rand()/RAND_MAX;
    for (auto& x:signs) x=(rand()&1)?1.f:-1.f;
    std::vector<float> rows((size_t)MB_NEXP*MB_DEG*MB_DIM);
    for (auto& x:rows) x=(float)rand()/RAND_MAX;
    std::vector<float> fac((size_t)MB_NEXP*3*MB_DEG);
    for (size_t e=0;e<MB_NEXP;e++) for(int i=0;i<MB_DEG;i++){
        fac[e*3*MB_DEG+i]=0.1f; fac[e*3*MB_DEG+MB_DEG+i]=0.01f; fac[e*3*MB_DEG+2*MB_DEG+i]=0.5f; }
    std::vector<uint64_t> codes((size_t)MB_NEXP*MB_CODE_WORDS);
    for (auto& c:codes) c=(static_cast<uint64_t>(rand()) << 32) ^ static_cast<uint64_t>(rand());

    float *dq,*dsigns,*drows,*dfac; uint64_t* dcodes; gpu_long_t* dout;
    CUDA_SAFE_CALL(cudaMalloc(&dq,q.size()*4));
    CUDA_SAFE_CALL(cudaMalloc(&dsigns,signs.size()*4));
    CUDA_SAFE_CALL(cudaMalloc(&drows,rows.size()*4));
    CUDA_SAFE_CALL(cudaMalloc(&dfac,fac.size()*4));
    CUDA_SAFE_CALL(cudaMalloc(&dcodes,codes.size()*sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc(&dout,nblk*sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dsigns,signs.data(),signs.size()*4,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(drows,rows.data(),rows.size()*4,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dfac,fac.data(),fac.size()*4,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dcodes,codes.data(),codes.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
    std::vector<gpu_long_t> hout(nblk);

    benchL2<<<nblk, BLOCK_SIZE>>>(dq, drows, dout);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(),dout,nblk*sizeof(gpu_long_t),cudaMemcpyDeviceToHost));
    report("exact_l2", hout, ns_per_cycle);

    size_t shm = MB_PADDED*sizeof(float) + MB_PADDED + (MB_PADDED<<2)
               + (size_t)MB_CODE_WORDS*sizeof(uint64_t) + 3*MB_DEG*sizeof(float);
    cudaFuncSetAttribute(benchRq, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    benchRq<<<nblk, BLOCK_SIZE, shm>>>(dq, dsigns, dcodes, dfac, dout);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(hout.data(),dout,nblk*sizeof(gpu_long_t),cudaMemcpyDeviceToHost));
    report("rabitq_batched", hout, ns_per_cycle);

    cudaFree(dq);cudaFree(dsigns);cudaFree(drows);cudaFree(dfac);cudaFree(dcodes);cudaFree(dout);
    return 0;
}
