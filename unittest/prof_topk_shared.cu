#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <random>
#include <string>
#include <vector>

#include "topk.cuh"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__,   \
                #call, cudaGetErrorString(err_));                       \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

// ── base-search pattern: BX=256, REG=1 ────────────────────────────────────
// Each iteration: merge CAND=64 candidates (threads 0..63 carry data,
// rest carry FLT_MAX), then pick 1 node.

template <unsigned BX>
__global__ void kernel_prof_topk_base(
    const float* __restrict__ g_dist,
    const uint32_t* __restrict__ g_idx,
    unsigned topk_size, unsigned cand, unsigned iters)
{
    constexpr unsigned REG = 1;
    constexpr unsigned N = BX * REG;

    extern __shared__ char smem[];
    char* ptr = smem;
    SharedBitonicTopK<BX, REG> buf(ptr, topk_size);
    __syncthreads();

    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / 32;

    for (unsigned iter = 0; iter < iters; iter++) {
        // merge
        float    d[REG];
        uint32_t x[REG];
        d[0] = (tid < cand) ? g_dist[iter * cand + tid] : FLT_MAX;
        x[0] = (tid < cand) ? g_idx[iter * cand + tid]  : 0xFFFFFFFF;
        buf.merge(d, x);

        // pick
        uint32_t pbuf[N];
        #pragma unroll
        for (unsigned i = 0; i < N; i++) pbuf[i] = 0xFFFFFFFF;
        unsigned found = buf.pick(pbuf, 1);
        if (warp_id == 0) {
            const bool has_pick = found > 0 && pbuf[0] != 0xFFFFFFFF;
            const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
            if (ballot != 0) {
                const int src = __ffs(ballot) - 1;
                __shfl_sync(0xffffffff, pbuf[0], src);
            }
        }
        __syncthreads();
    }
}

// ── warp-search pattern: BX=32, REG=2, NY warps per block ─────────────────

template <unsigned NY>
__global__ void kernel_prof_topk_warp(
    const float* __restrict__ g_dist,
    const uint32_t* __restrict__ g_idx,
    unsigned topk_size, unsigned cand, unsigned iters)
{
    constexpr unsigned BX  = 32;
    constexpr unsigned REG = 2;
    constexpr unsigned N   = BX * REG;

    extern __shared__ char smem[];
    char* ptr = smem;
    SharedBitonicTopK<BX, REG> buf(ptr, topk_size);
    __syncwarp();

    const unsigned lane = threadIdx.x;

    for (unsigned iter = 0; iter < iters; iter++) {
        // merge
        float    d[REG];
        uint32_t x[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = lane * REG + j;
            d[j] = (pos < cand) ? g_dist[iter * cand + pos] : FLT_MAX;
            x[j] = (pos < cand) ? g_idx[iter * cand + pos]  : 0xFFFFFFFF;
        }
        buf.merge(d, x);

        // pick
        uint32_t pbuf[N];
        #pragma unroll
        for (unsigned i = 0; i < N; i++) pbuf[i] = 0xFFFFFFFF;
        unsigned found = buf.pick(pbuf, 1);
        const bool has_pick = found > 0 && pbuf[0] != 0xFFFFFFFF;
        const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
        if (ballot != 0) {
            const int src = __ffs(ballot) - 1;
            __shfl_sync(0xffffffff, pbuf[0], src);
        }
        __syncwarp();
    }
}

struct ProfResult {
    const char* name;
    double us;
};

int main() {
    constexpr unsigned TOPK_SIZE = 256;
    constexpr unsigned CAND      = 64;
    constexpr unsigned ITERS     = 128;
    constexpr int      NBLOCKS   = 1024;
    constexpr int      NRUNS     = 5;

    // generate random candidate data: ITERS batches of CAND entries
    unsigned total = ITERS * CAND;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(0.0f, 1e6f);
    std::uniform_int_distribution<uint32_t> idist(0, 999999);

    std::vector<float> h_dist(total);
    std::vector<uint32_t> h_idx(total);
    for (unsigned i = 0; i < total; i++) {
        h_dist[i] = fdist(rng);
        h_idx[i]  = idist(rng);
    }

    float* d_dist;
    uint32_t* d_idx;
    CUDA_CHECK(cudaMalloc(&d_dist, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), total * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), total * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    ProfResult results[3];

    // ── base: BX=256, REG=1 ────────────────────────────────────────────────
    {
        constexpr unsigned BX = 256;
        size_t smem = SharedBitonicTopK<BX, 1>::shared_mem_bytes(TOPK_SIZE);
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_topk_base<BX>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_topk_base<BX><<<NBLOCKS, BX, smem>>>(
            d_dist, d_idx, TOPK_SIZE, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_topk_base<BX><<<NBLOCKS, BX, smem>>>(
                d_dist, d_idx, TOPK_SIZE, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        results[0] = {"base BX=256 REG=1", us};
    }

    // ── warp4: NY=4, BX=32, REG=2 ─────────────────────────────────────────
    {
        constexpr unsigned NY = 4;
        size_t smem = SharedBitonicTopK<32, 2>::shared_mem_bytes(TOPK_SIZE, NY);
        dim3 block(32, NY);
        int nblocks = (NBLOCKS + NY - 1) / NY;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_topk_warp<NY>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_topk_warp<NY><<<nblocks, block, smem>>>(
            d_dist, d_idx, TOPK_SIZE, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_topk_warp<NY><<<nblocks, block, smem>>>(
                d_dist, d_idx, TOPK_SIZE, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        results[1] = {"warp4 NY=4 REG=2", us};
    }

    // ── warp8: NY=8, BX=32, REG=2 ─────────────────────────────────────────
    {
        constexpr unsigned NY = 8;
        size_t smem = SharedBitonicTopK<32, 2>::shared_mem_bytes(TOPK_SIZE, NY);
        dim3 block(32, NY);
        int nblocks = (NBLOCKS + NY - 1) / NY;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_topk_warp<NY>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_topk_warp<NY><<<nblocks, block, smem>>>(
            d_dist, d_idx, TOPK_SIZE, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_topk_warp<NY><<<nblocks, block, smem>>>(
                d_dist, d_idx, TOPK_SIZE, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        results[2] = {"warp8 NY=8 REG=2", us};
    }

    // ── report ─────────────────────────────────────────────────────────────
    FILE* fp = fopen("report.txt", "a");
    std::string report;
    report += "prof_topk (topk=" + std::to_string(TOPK_SIZE)
            + " cand=" + std::to_string(CAND)
            + " iters=" + std::to_string(ITERS)
            + " blocks=" + std::to_string(NBLOCKS)
            + " runs=" + std::to_string(NRUNS) + ")\n";
    for (auto& r : results)
        report += "  " + std::string(r.name) + ": " + std::to_string(r.us) + " us\n";
    fputs(report.c_str(), fp);
    fputs(report.c_str(), stdout);
    fclose(fp);

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_idx));
    return 0;
}
