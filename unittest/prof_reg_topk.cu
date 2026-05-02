#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cfloat>
#include <climits>

#include "topk_reg.cuh"

// ═══════════════════════════════════════════════════════════════════════════

template <unsigned REGX, unsigned REGY>
__global__ void kernel_reg_bench(
    float* g_cand_dist, uint32_t* g_cand_idx,
    unsigned topk_size, unsigned cand_size, unsigned n_iters)
{
    const unsigned tid = threadIdx.x;

    RegBitonicTopK<REGX, REGY> buf;

    for (unsigned it = 0; it < n_iters; it++) {
        float d[REGY]; uint32_t x[REGY];
        #pragma unroll
        for (unsigned j = 0; j < REGY; j++) {
            unsigned pos = tid * REGY + j;
            d[j] = (pos < cand_size) ? g_cand_dist[it * cand_size + pos] : FLT_MAX;
            x[j] = (pos < cand_size) ? g_cand_idx[it * cand_size + pos]  : 0xFFFFFFFF;
        }
        buf.merge(d, x);

        uint32_t pbuf[REGY];
        #pragma unroll
        for (unsigned i = 0; i < REGY; i++) pbuf[i] = 0xFFFFFFFF;
        unsigned found = buf.pick(pbuf, 1);
        uint32_t picked = 0xFFFFFFFF;
        const bool has = found > 0 && pbuf[0] != 0xFFFFFFFF;
        const uint32_t ballot = __ballot_sync(0xffffffff, has);
        if (ballot != 0) picked = __shfl_sync(0xffffffff, pbuf[0], __ffs(ballot) - 1);
        (void)picked;
        __syncwarp();
    }
}

// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

template <unsigned REGX, unsigned REGY>
double run_bench(unsigned topk, unsigned cand, unsigned iters, unsigned nblocks, unsigned nruns) {
    kernel_reg_bench<REGX, REGY><<<nblocks, 32>>>(nullptr, nullptr, topk, cand, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Need actual data pointers — re-launch with real data below
    return 0;
}

int main() {
    constexpr unsigned TOPK = 256;
    constexpr unsigned CAND = 64;
    constexpr unsigned ITERS = 128;
    constexpr unsigned NBLOCKS = 1024;
    constexpr unsigned NRUNS = 5;
    constexpr unsigned TOTAL = ITERS * CAND;

    std::mt19937 rng(42);
    std::vector<float> h_dist(TOTAL);
    std::vector<uint32_t> h_idx(TOTAL);
    for (unsigned i = 0; i < TOTAL; i++) {
        h_dist[i] = static_cast<float>(rng() % 1000000);
        h_idx[i]  = rng() % 1000000;
    }

    float *d_dist; uint32_t *d_idx;
    CUDA_CHECK(cudaMalloc(&d_dist, TOTAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,  TOTAL * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), TOTAL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx,  h_idx.data(),  TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice));

    struct { const char* name; void(*launch)(float*, uint32_t*, unsigned, unsigned, unsigned); } configs[] = {
        {"reg<8,1>", [](float* dd, uint32_t* di, unsigned t, unsigned c, unsigned it)
            { kernel_reg_bench<8,1><<<NBLOCKS, 32>>>(dd, di, t, c, it); }},
        {"reg<8,2>", [](float* dd, uint32_t* di, unsigned t, unsigned c, unsigned it)
            { kernel_reg_bench<8,2><<<NBLOCKS, 32>>>(dd, di, t, c, it); }},
        {"reg<8,4>", [](float* dd, uint32_t* di, unsigned t, unsigned c, unsigned it)
            { kernel_reg_bench<8,4><<<NBLOCKS, 32>>>(dd, di, t, c, it); }},
        {"reg<8,8>", [](float* dd, uint32_t* di, unsigned t, unsigned c, unsigned it)
            { kernel_reg_bench<8,8><<<NBLOCKS, 32>>>(dd, di, t, c, it); }},
    };

    for (auto& cfg : configs) {
        // warmup
        cfg.launch(d_dist, d_idx, TOPK, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned r = 0; r < NRUNS; r++) {
            cfg.launch(d_dist, d_idx, TOPK, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        printf("%-18s %.1f us  (%d blocks x %d iters)\n", cfg.name, us, NBLOCKS, ITERS);
    }

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_idx));
    return 0;
}
