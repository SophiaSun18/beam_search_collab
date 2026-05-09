#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "l2.cuh"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__,   \
                #call, cudaGetErrorString(err_));                       \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

// ── base-search pattern: BX=256 (8 warps), each warp computes one distance ─
// Per iteration: up to CAND distances computed by BX/32 warps in parallel.

template <unsigned BX>
__global__ void kernel_prof_l2_base(
    const float* __restrict__ queries,  // [nq * dim]
    const float* __restrict__ vectors,  // [nvec * dim]
    float* __restrict__ out,            // [nq * cand]
    int dim, unsigned cand, unsigned iters)
{
    extern __shared__ float query_buf[];
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / 32;
    const unsigned lane_id = tid % 32;
    const int qid = blockIdx.x;

    // load query into shared memory
    for (unsigned i = tid; i < static_cast<unsigned>(dim); i += BX)
        query_buf[i] = queries[qid * dim + i];
    __syncthreads();

    for (unsigned iter = 0; iter < iters; iter++) {
        for (unsigned i = warp_id; i < cand; i += BX / 32) {
            unsigned vid = (qid * iters * cand + iter * cand + i) % 1000000;
            float d = l2_warp(dim, query_buf, vectors + vid * dim);
            if (lane_id == 0)
                out[qid * cand + i] = d;
        }
        __syncthreads();
    }
}

// ── warp-search pattern: NY warps, each warp processes sequentially ─────────

template <unsigned NY>
__global__ void kernel_prof_l2_warp(
    const float* __restrict__ queries,  // [nq * dim]
    const float* __restrict__ vectors,  // [nvec * dim]
    float* __restrict__ out,            // [nq * cand]
    int dim, unsigned cand, unsigned iters)
{
    extern __shared__ float query_buf[];
    const unsigned lane = threadIdx.x;
    const int qid = blockIdx.x * NY + threadIdx.y;

    // per-warp query buffer region
    float* my_query = query_buf + threadIdx.y * dim;

    for (unsigned i = lane; i < static_cast<unsigned>(dim); i += 32)
        my_query[i] = queries[qid * dim + i];
    __syncwarp();

    for (unsigned iter = 0; iter < iters; iter++) {
        for (unsigned i = 0; i < cand; i++) {
            unsigned vid = (qid * iters * cand + iter * cand + i) % 1000000;
            float d = l2_warp(dim, my_query, vectors + vid * dim);
            if (lane == 0)
                out[qid * cand + i] = d;
        }
        __syncwarp();
    }
}

struct ProfResult {
    const char* name;
    double us;
};

int main() {
    constexpr int      DIM     = 128;
    constexpr unsigned CAND    = 32;   // matches graph degree (only live slots)
    constexpr unsigned ITERS   = 128;
    constexpr int      NQ      = 1024;
    constexpr int      NVEC    = 1000000;
    constexpr int      NRUNS   = 5;

    // generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(0.0f, 1.0f);

    std::vector<float> h_queries(NQ * DIM);
    for (auto& v : h_queries) v = fdist(rng);

    std::vector<float> h_vectors(static_cast<size_t>(NVEC) * DIM);
    for (auto& v : h_vectors) v = fdist(rng);

    float *d_queries, *d_vectors, *d_out;
    CUDA_CHECK(cudaMalloc(&d_queries, NQ * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vectors, static_cast<size_t>(NVEC) * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, NQ * CAND * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), NQ * DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors.data(),
                          static_cast<size_t>(NVEC) * DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    ProfResult results[3];

    // ── base: BX=256, 1 query per block ────────────────────────────────────
    {
        constexpr unsigned BX = 256;
        size_t smem = DIM * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_l2_base<BX>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_l2_base<BX><<<NQ, BX, smem>>>(
            d_queries, d_vectors, d_out, DIM, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_l2_base<BX><<<NQ, BX, smem>>>(
                d_queries, d_vectors, d_out, DIM, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        results[0] = {"base BX=256", us};
    }

    // ── warp4: NY=4, dim3(32,4) ────────────────────────────────────────────
    {
        constexpr unsigned NY = 4;
        size_t smem = NY * DIM * sizeof(float);
        dim3 block(32, NY);
        int nblocks = (NQ + NY - 1) / NY;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_l2_warp<NY>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_l2_warp<NY><<<nblocks, block, smem>>>(
            d_queries, d_vectors, d_out, DIM, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_l2_warp<NY><<<nblocks, block, smem>>>(
                d_queries, d_vectors, d_out, DIM, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        results[1] = {"warp4 NY=4", us};
    }

    // ── warp8: NY=8, dim3(32,8) ────────────────────────────────────────────
    {
        constexpr unsigned NY = 8;
        size_t smem = NY * DIM * sizeof(float);
        dim3 block(32, NY);
        int nblocks = (NQ + NY - 1) / NY;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_l2_warp<NY>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_l2_warp<NY><<<nblocks, block, smem>>>(
            d_queries, d_vectors, d_out, DIM, CAND, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_l2_warp<NY><<<nblocks, block, smem>>>(
                d_queries, d_vectors, d_out, DIM, CAND, ITERS);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
        results[2] = {"warp8 NY=8", us};
    }

    // ── report ─────────────────────────────────────────────────────────────
    FILE* fp = fopen("report.txt", "a");
    std::string report;
    report += "prof_l2 (dim=" + std::to_string(DIM)
            + " cand=" + std::to_string(CAND)
            + " iters=" + std::to_string(ITERS)
            + " nq=" + std::to_string(NQ)
            + " nvec=" + std::to_string(NVEC)
            + " runs=" + std::to_string(NRUNS) + ")\n";
    for (auto& r : results)
        report += "  " + std::string(r.name) + ": " + std::to_string(r.us) + " us\n";
    fputs(report.c_str(), fp);
    fputs(report.c_str(), stdout);
    fclose(fp);

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
