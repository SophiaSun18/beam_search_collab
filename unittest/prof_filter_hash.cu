#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "filter_hash.cuh"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__,   \
                #call, cudaGetErrorString(err_));                       \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

// ── base-search pattern: BX=256, single hash filter per block ──────────────
// Each iteration: insert up to DEGREE new keys (strided across BX threads).
// Every reset_interval iterations: init + re-populate from frontier (topk_size
// entries).

template <unsigned BX>
__global__ void kernel_prof_hash_base(
    unsigned capacity,
    const uint32_t* __restrict__ g_keys,
    unsigned degree,
    unsigned topk_size,
    unsigned iters,
    unsigned reset_interval)
{
    extern __shared__ char smem[];
    char* ptr = smem;
    HashFilter hf(ptr, capacity);
    hf.init();
    __syncthreads();

    const unsigned tid = threadIdx.x;

    for (unsigned iter = 0; iter < iters; iter++) {
        // periodic reset + re-populate
        if ((iter + 1) % reset_interval == 0) {
            hf.init();
            __syncthreads();
            // re-populate: insert topk_size frontier members
            for (unsigned i = tid; i < topk_size; i += BX) {
                uint32_t key = g_keys[iter * degree + (i % degree)];
                hf.insert(key);
            }
        }
        __syncthreads();

        // insert degree neighbors (strided)
        for (unsigned i = tid; i < degree; i += BX) {
            hf.insert(g_keys[iter * degree + i]);
        }
        __syncthreads();
    }
}

// ── warp-search pattern: BX=32 per warp, NY warps per block ────────────────

template <unsigned NY>
__global__ void kernel_prof_hash_warp(
    unsigned capacity,
    const uint32_t* __restrict__ g_keys,
    unsigned degree,
    unsigned topk_size,
    unsigned iters,
    unsigned reset_interval)
{
    extern __shared__ char smem[];
    char* ptr = smem;
    // Each warp gets its own hash filter via threadIdx.y offset
    HashFilter hf(ptr, capacity);
    hf.init();

    // Warp-level sync
    __syncwarp();

    const unsigned lane = threadIdx.x;

    for (unsigned iter = 0; iter < iters; iter++) {
        if ((iter + 1) % reset_interval == 0) {
            hf.init();
            __syncwarp();
            for (unsigned i = lane; i < topk_size; i += 32) {
                uint32_t key = g_keys[iter * degree + (i % degree)];
                hf.insert(key);
            }
        }
        __syncwarp();

        for (unsigned i = lane; i < degree; i += 32) {
            hf.insert(g_keys[iter * degree + i]);
        }
        __syncwarp();
    }
}

struct ProfResult {
    const char* name;
    double us;
};

int main() {
    constexpr unsigned CAPACITY     = 1024;
    constexpr unsigned DEGREE       = 32;
    constexpr unsigned TOPK_SIZE    = 256;
    constexpr unsigned ITERS        = 128;
    constexpr unsigned RESET_INT    = 16;
    constexpr int      NBLOCKS      = 1024;
    constexpr int      NRUNS        = 5;

    // generate random keys
    unsigned total_keys = ITERS * DEGREE;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 999999);
    std::vector<uint32_t> h_keys(total_keys);
    for (auto& k : h_keys) k = dist(rng);

    uint32_t* d_keys;
    CUDA_CHECK(cudaMalloc(&d_keys, total_keys * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), total_keys * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    ProfResult results[3];

    // ── base: BX=256, 1 block ──────────────────────────────────────────────
    {
        constexpr unsigned BX = 256;
        size_t smem = HashFilter::shared_mem_bytes(CAPACITY);
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_hash_base<BX>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        // warmup
        kernel_prof_hash_base<BX><<<NBLOCKS, BX, smem>>>(
            CAPACITY, d_keys, DEGREE, TOPK_SIZE, ITERS, RESET_INT);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_hash_base<BX><<<NBLOCKS, BX, smem>>>(
                CAPACITY, d_keys, DEGREE, TOPK_SIZE, ITERS, RESET_INT);
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
        size_t smem = HashFilter::shared_mem_bytes(CAPACITY, NY);
        dim3 block(32, NY);
        int nblocks = (NBLOCKS + NY - 1) / NY;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_hash_warp<NY>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_hash_warp<NY><<<nblocks, block, smem>>>(
            CAPACITY, d_keys, DEGREE, TOPK_SIZE, ITERS, RESET_INT);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_hash_warp<NY><<<nblocks, block, smem>>>(
                CAPACITY, d_keys, DEGREE, TOPK_SIZE, ITERS, RESET_INT);
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
        size_t smem = HashFilter::shared_mem_bytes(CAPACITY, NY);
        dim3 block(32, NY);
        int nblocks = (NBLOCKS + NY - 1) / NY;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_prof_hash_warp<NY>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(smem)));

        kernel_prof_hash_warp<NY><<<nblocks, block, smem>>>(
            CAPACITY, d_keys, DEGREE, TOPK_SIZE, ITERS, RESET_INT);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NRUNS; r++) {
            kernel_prof_hash_warp<NY><<<nblocks, block, smem>>>(
                CAPACITY, d_keys, DEGREE, TOPK_SIZE, ITERS, RESET_INT);
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
    report += "prof_hash_filter (capacity=" + std::to_string(CAPACITY)
            + " degree=" + std::to_string(DEGREE)
            + " topk=" + std::to_string(TOPK_SIZE)
            + " iters=" + std::to_string(ITERS)
            + " reset=" + std::to_string(RESET_INT)
            + " blocks=" + std::to_string(NBLOCKS)
            + " runs=" + std::to_string(NRUNS) + ")\n";
    for (auto& r : results)
        report += "  " + std::string(r.name) + ": " + std::to_string(r.us) + " us\n";
    fputs(report.c_str(), fp);
    fputs(report.c_str(), stdout);
    fclose(fp);

    CUDA_CHECK(cudaFree(d_keys));
    return 0;
}
