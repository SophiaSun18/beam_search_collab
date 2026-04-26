#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include "include/topk.cuh"

// Test: bitonic_sort_full should correctly sort arbitrary unsorted data.

template <unsigned BX, unsigned REG>
__global__ void kernel_test_sort_full(
    float* g_in_dist, uint32_t* g_in_idx,
    float* g_out_dist, uint32_t* g_out_idx)
{
    constexpr unsigned N = BX * REG;
    __shared__ float    scratch_d[N];
    __shared__ uint32_t scratch_i[N];
    const unsigned tid = threadIdx.x;

    float    d[REG];
    uint32_t i[REG];
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = tid * REG + j;
        d[j] = g_in_dist[pos];
        i[j] = g_in_idx[pos];
    }

    SharedBitonicTopK<BX, REG>::bitonic_sort_full(d, i, scratch_d, scratch_i);

    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = tid * REG + j;
        g_out_dist[pos] = d[j];
        g_out_idx[pos]  = i[j];
    }
}

template <unsigned BX, unsigned REG>
bool run_sort_full_test(unsigned seed) {
    constexpr unsigned N = BX * REG;
    std::mt19937 rng(seed);

    std::vector<float> vals(N);
    std::vector<uint32_t> ids(N);
    for (unsigned i = 0; i < N; i++) {
        vals[i] = (float)i;
        ids[i]  = i;
    }
    // Fully shuffle (not bitonic)
    for (unsigned i = N - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(vals[i], vals[j]);
        std::swap(ids[i], ids[j]);
    }

    float *d_in_dist, *d_out_dist;
    uint32_t *d_in_idx, *d_out_idx;
    cudaMalloc(&d_in_dist,  N * sizeof(float));
    cudaMalloc(&d_in_idx,   N * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, N * sizeof(float));
    cudaMalloc(&d_out_idx,  N * sizeof(uint32_t));

    cudaMemcpy(d_in_dist, vals.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,  ids.data(),  N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Test ascending
    kernel_test_sort_full<BX, REG><<<1, BX>>>(
        d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();

    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_out_dist, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (unsigned i = 0; i < N; i++) {
        if (h_out[i] != (float)i) {
            printf("  FAIL at %u: expected %f, got %f\n", i, (float)i, h_out[i]);
            pass = false;
            break;
        }
    }

    cudaFree(d_in_dist); cudaFree(d_in_idx);
    cudaFree(d_out_dist); cudaFree(d_out_idx);
    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 300;

    struct { const char* name; bool (*fn)(unsigned); } tests[] = {
        {"sort_full<32,2>",  run_sort_full_test<32,2>},
        {"sort_full<32,4>",  run_sort_full_test<32,4>},
        {"sort_full<64,2>",  run_sort_full_test<64,2>},
        {"sort_full<64,4>",  run_sort_full_test<64,4>},
        {"sort_full<128,2>", run_sort_full_test<128,2>},
        {"sort_full<128,4>", run_sort_full_test<128,4>},
    };

    for (auto& t : tests) {
        bool pass = t.fn(seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
