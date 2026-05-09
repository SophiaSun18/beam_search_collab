#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include "topk_reg.cuh"

// Test: sort_warp_full should correctly sort arbitrary unsorted data.
// Mirrors test_bitonic_sort_full.cu for SharedBitonicTopK.

template <unsigned N>
__global__ void kernel_test_sort_warp_full(
    float* g_in_dist, uint32_t* g_in_idx,
    float* g_out_dist, uint32_t* g_out_idx)
{
    const unsigned tid = threadIdx.x;

    float    d[N];
    uint32_t i[N];
    #pragma unroll
    for (unsigned j = 0; j < N; j++) {
        unsigned pos = tid * N + j;
        d[j] = g_in_dist[pos];
        i[j] = g_in_idx[pos];
    }

    RegBitonicTopK<N, N>::template sort_warp_full<N>(d, i);

    #pragma unroll
    for (unsigned j = 0; j < N; j++) {
        unsigned pos = tid * N + j;
        g_out_dist[pos] = d[j];
        g_out_idx[pos]  = i[j];
    }
}

template <unsigned N>
bool run_sort_warp_full_test(unsigned seed) {
    constexpr unsigned TOTAL = 32 * N;
    std::mt19937 rng(seed);

    std::vector<float> vals(TOTAL);
    std::vector<uint32_t> ids(TOTAL);
    for (unsigned i = 0; i < TOTAL; i++) {
        vals[i] = (float)i;
        ids[i]  = i;
    }
    // Fully shuffle (not bitonic)
    for (unsigned i = TOTAL - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(vals[i], vals[j]);
        std::swap(ids[i], ids[j]);
    }

    float *d_in_dist, *d_out_dist;
    uint32_t *d_in_idx, *d_out_idx;
    cudaMalloc(&d_in_dist,  TOTAL * sizeof(float));
    cudaMalloc(&d_in_idx,   TOTAL * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, TOTAL * sizeof(float));
    cudaMalloc(&d_out_idx,  TOTAL * sizeof(uint32_t));

    cudaMemcpy(d_in_dist, vals.data(), TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,  ids.data(),  TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // sort_warp_full always produces ascending
    kernel_test_sort_warp_full<N><<<1, 32>>>(d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();

    std::vector<float> h_out(TOTAL);
    cudaMemcpy(h_out.data(), d_out_dist, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (unsigned i = 0; i < TOTAL; i++) {
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
    unsigned seed = 400;

    struct { const char* name; bool (*fn)(unsigned); } tests[] = {
        {"sort_warp_full<1>", run_sort_warp_full_test<1>},
        {"sort_warp_full<2>", run_sort_warp_full_test<2>},
        {"sort_warp_full<4>", run_sort_warp_full_test<4>},
        {"sort_warp_full<8>", run_sort_warp_full_test<8>},
    };

    for (auto& t : tests) {
        bool pass = t.fn(seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
