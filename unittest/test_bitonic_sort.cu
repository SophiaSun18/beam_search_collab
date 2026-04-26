#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include "include/topk.cuh"

// Test: bitonic_sort_block should correctly sort a bitonic sequence.
// We generate a random bitonic sequence (ascending then descending)
// of values 0.0..n-1 and verify the sort produces a fully sorted result.

template <unsigned BX, unsigned REG, bool asc>
__global__ void kernel_test_bitonic_sort(
    float* g_in_dist, uint32_t* g_in_idx,
    float* g_out_dist, uint32_t* g_out_idx)
{
    constexpr unsigned N = BX * REG;
    extern __shared__ char smem[];
    float*    scratch_d = (float*)smem;
    uint32_t* scratch_i = (uint32_t*)(scratch_d + N);

    const unsigned tid = threadIdx.x;

    float    d[REG];
    uint32_t i[REG];
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = tid * REG + j;
        d[j] = g_in_dist[pos];
        i[j] = g_in_idx[pos];
    }

    SharedBitonicTopK<BX, REG>::template bitonic_sort_block<asc>(d, i, scratch_d, scratch_i);

    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = tid * REG + j;
        g_out_dist[pos] = d[j];
        g_out_idx[pos]  = i[j];
    }
}

template <unsigned BX, unsigned REG>
bool run_bitonic_sort_test(unsigned seed) {
    constexpr unsigned N = BX * REG;
    std::mt19937 rng(seed);

    // Generate values 0..N-1
    std::vector<float> vals(N);
    std::vector<uint32_t> ids(N);
    for (unsigned i = 0; i < N; i++) {
        vals[i] = (float)i;
        ids[i]  = i;
    }

    // Shuffle then arrange as bitonic: sort first half ascending, second half descending
    for (unsigned i = N - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(vals[i], vals[j]);
        std::swap(ids[i], ids[j]);
    }

    std::vector<std::pair<float,uint32_t>> first_half(N/2), second_half(N/2);
    for (unsigned i = 0; i < N/2; i++) {
        first_half[i] = {vals[i], ids[i]};
        second_half[i] = {vals[N/2 + i], ids[N/2 + i]};
    }
    std::sort(first_half.begin(), first_half.end()); // ascending
    std::sort(second_half.begin(), second_half.end(),
        [](auto& a, auto& b){ return a.first > b.first; }); // descending

    std::vector<float> h_in_dist(N);
    std::vector<uint32_t> h_in_idx(N);
    for (unsigned i = 0; i < N/2; i++) {
        h_in_dist[i]       = first_half[i].first;
        h_in_idx[i]        = first_half[i].second;
        h_in_dist[N/2 + i] = second_half[i].first;
        h_in_idx[N/2 + i]  = second_half[i].second;
    }

    float *d_in_dist, *d_out_dist;
    uint32_t *d_in_idx, *d_out_idx;
    cudaMalloc(&d_in_dist,  N * sizeof(float));
    cudaMalloc(&d_in_idx,   N * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, N * sizeof(float));
    cudaMalloc(&d_out_idx,  N * sizeof(uint32_t));

    cudaMemcpy(d_in_dist, h_in_dist.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,  h_in_idx.data(),  N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    size_t smem_size = N * (sizeof(float) + sizeof(uint32_t));

    // Test ascending
    kernel_test_bitonic_sort<BX, REG, true><<<1, BX, smem_size>>>(
        d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();

    std::vector<float> h_out_dist(N);
    std::vector<uint32_t> h_out_idx(N);
    cudaMemcpy(h_out_dist.data(), d_out_dist, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_idx.data(),  d_out_idx,  N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (unsigned i = 0; i < N; i++) {
        if (h_out_dist[i] != (float)i) {
            printf("  FAIL asc at %u: expected %f, got %f\n", i, (float)i, h_out_dist[i]);
            pass = false;
            break;
        }
    }

    // Test descending
    kernel_test_bitonic_sort<BX, REG, false><<<1, BX, smem_size>>>(
        d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_dist.data(), d_out_dist, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < N; i++) {
        float expected = (float)(N - 1 - i);
        if (h_out_dist[i] != expected) {
            printf("  FAIL desc at %u: expected %f, got %f\n", i, expected, h_out_dist[i]);
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
    unsigned seed = 100;

    struct { const char* name; bool (*fn)(unsigned); } tests[] = {
        {"bitonic_sort<32,2>",   run_bitonic_sort_test<32,2>},
        {"bitonic_sort<32,4>",   run_bitonic_sort_test<32,4>},
        {"bitonic_sort<64,2>",   run_bitonic_sort_test<64,2>},
        {"bitonic_sort<64,4>",   run_bitonic_sort_test<64,4>},
        {"bitonic_sort<128,2>",  run_bitonic_sort_test<128,2>},
        {"bitonic_sort<128,4>",  run_bitonic_sort_test<128,4>},
    };

    for (auto& t : tests) {
        bool pass = t.fn(seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
