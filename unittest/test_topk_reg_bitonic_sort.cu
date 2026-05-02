#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include "topk_reg.cuh"

// Test: sort_warp and merge_warp on random bitonic sequences.
// Mirrors test_bitonic_sort.cu for SharedBitonicTopK.

// We use REGX=N, REGY=N so that the static helpers are instantiated for N.

template <unsigned N, bool asc>
__global__ void kernel_test_sort_warp(
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

    RegBitonicTopK<N, N>::template sort_warp<asc, N>(d, i);

    #pragma unroll
    for (unsigned j = 0; j < N; j++) {
        unsigned pos = tid * N + j;
        g_out_dist[pos] = d[j];
        g_out_idx[pos]  = i[j];
    }
}

template <unsigned N, bool asc>
__global__ void kernel_test_merge_warp(
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

    RegBitonicTopK<N, N>::template merge_warp<asc, N>(d, i);

    #pragma unroll
    for (unsigned j = 0; j < N; j++) {
        unsigned pos = tid * N + j;
        g_out_dist[pos] = d[j];
        g_out_idx[pos]  = i[j];
    }
}

// Generate a random bitonic sequence of values 0..TOTAL-1:
// first half ascending, second half descending.
template <unsigned TOTAL>
void make_bitonic(std::mt19937& rng,
                  std::vector<float>& dist, std::vector<uint32_t>& idx)
{
    dist.resize(TOTAL);
    idx.resize(TOTAL);
    for (unsigned i = 0; i < TOTAL; i++) {
        dist[i] = (float)i;
        idx[i]  = i;
    }
    // Shuffle
    for (unsigned i = TOTAL - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(dist[i], dist[j]);
        std::swap(idx[i], idx[j]);
    }
    // Split into ascending first half + descending second half
    std::vector<std::pair<float,uint32_t>> first(TOTAL/2), second(TOTAL/2);
    for (unsigned i = 0; i < TOTAL/2; i++) {
        first[i]  = {dist[i], idx[i]};
        second[i] = {dist[TOTAL/2 + i], idx[TOTAL/2 + i]};
    }
    std::sort(first.begin(), first.end());
    std::sort(second.begin(), second.end(),
        [](auto& a, auto& b){ return a.first > b.first; });

    for (unsigned i = 0; i < TOTAL/2; i++) {
        dist[i]           = first[i].first;
        idx[i]            = first[i].second;
        dist[TOTAL/2 + i] = second[i].first;
        idx[TOTAL/2 + i]  = second[i].second;
    }
}

template <unsigned N>
bool run_sort_warp_test(unsigned seed) {
    constexpr unsigned TOTAL = 32 * N;
    std::mt19937 rng(seed);

    std::vector<float> h_dist;
    std::vector<uint32_t> h_idx;
    make_bitonic<TOTAL>(rng, h_dist, h_idx);

    float *d_in_dist, *d_out_dist;
    uint32_t *d_in_idx, *d_out_idx;
    cudaMalloc(&d_in_dist,  TOTAL * sizeof(float));
    cudaMalloc(&d_in_idx,   TOTAL * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, TOTAL * sizeof(float));
    cudaMalloc(&d_out_idx,  TOTAL * sizeof(uint32_t));

    cudaMemcpy(d_in_dist, h_dist.data(), TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,  h_idx.data(),  TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bool pass = true;
    std::vector<float> h_out(TOTAL);

    // Test ascending
    kernel_test_sort_warp<N, true><<<1, 32>>>(d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out_dist, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < TOTAL; i++) {
        if (h_out[i] != (float)i) {
            printf("  FAIL asc at %u: expected %f, got %f\n", i, (float)i, h_out[i]);
            pass = false; break;
        }
    }

    // Test descending
    kernel_test_sort_warp<N, false><<<1, 32>>>(d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out_dist, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < TOTAL; i++) {
        float expected = (float)(TOTAL - 1 - i);
        if (h_out[i] != expected) {
            printf("  FAIL desc at %u: expected %f, got %f\n", i, expected, h_out[i]);
            pass = false; break;
        }
    }

    cudaFree(d_in_dist); cudaFree(d_in_idx);
    cudaFree(d_out_dist); cudaFree(d_out_idx);
    return pass;
}

template <unsigned N>
bool run_merge_warp_test(unsigned seed) {
    constexpr unsigned TOTAL = 32 * N;
    std::mt19937 rng(seed);

    std::vector<float> h_dist;
    std::vector<uint32_t> h_idx;
    make_bitonic<TOTAL>(rng, h_dist, h_idx);

    float *d_in_dist, *d_out_dist;
    uint32_t *d_in_idx, *d_out_idx;
    cudaMalloc(&d_in_dist,  TOTAL * sizeof(float));
    cudaMalloc(&d_in_idx,   TOTAL * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, TOTAL * sizeof(float));
    cudaMalloc(&d_out_idx,  TOTAL * sizeof(uint32_t));

    cudaMemcpy(d_in_dist, h_dist.data(), TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,  h_idx.data(),  TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bool pass = true;
    std::vector<float> h_out(TOTAL);

    // Test ascending
    kernel_test_merge_warp<N, true><<<1, 32>>>(d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out_dist, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < TOTAL; i++) {
        if (h_out[i] != (float)i) {
            printf("  FAIL asc at %u: expected %f, got %f\n", i, (float)i, h_out[i]);
            pass = false; break;
        }
    }

    // Test descending
    kernel_test_merge_warp<N, false><<<1, 32>>>(d_in_dist, d_in_idx, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out_dist, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < TOTAL; i++) {
        float expected = (float)(TOTAL - 1 - i);
        if (h_out[i] != expected) {
            printf("  FAIL desc at %u: expected %f, got %f\n", i, expected, h_out[i]);
            pass = false; break;
        }
    }

    cudaFree(d_in_dist); cudaFree(d_in_idx);
    cudaFree(d_out_dist); cudaFree(d_out_idx);
    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 200;

    struct { const char* name; bool (*fn)(unsigned); } tests[] = {
        {"sort_warp<1>",  run_sort_warp_test<1>},
        {"sort_warp<2>",  run_sort_warp_test<2>},
        {"sort_warp<4>",  run_sort_warp_test<4>},
        {"sort_warp<8>",  run_sort_warp_test<8>},
        {"merge_warp<1>", run_merge_warp_test<1>},
        {"merge_warp<2>", run_merge_warp_test<2>},
        {"merge_warp<4>", run_merge_warp_test<4>},
        {"merge_warp<8>", run_merge_warp_test<8>},
    };

    for (auto& t : tests) {
        bool pass = t.fn(seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
