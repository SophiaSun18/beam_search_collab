#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include "include/topk.cuh"

// Test: merge should retain the smallest elements.
// We fill the buffer with 0..topk-1 (shuffled, stored descending),
// then merge batches of N new elements. After all merges, the buffer
// should hold the smallest `topk` values from the union.

template <unsigned BX, unsigned REG>
__global__ void kernel_test_merge(
    float* g_buf_dist, uint32_t* g_buf_idx, unsigned topk,
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    float* g_out_dist, uint32_t* g_out_idx)
{
    constexpr unsigned N = BX * REG;
    extern __shared__ char smem[];
    char* ptr = smem;

    const unsigned tid = threadIdx.x;

    SharedBitonicTopK<BX, REG> buf(ptr, topk);
    __syncthreads();

    // Merge batches of N elements
    unsigned batches = n_inputs / N;
    for (unsigned b = 0; b < batches; b++) {
        float    dist[REG];
        uint32_t idx[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = b * N + tid * REG + j;
            dist[j] = g_input_dist[pos];
            idx[j]  = g_input_idx[pos];
        }
        buf.merge(dist, idx);
    }

    // Flush results
    buf.flush(g_out_dist, g_out_idx, topk);
}

template <unsigned BX, unsigned REG>
bool run_merge_test(unsigned topk, unsigned n_total, unsigned seed) {
    constexpr unsigned N = BX * REG;
    std::mt19937 rng(seed);

    // Generate all values: 0.0, 1.0, ..., n_total + topk - 1
    unsigned total = topk + n_total;
    std::vector<float> all_vals(total);
    std::vector<uint32_t> all_ids(total);
    for (unsigned i = 0; i < total; i++) {
        all_vals[i] = (float)i;
        all_ids[i]  = i;
    }
    // Shuffle
    for (unsigned i = total - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        if (i == j) { continue; }
        std::swap(all_vals[i], all_vals[j]);
        std::swap(all_ids[i], all_ids[j]);
    }

    // Buffer starts as FLT_MAX / 0xFFFFFFFF (empty)
    std::vector<float> h_buf_dist(topk, FLT_MAX);
    std::vector<uint32_t> h_buf_idx(topk, 0xFFFFFFFF);

    // All values go into input batches (unsorted within each batch)
    unsigned n_inputs = (total / N) * N; // round down to multiple of N
    std::vector<float> h_in_dist(n_inputs);
    std::vector<uint32_t> h_in_idx(n_inputs);
    for (unsigned i = 0; i < n_inputs; i++) {
        h_in_dist[i] = all_vals[i];
        h_in_idx[i]  = all_ids[i];
    }

    // Allocate device memory
    float *d_buf_dist, *d_in_dist, *d_out_dist;
    uint32_t *d_buf_idx, *d_in_idx, *d_out_idx;
    cudaMalloc(&d_buf_dist, topk * sizeof(float));
    cudaMalloc(&d_buf_idx,  topk * sizeof(uint32_t));
    cudaMalloc(&d_in_dist,  n_inputs * sizeof(float));
    cudaMalloc(&d_in_idx,   n_inputs * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, topk * sizeof(float));
    cudaMalloc(&d_out_idx,  topk * sizeof(uint32_t));

    cudaMemcpy(d_buf_dist, h_buf_dist.data(), topk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf_idx,  h_buf_idx.data(),  topk * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_dist,  h_in_dist.data(),  n_inputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,   h_in_idx.data(),   n_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice);

    size_t smem_size = SharedBitonicTopK<BX, REG>::shared_mem_bytes(topk);
    kernel_test_merge<BX, REG><<<1, BX, smem_size>>>(
        d_buf_dist, d_buf_idx, topk,
        d_in_dist, d_in_idx, n_inputs,
        d_out_dist, d_out_idx);
    cudaDeviceSynchronize();

    std::vector<float> h_out_dist(topk);
    std::vector<uint32_t> h_out_idx(topk);
    cudaMemcpy(h_out_dist.data(), d_out_dist, topk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_idx.data(),  d_out_idx,  topk * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_buf_dist); cudaFree(d_buf_idx);
    cudaFree(d_in_dist);  cudaFree(d_in_idx);
    cudaFree(d_out_dist); cudaFree(d_out_idx);

    // Verify: output should contain values 0..topk-1
    std::vector<float> sorted_out(h_out_dist.begin(), h_out_dist.end());
    std::sort(sorted_out.begin(), sorted_out.end());
    bool pass = true;
    for (unsigned i = 0; i < topk; i++) {
        if (sorted_out[i] != (float)i) {
            printf("  FAIL at position %u: expected %f, got %f\n", i, (float)i, sorted_out[i]);
            pass = false;
            break;
        }
    }
    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 42;

    struct { const char* name; bool (*fn)(unsigned, unsigned, unsigned); unsigned topk; unsigned n_total; } tests[] = {
        // BX=32, REG=2, N=64
        {"merge<32,2> topk=128  n=2560",  run_merge_test<32,2>,  128,  2560},
        {"merge<32,2> topk=256  n=5120",  run_merge_test<32,2>,  256,  5120},
        {"merge<32,2> topk=512  n=10240", run_merge_test<32,2>,  512,  10240},
        // BX=32, REG=4, N=128
        {"merge<32,4> topk=128  n=2560",  run_merge_test<32,4>,  128,  2560},
        {"merge<32,4> topk=256  n=5120",  run_merge_test<32,4>,  256,  5120},
        {"merge<32,4> topk=512  n=10240", run_merge_test<32,4>,  512,  10240},
        // BX=64, REG=2, N=128
        {"merge<64,2> topk=128  n=2560",  run_merge_test<64,2>,  128,  2560},
        {"merge<64,2> topk=256  n=5120",  run_merge_test<64,2>,  256,  5120},
        {"merge<64,2> topk=512  n=10240", run_merge_test<64,2>,  512,  10240},
        // BX=64, REG=4, N=256
        {"merge<64,4> topk=256  n=5120",  run_merge_test<64,4>,  256,  5120},
        {"merge<64,4> topk=512  n=10240", run_merge_test<64,4>,  512,  10240},
        {"merge<64,4> topk=1024 n=20480", run_merge_test<64,4>,  1024, 20480},
    };

    for (auto& t : tests) {
        bool pass = t.fn(t.topk, t.n_total, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
