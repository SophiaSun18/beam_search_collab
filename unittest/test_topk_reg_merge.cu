#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include "topk_reg.cuh"

template <unsigned REGX, unsigned REGY>
__global__ void kernel_test_reg_merge(
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    float* g_out_dist, uint32_t* g_out_idx, unsigned topk)
{
    constexpr unsigned N = 32 * REGY;  // batch size
    const unsigned lane = threadIdx.x;

    RegBitonicTopK<REGX, REGY> buf;

    unsigned batches = n_inputs / N;
    for (unsigned b = 0; b < batches; b++) {
        float    dist[REGY];
        uint32_t idx[REGY];
        #pragma unroll
        for (unsigned j = 0; j < REGY; j++) {
            unsigned pos = b * N + lane * REGY + j;
            dist[j] = g_input_dist[pos];
            idx[j]  = g_input_idx[pos];
        }
        buf.merge(dist, idx);
    }

    buf.flush(g_out_dist, g_out_idx, topk);
}

template <unsigned REGX, unsigned REGY>
bool run_reg_merge_test(unsigned n_total, unsigned seed) {
    constexpr unsigned TOPK = 32 * REGX;
    constexpr unsigned N = 32 * REGY;
    std::mt19937 rng(seed);

    unsigned total = TOPK + n_total;
    std::vector<float> all_vals(total);
    std::vector<uint32_t> all_ids(total);
    for (unsigned i = 0; i < total; i++) {
        all_vals[i] = static_cast<float>(i);
        all_ids[i]  = i;
    }
    for (unsigned i = total - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(all_vals[i], all_vals[j]);
        std::swap(all_ids[i], all_ids[j]);
    }

    unsigned n_inputs = (total / N) * N;
    std::vector<float> h_in_dist(n_inputs);
    std::vector<uint32_t> h_in_idx(n_inputs);
    for (unsigned i = 0; i < n_inputs; i++) {
        h_in_dist[i] = all_vals[i];
        h_in_idx[i]  = all_ids[i];
    }

    float *d_in_dist, *d_out_dist;
    uint32_t *d_in_idx, *d_out_idx;
    cudaMalloc(&d_in_dist,  n_inputs * sizeof(float));
    cudaMalloc(&d_in_idx,   n_inputs * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, TOPK * sizeof(float));
    cudaMalloc(&d_out_idx,  TOPK * sizeof(uint32_t));

    cudaMemcpy(d_in_dist, h_in_dist.data(), n_inputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx,  h_in_idx.data(),  n_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice);

    kernel_test_reg_merge<REGX, REGY><<<1, 32>>>(
        d_in_dist, d_in_idx, n_inputs,
        d_out_dist, d_out_idx, TOPK);
    cudaDeviceSynchronize();

    std::vector<float> h_out(TOPK);
    cudaMemcpy(h_out.data(), d_out_dist, TOPK * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in_dist); cudaFree(d_in_idx);
    cudaFree(d_out_dist); cudaFree(d_out_idx);

    std::sort(h_out.begin(), h_out.end());
    bool pass = true;
    for (unsigned i = 0; i < TOPK; i++) {
        if (h_out[i] != static_cast<float>(i)) {
            printf("  FAIL at %u: expected %f, got %f\n", i, static_cast<float>(i), h_out[i]);
            pass = false;
            break;
        }
    }
    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 42;

    struct { const char* name; bool (*fn)(unsigned, unsigned); unsigned n_total; } tests[] = {
        {"reg_merge<4,1> n=2560",  run_reg_merge_test<4, 1>,  2560},
        {"reg_merge<4,2> n=2560",  run_reg_merge_test<4, 2>,  2560},
        {"reg_merge<4,4> n=2560",  run_reg_merge_test<4, 4>,  2560},
        {"reg_merge<8,1> n=5120",  run_reg_merge_test<8, 1>,  5120},
        {"reg_merge<8,2> n=5120",  run_reg_merge_test<8, 2>,  5120},
        {"reg_merge<8,4> n=5120",  run_reg_merge_test<8, 4>,  5120},
        {"reg_merge<8,8> n=5120",  run_reg_merge_test<8, 8>,  5120},
        {"reg_merge<16,2> n=10240", run_reg_merge_test<16, 2>, 10240},
        {"reg_merge<16,4> n=10240", run_reg_merge_test<16, 4>, 10240},
    };

    for (auto& t : tests) {
        bool pass = t.fn(t.n_total, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
