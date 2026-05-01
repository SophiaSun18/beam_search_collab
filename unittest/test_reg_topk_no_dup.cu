#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include "topk_reg.cuh"

template <unsigned REGX, unsigned REGY>
__global__ void kernel_test_reg_no_dup(
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    uint32_t* g_picked, unsigned* g_picked_count)
{
    constexpr unsigned N = 32 * REGY;
    const unsigned lane = threadIdx.x;

    RegBitonicTopK<REGX, REGY> buf;

    unsigned batches = n_inputs / N;
    unsigned picked_count = 0;

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

        // pick 1 node
        uint32_t pbuf[N];
        #pragma unroll
        for (unsigned j = 0; j < N; ++j) pbuf[j] = 0xFFFFFFFF;
        unsigned found = buf.pick(pbuf, 1);
        uint32_t picked = 0xFFFFFFFF;
        if (found > 0 && pbuf[0] != 0xFFFFFFFF) {
            picked = pbuf[0];
        }
        // Broadcast from lane 0 (pick writes to lane 0)
        picked = __shfl_sync(0xffffffff, picked, 0);
        if (lane == 0 && found > 0) {
            g_picked[picked_count] = picked;
            picked_count++;
        }
        __syncwarp();
    }

    if (lane == 0) {
        *g_picked_count = picked_count;
    }
}

template <unsigned REGX, unsigned REGY>
bool run_reg_no_dup_test(unsigned n_total, unsigned seed) {
    constexpr unsigned N = 32 * REGY;
    std::mt19937 rng(seed);

    unsigned n_inputs = (n_total / N) * N;
    std::vector<float> h_dist(n_inputs);
    std::vector<uint32_t> h_idx(n_inputs);
    for (unsigned i = 0; i < n_inputs; i++) {
        h_dist[i] = static_cast<float>(i);
        h_idx[i]  = i;
    }
    for (unsigned i = n_inputs - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(h_dist[i], h_dist[j]);
        std::swap(h_idx[i], h_idx[j]);
    }

    unsigned max_picks = n_inputs / N;

    float *d_dist;
    uint32_t *d_idx, *d_picked;
    unsigned *d_picked_count;
    cudaMalloc(&d_dist, n_inputs * sizeof(float));
    cudaMalloc(&d_idx,  n_inputs * sizeof(uint32_t));
    cudaMalloc(&d_picked, max_picks * sizeof(uint32_t));
    cudaMalloc(&d_picked_count, sizeof(unsigned));

    cudaMemcpy(d_dist, h_dist.data(), n_inputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,  h_idx.data(),  n_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_picked, 0, max_picks * sizeof(uint32_t));
    cudaMemset(d_picked_count, 0, sizeof(unsigned));

    kernel_test_reg_no_dup<REGX, REGY><<<1, 32>>>(
        d_dist, d_idx, n_inputs,
        d_picked, d_picked_count);
    cudaDeviceSynchronize();

    unsigned h_picked_count = 0;
    std::vector<uint32_t> h_picked(max_picks);
    cudaMemcpy(&h_picked_count, d_picked_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_picked.data(), d_picked, max_picks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_dist); cudaFree(d_idx);
    cudaFree(d_picked); cudaFree(d_picked_count);

    std::set<uint32_t> seen;
    bool pass = true;
    for (unsigned i = 0; i < h_picked_count; i++) {
        if (seen.count(h_picked[i])) {
            printf("  FAIL: duplicate node ID %u at pick %u\n", h_picked[i], i);
            pass = false;
            break;
        }
        seen.insert(h_picked[i]);
    }

    if (pass && h_picked_count == 0) {
        printf("  FAIL: no nodes were picked\n");
        pass = false;
    }

    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 200;

    struct { const char* name; bool (*fn)(unsigned, unsigned); unsigned n_total; } tests[] = {
        {"reg_no_dup<4,1> n=256",   run_reg_no_dup_test<4, 1>,   256},
        {"reg_no_dup<4,2> n=512",   run_reg_no_dup_test<4, 2>,   512},
        {"reg_no_dup<8,1> n=256",   run_reg_no_dup_test<8, 1>,   256},
        {"reg_no_dup<8,2> n=512",   run_reg_no_dup_test<8, 2>,   512},
        {"reg_no_dup<8,4> n=1024",  run_reg_no_dup_test<8, 4>,   1024},
        {"reg_no_dup<8,8> n=2048",  run_reg_no_dup_test<8, 8>,   2048},
        {"reg_no_dup<16,2> n=512",  run_reg_no_dup_test<16, 2>,  512},
        {"reg_no_dup<16,4> n=1024", run_reg_no_dup_test<16, 4>,  1024},
    };

    for (auto& t : tests) {
        bool pass = t.fn(t.n_total, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
