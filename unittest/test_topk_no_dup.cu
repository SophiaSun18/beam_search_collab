#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include "include/topk.cuh"

// Test: interleaved pick/merge must never yield duplicate node IDs.
// A shuffled list of (dist, id) pairs is merged in batches of N.
// Between each merge, pick(1) yields one node, recorded in global memory.
// Host verifies no duplicates across all yielded nodes.

template <unsigned BX, unsigned REG>
__global__ void kernel_test_no_dup(
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    unsigned topk,
    uint32_t* g_picked, unsigned* g_picked_count)
{
    constexpr unsigned N = BX * REG;
    extern __shared__ char smem[];
    char* ptr = smem;

    const unsigned tid = threadIdx.x;

    SharedBitonicTopK<BX, REG> buf(ptr, topk);
    __syncthreads();

    unsigned batches = n_inputs / N;
    unsigned picked_count = 0;

    for (unsigned b = 0; b < batches; b++) {
        // merge next batch
        float    dist[REG];
        uint32_t idx[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = b * N + tid * REG + j;
            dist[j] = g_input_dist[pos];
            idx[j]  = g_input_idx[pos];
        }
        buf.merge(dist, idx);

        // pick 1 node
        uint32_t pbuf[N];
        unsigned found = buf.pick(pbuf, 1);
        if (tid == 0 && found > 0) {
            g_picked[picked_count] = pbuf[0];
            picked_count++;
        }
        __syncthreads();
    }

    if (tid == 0) {
        *g_picked_count = picked_count;
    }
}

template <unsigned BX, unsigned REG>
bool run_no_dup_test(unsigned topk, unsigned n_total, unsigned seed) {
    constexpr unsigned N = BX * REG;
    std::mt19937 rng(seed);

    // generate shuffled (dist, id) pairs
    unsigned n_inputs = (n_total / N) * N;  // round down to multiple of N
    std::vector<float> h_dist(n_inputs);
    std::vector<uint32_t> h_idx(n_inputs);
    for (unsigned i = 0; i < n_inputs; i++) {
        h_dist[i] = (float)i;
        h_idx[i]  = i;
    }
    // shuffle
    for (unsigned i = n_inputs - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(h_dist[i], h_dist[j]);
        std::swap(h_idx[i], h_idx[j]);
    }

    unsigned max_picks = n_inputs / N;  // one pick per batch

    // allocate device memory
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

    size_t smem_size = SharedBitonicTopK<BX, REG>::shared_mem_bytes(topk);
    kernel_test_no_dup<BX, REG><<<1, BX, smem_size>>>(
        d_dist, d_idx, n_inputs, topk,
        d_picked, d_picked_count);
    cudaDeviceSynchronize();

    // read back results
    unsigned h_picked_count = 0;
    std::vector<uint32_t> h_picked(max_picks);
    cudaMemcpy(&h_picked_count, d_picked_count, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_picked.data(), d_picked, max_picks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_dist); cudaFree(d_idx);
    cudaFree(d_picked); cudaFree(d_picked_count);

    // verify no duplicates
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

    struct { const char* name; bool (*fn)(unsigned, unsigned, unsigned); unsigned topk; unsigned n_total; } tests[] = {
        {"no_dup<32,2> topk=128 n=256",   run_no_dup_test<32,2>,   128, 256},
        {"no_dup<32,2> topk=256 n=512",   run_no_dup_test<32,2>,   256, 512},
        {"no_dup<32,4> topk=128 n=256",   run_no_dup_test<32,4>,   128, 256},
        {"no_dup<32,4> topk=256 n=512",   run_no_dup_test<32,4>,   256, 512},
        {"no_dup<32,4> topk=512 n=1024",  run_no_dup_test<32,4>,   512, 1024},
        {"no_dup<64,2> topk=128 n=256",   run_no_dup_test<64,2>,   128, 256},
        {"no_dup<64,4> topk=256 n=512",   run_no_dup_test<64,4>,   256, 512},
        {"no_dup<64,4> topk=512 n=1024",  run_no_dup_test<64,4>,   512, 1024},
    };

    for (auto& t : tests) {
        bool pass = t.fn(t.topk, t.n_total, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
