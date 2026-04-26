#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include "include/topk.cuh"

// Test: after multiple merges, no element id should appear twice in the buffer.
// We generate values 0..total-1 with distinct ids, merge them in,
// then check the output has no duplicate ids.

template <unsigned N, unsigned BX, unsigned REG>
__global__ void kernel_test_no_dup(
    float* g_buf_dist, uint32_t* g_buf_idx, unsigned topk,
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    float* g_out_dist, uint32_t* g_out_idx)
{
    extern __shared__ char smem[];
    float*    s_dist = (float*)smem;
    uint32_t* s_idx  = (uint32_t*)(s_dist + topk);

    const unsigned tid = threadIdx.x;

    for (unsigned i = tid; i < topk; i += BX) {
        s_dist[i] = g_buf_dist[i];
        s_idx[i]  = g_buf_idx[i];
    }
    __syncthreads();

    SharedBitonicTopK<N, BX, REG> buf(topk, s_dist, s_idx);

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

    buf.flush(g_out_dist, g_out_idx, topk);
}

template <unsigned N, unsigned BX, unsigned REG>
bool run_no_dup_test(unsigned topk, unsigned n_total, unsigned seed) {
    static_assert(N == BX * REG);
    std::mt19937 rng(seed);

    unsigned total = topk + n_total;
    std::vector<float> all_vals(total);
    std::vector<uint32_t> all_ids(total);
    for (unsigned i = 0; i < total; i++) {
        all_vals[i] = (float)i;
        all_ids[i]  = i;  // distinct ids
    }
    for (unsigned i = total - 1; i > 0; i--) {
        unsigned j = rng() % (i + 1);
        std::swap(all_vals[i], all_vals[j]);
        std::swap(all_ids[i], all_ids[j]);
    }

    // Buffer: first topk, sorted descending
    std::vector<std::pair<float,uint32_t>> buf_pairs(topk);
    for (unsigned i = 0; i < topk; i++)
        buf_pairs[i] = {all_vals[i], all_ids[i]};
    std::sort(buf_pairs.begin(), buf_pairs.end(),
        [](auto& a, auto& b){ return a.first > b.first; });

    std::vector<float> h_buf_dist(topk);
    std::vector<uint32_t> h_buf_idx(topk);
    for (unsigned i = 0; i < topk; i++) {
        h_buf_dist[i] = buf_pairs[i].first;
        h_buf_idx[i]  = buf_pairs[i].second;
    }

    unsigned n_inputs = (n_total / N) * N;
    std::vector<float> h_in_dist(n_inputs);
    std::vector<uint32_t> h_in_idx(n_inputs);
    for (unsigned b = 0; b < n_inputs / N; b++) {
        std::vector<std::pair<float,uint32_t>> batch(N);
        for (unsigned i = 0; i < N; i++) {
            unsigned src = topk + b * N + i;
            batch[i] = {all_vals[src], all_ids[src]};
        }
        std::sort(batch.begin(), batch.end());
        for (unsigned i = 0; i < N; i++) {
            h_in_dist[b * N + i] = batch[i].first;
            h_in_idx[b * N + i]  = batch[i].second;
        }
    }

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

    size_t smem_size = topk * (sizeof(float) + sizeof(uint32_t));
    kernel_test_no_dup<N, BX, REG><<<1, BX, smem_size>>>(
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

    // Check no duplicate ids
    std::set<uint32_t> seen;
    bool pass = true;
    for (unsigned i = 0; i < topk; i++) {
        if (seen.count(h_out_idx[i])) {
            printf("  FAIL: duplicate id %u at position %u\n", h_out_idx[i], i);
            pass = false;
            break;
        }
        seen.insert(h_out_idx[i]);
    }

    // Also verify ids match their distances
    for (unsigned i = 0; i < topk && pass; i++) {
        if (h_out_dist[i] != (float)h_out_idx[i]) {
            printf("  FAIL: dist/idx mismatch at %u: dist=%f idx=%u\n",
                   i, h_out_dist[i], h_out_idx[i]);
            pass = false;
        }
    }

    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 200;

    struct { const char* name; bool (*fn)(unsigned, unsigned, unsigned); unsigned topk; unsigned n_total; } tests[] = {
        {"no_dup<64,32,2> topk=128 n=256",   run_no_dup_test<64,32,2>,   128, 256},
        {"no_dup<64,32,2> topk=256 n=512",   run_no_dup_test<64,32,2>,   256, 512},
        {"no_dup<128,32,4> topk=128 n=256",  run_no_dup_test<128,32,4>,  128, 256},
        {"no_dup<128,32,4> topk=256 n=512",  run_no_dup_test<128,32,4>,  256, 512},
        {"no_dup<128,32,4> topk=512 n=1024", run_no_dup_test<128,32,4>,  512, 1024},
        {"no_dup<128,64,2> topk=128 n=256",  run_no_dup_test<128,64,2>,  128, 256},
        {"no_dup<256,64,4> topk=256 n=512",  run_no_dup_test<256,64,4>,  256, 512},
        {"no_dup<256,64,4> topk=512 n=1024", run_no_dup_test<256,64,4>,  512, 1024},
    };

    for (auto& t : tests) {
        bool pass = t.fn(t.topk, t.n_total, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
