#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>
#include <vector>
#include <cfloat>
#include "topk.cuh"
#include "l2.cuh"

// Test rerank with l2_warp closure — same pattern as gpu_search_rabitq.cuh

template <unsigned BX, unsigned REG>
__global__ void kernel_test_rerank_l2(
    const float* __restrict__ d_vectors,  // [N, dim]
    const float* __restrict__ d_query,    // [dim]
    int dim,
    unsigned topk_size,
    unsigned K,
    uint32_t* out_idx,
    float* out_dist)
{
    extern __shared__ char smem[];
    char* ptr = smem;
    SharedBitonicTopK<BX, REG> topk(ptr, topk_size);

    // query in shared memory (same as gpu_search_rabitq)
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    float* query_buf = reinterpret_cast<float*>(ptr);
    ptr += dim * sizeof(float);

    const unsigned lane = threadIdx.x;
    for (unsigned i = lane; i < (unsigned)dim; i += BX)
        query_buf[i] = d_query[i];
    __syncwarp();

    // Insert nodes with fake (reversed) distances
    unsigned chunks = topk_size / (BX * REG);
    for (unsigned c = 0; c < chunks; c++) {
        float d[REG];
        uint32_t x[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned node = c * BX * REG + lane * REG + j;
            d[j] = (node < topk_size) ? static_cast<float>(topk_size - node) : FLT_MAX;
            x[j] = (node < topk_size) ? node : 0xFFFFFFFF;
        }
        topk.merge(d, x);
    }
    __syncwarp();

    // Rerank with l2_warp — same closure pattern as gpu_search_rabitq
    topk.rerank([&](uint32_t node) {
        return l2_warp(dim, query_buf, d_vectors + static_cast<size_t>(node) * dim);
    });
    __syncwarp();

    topk.flush(out_dist, out_idx, K);
}

bool test_l2_rerank(unsigned topk_size, unsigned K, int dim) {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Generate random vectors and query
    std::vector<float> vectors(topk_size * dim);
    std::vector<float> query(dim);
    for (auto& v : vectors) v = normal(rng);
    for (auto& v : query) v = normal(rng);

    // Compute true L2 distances on CPU
    std::vector<float> true_dists(topk_size);
    for (unsigned i = 0; i < topk_size; i++) {
        float d = 0;
        for (int j = 0; j < dim; j++) {
            float diff = query[j] - vectors[i * dim + j];
            d += diff * diff;
        }
        true_dists[i] = d;
    }

    // Upload
    float *d_vectors, *d_query, *d_out_dist;
    uint32_t* d_out_idx;
    cudaMalloc(&d_vectors, topk_size * dim * sizeof(float));
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_out_idx, K * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, K * sizeof(float));
    cudaMemcpy(d_vectors, vectors.data(), topk_size * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query.data(), dim * sizeof(float), cudaMemcpyHostToDevice);

    constexpr unsigned BX = 32, REG = 1;
    size_t smem = SharedBitonicTopK<BX, REG>::shared_mem_bytes_per_instance(topk_size)
                + dim * sizeof(float);  // query_buf
    kernel_test_rerank_l2<BX, REG><<<1, BX, smem>>>(
        d_vectors, d_query, dim, topk_size, K, d_out_idx, d_out_dist);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }

    std::vector<uint32_t> h_idx(K);
    std::vector<float> h_dist(K);
    cudaMemcpy(h_idx.data(), d_out_idx, K * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist.data(), d_out_dist, K * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: K smallest true L2 distances
    std::vector<std::pair<float, uint32_t>> sorted_all(topk_size);
    for (unsigned i = 0; i < topk_size; i++)
        sorted_all[i] = {true_dists[i], i};
    std::sort(sorted_all.begin(), sorted_all.end());

    std::set<uint32_t> expected_set, got_set;
    for (unsigned i = 0; i < K; i++) {
        expected_set.insert(sorted_all[i].second);
        got_set.insert(h_idx[i]);
    }

    bool pass = (expected_set == got_set);
    if (!pass) {
        printf("  FAIL: sets differ\n");
        printf("  Got (first 5):      ");
        for (unsigned i = 0; i < 5 && i < K; i++)
            printf("(node=%u, dist=%.4f) ", h_idx[i], h_dist[i]);
        printf("\n  Expected (first 5): ");
        for (unsigned i = 0; i < 5 && i < K; i++)
            printf("(node=%u, dist=%.4f) ", sorted_all[i].second, sorted_all[i].first);
        printf("\n");

        // Check how many match
        unsigned matches = 0;
        for (auto id : got_set)
            if (expected_set.count(id)) matches++;
        printf("  Overlap: %u/%u\n", matches, K);
    }

    // Check distances are close to CPU-computed values
    bool dist_ok = true;
    for (unsigned i = 0; i < K; i++) {
        float expected_d = true_dists[h_idx[i]];
        float rel = fabsf(h_dist[i] - expected_d) / (expected_d + 1e-10f);
        if (rel > 1e-4f) {
            printf("  dist mismatch at pos %u: node=%u got=%.6f cpu=%.6f rel=%.6f\n",
                   i, h_idx[i], h_dist[i], expected_d, rel);
            dist_ok = false;
            if (i > 3) break;
        }
    }

    cudaFree(d_vectors); cudaFree(d_query); cudaFree(d_out_idx); cudaFree(d_out_dist);
    return pass && dist_ok;
}

int main() {
    int total = 0, passed = 0;

    printf("l2_warp rerank (topk=64, K=10, dim=128):\n");
    total++; if (test_l2_rerank(64, 10, 128)) { printf("  PASS\n"); passed++; }

    printf("l2_warp rerank (topk=128, K=32, dim=128):\n");
    total++; if (test_l2_rerank(128, 32, 128)) { printf("  PASS\n"); passed++; }

    printf("l2_warp rerank (topk=256, K=100, dim=128):\n");
    total++; if (test_l2_rerank(256, 100, 128)) { printf("  PASS\n"); passed++; }

    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
