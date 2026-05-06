#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>
#include <vector>
#include <cfloat>
#include "topk.cuh"

// Flush ALL elements to see the full sorted buffer
template <unsigned REGX>
__global__ void kernel_flush_all(
    const float* true_dists,
    float* out_dist,
    uint32_t* out_idx)
{
    constexpr unsigned TOPK = 32 * REGX;
    const unsigned lane = threadIdx.x & 31;

    char* dummy = nullptr;
    RegBitonicTopK<REGX, REGX> topk(dummy, TOPK);

    // Insert with reversed distances
    float d[REGX];
    uint32_t x[REGX];
    #pragma unroll
    for (unsigned j = 0; j < REGX; j++) {
        unsigned node = lane * REGX + j;
        d[j] = static_cast<float>(TOPK - node);
        x[j] = node;
    }
    topk.merge(d, x);

    // Rerank
    topk.rerank([&](uint32_t node) {
        return true_dists[node];
    });

    // Flush everything
    topk.flush(out_dist, out_idx, TOPK);
}

template <unsigned REGX>
bool test_reg_rerank(unsigned K) {
    constexpr unsigned TOPK = 32 * REGX;
    std::mt19937 rng(123);
    std::vector<float> true_dists(TOPK);
    for (unsigned i = 0; i < TOPK; i++)
        true_dists[i] = static_cast<float>(rng() % 10000);

    float* d_true;
    uint32_t* d_out_idx;
    float* d_out_dist;
    cudaMalloc(&d_true, TOPK * sizeof(float));
    cudaMalloc(&d_out_idx, TOPK * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, TOPK * sizeof(float));
    cudaMemcpy(d_true, true_dists.data(), TOPK * sizeof(float), cudaMemcpyHostToDevice);

    kernel_flush_all<REGX><<<1, 32>>>(d_true, d_out_dist, d_out_idx);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_idx(TOPK);
    std::vector<float> h_dist(TOPK);
    cudaMemcpy(h_idx.data(), d_out_idx, TOPK * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist.data(), d_out_dist, TOPK * sizeof(float), cudaMemcpyDeviceToHost);

    // Check distances match true values
    bool dists_ok = true;
    for (unsigned i = 0; i < TOPK; i++) {
        if (h_dist[i] != true_dists[h_idx[i]]) {
            printf("  dist mismatch at pos %u: node=%u got=%.0f expected=%.0f\n",
                   i, h_idx[i], h_dist[i], true_dists[h_idx[i]]);
            dists_ok = false;
            if (i > 3) break;
        }
    }

    // Check the K smallest are in the tail (flush outputs from tail of descending)
    // flush[0] = buffer[TOPK - TOPK + 0] = buffer[0] = largest (descending)
    // flush[TOPK-1] = buffer[TOPK-1] = smallest
    // So flush output is descending. The last K elements should be the K smallest.
    std::set<uint32_t> expected_set, got_set;
    std::vector<std::pair<float, uint32_t>> sorted_all(TOPK);
    for (unsigned i = 0; i < TOPK; i++)
        sorted_all[i] = {true_dists[i], i};
    std::sort(sorted_all.begin(), sorted_all.end());

    for (unsigned i = 0; i < K; i++) {
        expected_set.insert(sorted_all[i].second);
        got_set.insert(h_idx[TOPK - K + i]);  // last K from flush = smallest K
    }

    bool set_ok = (expected_set == got_set);
    if (!set_ok) {
        printf("  Set mismatch! Last %u elements of flush don't match %u smallest\n", K, K);
        printf("  Last 5 from flush: ");
        for (unsigned i = TOPK - 5; i < TOPK; i++)
            printf("(node=%u,d=%.0f) ", h_idx[i], h_dist[i]);
        printf("\n  Expected smallest 5: ");
        for (unsigned i = 0; i < 5; i++)
            printf("(node=%u,d=%.0f) ", sorted_all[i].second, sorted_all[i].first);
        printf("\n");
    }

    // Check all nodes are unique
    std::set<uint32_t> all_nodes(h_idx.begin(), h_idx.end());
    bool unique = (all_nodes.size() == TOPK);
    if (!unique)
        printf("  Not all nodes unique! Got %zu unique out of %u\n", all_nodes.size(), TOPK);

    // Check full buffer is descending (flush output)
    bool descending = true;
    for (unsigned i = 1; i < TOPK; i++) {
        if (h_dist[i] > h_dist[i-1]) {
            printf("  Not descending at pos %u: %.0f > %.0f\n", i, h_dist[i], h_dist[i-1]);
            descending = false;
            break;
        }
    }

    printf("  dists_correct=%s unique=%s descending=%s set_ok=%s\n",
           dists_ok?"Y":"N", unique?"Y":"N", descending?"Y":"N", set_ok?"Y":"N");

    cudaFree(d_true); cudaFree(d_out_idx); cudaFree(d_out_dist);
    return dists_ok && unique && descending && set_ok;
}

int main() {
    int total = 0, passed = 0;

    printf("RegBitonicTopK<2,2> (TOPK=64, check K=16):\n");
    total++; if (test_reg_rerank<2>(16)) { printf("  PASS\n"); passed++; }

    printf("RegBitonicTopK<4,4> (TOPK=128, check K=32):\n");
    total++; if (test_reg_rerank<4>(32)) { printf("  PASS\n"); passed++; }

    printf("RegBitonicTopK<8,8> (TOPK=256, check K=64):\n");
    total++; if (test_reg_rerank<8>(64)) { printf("  PASS\n"); passed++; }

    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
