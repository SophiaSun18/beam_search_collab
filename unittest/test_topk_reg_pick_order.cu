#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include "topk_reg.cuh"
#include "topk_shared.cuh"

// Stress test: interleave merge+pick on both SharedBitonicTopK and RegBitonicTopK
// with identical inputs. Verify they pick the same nodes in the same order.

template <unsigned BX, unsigned REG>
__global__ void kernel_shared_merge_pick(
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
        float    dist[REG];
        uint32_t idx[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = b * N + tid * REG + j;
            dist[j] = g_input_dist[pos];
            idx[j]  = g_input_idx[pos];
        }
        buf.merge(dist, idx);

        uint32_t pbuf[N];
        #pragma unroll
        for (unsigned j = 0; j < N; ++j) pbuf[j] = 0xFFFFFFFF;
        unsigned found = buf.pick(pbuf, 1);
        uint32_t picked = 0xFFFFFFFF;
        const bool has_pick = found > 0 && pbuf[0] != 0xFFFFFFFF;
        const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
        if (ballot != 0) {
            const int src_lane = __ffs(ballot) - 1;
            picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
        }
        if (tid == 0 && found > 0) {
            g_picked[picked_count] = picked;
            picked_count++;
        }
        __syncthreads();
    }

    if (tid == 0) *g_picked_count = picked_count;
}

template <unsigned REGX, unsigned REGY>
__global__ void kernel_reg_merge_pick(
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

        uint32_t pbuf[REGY];
        #pragma unroll
        for (unsigned j = 0; j < REGY; ++j) pbuf[j] = 0xFFFFFFFF;
        unsigned found = buf.pick(pbuf, 1);
        uint32_t picked = 0xFFFFFFFF;
        const bool has_pick = found > 0 && pbuf[0] != 0xFFFFFFFF;
        const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
        if (ballot != 0) {
            const int src_lane = __ffs(ballot) - 1;
            picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
        }
        if (lane == 0 && found > 0) {
            g_picked[picked_count] = picked;
            picked_count++;
        }
        __syncwarp();
    }

    if (lane == 0) *g_picked_count = picked_count;
}

// REGX=8, REGY=2 → topk=256, batch=64. Compare with SharedBitonicTopK<32,2> topk=256.
bool run_pick_order_test(unsigned n_total, unsigned seed) {
    constexpr unsigned REGX = 8;
    constexpr unsigned REGY = 2;
    constexpr unsigned BX = 32;
    constexpr unsigned REG = REGY;
    constexpr unsigned N = BX * REG;  // 64, same batch size
    constexpr unsigned TOPK = 32 * REGX;  // 256

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
    uint32_t *d_idx;
    uint32_t *d_picked_shared, *d_picked_reg;
    unsigned *d_count_shared, *d_count_reg;
    cudaMalloc(&d_dist, n_inputs * sizeof(float));
    cudaMalloc(&d_idx,  n_inputs * sizeof(uint32_t));
    cudaMalloc(&d_picked_shared, max_picks * sizeof(uint32_t));
    cudaMalloc(&d_picked_reg,    max_picks * sizeof(uint32_t));
    cudaMalloc(&d_count_shared, sizeof(unsigned));
    cudaMalloc(&d_count_reg,    sizeof(unsigned));

    cudaMemcpy(d_dist, h_dist.data(), n_inputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,  h_idx.data(),  n_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_picked_shared, 0xFF, max_picks * sizeof(uint32_t));
    cudaMemset(d_picked_reg,    0xFF, max_picks * sizeof(uint32_t));
    cudaMemset(d_count_shared, 0, sizeof(unsigned));
    cudaMemset(d_count_reg,    0, sizeof(unsigned));

    // Run shared version
    size_t smem = SharedBitonicTopK<BX, REG>::shared_mem_bytes(TOPK);
    kernel_shared_merge_pick<BX, REG><<<1, BX, smem>>>(
        d_dist, d_idx, n_inputs, TOPK,
        d_picked_shared, d_count_shared);
    cudaDeviceSynchronize();

    // Run reg version
    kernel_reg_merge_pick<REGX, REGY><<<1, 32>>>(
        d_dist, d_idx, n_inputs,
        d_picked_reg, d_count_reg);
    cudaDeviceSynchronize();

    // Read back
    unsigned count_shared = 0, count_reg = 0;
    std::vector<uint32_t> picked_shared(max_picks), picked_reg(max_picks);
    cudaMemcpy(&count_shared, d_count_shared, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_reg,    d_count_reg,    sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(picked_shared.data(), d_picked_shared, max_picks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(picked_reg.data(),    d_picked_reg,    max_picks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_dist); cudaFree(d_idx);
    cudaFree(d_picked_shared); cudaFree(d_picked_reg);
    cudaFree(d_count_shared); cudaFree(d_count_reg);

    // Compare — print all picks side-by-side
    bool pass = true;
    if (count_shared != count_reg) {
        printf("  pick count: shared=%u reg=%u\n", count_shared, count_reg);
        pass = false;
    }

    unsigned max_count = (count_shared > count_reg) ? count_shared : count_reg;
    unsigned print_limit = (max_count < 30) ? max_count : 30;
    for (unsigned i = 0; i < print_limit; i++) {
        uint32_t s = (i < count_shared) ? picked_shared[i] : 0xFFFFFFFF;
        uint32_t r = (i < count_reg)    ? picked_reg[i]    : 0xFFFFFFFF;
        const char* tag = (s != r) ? " <-- MISMATCH" : "";
        printf("  pick %3u: shared=%6u  reg=%6u%s\n", i, s, r, tag);
        if (s != r) pass = false;
    }

    if (pass && count_shared == 0) {
        printf("  FAIL: no nodes were picked\n");
        pass = false;
    }

    return pass;
}

// ── flush comparison test ───────────────────────────────────────────────────
// Merge N batches, pick 1 each, then flush K results. Compare shared vs reg.

template <unsigned BX, unsigned REG>
__global__ void kernel_shared_flush_test(
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    unsigned topk, unsigned K,
    float* g_out_dist, uint32_t* g_out_idx)
{
    constexpr unsigned N = BX * REG;
    extern __shared__ char smem[];
    char* ptr = smem;
    const unsigned tid = threadIdx.x;

    SharedBitonicTopK<BX, REG> buf(ptr, topk);
    __syncthreads();

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

        // pick 1
        uint32_t pbuf[N];
        for (unsigned i = 0; i < N; ++i) pbuf[i] = 0xFFFFFFFF;
        buf.pick(pbuf, 1);

        // simulate hash reset: read all ids
        if ((b + 1) % 16 == 0) {
            for (unsigned i = tid; i < topk; i += BX) {
                volatile uint32_t node = buf.id(i);
                (void)node;
            }
        }
        __syncthreads();
    }

    buf.flush(g_out_dist, g_out_idx, K);
}

template <unsigned REGX, unsigned REGY>
__global__ void kernel_reg_flush_test(
    float* g_input_dist, uint32_t* g_input_idx, unsigned n_inputs,
    unsigned K,
    float* g_out_dist, uint32_t* g_out_idx)
{
    constexpr unsigned N = 32 * REGY;
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

        // pick 1
        uint32_t pbuf[REGY];
        for (unsigned i = 0; i < REGY; ++i) pbuf[i] = 0xFFFFFFFF;
        buf.pick(pbuf, 1);

        // simulate hash reset: read all ids
        if ((b + 1) % 16 == 0) {
            for (unsigned i = lane; i < buf.capacity(); i += 32) {
                volatile uint32_t node = buf.id(i);
                (void)node;
            }
        }
        __syncwarp();
    }

    buf.flush(g_out_dist, g_out_idx, K);
}

bool run_flush_test(unsigned n_total, unsigned K, unsigned seed) {
    constexpr unsigned REGX = 8;
    constexpr unsigned REGY = 2;
    constexpr unsigned BX = 32;
    constexpr unsigned REG = REGY;
    constexpr unsigned N = BX * REG;
    constexpr unsigned TOPK = 32 * REGX;

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

    float *d_dist;
    uint32_t *d_idx;
    float *d_out_dist_s, *d_out_dist_r;
    uint32_t *d_out_idx_s, *d_out_idx_r;
    cudaMalloc(&d_dist, n_inputs * sizeof(float));
    cudaMalloc(&d_idx,  n_inputs * sizeof(uint32_t));
    cudaMalloc(&d_out_dist_s, K * sizeof(float));
    cudaMalloc(&d_out_idx_s,  K * sizeof(uint32_t));
    cudaMalloc(&d_out_dist_r, K * sizeof(float));
    cudaMalloc(&d_out_idx_r,  K * sizeof(uint32_t));

    cudaMemcpy(d_dist, h_dist.data(), n_inputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,  h_idx.data(),  n_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice);

    size_t smem = SharedBitonicTopK<BX, REG>::shared_mem_bytes(TOPK);
    kernel_shared_flush_test<BX, REG><<<1, BX, smem>>>(
        d_dist, d_idx, n_inputs, TOPK, K,
        d_out_dist_s, d_out_idx_s);
    cudaDeviceSynchronize();

    kernel_reg_flush_test<REGX, REGY><<<1, 32>>>(
        d_dist, d_idx, n_inputs, K,
        d_out_dist_r, d_out_idx_r);
    cudaDeviceSynchronize();

    std::vector<float> out_dist_s(K), out_dist_r(K);
    std::vector<uint32_t> out_idx_s(K), out_idx_r(K);
    cudaMemcpy(out_dist_s.data(), d_out_dist_s, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_idx_s.data(),  d_out_idx_s,  K * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_dist_r.data(), d_out_dist_r, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_idx_r.data(),  d_out_idx_r,  K * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_dist); cudaFree(d_idx);
    cudaFree(d_out_dist_s); cudaFree(d_out_idx_s);
    cudaFree(d_out_dist_r); cudaFree(d_out_idx_r);

    // Sort both by distance for comparison
    std::vector<std::pair<float,uint32_t>> sorted_s(K), sorted_r(K);
    for (unsigned i = 0; i < K; i++) {
        sorted_s[i] = {out_dist_s[i], out_idx_s[i]};
        sorted_r[i] = {out_dist_r[i], out_idx_r[i]};
    }
    std::sort(sorted_s.begin(), sorted_s.end());
    std::sort(sorted_r.begin(), sorted_r.end());

    bool pass = true;
    unsigned print_limit = (K < 20) ? K : 20;
    for (unsigned i = 0; i < print_limit; i++) {
        const char* tag = (sorted_s[i] != sorted_r[i]) ? " <-- MISMATCH" : "";
        printf("  [%3u] shared=(%6.0f, %6u)  reg=(%6.0f, %6u)%s\n",
               i, sorted_s[i].first, sorted_s[i].second,
               sorted_r[i].first, sorted_r[i].second, tag);
        if (sorted_s[i] != sorted_r[i]) pass = false;
    }
    // Check rest silently
    for (unsigned i = print_limit; i < K; i++) {
        if (sorted_s[i] != sorted_r[i]) {
            printf("  [%3u] shared=(%6.0f, %6u)  reg=(%6.0f, %6u) <-- MISMATCH\n",
                   i, sorted_s[i].first, sorted_s[i].second,
                   sorted_r[i].first, sorted_r[i].second);
            pass = false;
            break;
        }
    }

    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 500;

    printf("=== pick order tests ===\n");
    struct { const char* name; unsigned n_total; } pick_tests[] = {
        {"pick_order n=512",   512},
        {"pick_order n=1024",  1024},
        {"pick_order n=4096",  4096},
        {"pick_order n=16384", 16384},
    };
    for (auto& t : pick_tests) {
        bool pass = run_pick_order_test(t.n_total, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    printf("\n=== flush tests (merge+pick+id+flush) ===\n");
    struct { const char* name; unsigned n_total; unsigned K; } flush_tests[] = {
        {"flush n=4096 K=10",   4096,  10},
        {"flush n=4096 K=100",  4096,  100},
        {"flush n=16384 K=100", 16384, 100},
    };
    for (auto& t : flush_tests) {
        printf("%s:\n", t.name);
        bool pass = run_flush_test(t.n_total, t.K, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
