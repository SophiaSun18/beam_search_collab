#include "fastscan.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err_));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

static constexpr int TEST_NUM_BITS = 6;

// ── test kernel ─────────────────────────────────────────────────────────────

__global__ void kernel_test_fastscan(
    const float* __restrict__ d_query,
    const float* __restrict__ d_signs,
    const uint32_t* __restrict__ d_neighbors,
    const float* __restrict__ d_factors,
    int dim, int num_neighbors, int degree,
    uint32_t dither_seed,
    float exact_dist,
    float* __restrict__ d_output)
{
    extern __shared__ char smem[];
    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    float* query_buf = reinterpret_cast<float*>(ptr);
    ptr += dim * sizeof(float);

    unsigned lane = threadIdx.x;
    for (unsigned i = lane; i < static_cast<unsigned>(dim); i += 32)
        query_buf[i] = d_query[i];
    __syncwarp();

    RabitqQuantizer<32> quantizer(d_signs, dither_seed);
    RabitQFastScan<32, TEST_NUM_BITS> fs(ptr, query_buf, dim, quantizer);
    fs.dot(d_neighbors, num_neighbors, d_factors, degree,
           exact_dist, d_output);
}

// ── CPU reference ───────────────────────────────────────────────────────────

static int cpu_padded_dim(int dim)
{
    int p = 1;
    while (p < dim) p <<= 1;
    return p;
}

static void cpu_wht(float* data, int n)
{
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = 0; j < len; ++j) {
                float u = data[i + j], v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
}

static uint32_t cpu_hash(uint32_t a, uint32_t b)
{
    a ^= b;
    a ^= a >> 16;
    a *= 0x45d9f3b;
    a ^= a >> 16;
    a *= 0x45d9f3b;
    a ^= a >> 16;
    return a;
}

static float cpu_uniform_01(uint32_t h)
{
    return static_cast<float>(h & 0xFFFFFF) / 16777216.0f;
}

static void cpu_quantize(
    const float* query, const float* signs, int dim, int num_bits,
    uint32_t dither_seed, uint8_t* codes_out, int pdim,
    float* low_out, float* width_out, int32_t* sum_q_out)
{
    std::vector<float> rot(pdim);
    for (int d = 0; d < dim; ++d) rot[d] = query[d] * signs[d];
    for (int d = dim; d < pdim; ++d) rot[d] = 0.0f;
    cpu_wht(rot.data(), pdim);

    float lo = rot[0], hi = rot[0];
    for (int d = 1; d < pdim; ++d) {
        lo = std::min(lo, rot[d]);
        hi = std::max(hi, rot[d]);
    }
    float w = (hi - lo) / ((1 << num_bits) - 1);
    float inv_w = 1.0f / w;
    int max_val = (1 << num_bits) - 1;
    int32_t sum = 0;
    for (int d = 0; d < pdim; ++d) {
        float x = (rot[d] - lo) * inv_w;
        float u = cpu_uniform_01(
            cpu_hash(dither_seed, static_cast<uint32_t>(d)));
        int q = static_cast<int>(floorf(x + u));
        if (q < 0) q = 0;
        if (q > max_val) q = max_val;
        codes_out[d] = static_cast<uint8_t>(q);
        sum += q;
    }
    *low_out = lo;
    *width_out = w;
    *sum_q_out = sum;
}

static uint32_t cpu_raw_dot(
    const uint8_t* codes, const uint32_t* nbr, int pdim)
{
    uint32_t sum = 0;
    for (int d = 0; d < pdim; ++d) {
        int w = d / 32;
        int bit = d % 32;
        if ((nbr[w] >> bit) & 1)
            sum += codes[d];
    }
    return sum;
}

static float cpu_estimate(
    uint32_t raw_sum, int32_t sum_q, float width, float low_val,
    float tx, float fdq, float fvq, float exact_dist)
{
    float ip = static_cast<float>(
        (static_cast<int32_t>(raw_sum) << 1) - sum_q);
    float est = tx + exact_dist + fdq * width * ip + fvq * low_val;
    if (tx == FLT_MAX || !std::isfinite(est))
        est = FLT_MAX;
    return est;
}

// ── fuzz test ───────────────────────────────────────────────────────────────

static int fuzz(unsigned seed, int num_trials)
{
    srand(seed);
    int fail = 0;

    for (int trial = 0; trial < num_trials; ++trial) {
        int log2_dim = 5 + (rand() % 6);
        int dim = 1 << log2_dim;
        int pdim = cpu_padded_dim(dim);
        int wpp = pdim / 32;
        int num_neighbors = 1 + (rand() % 32);
        int degree = 32;
        uint32_t dither_seed = static_cast<uint32_t>(rand());
        float exact_dist = static_cast<float>(rand()) / RAND_MAX * 10.0f;

        std::vector<float> query(dim);
        for (auto& v : query)
            v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

        std::vector<float> signs(pdim);
        for (auto& s : signs) s = (rand() & 1) ? 1.0f : -1.0f;

        std::vector<uint32_t> neighbors(num_neighbors * wpp);
        for (auto& w : neighbors) w = static_cast<uint32_t>(rand());

        // factors layout: [tx(degree) | fdq(degree) | fvq(degree)]
        std::vector<float> factors(3 * degree, 0.0f);
        for (int n = 0; n < num_neighbors; ++n) {
            factors[n]              = static_cast<float>(rand()) / RAND_MAX * 5.0f;
            factors[degree + n]     = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            factors[2 * degree + n] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        // CPU reference
        std::vector<uint8_t> cpu_codes(pdim);
        float low_val, width;
        int32_t sum_q;
        cpu_quantize(query.data(), signs.data(), dim, TEST_NUM_BITS,
                     dither_seed, cpu_codes.data(), pdim,
                     &low_val, &width, &sum_q);

        std::vector<float> expected(num_neighbors);
        for (int n = 0; n < num_neighbors; ++n) {
            uint32_t raw = cpu_raw_dot(
                cpu_codes.data(), neighbors.data() + n * wpp, pdim);
            expected[n] = cpu_estimate(
                raw, sum_q, width, low_val,
                factors[n], factors[degree + n], factors[2 * degree + n],
                exact_dist);
        }

        // GPU
        float* d_query;
        float* d_signs;
        uint32_t* d_neighbors;
        float* d_factors;
        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_signs, pdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_neighbors,
            num_neighbors * wpp * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_factors, 3 * degree * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output,
            num_neighbors * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_query, query.data(),
            dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_signs, signs.data(),
            pdim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighbors, neighbors.data(),
            num_neighbors * wpp * sizeof(uint32_t),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_factors, factors.data(),
            3 * degree * sizeof(float), cudaMemcpyHostToDevice));

        size_t smem_bytes = alignof(float) - 1 + dim * sizeof(float)
            + alignof(float) - 1 + pdim * sizeof(float)
            + alignof(uint32_t) - 1
            + TEST_NUM_BITS * wpp * sizeof(uint32_t)
            + sizeof(float) + sizeof(float) + sizeof(int32_t);

        kernel_test_fastscan<<<1, 32, smem_bytes>>>(
            d_query, d_signs, d_neighbors, d_factors,
            dim, num_neighbors, degree, dither_seed,
            exact_dist, d_output);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> output(num_neighbors);
        CUDA_CHECK(cudaMemcpy(output.data(), d_output,
            num_neighbors * sizeof(float),
            cudaMemcpyDeviceToHost));

        for (int n = 0; n < num_neighbors; ++n) {
            float diff = fabsf(output[n] - expected[n]);
            float rel = (expected[n] != 0.0f)
                ? diff / fabsf(expected[n]) : diff;
            if (rel > 1e-5f && diff > 1e-5f) {
                printf("FAIL trial=%d dim=%d nbr=%d: "
                       "got %.6f expected %.6f (diff=%.2e)\n",
                       trial, dim, n, output[n], expected[n], diff);
                fail = 1;
            }
        }

        CUDA_CHECK(cudaFree(d_query));
        CUDA_CHECK(cudaFree(d_signs));
        CUDA_CHECK(cudaFree(d_neighbors));
        CUDA_CHECK(cudaFree(d_factors));
        CUDA_CHECK(cudaFree(d_output));
    }
    return fail;
}

int main()
{
    int fail = 0;

    fail |= fuzz(42, 200);
    printf("%s fuzz (seed=42, 200 trials)\n", fail ? "FAIL" : "PASS");

    int prev = fail;
    fail |= fuzz(123, 200);
    printf("%s fuzz (seed=123, 200 trials)\n",
           (fail != prev) ? "FAIL" : "PASS");

    return fail ? 1 : 0;
}
