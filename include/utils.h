#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath> 
#include <chrono>
#include <cassert>
#include <cstring>
#include <cfloat>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <omp.h>

#include "metric.hpp"
#if !defined(__CUDACC__) && (defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__))
#include <immintrin.h>
#endif

using namespace std;
typedef uint32_t vidType;

template <typename T>
class both_pqueue_t;

#define QG_BQUERY 6

inline int* read_ivecs(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open ivecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    size_t num = file_size / ((d + 1) * sizeof(int));
    int* data = new int[num * d];

    for (size_t i = 0; i < num; ++i) {
        int dim;
        input.read((char*)&dim, sizeof(int));
        assert(dim == d);
        input.read((char*)(data + i * d), sizeof(int) * d);
    }

    d_out = d;
    n_out = num;
    return data;
}

inline float* read_fvecs(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open fvecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    size_t num = file_size / (d * sizeof(float) + sizeof(int));
    float* data = new float[num * d];

    for (size_t i = 0; i < num; ++i) {
        int dim;
        input.read((char*)&dim, sizeof(int));
        assert(dim == d);
        input.read((char*)(data + i * d), sizeof(float) * d);
    }

    d_out = d;
    n_out = num;
    return data;
}

template <typename T = float>
T* read_bin(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    printf("Reading bin file: %s\n", filename);
    if (!input) throw std::runtime_error("Cannot open bin file");
    int32_t num = 0, dim = 0;
    input.read(reinterpret_cast<char*>(&num), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (num <= 0 || dim <= 0)
        throw std::runtime_error("Invalid dimensions read from file");
    size_t total = static_cast<size_t>(num) * dim;
    size_t row_bytes = dim * sizeof(T);
    T *data;
    if (row_bytes % 64 == 0)
        data = static_cast<T*>(std::aligned_alloc(64, sizeof(T) * total));
    else if (row_bytes % 32 == 0)
        data = static_cast<T*>(std::aligned_alloc(32, sizeof(T) * total));
    else
        data = new T[total];
    input.read(reinterpret_cast<char*>(data), sizeof(T) * total);
    //for (size_t i = 0; i < num; i++) in.read((char*)(data+i*dim), sizeof(T)*dim);
    if (!input) throw std::runtime_error("Error reading bin vector data");
    n_out = size_t(num);
    d_out = size_t(dim);
    input.close();
    return data;
}

template <typename T>
void release_bin_buffer(T* ptr, size_t row_dim) {
    if (ptr == nullptr) return;
    const size_t row_bytes = row_dim * sizeof(T);
    if (row_bytes % 64 == 0 || row_bytes % 32 == 0) {
        std::free(ptr);
    } else {
        delete[] ptr;
    }
}

template <typename T>
void write_bin(const char* filename, const T* data, size_t n, size_t d) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open output bin file");
    int32_t nn = static_cast<int32_t>(n);
    int32_t dd = static_cast<int32_t>(d);
    output.write(reinterpret_cast<const char*>(&nn), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(&dd), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(data), sizeof(T) * n * d);
    if (!output) throw std::runtime_error("Error writing bin data");
}

struct Graph {
    size_t n;
    size_t degree;
    vector<vidType> edges;  // size = n * degree

    inline const vidType* get_neighbors(int node) const {
        return edges.data() + node * degree;
    }

    void load_graph_index(const char* gFile) {
        size_t nt = 0, deg = 0;
        vidType* edge_data = read_bin<vidType>(gFile, nt, deg);

        n = nt;
        degree = deg;
        edges.assign(edge_data, edge_data + n * degree);

        std::cout << "Loading graph index from " << gFile 
                  << ": ntotal = " << n << ", maxDeg = " << degree << std::endl;

        release_bin_buffer(edge_data, deg);
    }

    void load_qg_index(const char* filename,
                       size_t num_points, size_t dimension, size_t degree_bound=32,
                       float* data_out = nullptr)
    {
        // Reproduce QG offset computation
        auto next_pow2 = [](size_t v) -> size_t {
            size_t p = 1;
            while (p < v) p <<= 1;
            return p;
        };
        size_t padded_dim    = next_pow2(dimension);
        size_t code_offset   = dimension;  // in floats
        size_t factor_offset = code_offset + padded_dim / 64 * 2 * degree_bound;
        size_t neighbor_offset = factor_offset + 3 * degree_bound; // sizeof(Factor)=3 floats
        size_t row_offset    = neighbor_offset + degree_bound;

        size_t data_bytes = sizeof(float) * num_points * row_offset;
        size_t rotator_bytes = sizeof(float) * padded_dim;
        size_t expected_size = sizeof(vidType) + data_bytes + rotator_bytes;

        std::ifstream input(filename, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open QG index file");

        // Verify file size
        input.seekg(0, std::ios::end);
        size_t file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (file_size != expected_size) {
            throw std::runtime_error(
                "QG index file size mismatch: expected " + std::to_string(expected_size) +
                " but got " + std::to_string(file_size) +
                ". Check num_points/dimension/degree_bound.");
        }

        // 1. Read entry point (stored but we just note it)
        vidType entry_point = 0;
        input.read(reinterpret_cast<char*>(&entry_point), sizeof(vidType));
        std::cout << "QG entry point: " << entry_point << "\n";

        // 2. Read rows, extracting only raw vectors and neighbor IDs
        n = num_points;
        degree = degree_bound;
        edges.resize(n * degree);

        size_t row_bytes = row_offset * sizeof(float);
        std::vector<char> row_buf(row_bytes);

        for (size_t i = 0; i < n; ++i) {
            input.read(row_buf.data(), row_bytes);

            // Extract raw vector if requested
            if (data_out) {
                std::memcpy(data_out + i * dimension,
                            row_buf.data(),
                            dimension * sizeof(float));
            }

            // Extract neighbor IDs (stored as PID/uint32_t reinterpreted in the float array)
            std::memcpy(edges.data() + i * degree,
                        row_buf.data() + neighbor_offset * sizeof(float),
                        degree * sizeof(vidType));
        }

        input.close();
        std::cout << "Loaded QG index from " << filename
                  << ": n=" << n << ", degree=" << degree
                  << ", dim=" << dimension << "\n";
    }

    void prefetch_node(vidType node) {
        const vidType* start = this->get_neighbors(node);

        // compute the number of cache lines needed to load the next node
        constexpr size_t ELEMS_PER_CL = 64 / sizeof(vidType);
        size_t nlines = (degree + ELEMS_PER_CL - 1) / ELEMS_PER_CL;

        for (size_t i = 0; i < nlines; i++) {
            __builtin_prefetch((start + i * ELEMS_PER_CL));
        }
    }
};

struct RunStats {
    double runtime;
    double latency;
    double throughput;
    double recall;
    int64_t total_dc;
    int64_t total_ds;
    int64_t total_extra_dc;
    int64_t total_full_dc;
    int64_t total_iter;
    int64_t total_pruned;
    // fine-grained timers
    double time_search_total = 0;     // total time spent inside search function
    double time_dist_comp = 0;
    double time_nbr_filter = 0;       // fetching neighbor factors
    double time_dist_est = 0;         // actual distance estimation computation
    double time_queue_mgmt = 0;       // frontier + result_pool queue operations
    double time_pruning = 0;          // candidate buffer handling + pruning
    double time_candidate_buffer_mgmt = 0; // curr_candidate clear/push/resize overhead
    double time_nth_element = 0;      // std::nth_element call overhead
    double time_hash_filter = 0;      // is_visited check + add
    int64_t total_estimate_events = 0;
    int64_t total_unique_estimated_nodes = 0;
    int64_t total_duplicate_estimates = 0;
    int64_t total_step_efficiency_events = 0;
    double total_step_efficiency = 0;
    unordered_map<vidType, size_t> estimated_count_by_node;
    unordered_map<size_t, size_t> estimate_count_histogram;
    vector<double> step_efficiency_sums_by_step;
    vector<size_t> step_efficiency_counts_by_step;
};

struct GpuRunStats {
    RunStats common{};
};

struct SearchResult {
    vector<int> neighbors;
    size_t dist_cmps;
    size_t dist_est;
    size_t extra_dist_cmps;
    size_t iters;
    size_t neighbor_pruned;
    // fine-grained timers (seconds)
    double time_search_total = 0;     // total time spent inside search function
    double time_dist_comp = 0;        // exact distance computations
    double time_nbr_filter = 0;       // fetching neighbor factors
    double time_dist_est = 0;         // actual distance estimation computation
    double time_queue_mgmt = 0;       // frontier + result_pool queue operations
    double time_pruning = 0;          // candidate buffer handling + pruning
    double time_candidate_buffer_mgmt = 0; // curr_candidate clear/push/resize overhead
    double time_nth_element = 0;      // std::nth_element call overhead
    double time_hash_filter = 0;      // is_visited check + add
    size_t total_estimate_events = 0;
    size_t total_unique_estimated_nodes = 0;
    size_t total_duplicate_estimates = 0;
    unordered_map<vidType, size_t> estimated_count_by_node;
    unordered_map<size_t, size_t> estimate_count_histogram;
    size_t total_step_efficiency_events = 0;
    double total_step_efficiency = 0;
    vector<double> step_efficiency_by_step;
};

// ─── Scalar fallback distance ───────────────────────────────────────
inline float compute_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float compute_ip_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return -sum;
}

#ifdef __AVX2__
// from DiskANN
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline float compute_distance_vec256(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[8j:8j+7], b[8j:8j+7]
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    float dist = _mm256_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 8; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

inline float compute_ip_distance_vec256(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        if (j + 1 < niters) {
            _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
    }
    float ip = _mm256_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 8; i < dim; ++i) {
        ip += a[i] * b[i];
    }
    return -ip;
}
#endif // __AVX2__

#ifdef __AVX512F__
// adapted from DiskANN
inline float compute_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 16);
    __m512 sum = _mm512_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[16j:16j+15], b[16j:16j+15]
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 16 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 16 * (j + 1)), _MM_HINT_T0);
        }
        __m512 a_vec = _mm512_loadu_ps(a + 16 * j);
        __m512 b_vec = _mm512_loadu_ps(b + 16 * j);
        __m512 tmp_vec = _mm512_sub_ps(a_vec, b_vec);
        sum = _mm512_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    float dist = _mm512_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 16; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

inline float compute_ip_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 16);
    __m512 sum = _mm512_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        if (j + 1 < niters) {
            _mm_prefetch((char *)(a + 16 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 16 * (j + 1)), _MM_HINT_T0);
        }
        __m512 a_vec = _mm512_loadu_ps(a + 16 * j);
        __m512 b_vec = _mm512_loadu_ps(b + 16 * j);
        sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
    }
    float ip = _mm512_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 16; i < dim; ++i) {
        ip += a[i] * b[i];
    }
    return -ip;
}
#endif // __AVX512F__

// ─── Best-available distance function ───────────────────────────────
inline float compute_distance(MetricType metric, int dim, const float* __restrict__ a, const float* __restrict__ b) {
    if (metric == METRIC_IP) {
#ifdef __AVX512F__
        return compute_ip_distance_vec512(dim, a, b);
#elif defined(__AVX2__)
        return compute_ip_distance_vec256(dim, a, b);
#else
        return compute_ip_distance_scalar(dim, a, b);
#endif
    }
#ifdef __AVX512F__
    return compute_distance_vec512(dim, a, b);
#elif defined(__AVX2__)
    return compute_distance_vec256(dim, a, b);
#else
    return compute_distance_scalar(dim, a, b);
#endif
}

inline float compute_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    return compute_distance(g_metric_type, dim, a, b);
}

inline void get_sign_bit_vec(const float* n1, const float* n2, uint32_t* out, size_t dim) {
    size_t words = (dim + 31) / 32;
    memset(out, 0, words * sizeof(uint32_t));

    for (size_t k = 0; k < dim; k++) {
        if (n2[k] > n1[k]) {
            out[k / 32] |= (1U << (31 - (k % 32)));
        }
    }
}

inline int get_matching_bits(const uint32_t* a, const uint32_t* b, size_t words) {
    int dist = 0;
    // #pragma omp simd reduction(- : dist) aligned(a, b : 8)
    for (int i = 0; i < static_cast<int>(words); ++i)
        dist -= __builtin_popcount(a[i] ^ b[i]);
    return dist;
}

// compute recall@K for a single query
inline float recall_at_k(const vector<int>& ann_result, const int* gt_ptr, int K, int gt_k) {
    unordered_set<int> ann_set(ann_result.begin(), ann_result.begin() + K);

    int hit = 0;
    for (int i = 0; i < gt_k; i++) {
        if (ann_set.count(gt_ptr[i])) hit++;
    }
    return float(hit) / K;
}

// compute average recall for all queries
inline float compute_avg_recall(const vector<std::vector<int>>& all_results,
                                const int* all_groundtruth, size_t n_query, int K, int gt_k) {
    float total_recall = 0.0f;
    for (size_t q = 0; q < n_query; q++) {
        const int* gt_ptr = all_groundtruth + q * gt_k;
        total_recall += recall_at_k(all_results[q], gt_ptr, K, gt_k);
    }
    return total_recall / n_query * 100.0f;
}

template <typename T>
float compute_recall(const T* predicted, const int* groundtruth,
                     size_t nq, int topk, int gt_k) {
    size_t correct = 0;
    const int eval_k = std::min(topk, gt_k);
    for (size_t i = 0; i < nq; ++i) {
        for (int j = 0; j < topk; ++j) {
            int pred = int(predicted[i * topk + j]);
            for (int k = 0; k < eval_k; ++k) {
                int gt = groundtruth[i * gt_k + k];
                if (pred == gt) {
                    ++correct;
                    break;
                }
            }
        }
    }
    return float(correct) / (nq * topk);
}

// Fast Walsh-Hadamard Transform (in-place, power-of-2 B, B >= 8)
inline void fwht_inplace(float* data, size_t B) {
#ifdef __AVX512F__
    // ---- AVX-512: 16 floats per register ----

    // len=1: butterfly on adjacent pairs
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_permute_ps(v, 0xB1); // [1,0,3,2,...,15,14]
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xAAAA, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len=2: butterfly on pairs 2 apart
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_permute_ps(v, 0x4E); // [2,3,0,1,6,7,4,5,...]
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xCCCC, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len=4: butterfly on pairs 4 apart (swap 128-bit lanes pairwise)
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_shuffle_f32x4(v, v, 0xB1); // swap 128-bit lane pairs
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xF0F0, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len=8: butterfly on pairs 8 apart (swap 256-bit halves)
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_shuffle_f32x4(v, v, 0x4E); // swap 256-bit halves
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xFF00, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len >= 16: pairs in separate registers
    for (size_t len = 16; len < B; len <<= 1) {
        for (size_t i = 0; i < B; i += len << 1) {
            for (size_t j = 0; j < len; j += 16) {
                __m512 u = _mm512_load_ps(data + i + j);
                __m512 v = _mm512_load_ps(data + i + j + len);
                _mm512_store_ps(data + i + j,       _mm512_add_ps(u, v));
                _mm512_store_ps(data + i + j + len,  _mm512_sub_ps(u, v));
            }
        }
    }

    // // Scale by 1/sqrt(B)
    // __m512 sc = _mm512_set1_ps(1.0f / std::sqrt(static_cast<float>(B)));
    // for (size_t i = 0; i < B; i += 16)
    //     _mm512_store_ps(data + i, _mm512_mul_ps(_mm512_load_ps(data + i), sc));

#elif defined(__AVX2__)
    // ---- AVX2: 8 floats per register ----

    // len=1: butterfly on adjacent pairs
    for (size_t i = 0; i < B; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 s = _mm256_permute_ps(v, 0xB1); // [1,0,3,2,5,4,7,6]
        __m256 r = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xAA);
        _mm256_store_ps(data + i, r);
    }

    // len=2: butterfly on pairs 2 apart
    for (size_t i = 0; i < B; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 s = _mm256_permute_ps(v, 0x4E); // [2,3,0,1,6,7,4,5]
        __m256 r = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xCC);
        _mm256_store_ps(data + i, r);
    }

    // len=4: butterfly on pairs 4 apart (swap 128-bit halves)
    for (size_t i = 0; i < B; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 s = _mm256_permute2f128_ps(v, v, 0x01);
        __m256 r = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xF0);
        _mm256_store_ps(data + i, r);
    }

    // len >= 8: pairs in separate registers
    for (size_t len = 8; len < B; len <<= 1) {
        for (size_t i = 0; i < B; i += len << 1) {
            for (size_t j = 0; j < len; j += 8) {
                __m256 u = _mm256_load_ps(data + i + j);
                __m256 v = _mm256_load_ps(data + i + j + len);
                _mm256_store_ps(data + i + j,       _mm256_add_ps(u, v));
                _mm256_store_ps(data + i + j + len,  _mm256_sub_ps(u, v));
            }
        }
    }

    // Scale by 1/sqrt(B)
    // __m256 sc = _mm256_set1_ps(1.0f / std::sqrt(static_cast<float>(B)));
    // for (size_t i = 0; i < B; i += 8)
    //     _mm256_store_ps(data + i, _mm256_mul_ps(_mm256_load_ps(data + i), sc));

#else
    // ---- Scalar fallback ----
    for (size_t len = 1; len < B; len <<= 1) {
        for (size_t i = 0; i < B; i += len << 1) {
            for (size_t j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    // float scale = 1.0f / std::sqrt(static_cast<float>(B));
    // for (size_t i = 0; i < B; i++) data[i] *= scale;
#endif
}

inline void rotate_vector(const float* src, float* dst, const float* signs, 
                         size_t dim, size_t padded_dim) {
    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= dim; i += 16) {
        __m512 v_src   = _mm512_loadu_ps(src + i);
        __m512 v_signs = _mm512_loadu_ps(signs + i);
        _mm512_store_ps(dst + i, _mm512_mul_ps(v_src, v_signs));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= dim; i += 8) {
        __m256 v_src   = _mm256_loadu_ps(src + i);
        __m256 v_signs = _mm256_loadu_ps(signs + i);
        _mm256_store_ps(dst + i, _mm256_mul_ps(v_src, v_signs));
    }
#endif
    for (; i < dim; i++) {
        dst[i] = src[i] * signs[i];
    }
    if (padded_dim > dim) {
        std::fill(dst + dim, dst + padded_dim, 0.0f);
    }
    fwht_inplace(dst, padded_dim);
}

// adapted from SymphonyQG
inline auto compute_centroid(
    const float* data, size_t num_points, size_t dim, size_t num_threads)
{
    std::vector<std::vector<double>> all_results(num_threads, std::vector<double>(dim, 0));

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        std::vector<double>& cur_results = all_results[tid];
        const float* cur_data = data + (dim * i);
        for (size_t k = 0; k < dim; ++k) {
            cur_results[k] += cur_data[k];
        }
    }

    std::vector<double> centroid_d(dim, 0);
    std::vector<float> centroid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            centroid_d[i] += one_res[i];
        }
    }
    double inv_num_points = 1.0 / static_cast<double>(num_points);
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = static_cast<float>(centroid_d[i] * inv_num_points);
    }

    return centroid;
}

inline vidType exact_nn(const float* data, const float* query, size_t num_points,
    size_t dim, size_t num_threads)
{
    std::vector<std::pair<vidType, float>> best_entries(num_threads, {0, FLT_MAX});

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        auto& cur_entry = best_entries[tid];
        const float* cur_data = data + (dim * i);

        float distance = compute_distance(dim, cur_data, query);
        if (distance < cur_entry.second) {
            cur_entry.first = static_cast<vidType>(i);
            cur_entry.second = distance;
        }
    }

    vidType nearest_neighbor = 0;
    float min_dist = FLT_MAX;
    for (auto& candi : best_entries) {
        if (candi.second < min_dist) {
            nearest_neighbor = candi.first;
            min_dist = candi.second;
        }
    }
    return nearest_neighbor;
}

inline void append_stats_to_csv(const std::string& filename,
    int k, int beam, float prune_ratio, float approx_ratio,
    const RunStats& stats, int l2_iter_threshold = 0, int l2_end_iter_threshold = 0,
    int l2_period = 0, int rerank_top_m = 1)
{
    bool file_exists = std::filesystem::exists(filename);

    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open CSV file " << filename << "\n";
        return;
    }

    // Write header only if file does not exist
    if (!file_exists) {
        out << "TopK,"
            << "Beam_size,"
            << "Prune_ratio,"
            << "Approx_ratio,"
            << "L2_iter_threshold,"
            << "L2_end_iter_threshold,"
            << "L2_period,"
            << "Rerank_top_m,"
            << "Runtime,"
            << "Latency,"
            << "Throughput,"
            << "Recall,"
            << "Total_dist_comp,"
            << "Total_dist_est,"
            << "Total_extra_dist_comp,"
            << "Total_full_dist_comp,"
            << "Total_iter,"
            << "Total_pruned,"
            << "Time_search_total,"
            << "Time_dist_comp,"
            << "Time_nbr_filter,"
            << "Time_dist_est,"
            << "Time_queue_management,"
            << "Time_pruning,"
            << "Time_candidate_buffer_management,"
            << "Time_nth_element,"
            << "Time_hash_filter\n";
    }

    out << k << ","
        << beam << ","
        << prune_ratio << ","
        << approx_ratio << ","
        << l2_iter_threshold << ","
        << l2_end_iter_threshold << ","
        << l2_period << ","
        << rerank_top_m << ","
        << stats.runtime << ","
        << stats.latency << ","
        << stats.throughput << ","
        << stats.recall << ","
        << stats.total_dc << ","
        << stats.total_ds << ","
        << stats.total_extra_dc << ","
        << stats.total_full_dc << ","
        << stats.total_iter << ","
        << stats.total_pruned << ","
        << stats.time_search_total << ","
        << stats.time_dist_comp << ","
        << stats.time_nbr_filter << ","
        << stats.time_dist_est << ","
        << stats.time_queue_mgmt << ","
        << stats.time_pruning << ","
        << stats.time_candidate_buffer_mgmt << ","
        << stats.time_nth_element << ","
        << stats.time_hash_filter << "\n";

    out.close();
}

inline void append_gpu_stats_to_csv(const std::string& filename,
    int k, int beam, const GpuRunStats& gpu_stats,
    float prune_ratio = 0.0f, float approx_ratio = 0.0f)
{
    bool file_exists = std::filesystem::exists(filename);

    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open CSV file " << filename << "\n";
        return;
    }

    if (!file_exists) {
        out << "TopK,"
            << "Beam_size,"
            << "Prune_ratio,"
            << "Approx_ratio,"
            << "L2_iter_threshold,"
            << "L2_end_iter_threshold,"
            << "L2_period,"
            << "Rerank_top_m,"
            << "Runtime,"
            << "Latency,"
            << "Throughput,"
            << "Recall,"
            << "Total_dist_comp,"
            << "Total_dist_est,"
            << "Total_extra_dist_comp,"
            << "Total_full_dist_comp,"
            << "Total_iter,"
            << "Total_pruned,"
            << "Time_search_total,"
            << "Time_dist_comp,"
            << "Time_nbr_filter,"
            << "Time_dist_est,"
            << "Time_queue_management,"
            << "Time_pruning,"
            << "Time_candidate_buffer_management,"
            << "Time_nth_element,"
            << "Time_hash_filter\n";
    }

    const RunStats& stats = gpu_stats.common;
    out << k << ","
        << beam << ","
        << prune_ratio << ","
        << approx_ratio << ","
        << 0 << ","
        << 0 << ","
        << 0 << ","
        << 1 << ","
        << stats.runtime << ","
        << stats.latency << ","
        << stats.throughput << ","
        << stats.recall << ","
        << stats.total_dc << ","
        << stats.total_ds << ","
        << stats.total_extra_dc << ","
        << stats.total_full_dc << ","
        << stats.total_iter << ","
        << stats.total_pruned << ","
        << stats.time_search_total << ","
        << stats.time_dist_comp << ","
        << stats.time_nbr_filter << ","
        << stats.time_dist_est << ","
        << stats.time_queue_mgmt << ","
        << stats.time_pruning << ","
        << stats.time_candidate_buffer_mgmt << ","
        << stats.time_nth_element << ","
        << stats.time_hash_filter << "\n";

    out.close();
}

inline void merge_estimate_histograms(unordered_map<size_t, size_t>& dst,
                                      const unordered_map<size_t, size_t>& src) {
    for (const auto& [estimate_count, node_count] : src) {
        dst[estimate_count] += node_count;
    }
}

inline void merge_step_efficiency_vectors(vector<double>& sum_dst,
                                          vector<size_t>& count_dst,
                                          const vector<double>& src) {
    if (src.empty()) return;
    if (sum_dst.size() < src.size()) {
        sum_dst.resize(src.size(), 0.0);
        count_dst.resize(src.size(), 0);
    }
    for (size_t i = 0; i < src.size(); ++i) {
        sum_dst[i] += src[i];
        count_dst[i]++;
    }
}

inline unordered_map<size_t, size_t> build_estimate_histogram(
    const unordered_map<vidType, size_t>& estimate_counts) {
    unordered_map<size_t, size_t> histogram;
    for (const auto& [node, count] : estimate_counts) {
        (void)node;
        histogram[count]++;
    }
    return histogram;
}

inline size_t count_duplicate_estimates(
    const unordered_map<vidType, size_t>& estimate_counts) {
    size_t duplicates = 0;
    for (const auto& [node, count] : estimate_counts) {
        (void)node;
        if (count > 1) duplicates += count - 1;
    }
    return duplicates;
}

inline vidType debug_candidate_node(const std::pair<vidType, float>& candidate) {
    return candidate.first;
}

inline float debug_candidate_dist(const std::pair<vidType, float>& candidate) {
    return candidate.second;
}

template <typename CandidateT>
inline vidType debug_candidate_node(const CandidateT& candidate) {
    return candidate.vid;
}

template <typename CandidateT>
inline float debug_candidate_dist(const CandidateT& candidate) {
    return candidate.dist;
}

inline void debug_print_candidate_entry(const std::pair<vidType, float>& candidate) {
    cout << "    node=" << debug_candidate_node(candidate)
         << ", est_dist=" << fixed << setprecision(6) << debug_candidate_dist(candidate) << "\n";
}

template <typename CandidateT>
inline void debug_print_candidates(
    int iter, vidType cur_node, const vector<CandidateT>& candidates) {
    cout << "[DEBUG] iteration " << iter << "\n";
    cout << "  expanding node: " << cur_node << "\n";
    cout << "  candidate buffer:";
    if (candidates.empty()) {
        cout << " []\n";
        return;
    }
    cout << "\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        cout << "    idx=" << i
             << ", node=" << debug_candidate_node(candidate)
             << ", dist=" << fixed << setprecision(6) << debug_candidate_dist(candidate) << "\n";
    }
}

template <typename QueueT>
inline void debug_print_beam_state(QueueT& frontier) {
    cout << "  beam state:";
    if (frontier.size() == 0) {
        cout << " []\n";
        return;
    }
    cout << "\n";
    for (int i = 0; i < frontier.size(); i++) {
        cout << "    idx=" << i
             << ", node=" << frontier[i]
             << ", dist=" << fixed << setprecision(6) << frontier.get_dist(i)
             << ", expanded=" << (frontier.is_expanded(i) ? "true" : "false") << "\n";
    }
}

template <typename T>
inline void debug_print_beam_state(both_pqueue_t<T>& frontier) {
    cout << "  beam state:";
    if (frontier.size() == 0) {
        cout << " []\n";
        return;
    }
    cout << "\n";
    for (int i = 0; i < frontier.size(); i++) {
        cout << "    idx=" << i
             << ", node=" << frontier[i]
             << ", rabitq_dist=" << fixed << setprecision(6) << frontier.get_dist(i);
        if (frontier.has_l2_dist(i)) {
            cout << ", l2_dist=" << fixed << setprecision(6) << frontier.get_l2_dist(i);
        } else {
            cout << ", l2_dist=NA";
        }
        cout << ", expanded=" << (frontier.is_expanded(i) ? "true" : "false") << "\n";
    }
}

template <typename QueueT>
inline void debug_print_final_results(QueueT& result_pool) {
    cout << "[DEBUG] final result pool";
    if (result_pool.size() == 0) {
        cout << " []\n";
        return;
    }
    cout << "\n";
    for (int i = 0; i < result_pool.size(); i++) {
        cout << "  idx=" << i
             << ", node=" << result_pool[i]
             << ", dist=" << fixed << setprecision(6) << result_pool.get_dist(i) << "\n";
    }
}

inline void dump_estimate_stats(const std::string& filename, const RunStats& stats) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open estimate stats file " << filename << "\n";
        return;
    }

    out << "# Estimate reuse stats\n";
    out << "total_estimate_events," << stats.total_estimate_events << "\n";
    out << "total_unique_estimated_nodes," << stats.total_unique_estimated_nodes << "\n";
    out << "total_duplicate_estimates," << stats.total_duplicate_estimates << "\n";
    out << "avg_estimates_per_unique_node,"
        << (stats.total_unique_estimated_nodes > 0
                ? (double)stats.total_estimate_events / stats.total_unique_estimated_nodes
                : 0.0)
        << "\n";

    std::vector<std::pair<size_t, size_t>> histogram(
        stats.estimate_count_histogram.begin(), stats.estimate_count_histogram.end());
    std::sort(histogram.begin(), histogram.end());

    out << "\n# Histogram\n";
    out << "estimate_count,node_count\n";
    for (const auto& [estimate_count, node_count] : histogram) {
        out << estimate_count << "," << node_count << "\n";
    }

    if (!stats.estimated_count_by_node.empty()) {
        std::vector<std::pair<vidType, size_t>> node_counts(
            stats.estimated_count_by_node.begin(), stats.estimated_count_by_node.end());
        std::sort(node_counts.begin(), node_counts.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        out << "\n# Node counts\n";
        out << "node_id,estimate_count\n";
        for (const auto& [node_id, estimate_count] : node_counts) {
            out << node_id << "," << estimate_count << "\n";
        }
    } else {
        out << "\n# Node counts\n";
        out << "# omitted because estimate stats are aggregated per query,\n";
        out << "# so node_id counts are no longer meaningful across the whole run.\n";
    }

    out.close();
}

inline void dump_step_efficiency_stats(const std::string& filename, const RunStats& stats) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open step efficiency stats file " << filename << "\n";
        return;
    }

    out << "# Step efficiency stats\n";
    out << "total_step_efficiency_events," << stats.total_step_efficiency_events << "\n";
    out << "total_step_efficiency," << stats.total_step_efficiency << "\n";
    out << "avg_step_efficiency,"
        << (stats.total_step_efficiency_events > 0
                ? stats.total_step_efficiency / stats.total_step_efficiency_events
                : 0.0)
        << "\n";
    out << "max_recorded_step," << stats.step_efficiency_sums_by_step.size() << "\n";

    out << "\n# Histogram\n";
    out << "step,avg_step_efficiency,query_count\n";
    for (size_t i = 0; i < stats.step_efficiency_sums_by_step.size(); ++i) {
        const size_t count = i < stats.step_efficiency_counts_by_step.size()
                                 ? stats.step_efficiency_counts_by_step[i]
                                 : 0;
        const double avg = count > 0 ? stats.step_efficiency_sums_by_step[i] / count : 0.0;
        out << (i + 1) << "," << avg << "," << count << "\n";
    }

    out.close();
}

// Dump a rabitq table (.bin) with per-neighbor breakdown:
//   bits as hex, popcount, and the 3 factor floats (triple_x, factor_dq, factor_vq)
inline void dump_rabitq_table(const std::string& bin_file, const std::string& txt_file,
                              int dim, int degree, int max_nodes = 0) {
    int bitcode_words = dim / 64;
    int words_per_neighbor = bitcode_words + 2;
    int words_per_node = degree * words_per_neighbor;

    std::ifstream input(bin_file, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open: " + bin_file);

    int32_t n, d;
    input.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    if (d != words_per_node)
        throw std::runtime_error("words_per_node mismatch: header=" + std::to_string(d) +
                                 " expected=" + std::to_string(words_per_node));

    std::vector<uint64_t> data(static_cast<size_t>(n) * d);
    input.read(reinterpret_cast<char*>(data.data()), sizeof(uint64_t) * n * d);
    input.close();

    int dump_n = (max_nodes > 0 && max_nodes < n) ? max_nodes : n;

    std::ofstream out(txt_file);
    if (!out) throw std::runtime_error("Cannot open: " + txt_file);

    out << "# " << bin_file << "\n";
    out << "# n=" << n << " words_per_node=" << d
        << " dim=" << dim << " degree=" << degree << "\n";
    out << "# Per neighbor: [" << bitcode_words << " bitcode u64] "
        << "[2 factor u64 = 3 floats: tx, dq, vq]\n\n";

    for (int i = 0; i < dump_n; ++i) {
        out << "=== Node " << i << " ===\n";
        const uint64_t* node_ptr = data.data() + static_cast<size_t>(i) * words_per_node;

        for (int j = 0; j < degree; ++j) {
            const uint64_t* nbr_ptr = node_ptr + j * words_per_neighbor;

            // Popcount across all bitcode words
            int popcount = 0;
            for (int w = 0; w < bitcode_words; ++w)
                popcount += __builtin_popcountll(nbr_ptr[w]);

            // Factors: 3 floats packed in 2 uint64_t
            const float* f = reinterpret_cast<const float*>(nbr_ptr + bitcode_words);

            out << "  nbr[" << std::setw(2) << j << "]: bits=[";
            for (int w = 0; w < bitcode_words; ++w) {
                if (w > 0) out << " ";
                out << "0x" << std::hex << std::setfill('0')
                    << std::setw(16) << nbr_ptr[w] << std::dec;
            }
            out << "] popcount=" << std::setw(3) << popcount
                << " tx=" << std::setw(12) << std::fixed << std::setprecision(6) << f[0]
                << " dq=" << std::setw(12) << f[1]
                << " vq=" << std::setw(12) << f[2] << "\n";
        }
        out << "\n";
    }
    if (dump_n < n) out << "... (" << (n - dump_n) << " more nodes)\n";
    out.close();

    std::cout << "Dumped rabitq table " << bin_file << " to " << txt_file
              << " (" << dump_n << "/" << n << " nodes)\n";
}

static inline void prefetch_l2(const void* addr) {
#if !defined(__CUDACC__) && defined(__SSE2__)
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l2(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}
