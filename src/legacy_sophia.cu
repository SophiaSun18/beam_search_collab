// legacy_sophia.cu — Flattened from sophia branch
// All sophia headers and sources inlined into a single compilation unit.
// Exposes run_sophia(json) for use by the merged gpu_cli.

#pragma nv_diag_suppress 177

// ═══════════════════════════════════════════════════════════════════════
// ── include/utils.h (inlined) ──
// ═══════════════════════════════════════════════════════════════════════


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
#include <filesystem>
#include <iomanip>
#include <omp.h>
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

        delete[] edge_data;
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
    int64_t gpu_l2_calls = 0;
    int64_t gpu_l2_cycles = 0;
    int64_t gpu_est_calls = 0;
    int64_t gpu_est_cycles = 0;
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
    return _mm256_reduce_add_ps(sum);
}
#endif // __AVX2__

#ifdef __AVX512F__
// adapted from DiskANN
inline float compute_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    // assume size is divisible by 16
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
    return _mm512_reduce_add_ps(sum);
}
#endif // __AVX512F__

// ─── Best-available distance function ───────────────────────────────
inline float compute_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
#ifdef __AVX512F__
    return compute_distance_vec512(dim, a, b);
#elif defined(__AVX2__)
    return compute_distance_vec256(dim, a, b);
#else
    return compute_distance_scalar(dim, a, b);
#endif
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
    size_t total_dups = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<int> gt_set;
        gt_set.reserve(static_cast<size_t>(gt_k));
        for (int k = 0; k < gt_k; k++)
            gt_set.insert(groundtruth[i * gt_k + k]);
        std::unordered_set<int> seen;
        seen.reserve(static_cast<size_t>(topk));
        for (int j = 0; j < topk; j++) {
            int pred = static_cast<int>(predicted[i * topk + j]);
            if (pred == -1) continue;
            if (!seen.insert(pred).second) { total_dups++; continue; }
            if (gt_set.count(pred) != 0) correct++;
        }
    }
    if (total_dups > 0)
        std::cout << "  duplicates: " << total_dups << " total ("
                  << static_cast<double>(total_dups) / nq << " avg/query)" << std::endl;
    return static_cast<float>(correct) / (nq * std::min(topk, gt_k));
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
#ifdef __AVX512F__
    for (size_t i = 0; i + 16 <= dim; i += 16) {
        __m512 v_src   = _mm512_load_ps(src + i);
        __m512 v_signs = _mm512_loadu_ps(signs + i);
        _mm512_store_ps(dst + i, _mm512_mul_ps(v_src, v_signs));
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i + 8 <= dim; i += 8) {
        __m256 v_src   = _mm256_load_ps(src + i);
        __m256 v_signs = _mm256_loadu_ps(signs + i);
        _mm256_store_ps(dst + i, _mm256_mul_ps(v_src, v_signs));
    }
#else
    for (size_t i = 0; i < dim; i++) {
        dst[i] = src[i] * signs[i];
    }
#endif
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
            << "Time_hash_filter,"
            << "\n";
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
    int k, int beam, const GpuRunStats& gpu_stats)
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
            << "Time_hash_filter,"
            << "Gpu_l2_calls,"
            << "Gpu_l2_cycles,"
            << "Gpu_est_calls,"
            << "Gpu_est_cycles,"
            << "\n";
    }

    const RunStats& stats = gpu_stats.common;
    out << k << ","
        << beam << ","
        << 0 << ","
        << 0 << ","
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
        << stats.time_hash_filter << ","
        << gpu_stats.gpu_l2_calls << ","
        << gpu_stats.gpu_l2_cycles << ","
        << gpu_stats.gpu_est_calls << ","
        << gpu_stats.gpu_est_cycles << "\n";

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

inline void debug_print_candidate_entry(const std::pair<vidType, float>& candidate) {
    cout << "    node=" << candidate.first
         << ", est_dist=" << fixed << setprecision(6) << candidate.second << "\n";
}

template <typename CandidateT>
inline void debug_print_candidate_entry(const CandidateT& candidate) {
    cout << "    node=" << candidate.vid
         << ", est_dist=" << fixed << setprecision(6) << candidate.dist << "\n";
}

template <typename CandidateT>
inline void debug_print_candidates(
    int iter, vidType cur_node, const vector<CandidateT>& candidates) {
    cout << "[DEBUG] iteration " << iter << "\n";
    cout << "  expanding node: " << cur_node << "\n";
    cout << "  neighbors:";
    if (candidates.empty()) {
        cout << " []\n";
        return;
    }
    cout << "\n";
    for (const auto& candidate : candidates) {
        debug_print_candidate_entry(candidate);
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
    cout << "[DEBUG] final result pool:";
    if (result_pool.size() == 0) {
        cout << " []\n";
        return;
    }
    cout << "\n";
    for (int i = 0; i < result_pool.size(); i++) {
        cout << "  rank=" << i
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


// ═══════════════════════════════════════════════════════════════════════
// ── include/utils.cuh (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

#include <cfloat>
#include <cassert>
#include <climits>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define FULL_MASK     0xffffffff
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define GPU_HASH_BASE_BITLEN 10
#define GPU_HASH_BITLEN_FOR_TOPK(topk)                                            \
  (((topk) * 4 > 3 * (1 << GPU_HASH_BASE_BITLEN)) ? (GPU_HASH_BASE_BITLEN + 1) : GPU_HASH_BASE_BITLEN)
   
//#define FLT_MAX 3.40282347e+38
#ifndef GPU_SEARCH_STATS
#define GPU_SEARCH_STATS 1
#endif

#ifndef GPU_DISTANCE_TIMING
#define GPU_DISTANCE_TIMING 0
#endif

#define MAX_INDEX UINT_MAX

#define SHFL_DOWN(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define SHFL(val, lane) __shfl_sync(FULL_MASK, val, lane)
#define SHFL_XOR(val, mask) __shfl_xor_sync(FULL_MASK, val, mask)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 900)
#define GPU_MAX_WARPS_PER_SM 48
#else
#define GPU_MAX_WARPS_PER_SM 64
#endif

#define GPU_MIN_BLOCKS_PER_SM_FOR(block_size) (GPU_MAX_WARPS_PER_SM / WARPS_PER_BLOCK)
#define GPU_LAUNCH_BOUNDS(block_size) __launch_bounds__((block_size), GPU_MIN_BLOCKS_PER_SM_FOR(block_size))

typedef unsigned long long gpu_long_t; // for counters
typedef uint32_t vidType;
using vid_t = vidType;

__device__ __forceinline__ gpu_long_t gpu_globaltimer() {
  gpu_long_t ticks;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
  return ticks;
}

enum GpuDistanceTimingIndex : int {
  GPU_TIMING_L2_CALLS = 0,
  GPU_TIMING_L2_CYCLES = 1,
  GPU_TIMING_EST_CALLS = 2,
  GPU_TIMING_EST_CYCLES = 3,
  GPU_TIMING_EST_REGION_CANDIDATES = 4,
  GPU_TIMING_EST_REGION_CYCLES = 5,
  GPU_TIMING_QUERY_PREP_CYCLES = 6,
  GPU_TIMING_SORT_CYCLES = 7,
  GPU_TIMING_PICK_CYCLES = 8,
  GPU_TIMING_VISITED_CYCLES = 9,
  GPU_TIMING_STAGE_CYCLES = 10,
  GPU_TIMING_FINAL_SORT_CYCLES = 11,
  GPU_TIMING_RESULT_POOL_CYCLES = 12,
  GPU_TIMING_TENSOR_QUERY_TILE_CYCLES = 13,
  GPU_TIMING_TENSOR_UNPACK_CYCLES = 14,
  GPU_TIMING_TENSOR_MMA_CYCLES = 15,
  GPU_TIMING_TENSOR_FINAL_FACTOR_CYCLES = 16,
  GPU_TIMING_COUNT = 17
};

enum GpuSearchStatIndex : int {
  GPU_SEARCH_STAT_DIST_CMPS = 0,
  GPU_SEARCH_STAT_DIST_EST = 1,
  GPU_SEARCH_STAT_ITERATIONS = 2,
  GPU_SEARCH_STAT_COUNT = 3
};

struct QueryFactors {
  float* rotated_query = nullptr;
  uint8_t* quantized_query = nullptr;
  uint8_t* lut = nullptr;
  float low_val = 0.0f;
  float high_val = 0.0f;
  float width = 0.0f;
  int32_t sum_q = 0;
};

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);              \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                           \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if ( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
    } }

template <typename T>
class GraphGPU {
protected:
  vidType nv;           // number of vertices
  vidType maxDeg;       // maximun degree
  vidType *d_edges;
public:
  GraphGPU(int n, int d, vidType *h_edges) : nv(n), maxDeg(d) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges, n * d * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, n * d * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  inline __device__ __host__ vidType V() { return nv; }
  inline __device__ __host__ vidType get_max_degree() { return maxDeg; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_edges + vid * maxDeg; }
};

template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T* a, const T* b) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T val = 0.; 
  for(int i = thread_lane; i < dim; i += WARP_SIZE)
    val += (a[i] - b[i]) * (a[i] - b[i]);
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8); 
  sum += SHFL_DOWN(sum, 4); 
  sum += SHFL_DOWN(sum, 2); 
  sum += SHFL_DOWN(sum, 1); 
  sum  = SHFL(sum, 0); 
  return sum;
}

/*-------------------------------------------- sort --------------------------------------------*/
typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

template <typename T>
__device__ inline void print_buffer(T* buffer, uint32_t buffer_size, uint32_t index_mask=0x80000000) {
    if constexpr (std::is_same<T, INDEX_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%d, ", buffer[i] & ~index_mask);
        }
    } else if constexpr (std::is_same<T, DISTANCE_T>::value || std::is_same<T, DATA_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%f, ", buffer[i]);
        }
    } else {
        printf("Unsupported type\n");
        assert(false);
    }
    printf("\n");
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc) {
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc) {
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0;
    k0               = k1;
    k1               = tmp_k;
    const auto tmp_v = v0;
    v0               = v1;
    v1               = tmp_v;
  }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc) {
  if (_N == 2) {
    constexpr unsigned N = 2;
    const auto lane_id   = threadIdx.x % warp_size;

    if (range == 1) {
      const auto p = ((lane_id & 1) == 0);
      swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
      return;
    }

    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
      }
    }
    const auto p = ((lane_id & b) == 0);
    swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
  } else if (_N == 4) {
    constexpr unsigned N = 4;
    const auto lane_id = threadIdx.x % warp_size;
    if (range == 1) {
      for (std::uint32_t b = 2; b <= N; b <<= 1) {
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
          #pragma unroll
          for (std::uint32_t i = 0; i < N; i++) {
            std::uint32_t j = i ^ c;
            if (i >= j) continue;
            const auto line_id = i + (N * lane_id);
            const auto p       = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
            swap_if_needed(k[i], v[i], k[j], v[j], p);
          }
        }
      }
      return;
    }

    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed(k[i], v[i], c, p);
      }
    }
    const auto p = ((lane_id & b) == 0);
    for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
    #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        std::uint32_t j = i ^ c;
        if (i >= j) continue;
        swap_if_needed(k[i], v[i], k[j], v[j], p);
      }
    }
  }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
  warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // sort 1
  if (warp_id > 0) return;
  if (CANDIDATE_BUFFER_SIZE != 64) {
    printf("CANDIDATE_BUFFER_SIZE must be 64\n");
    assert(false);
  }
  // constexpr unsigned N_1 = 2;
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];

  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort_inverse(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // sort 1
  if (warp_id > 0) { return; }
  if (CANDIDATE_BUFFER_SIZE != 64) {
    printf("CANDIDATE_BUFFER_SIZE must be 64\n");
    assert(false);
  }
  // constexpr unsigned N_1 = 2;
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];

  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
    if (j < CANDIDATE_BUFFER_SIZE) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_128_by_bitonic_sort(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // Sort part 1
  if (warp_id > 0) return;
  assert (CANDIDATE_BUFFER_SIZE <= 64);
  
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];
  auto candidate_distances = result_distances_ptr + internal_topk;
  auto candidate_indices = result_indices_ptr + internal_topk;
  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE && j < internal_topk) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }

  // Sort part 2
  // constexpr unsigned N_2 = 4;
  DISTANCE_T key_2[N_2];
  INDEX_T val_2[N_2];
  if (first) {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < internal_topk) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = MAX_INDEX;
      }
    }
    /* Warp Sort */
    warp_sort<float, uint32_t, N_2>(key_2, val_2);
  } else {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      if (j < internal_topk) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = MAX_INDEX;
      }
    }
  }

  /* Merge candidates */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;  // [0:MAX_ITOPK-1]
    unsigned k = internal_topk - 1 - j;
    if (k >= internal_topk || k >= CANDIDATE_BUFFER_SIZE) continue;
    auto candidate_key = candidate_distances[k];
    if (key_2[i] > candidate_key) {
      key_2[i] = candidate_key;
      val_2[i] = candidate_indices[k];
    }
  }
  /* Warp Merge */
  warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

  /* Store new itopk results */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;
    if (j < internal_topk) {
      result_distances_ptr[j] = key_2[i];
      result_indices_ptr[j]   = val_2[i];
    }
  }

  (void)RESULT_BUFFER_SIZE;
  (void)first;
}

__device__ inline int merge_path_partition(
    const DISTANCE_T* a,
    int a_count,
    const DISTANCE_T* b,
    int b_count,
    int diag) {
  int low = max(0, diag - b_count);
  int high = min(diag, a_count);

  while (low <= high) {
    const int a_idx = (low + high) >> 1;
    const int b_idx = diag - a_idx;

    const DISTANCE_T a_left = (a_idx > 0) ? a[a_idx - 1] : -FLT_MAX;
    const DISTANCE_T a_right = (a_idx < a_count) ? a[a_idx] : FLT_MAX;
    const DISTANCE_T b_left = (b_idx > 0) ? b[b_idx - 1] : -FLT_MAX;
    const DISTANCE_T b_right = (b_idx < b_count) ? b[b_idx] : FLT_MAX;

    if (a_left > b_right) {
      high = a_idx - 1;
    } else if (b_left > a_right) {
      low = a_idx + 1;
    } else {
      return a_idx;
    }
  }

  return low;
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_candidate_sort_and_merge(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  __shared__ DISTANCE_T candidate_snapshot_dist_shared[64];
  __shared__ INDEX_T candidate_snapshot_index_shared[64];
  __shared__ DISTANCE_T merged_topk_dist_shared[N_2 * WARP_SIZE];
  __shared__ INDEX_T merged_topk_index_shared[N_2 * WARP_SIZE];

  auto candidate_indices = result_indices_ptr + internal_topk;
  auto candidate_distances = result_distances_ptr + internal_topk;

  candidate_by_bitonic_sort<N_1, 0>(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE);
  __syncthreads();

  for (unsigned i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
    candidate_snapshot_dist_shared[i] = candidate_distances[i];
    candidate_snapshot_index_shared[i] = candidate_indices[i];
  }
  __syncthreads();

  constexpr int kItemsPerThread = 4;
  int active_threads = (internal_topk + kItemsPerThread - 1) / kItemsPerThread;
  active_threads = min(static_cast<int>(blockDim.x),
                       ((active_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);

  if (threadIdx.x < active_threads) {
    const int out_begin =
        min(internal_topk, (static_cast<int>(threadIdx.x) * internal_topk) / active_threads);
    const int out_end =
        min(internal_topk, (static_cast<int>(threadIdx.x + 1) * internal_topk) / active_threads);

    const int top_begin = merge_path_partition(
        result_distances_ptr, static_cast<int>(internal_topk),
        candidate_snapshot_dist_shared, static_cast<int>(CANDIDATE_BUFFER_SIZE),
        out_begin);
    const int top_end = merge_path_partition(
        result_distances_ptr, static_cast<int>(internal_topk),
        candidate_snapshot_dist_shared, static_cast<int>(CANDIDATE_BUFFER_SIZE),
        out_end);

    int top_i = top_begin;
    int cand_i = out_begin - top_begin;
    const int cand_end = out_end - top_end;

    for (int out_i = out_begin; out_i < out_end; ++out_i) {
      const bool take_topk =
          (cand_i >= cand_end) ||
          (top_i < top_end &&
           result_distances_ptr[top_i] <= candidate_snapshot_dist_shared[cand_i]);

      if (take_topk) {
        merged_topk_dist_shared[out_i] = result_distances_ptr[top_i];
        merged_topk_index_shared[out_i] = result_indices_ptr[top_i];
        ++top_i;
      } else {
        merged_topk_dist_shared[out_i] = candidate_snapshot_dist_shared[cand_i];
        merged_topk_index_shared[out_i] = candidate_snapshot_index_shared[cand_i];
        ++cand_i;
      }
    }
  }
  __syncthreads();

  for (unsigned i = threadIdx.x; i < internal_topk; i += blockDim.x) {
    result_distances_ptr[i] = merged_topk_dist_shared[i];
    result_indices_ptr[i] = merged_topk_index_shared[i];
  }
  __syncthreads();

  (void)RESULT_BUFFER_SIZE;
  (void)first;
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_sort_candidates_and_merge(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  if constexpr (N_2 == 2 || N_2 == 4) {
    topk_128_by_bitonic_sort<N_1, N_2>(
        result_indices_ptr, result_distances_ptr, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, internal_topk, first);
  } else {
    topk_by_candidate_sort_and_merge<N_1, N_2>(
        result_indices_ptr, result_distances_ptr, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, internal_topk, first);
  }
}

#define SMALL_HASH_RESET_INTERVAL 16

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
  return 1 << bitlen;
}

__device__ inline void hashtable_init(INDEX_T* const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
  if (threadIdx.x < FIRST_TID) return;
  for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = MAX_INDEX;
  }
}

__device__ inline uint32_t hashtable_insert(INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    // Open addressing is used for collision resolution
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    // Linear probing
    INDEX_T index             = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = atomicCAS(&table[index], ~static_cast<INDEX_T>(0), key);
        if (old == ~static_cast<INDEX_T>(0)) {
            return 1;
        } else if (old == key) {
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
  return 0;
}

__device__ inline uint32_t hashtable_contains(INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    INDEX_T index             = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    // only check if the slot is filled, no actual insertion
    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = table[index];
        if (old == key) return 1;
        if (old == ~static_cast<INDEX_T>(0)) return 0;
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ inline void hashtable_restore(   
    INDEX_T* const table, 
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashtable_insert(table, BITLEN, key);
  }
}

__device__ inline void hashtable_restore_expand_only(
    INDEX_T* const table,
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    const INDEX_T raw = itopk_indices[i];
    // only restore nodes that are expanded already (MSB is 0)
    if ((raw & index_msb_1_mask) == 0 || raw == MAX_INDEX) continue;
    const INDEX_T key = raw & ~index_msb_1_mask;
    hashtable_insert(table, BITLEN, key);
  }
}

// select top-N un-expanded nodes in the priority queue of size M
static __device__ uint32_t pick_expanders(int N, INDEX_T* output_list, int M, INDEX_T* pri_queue) {
  uint32_t num_exp = 0; // number of expanders actually seleceted; must be <= N
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  uint32_t itopk_max = M;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }   // round up to be multiple of 32
  for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    INDEX_T index;
    int new_parent = 0;
    if (j < M) {    // within the priority queue bound
      index = pri_queue[j];
      if ((index & index_msb_1_mask) == 0) {   // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_exp;
      if (i < N) {
        output_list[i] = j;
        //if (blockIdx.x == 0) printf("pick %d-th node %d as the %d-th parent\n", j, pri_queue[j], i);
        // set most significant bit as used node
        pri_queue[j] |= index_msb_1_mask;
      }
    }
    num_exp += __popc(ballot_mask);
    if (num_exp >= N) { break; }
  }
  return num_exp;
}

static __device__ inline void result_pool_recompute_worst(
    const DISTANCE_T* pool_distance,
    uint32_t pool_size,
    uint32_t* worst_idx,
    DISTANCE_T* worst_dist) {
  if (pool_size == 0) {
    *worst_idx = 0;
    *worst_dist = -FLT_MAX;
    return;
  }
  uint32_t local_worst_idx = 0;
  DISTANCE_T local_worst_dist = pool_distance[0];
  for (uint32_t i = 1; i < pool_size; ++i) {
    if (pool_distance[i] > local_worst_dist) {
      local_worst_dist = pool_distance[i];
      local_worst_idx = i;
    }
  }
  *worst_idx = local_worst_idx;
  *worst_dist = local_worst_dist;
}

static __device__ inline void result_pool_push_unique_unsorted(
    INDEX_T* pool_index,
    DISTANCE_T* pool_distance,
    uint32_t* pool_size,
    uint32_t* worst_idx,
    DISTANCE_T* worst_dist,
    int capacity,
    INDEX_T node,
    DISTANCE_T distance) {
  uint32_t size = *pool_size;

  if (size < static_cast<uint32_t>(capacity)) {
    pool_index[size] = node;
    pool_distance[size] = distance;
    *pool_size = size + 1;
    if (size == 0 || distance > *worst_dist) {
      *worst_idx = size;
      *worst_dist = distance;
    }
    return;
  }

  if (distance >= *worst_dist) return;

  pool_index[*worst_idx] = node;
  pool_distance[*worst_idx] = distance;
  result_pool_recompute_worst(pool_distance, size, worst_idx, worst_dist);
}

static __device__ inline void result_pool_sort_small(
    INDEX_T* pool_index,
    DISTANCE_T* pool_distance,
    uint32_t pool_size) {
  for (uint32_t i = 1; i < pool_size; ++i) {
    INDEX_T key_index = pool_index[i];
    DISTANCE_T key_dist = pool_distance[i];
    int j = static_cast<int>(i) - 1;
    while (j >= 0 && pool_distance[j] > key_dist) {
      pool_distance[j + 1] = pool_distance[j];
      pool_index[j + 1] = pool_index[j];
      --j;
    }
    pool_distance[j + 1] = key_dist;
    pool_index[j + 1] = key_index;
  }
}

/*-------------------------------------------- rabitq --------------------------------------------*/
static __device__ inline void rotate_vector_gpu(const float* src, float* dst,
   const float* signs, int dim, int padded_dim) {

  int tid = threadIdx.x;
  
  for (size_t i = tid; i < dim; i += blockDim.x) {
    dst[i] = src[i] * signs[i];
  }
  for (int i = dim + tid; i < padded_dim; i += blockDim.x) {
    dst[i] = 0.0f;
  }
  __syncthreads();

  // Fast Walsh-Hadamard Transform (in-place, power-of-2 B, B >= 8)
  for (size_t len = 1; len < padded_dim; len <<= 1) {
    int pairs = padded_dim >> 1;
    for (int pair_idx = tid; pair_idx < pairs; pair_idx += blockDim.x) {
      int block = pair_idx / len;
      int j = pair_idx % len;
      int base = block * (len << 1);

      int idx1 = base + j;
      int idx2 = idx1 + len;

      float u = dst[idx1];
      float v = dst[idx2];
      dst[idx1] = u + v;
      dst[idx2] = u - v;
    }
    __syncthreads();
  }
}

static __device__ void query_prepare_gpu(const float* query_raw, QueryFactors& scratch,
    const float* signs_ptr, int dim, int padded_dim) {

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  // rotate the query
  rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
  __syncthreads();

  __shared__ float warp_min[WARPS_PER_BLOCK];
  __shared__ float warp_max[WARPS_PER_BLOCK];
  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  // data range: find the lowest and highest dimension in the rotated query
  for (size_t i = tid; i < padded_dim; i += blockDim.x) {
      float tmp = scratch.rotated_query[i];
      local_min = fminf(local_min, tmp);
      local_max = fmaxf(local_max, tmp);
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
    local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
  }

  if (lane_id == 0) {
    warp_min[warp_id] = local_min;
    warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
    local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
      local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
      scratch.low_val = local_min;
      scratch.high_val = local_max;
      scratch.width = (scratch.high_val - scratch.low_val) / ((1 << QG_BQUERY) - 1);
    }
  }
  __syncthreads();

  // quantize the query 
  float inv_width = 1.0F / scratch.width;

  __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
  int32_t local_sum = 0;
  for (size_t i = tid; i < padded_dim; i += blockDim.x) {
      uint8_t cur = static_cast<uint8_t>(lroundf(((scratch.rotated_query[i] - scratch.low_val) * inv_width) + 0.5));
      scratch.quantized_query[i] = cur;
      local_sum += cur;
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
  }

  if (lane_id == 0) {
    warp_sum[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
      scratch.sum_q = local_sum;
    }
  }
  __syncthreads();
}

static __device__ inline DISTANCE_T estimated_distance_gpu(const QueryFactors& qf,
  const uint64_t* packed_code, float triple_x, float factor_dq, float factor_vq,
  float exact_dist, int padded_dim_) {

  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;

  if (triple_x == FLT_MAX) return 0.0f;

  int32_t local_sum = 0;
  for (int word = 0; word < padded_dim_ / 64; ++word) {
      uint64_t bits = packed_code[word];
      int base = word * 64;

      int d0 = base + lane_id;
      if ((bits >> lane_id) & 1ULL)
          local_sum += static_cast<int32_t>(qf.quantized_query[d0]);

      int d1 = d0 + 32;
      if (d1 < padded_dim_ && ((bits >> (lane_id + 32)) & 1ULL))
          local_sum += static_cast<int32_t>(qf.quantized_query[d1]);
  }

  // reduce the sum
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  float result_float = static_cast<float>((local_sum << 1) - qf.sum_q);
  DISTANCE_T est_dist = triple_x + exact_dist;
  est_dist += factor_dq * qf.width * result_float;
  est_dist += factor_vq * qf.low_val;
  return est_dist;
}


static __device__ __forceinline__ int fastscan_lane_pack_index(int lane_mod16) {
  return ((lane_mod16 & 7) << 1) | (lane_mod16 >> 3);
}

static __device__ __forceinline__ uint8_t fastscan_decode_code_gpu(
    const uint8_t* packed_codes_block, int lane_id, int codebook_idx) {
  const int pair_idx = codebook_idx >> 1;
  const int nibble_group = codebook_idx & 1;
  const int lane_mod16 = lane_id & 15;
  const int byte_idx = fastscan_lane_pack_index(lane_mod16);
  const uint8_t packed =
      packed_codes_block[pair_idx * 32 + nibble_group * 16 + byte_idx];
  return (lane_id < 16) ? static_cast<uint8_t>(packed & 0x0f)
                        : static_cast<uint8_t>(packed >> 4);
}

static __device__ inline void pack_lut_gpu(
    int dim, const uint8_t* __restrict__ quantized_query, uint8_t* __restrict__ lut) {
  const int tid = threadIdx.x;
  const int num_codebook = dim >> 2;

  for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
    const uint8_t* query_chunk = quantized_query + (cb << 2);
    uint8_t* lut_chunk = lut + (cb << 4);

    lut_chunk[0] = 0;
    for (int mask = 1; mask < 16; ++mask) {
      const int lowbit = mask & -mask;
      const int pos = (lowbit == 8) ? 0 : ((lowbit == 4) ? 1 : ((lowbit == 2) ? 2 : 3));
      lut_chunk[mask] = static_cast<uint8_t>(lut_chunk[mask - lowbit] + query_chunk[pos]);
    }
  }
}

static __device__ inline void scan_all_neighbors_32_gpu(
    const QueryFactors& qf, const uint8_t* packed_codes_block,
    const float* triple_x, const float* factor_dq, const float* factor_vq,
    float exact_dist, int padded_dim, float* appro_dist, int active_neighbors = WARP_SIZE) {
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  if (lane_id >= active_neighbors) return;

  const int num_codebook = padded_dim >> 2;
  uint32_t raw_sum = 0;

  for (int cb = 0; cb < num_codebook; ++cb) {
    const uint8_t code = fastscan_decode_code_gpu(packed_codes_block, lane_id, cb);
    raw_sum += static_cast<uint32_t>(qf.lut[(cb << 4) + code]);
  }

  const float result_float = static_cast<float>((static_cast<int32_t>(raw_sum) << 1) - qf.sum_q);
  float est_dist = triple_x[lane_id] + exact_dist;
  est_dist += factor_dq[lane_id] * qf.width * result_float;
  est_dist += factor_vq[lane_id] * qf.low_val;
  appro_dist[lane_id] = est_dist;
}

// ═══════════════════════════════════════════════════════════════════════
// ── include/topk.cuh (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

#include <concepts>
#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>

/*
 * TopKBuffer — self-contained warp-collaborative sorted priority queue.
 * Owns its sorted (distance, index) storage internally.
 * All methods called in warp lockstep.
 */
template <typename B, unsigned REG>
concept TopKBuffer = requires(
    B              	buf,
    float           (&dist)[REG],
    uint32_t        (&idx)[REG],
    unsigned       	n
) {
    // Merge per-thread REG candidates (unsorted) into internal storage.
    { buf.merge(dist, idx) } -> std::same_as<void>;

    // Find up to `n` unexpanded nodes (MSB==0), mark them expanded,
    // write their positions to parent_list. Returns count found.
    { buf.pick(idx, n) } -> std::convertible_to<unsigned>;

    // Write internal storage out to result arrays (e.g. global memory).
    { buf.flush((float*)nullptr, (uint32_t*)nullptr, n) } -> std::same_as<void>;
};

/**
 *  @class SharedBitonicTopK shared memory topk buffer for bitonic sorting
 *  @tparam BX number of threads in the block
 *  @tparam REG rough register pressure parameter per thread
 */
template <unsigned BX, unsigned REG>
class SharedBitonicTopK {
    private:
    static constexpr unsigned N = BX * REG;
    unsigned  topk;       // the number of elements in this buffer
    float*    topk_dist;  // points into shared memory, stored descending
    uint32_t* topk_idx;   // points into shared memory, stored descending
    public:
    /**
     * @brief Construct a top-k buffer backed by shared memory.
     *
     * The caller is responsible for allocating shared memory arrays of at
     * least `topk` elements each. The buffer must be pre-filled in
     * descending order by distance before the first merge call.
     *
     * @param topk       Number of elements the buffer holds.
     * @param topk_dist  Shared memory array of distances, size >= topk.
     * @param topk_idx   Shared memory array of indices, size >= topk.
     */
    __device__ SharedBitonicTopK(unsigned topk, float* topk_dist, uint32_t* topk_idx)
        : topk(topk), topk_dist(topk_dist), topk_idx(topk_idx) {}
    /**
     * @brief Merge candidates into the descending internal buffer.
     *
     * Input may be unsorted — the buffer sorts it internally. The internal
     * buffer must be sorted descending and its size must be a multiple of N.
     *
     * @param dist  Per-thread REG distances (unsorted).
     * @param idx   Per-thread REG indices, corresponding to dist.
     */
    __device__ void merge(float (&dist)[REG], uint32_t (&idx)[REG]);
    /**
     * @brief Pick up to n unmarked nodes with the smallest distances.
     *
     * Scans the buffer from the tail (smallest distances). Unmarked means
     * MSB of the index is 0 and the index is not 0xFFFFFFFF. Found nodes
     * are marked (MSB set to 1). Only warp 0 (threads 0..31) participates.
     *
     * @param idx   Output array receiving the buffer positions of picked nodes.
     * @param n     Maximum number of nodes to pick.
     * @return      Actual number of nodes picked (<= n).
     */
    __device__ unsigned pick(uint32_t (&idx)[N], unsigned n);
    /**
     * @brief Flush the n smallest elements to global memory.
     *
     * Reads from the tail of the descending buffer. Clears the MSB mark
     * on output indices. All BX threads participate with strided access.
     *
     * @param dist  Output distance array in global memory, size >= n.
     * @param idx   Output index array in global memory, size >= n.
     * @param n     Number of elements to write, must be <= topk.
     */
    __device__ void flush(float* dist, uint32_t* idx, unsigned n);
    // -- helpers
    template <bool asc = true>
    __device__ static inline void bitonic_sort_local(
            float (&d)[REG], uint32_t (&i)[REG]);
    template <bool asc = true>
    __device__ static inline void bitonic_sort_block(
            float (&d)[REG], uint32_t (&i)[REG], float* scratch_d, uint32_t* scratch_i);
    __device__ static inline void bitonic_sort_full(
            float (&d)[REG], uint32_t (&i)[REG], float* scratch_d, uint32_t* scratch_i);
    __device__ static inline void butterfly(
            float (&a_d)[REG], uint32_t (&a_i)[REG], float (&b_d)[REG], uint32_t (&b_i)[REG]);
};

template <unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<BX, REG>::merge(float (&dist)[REG], uint32_t (&idx)[REG]) {
    // Scratch for cross-warp bitonic sort steps. When BX <= 32 (single warp),
    // the cross-warp path is dead code and the compiler eliminates these.
    __shared__ float    scratch_d[N];
    __shared__ uint32_t scratch_i[N];
    const unsigned tid = threadIdx.x;
    const unsigned chunks = topk / N;
    // [1] Sort incoming candidates ascending (from arbitrary order).
    bitonic_sort_full(dist, idx, scratch_d, scratch_i);
    for (unsigned c = 0; c < chunks; c++) {
        // [2] Load chunk c into registers
        float    dist_[REG];
        uint32_t idx_[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = c * N + tid * REG + j;
            dist_[j] = topk_dist[pos];
            idx_[j]  = topk_idx[pos];
        }
        // [3] Butterfly: dist gets min (lower half), dist_ gets max (upper half)
        butterfly(dist, idx, dist_, idx_);
        // [5] If not first chunk, sort upper half descending and flush to previous chunk
        if (c > 0) {
            bitonic_sort_block<false>(dist_, idx_, scratch_d, scratch_i);
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = (c - 1) * N + tid * REG + j;
                topk_dist[pos] = dist_[j];
                topk_idx[pos]  = idx_[j];
            }
            __syncthreads();
        }
        // [6] Sort lower half ascending, ready for next chunk
        bitonic_sort_block<true>(dist, idx, scratch_d, scratch_i);
    }
    // [7] Flush last chunk descending
    bitonic_sort_block<false>(dist, idx, scratch_d, scratch_i);
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = (chunks - 1) * N + tid * REG + j;
        topk_dist[pos] = dist[j];
        topk_idx[pos]  = idx[j];
    }
    __syncthreads();
}

template <unsigned BX, unsigned REG>
__device__ unsigned SharedBitonicTopK<BX, REG>::pick(uint32_t (&out)[N], unsigned n) {
    constexpr uint32_t MSB = 0x80000000;
    const unsigned tid = threadIdx.x;
    unsigned found = 0;

    // Smallest elements at the tail (descending buffer). Scan backwards.
    unsigned base = (topk > 32) ? topk - 32 : 0;
    while (found < n) {
        uint32_t val = 0;
        bool unmarked = false;
        if (tid < 32 && base + tid < topk) {
            val = topk_idx[base + tid];
            unmarked = ((val & MSB) == 0) && (val != 0xFFFFFFFF);
        }

        uint32_t ballot = __ballot_sync(0xffffffff, unmarked);
        if (ballot == 0) {
            if (base == 0) break;
            base = (base > 32) ? base - 32 : 0;
            continue;
        }

        unsigned count = __popc(ballot);
        unsigned lane_prefix = __popc(ballot & ((1u << tid) - 1));

        if (unmarked && found + lane_prefix < n) {
            out[found + lane_prefix] = base + tid;
            topk_idx[base + tid] = val | MSB;
        }

        found += count;
        if (base == 0) break;
        base = (base > 32) ? base - 32 : 0;
    }

    return (found < n) ? found : n;
}

template <unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<BX, REG>::flush(float* dist, uint32_t* idx, unsigned n) {
    const unsigned tid = threadIdx.x;
    // Copy the n smallest elements (at the tail of the descending buffer).
    for (unsigned i = tid; i < n; i += BX) {
        unsigned pos = topk - n + i;
        dist[i] = topk_dist[pos];
        idx[i]  = topk_idx[pos] & ~0x80000000u;  // clear MSB mark
    }
}

/**
 * @brief Full bitonic sort of BX*REG key-value pairs from arbitrary order.
 *        Always produces ascending output.
 *
 * Unlike bitonic_sort_block (which assumes bitonic input), this runs
 * the complete bitonic sorting network from unsorted data.
 * Direction: ((global_pos & len) == 0) — naturally ascending at top level
 * since global_pos < TOTAL = len at the final pass.
 *
 * @param scratch_d   Shared memory scratch for floats, size >= N.
 * @param scratch_i   Shared memory scratch for uint32_ts, size >= N.
 */
template <unsigned BX, unsigned REG>
__device__ inline void SharedBitonicTopK<BX, REG>::bitonic_sort_full(
    float (&d)[REG], uint32_t (&i)[REG],
    float* scratch_d, uint32_t* scratch_i)
{
    const unsigned tid = threadIdx.x;
    constexpr unsigned TOTAL = BX * REG;

    for (unsigned len = 2; len <= TOTAL; len <<= 1) {
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            if (stride >= REG) {
                unsigned thread_stride = stride / REG;
                if (thread_stride < 32) {
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                        uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                        unsigned gp = tid * REG + j;
                        bool dir = ((gp & len) == 0);
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                } else {
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        scratch_d[tid * REG + j] = d[j];
                        scratch_i[tid * REG + j] = i[j];
                    }
                    __syncthreads();
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = scratch_d[partner * REG + j];
                        uint32_t pi = scratch_i[partner * REG + j];
                        unsigned gp = tid * REG + j;
                        bool dir = ((gp & len) == 0);
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                    __syncthreads();
                }
            } else {
                #pragma unroll
                for (unsigned k = 0; k < REG; k++) {
                    unsigned j = k ^ stride;
                    if (j > k) {
                        unsigned gp = tid * REG + k;
                        bool dir = ((gp & len) == 0);
                        if ((d[k] > d[j]) == dir) {
                            float    td = d[k]; d[k] = d[j]; d[j] = td;
                            uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Pairwise min/max split on REG register pairs.
 *
 * After the call, a[j] holds min(a[j], b[j]) and b[j] holds max(a[j], b[j])
 * for every j in [0, REG). When both inputs are sorted halves of a bitonic
 * sequence, the result is two bitonic subsequences where every element in a
 * is <= every element in b.
 */
template <unsigned BX, unsigned REG>
__device__ inline void SharedBitonicTopK<BX, REG>::butterfly(
    float (&a_d)[REG], uint32_t (&a_i)[REG],
    float (&b_d)[REG], uint32_t (&b_i)[REG])
{
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        if (a_d[j] > b_d[j]) {
            float    td = a_d[j]; a_d[j] = b_d[j]; b_d[j] = td;
            uint32_t ti = a_i[j]; a_i[j] = b_i[j]; b_i[j] = ti;
        }
    }
}

/**
 * @brief Bitonic sort of REG key-value pairs within a single thread.
 *
 * Applies the full bitonic sorting network over the REG elements held in
 * registers by one thread. No cross-thread communication.
 *
 * @tparam asc  true for ascending output, false for descending.
 */
template <unsigned BX, unsigned REG>
template <bool asc>
__device__ inline void SharedBitonicTopK<BX, REG>::bitonic_sort_local(float (&d)[REG], uint32_t (&i)[REG]) {
    #pragma unroll
    for (unsigned len = 2; len <= REG; len <<= 1) {
        #pragma unroll
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            #pragma unroll
            for (unsigned k = 0; k < REG; k++) {
                unsigned j = k ^ stride;
                if (j > k) {
                    bool dir = ((k & len) == 0) == asc;
                    if ((d[k] > d[j]) == dir) {
                        float    td = d[k]; d[k] = d[j]; d[j] = td;
                        uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                    }
                }
            }
        }
    }
}

/**
 * @brief Bitonic sort of BX*REG key-value pairs distributed across BX threads.
 *
 * Each thread holds REG elements. The sort first calls bitonic_sort_local,
 * then performs cross-thread merge steps: intra-warp steps use __shfl_xor_sync,
 * cross-warp steps go through the shared-memory scratch buffers.
 *
 * @tparam asc        true for ascending output, false for descending.
 * @param scratch_d   Shared memory scratch for floats, size >= N.
 * @param scratch_i   Shared memory scratch for uint32_ts, size >= N.
 */
template <unsigned BX, unsigned REG>
template <bool asc>
__device__ inline void SharedBitonicTopK<BX, REG>::bitonic_sort_block(
    float (&d)[REG], uint32_t (&i)[REG],
    float* scratch_d, uint32_t* scratch_i)
{
    const unsigned tid = threadIdx.x;
    constexpr unsigned TOTAL = BX * REG;
    bitonic_sort_local<asc>(d, i);
    for (unsigned len = 2 * REG; len <= TOTAL; len <<= 1) {
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            if (stride >= REG) {
                unsigned thread_stride = stride / REG;
                if (thread_stride < 32) {
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                        uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                        unsigned global_pos = tid * REG + j;
                        bool dir = ((global_pos & len) == 0) == asc;
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                } else {
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        scratch_d[tid * REG + j] = d[j];
                        scratch_i[tid * REG + j] = i[j];
                    }
                    __syncthreads();
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = scratch_d[partner * REG + j];
                        uint32_t pi = scratch_i[partner * REG + j];
                        unsigned global_pos = tid * REG + j;
                        bool dir = ((global_pos & len) == 0) == asc;
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                    __syncthreads();
                }
            } else {
                #pragma unroll
                for (unsigned k = 0; k < REG; k++) {
                    unsigned j = k ^ stride;
                    if (j > k) {
                        unsigned global_pos_k = tid * REG + k;
                        bool dir = ((global_pos_k & len) == 0) == asc;
                        if ((d[k] > d[j]) == dir) {
                            float    td = d[k]; d[k] = d[j]; d[j] = td;
                            uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                        }
                    }
                }
            }
        }
    }
}



// ═══════════════════════════════════════════════════════════════════════
// ── include/distance.cuh (inlined) ──
// ═══════════════════════════════════════════════════════════════════════


// [flattened] #include "utils.cuh"
#include <curand_kernel.h>
#define MAX_DIM 128

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_full(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    constexpr unsigned reg_nelem = (MAX_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff[reg_nelem];
    unsigned int full_mask = 0xffffffff;
    DISTANCE_T norm2 = 0;
    if (valid_child) {
        // Load
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;
            dl_buff[e] = reinterpret_cast<const float4*>(child_data_ptr)[k / 4];
        }
        // Compute
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;
            DISTANCE_T d = query_ptr[k];
            norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
            d = query_ptr[k + 1];
            norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
            d = query_ptr[k + 2];
            norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
            d = query_ptr[k + 3];
            norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

        }
    }
    unsigned team_mask = __ballot_sync(0xffffffff, valid_child);
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE, uint32_t RATIO = 2>
__device__ DISTANCE_T compute_similarity_vector_load_partial(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child,
    DISTANCE_T threshold = 9999999999
) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;

    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte, vlen == 4
    constexpr unsigned NELEM = ((MAX_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen) + RATIO - 1) / RATIO * (RATIO - 1);
    constexpr unsigned reg_nelem = NELEM > 1U ? NELEM : 1U;
    float4 dl_buff[reg_nelem];
    unsigned int full_mask = 0xffffffff;
    DISTANCE_T norm2 = 0;
    if (valid_child) {
        // Load
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;
            dl_buff[e] = reinterpret_cast<const float4*>(child_data_ptr)[k / 4];
        }
        
        // Compute
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;

            DISTANCE_T d = query_ptr[k];
            norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
            d = query_ptr[k + 1];
            norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
            d = query_ptr[k + 2];
            norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
            d = query_ptr[k + 3];
            norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

        }
    }
    unsigned team_mask = __ballot_sync(0xffffffff, valid_child);
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team(
    uint32_t dim,
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    unsigned team_id  = threadIdx.x / TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    constexpr unsigned reg_nelem = (MAX_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff[reg_nelem];
    DISTANCE_T norm2 = 0;
    // Load
    #pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
        const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dim) break;
        dl_buff[e] = reinterpret_cast<float4*>(child_data_ptr)[k / 4];
    }
    // Compute
    #pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
        const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dim) break;
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
        d = query_ptr[k + 2];
        norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
        d = query_ptr[k + 3];
        norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);
    }

    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team_partial_float4(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask,
    uint32_t e) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    float4 dl_buff;
    DISTANCE_T norm2 = 0;
    
    const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
    if (k < dim) {
        // Load
        dl_buff = reinterpret_cast<const float4*>(child_data_ptr)[k / 4];
        // Compute
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff.x) * (d - dl_buff.x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff.y) * (d - dl_buff.y);
        d = query_ptr[k + 2];
        norm2 += (d - dl_buff.z) * (d - dl_buff.z);
        d = query_ptr[k + 3];
        norm2 += (d - dl_buff.w) * (d - dl_buff.w);
    }
    // Reduce
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team_partial_float2(
    uint32_t dim,
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask,
    uint32_t e) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    unsigned team_id  = threadIdx.x / TEAM_SIZE;
    float2 dl_buff;
    DISTANCE_T norm2 = 0;
    const uint32_t k = (lane_id + (TEAM_SIZE * e)) * 2;
    if (k < dim) {
        // Load
        dl_buff = reinterpret_cast<float2*>(child_data_ptr)[k / 2];
        // Compute
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff.x) * (d - dl_buff.x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff.y) * (d - dl_buff.y);
    }
    // Reduce
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }
    return norm2;
}

// map top10 nodes and fetch neighbors and calculate distance
template<uint32_t TEAM_SIZE>
__device__ void compute_distance_to_maped_top10_nodes(
    uint32_t dim,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    const DATA_T* query_ptr,
    const DATA_T* dataset_ptr,
    INDEX_T* graph_ptr,
    INDEX_T* top10_ptr,
    INDEX_T* seed_map_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t DATASET_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t SEED_CONFIG,
    uint32_t HASH_TABLE_CONFIG,
    uint32_t SEED_TOPK_SIZE) {
    const INDEX_T invalid_index        = MAX_INDEX;
    // Load top N nodes
    INDEX_T top1 = top10_ptr[blockIdx.y * SEED_TOPK_SIZE];
    // Map top1 nodes to dataset
    INDEX_T parent_id = seed_map_ptr[top1];
    // fetch neighbors and insert to hash table
    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        INDEX_T child_id = graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
        if (HASH_TABLE_CONFIG) {
            if (child_id != invalid_index) {
                if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0) {
                    child_id = invalid_index;
                    candidate_distances_ptr[i] = FLT_MAX;
                } else {
                    candidate_distances_ptr[i] = 0;
                }
            }
            candidate_indices_ptr[i] = child_id;
        }
    }
    __syncthreads();

    // calculate distance
    for (uint32_t tid = threadIdx.x; tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE; tid += blockDim.x) {
        const auto i       = tid / TEAM_SIZE;
        const bool valid_i = (i < (CANDIDATE_BUFFER_SIZE));
        INDEX_T child_id   = invalid_index;
        if (valid_i) { child_id = candidate_indices_ptr[i]; }
        DISTANCE_T norm2 = compute_similarity_vector_load_full<TEAM_SIZE>(dim, dataset_ptr, 
                                    query_ptr, child_id, child_id != invalid_index);
        //if (tid % TEAM_SIZE == 0 && child_id != invalid_index) atomicAdd(counter, 1);
        // Store the distance
        const unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (valid_i && lane_id == 0) {
            if (child_id != invalid_index) {
                candidate_distances_ptr[i] = norm2;
            } else {
                candidate_distances_ptr[i] = FLT_MAX;
            }
        }
    }    
}

#define NUM_DIST 1

template <uint32_t TEAM_SIZE>
__device__ void compute_distance_to_random_nodes(
    uint32_t dim,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    const DATA_T* query_ptr,
    const DATA_T* dataset_ptr,
    uint32_t DATASET_SIZE,
    uint32_t max_i, // number of starting nodes to pick
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t HASH_TABLE_CONFIG) {
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }
  for (uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += blockDim.x / TEAM_SIZE) {
    const bool valid_i = (i < max_i);
    INDEX_T best_index_team_local;
    DISTANCE_T best_norm2_team_local = FLT_MAX;
    for (uint32_t j = 0; j < NUM_DIST; j++) {
      // Select a node randomly and compute the distance to it
      INDEX_T seed_index;
      if (valid_i) {
        uint32_t gid = threadIdx.x + (blockDim.x * (i + (max_i * j)));
        curandState state;
        curand_init(clock64(), gid, 0, &state);
        seed_index = curand_uniform(&state) * DATASET_SIZE;
      }
      auto norm2 = compute_similarity_vector_load_full<TEAM_SIZE>(dim, dataset_ptr,
          query_ptr, seed_index, valid_i);
      //if (valid_i) atomicAdd(counter, 1);
      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }
    const unsigned lane_id = threadIdx.x % TEAM_SIZE;
    if (valid_i && lane_id == 0) {
      if (HASH_TABLE_CONFIG) {
        if (hashtable_insert(visited_hash_ptr, BITLEN, best_index_team_local)) {
          candidate_distances_ptr[i] = best_norm2_team_local;
          candidate_indices_ptr[i]   = best_index_team_local;
        } else {
          candidate_distances_ptr[i] = FLT_MAX;
          candidate_indices_ptr[i]   = MAX_INDEX;
        }
      } else {
        candidate_distances_ptr[i] = best_norm2_team_local;
        candidate_indices_ptr[i]   = best_index_team_local;
      }
    }
  }
}

template <uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK_SIZE>
__device__ void compute_distance_to_child_nodes(
    uint32_t dim,
    INDEX_T* parent_buffer,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    INDEX_T* internal_topk_list,
    const DATA_T* query_ptr,
    const DATA_T* d_dataset_ptr,
    INDEX_T* d_graph_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t SEARCH_WIDTH,
    uint32_t CANDIDATE_BUFFER_SIZE,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t* counter,
    uint32_t THRESHOLD_CONFIG,
    uint32_t HASH_TABLE_CONFIG) {

  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  const INDEX_T invalid_index        = MAX_INDEX;
  // Read child indices of parents from knn graph and check if the distance computaiton is necessary.
  for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
    const INDEX_T smem_parent_id = parent_buffer[0];
    const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
    INDEX_T child_id = d_graph_ptr[(int64_t(GRAPH_DEGREE) * parent_id) + i];
    //if (blockIdx.x == 0 && threadIdx.x < 4) printf("block %d, thread %d expanding node %d 's %d th child node %d \n", blockIdx.x, i, parent_id, i, child_id);
    if (!HASH_TABLE_CONFIG) {
      // check with internal topk
      for (int32_t internal_topk_ptr = 0; internal_topk_ptr < INTERNAL_TOPK_SIZE; internal_topk_ptr++) {
        INDEX_T internal_topk_id = internal_topk_list[internal_topk_ptr] & ~index_msb_1_mask;
        if (child_id == internal_topk_id) {
          child_id = invalid_index;
          break;
        }
      }
      if (child_id != invalid_index) {
        candidate_distances_ptr[i] = 0;
      }
      candidate_indices_ptr[i] = child_id;
    } else {
      if (child_id != invalid_index) {
        if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0) {
          child_id = invalid_index;
          candidate_distances_ptr[i] = FLT_MAX;
        } else {
          candidate_distances_ptr[i] = 0;
        }
      }
      candidate_indices_ptr[i] = child_id;
    }
  }
  // Compute half of the distance to child nodes
  __syncthreads();
  if (THRESHOLD_CONFIG == 0) {
    std::uint32_t max_i = GRAPH_DEGREE * SEARCH_WIDTH;
    for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x) {
      const auto i       = tid / TEAM_SIZE;
      const bool valid_i = (i < (GRAPH_DEGREE * SEARCH_WIDTH));
      INDEX_T child_id   = invalid_index;
      if (valid_i) { child_id = candidate_indices_ptr[i]; }
      // each team of threads do a distance computation
      DISTANCE_T norm2 = compute_similarity_vector_load_full<TEAM_SIZE>(dim, 
                         d_dataset_ptr, query_ptr, child_id, child_id != invalid_index);
      if (tid % TEAM_SIZE == 0 && child_id != invalid_index) atomicAdd(counter, 1);
      //if (blockIdx.x == 0 && tid % TEAM_SIZE == 0) printf("query 0: tid %d, team %d: child id %d 's distance %f\n", tid, i, child_id, norm2);
      // Store the distance
      const unsigned lane_id = threadIdx.x % TEAM_SIZE;
      if (valid_i && lane_id == 0) {
        if (child_id != invalid_index) {
          candidate_distances_ptr[i] = norm2;
        } else {
          candidate_distances_ptr[i] = FLT_MAX;
        }
      }
    }
  } else {
    DISTANCE_T threshold = candidate_distances_ptr[-1];
    std::uint32_t max_i = GRAPH_DEGREE * SEARCH_WIDTH;
    for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x) {
      const auto i       = tid / TEAM_SIZE;
      const bool valid_i = (i < (GRAPH_DEGREE * SEARCH_WIDTH));
      INDEX_T child_id   = invalid_index;
      if (valid_i) { child_id = candidate_indices_ptr[i]; }
      DISTANCE_T norm2 = compute_similarity_vector_load_partial<TEAM_SIZE, 4/*compute 75%*/>
        (dim, d_dataset_ptr, query_ptr, child_id, child_id != invalid_index);
      // Store the distance
      const unsigned lane_id = threadIdx.x % TEAM_SIZE;
      if (valid_i && lane_id == 0) {
        if (child_id != invalid_index && norm2 < threshold) {
          candidate_distances_ptr[i] = norm2;
        } else {
          candidate_distances_ptr[i] = FLT_MAX;
        }
      }
    }

    // atomic add
    __syncthreads();
    const uint32_t _TEAM_SIZE = TEAM_SIZE;
    uint8_t _lane_id = threadIdx.x % _TEAM_SIZE;
    uint8_t _team_id = threadIdx.x / _TEAM_SIZE;
    unsigned _team_mask = 0xff << (_team_id * _TEAM_SIZE);
    uint32_t _offset_e_maxi = ((dim + _TEAM_SIZE * 4 - 1) / (_TEAM_SIZE * 4) + 4 -1) / 4;
    uint32_t _max_i = _offset_e_maxi * CANDIDATE_BUFFER_SIZE;
    while(true) {   
      uint32_t i = 0;
      if(_lane_id == 0) {
        while(true) {
          i =  atomicAdd(counter, 1);
          if(i >= _max_i) {
            break;
          }
          if(candidate_distances_ptr[i % CANDIDATE_BUFFER_SIZE] != FLT_MAX) {
            break;
          }
        }
      }
      i = __shfl_sync(_team_mask, i, _team_id * _TEAM_SIZE);
      if(i >= _max_i) {
        break;
      }
      uint32_t vector_offset = i / CANDIDATE_BUFFER_SIZE;
      i = i % CANDIDATE_BUFFER_SIZE;
      INDEX_T child_id = candidate_indices_ptr[i];
      // Distance calculation
      uint32_t e = _offset_e_maxi * (4 - 1);
      DISTANCE_T partial_norm2 = compute_similarity_vector_load_by_team_partial_float4<_TEAM_SIZE>(dim,
          d_dataset_ptr, query_ptr, child_id, _team_mask, e + vector_offset); // vector offset
      if (_lane_id == 0) {
        DISTANCE_T saved_norm2 = candidate_distances_ptr[i];
        if(saved_norm2 + partial_norm2 > threshold) {
          candidate_distances_ptr[i] = FLT_MAX;
        } else {
          candidate_distances_ptr[i] = saved_norm2 + partial_norm2;
        }
      }  
    }
  }
}

// template <uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
// __device__ inline void compute_distance_to_child_nodes_team_with_direction(
//     uint32_t dim,
//     INDEX_T* parent_buffer,
//     INDEX_T* internal_topk_list,
//     INDEX_T* candidate_indices_ptr,
//     DISTANCE_T* candidate_distances_ptr,
//     const DATA_T* query_ptr,
//     const DATA_T* d_dataset_ptr,
//     INDEX_T* d_graph_ptr,
//     uint32_t* d_sign_bit_ptr,
//     uint32_t GRAPH_DEGREE,
//     uint32_t CANDIDATE_BUFFER_SIZE,
//     uint32_t PRUNE_CONFIG,
//     float PRUNE_RATIO) {
//     constexpr INDEX_T index_msb_1_mask = 0x80000000;
//     const INDEX_T invalid_index        = MAX_INDEX;
//     const INDEX_T smem_parent_id = parent_buffer[0];
//     const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
//     int8_t lane_id = threadIdx.x % 32;
//     uint32_t sign_bit_vector_size = dim * GRAPH_DEGREE / 32; // 256 for sift-128
//     constexpr uint32_t packed_vector_dim_size = (MAX_DIM + 31) / 32;
//     int32_t query_sign_bit[packed_vector_dim_size];

//     uint32_t VECTOR_DIM_32 = (dim + 31) & ~31;
//     for (uint32_t i = threadIdx.x; i < VECTOR_DIM_32; i += blockDim.x) {
//         const bool valid_i = i < dim;
//         uint32_t sign_bit = query_ptr[i] > d_dataset_ptr[parent_id * dim + i];

//         sign_bit <<= (31 - lane_id);
//         unsigned active_threads = __ballot_sync(0xffffffff, valid_i);
//         __syncwarp(active_threads);

//         for (int offset = 16; offset > 0; offset /= 2) {
//             sign_bit |= __shfl_xor_sync(0xffffffff, sign_bit, offset);
//         }
//         query_sign_bit[i / 32] = sign_bit;
//         __syncwarp(active_threads);
//     }

//     for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
//         // const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
//         INDEX_T child_id = d_graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
//         // check with internal topk
//         for (int32_t internal_topk_ptr = 0; internal_topk_ptr < INTERNAL_TOPK; internal_topk_ptr++) {
//             INDEX_T internal_topk_id = internal_topk_list[internal_topk_ptr] & ~index_msb_1_mask;
//             if(child_id == internal_topk_id) {
//                 child_id = invalid_index;
//                 break;
//             }
//         }
//         if (child_id != invalid_index) {
//             if(PRUNE_CONFIG == RANDOM_PRUNE) {
//                 uint32_t gid = threadIdx.x + blockDim.x * i;
//                 curandState state;
//                 curand_init(clock64(), gid, 0, &state);
//                 DISTANCE_T direction = (curand_uniform(&state) - 0.5) * dim;
//                 candidate_distances_ptr[i] = direction;
//             } else if(PRUNE_CONFIG == SIGN_BIT_PRUNE) {
//                 int direction = 0; 
//                 for (int j = 0; j < packed_vector_dim_size; j++) {
//                     direction -= __popcll(query_sign_bit[j] ^ d_sign_bit_ptr[parent_id * sign_bit_vector_size + i * packed_vector_dim_size + j]);
//                 }
//                 candidate_distances_ptr[i] = static_cast<float>(direction);
//             }
//         } else {
//             candidate_distances_ptr[i] = -99999999.0;
//         }
//         candidate_indices_ptr[i] = child_id;
//     }
//     __syncwarp(0xffffffff);
//     // sort
//     candidate_by_bitonic_sort_inverse<2, INTERNAL_TOPK / 32>(
//             candidate_indices_ptr,
//             candidate_distances_ptr,
//             CANDIDATE_BUFFER_SIZE
//         );

//     __syncwarp(0xffffffff);

//     uint32_t max_tid = CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO + CANDIDATE_BUFFER_SIZE - CANDIDATE_BUFFER_SIZE * PRUNE_RATIO;
//     for (uint32_t tid = threadIdx.x; tid < max_tid ; tid += blockDim.x) {
//         if (tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO) {
//             uint32_t i = tid / TEAM_SIZE;
//             INDEX_T child_id = candidate_indices_ptr[i];
//             // Distance calculation
//             DISTANCE_T norm2 = compute_similarity_vector_load_full<TEAM_SIZE>
//                                (dim, d_dataset_ptr, query_ptr, child_id, child_id != invalid_index);
//             //if (tid % TEAM_SIZE == 0 && child_id != invalid_index) atomicAdd(counter, 1);
//             uint8_t lane_id = threadIdx.x % TEAM_SIZE;
//             if (lane_id == 0) {
//                 if(child_id != invalid_index) {
//                     candidate_distances_ptr[i] = norm2;
//                 } else {
//                     candidate_distances_ptr[i] = FLT_MAX;
//                 }
//             }
//         } else {
//             uint32_t i = tid - CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO + CANDIDATE_BUFFER_SIZE * PRUNE_RATIO;
//             candidate_distances_ptr[i] = FLT_MAX;
//         }
//     }
// }


// ═══════════════════════════════════════════════════════════════════════
// ── include/hash_filter.hpp (inlined) ──
// ═══════════════════════════════════════════════════════════════════════


#include <vector>
#include <unordered_set>
#include <stdint.h>
#include <string.h>
#include <cfloat>
#include <limits>

template <typename T>
class hash_filter {
private:
  int bits;
  int capacity;
  std::vector<T> seen;
  // std::hash<T> hasher;
  std::unordered_set<T> stl_hash_;
/*
parlay uses:
inline uint64_t hash64_2(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}
*/
  size_t total_inserts = 0;
  size_t total_collisions = 0;
  static constexpr T kEmptyValue = std::numeric_limits<T>::max();

public:
  hash_filter(int L) {
    int bit_size = 0;
    int tmp = L;
    while (tmp != 0) { bit_size++; tmp >>= 1; }
    bits = (bit_size + 4) / 2 + 3;
    capacity = 1 << bits;
    seen.resize(capacity);
    fill(seen.begin(), seen.end(), kEmptyValue);
  }
  int get_hash(T thing) {
    //return hasher(thing) & ((1 << bits)-1);
    return thing & ((1 << bits)-1);
  }

  bool contains(T thing) {
    T val = seen[get_hash(thing)];
    if (val == thing) return true;
    if (val == kEmptyValue) return false;
    return stl_hash_.find(thing) != stl_hash_.end();
  }
  bool add(T thing) {
    T& val = seen[get_hash(thing)];
    if (val == thing) return false;
    total_inserts++;
    if (val != kEmptyValue) total_collisions++;  // slot occupied by different ID, spill to overflow
    else { val = thing; return true; }
    stl_hash_.insert(thing);
    return true;
  }

  size_t get_overflow_size() const { return stl_hash_.size(); }
  size_t get_total_inserts() const { return total_inserts; }
  size_t get_total_collisions() const { return total_collisions; }
  double get_collision_ratio() const { return total_inserts > 0 ? (double)total_collisions / total_inserts : 0.0; }

  void clear() {
    fill(seen.begin(), seen.end(), -1);
    stl_hash_.clear();
  }
};


// ═══════════════════════════════════════════════════════════════════════
// ── include/pqueue.hpp (inlined) ──
// ═══════════════════════════════════════════════════════════════════════


#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <cfloat>
#include <cassert>

template <typename T>
class pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  int next_idx;
  std::vector<T> vid_queue;
  std::vector<float> distances;
  std::vector<uint8_t> expanded;
  std::vector<T> vid_queue2;
  std::vector<float> distances2;
  std::vector<uint8_t> expanded2;

public:
  pqueue_t(int L) {
    next_idx = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(2*L);
    distances.resize(2*L);
    expanded.resize(2*L);
    vid_queue2.resize(2*L);
    distances2.resize(2*L);
    expanded2.resize(2*L);
    std::fill(expanded.begin(), expanded.end(), 0);
  }
  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : distances[queue_size-1]; }
  float get_dist(int idx) const { return distances[idx]; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }

  void set_expanded(int idx) { expanded[idx] = 1; }
  void set_unexpanded(int idx) { expanded[idx] = 0; }

  void inc_next_index() { next_idx++; }

  bool split_queue(std::vector<pqueue_t> &lqs) {
    if (queue_size == 0) return false;
    int num_tasks = 0;
    size_t j = 0;
    for (int i = next_idx; i < queue_size; i++) {
      if (expanded[i]) continue;
      //auto loc = lqs[j].push(vid_queue[i], distances[i]);
      auto loc = lqs[j].insert(vid_queue[i], distances[i]);
      if (loc >= 0) {
        set_expanded(i);
        num_tasks ++;
      } else break;
      j ++; // round robin assignment
      if (j == lqs.size()) j = 0;
    }
    if (num_tasks == 0) return false;
    return true;
  }

  int merge_queues(std::vector<pqueue_t> &lqs) {
    // linear insertion
    // TODO: do 'merge' instead to improve performance
    for (size_t i = 0; i < lqs.size(); i++) {
      for (int j = 0; j < lqs[i].size(); j++) {
        auto loc = push(lqs[i][j], lqs[i].get_dist(j));
        if (loc == -2) { // Duplicate
          if (!lqs[i].is_expanded(j) && is_expanded(loc)) {
            set_unexpanded(loc);
            if (loc < next_idx) next_idx = loc;
          }
        } else if (loc >= 0) {
          if (lqs[i].is_expanded(j)) {
            set_expanded(loc);
            if (loc < next_idx) next_idx++;
          } else if (loc < next_idx) next_idx = loc;
        }
      }
    }
    return next_idx;
  }

  int insert(T vid, float dist) {
    if (queue_size == queue_capacity) return -1;
    vid_queue[queue_size] = vid;
    distances[queue_size] = dist;
    expanded[queue_size] = 0;
    return queue_size ++;
  }
  int push(std::pair<float, T> element) {
    auto vid = element.second;
    auto dist = element.first;
    return push(vid, dist);
  }
  int push(T vid, float dist) {
    //printf("pushing %d with dist %f into the queue\n", vid, dist);
    if (0 == queue_size) {
      vid_queue[0] = vid;
      queue_size = 1;
      distances[0] = dist;
      expanded[0] = 0; // not expanded yet
      return 0;
    }
    // Find the insert location
    auto queue_start = &distances[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc != queue_size) {
      if (vid == vid_queue[insert_loc]) { // Duplicate, skip
        return -2;
      }
      if (queue_size >= queue_capacity) { // Queue is full
        --queue_size;
      }
    } else { // insert_loc == queue_size, insert at the end?
      if (queue_size < queue_capacity) { // Queue is not full
        // Insert at the end
        vid_queue[insert_loc] = vid;
        distances[insert_loc] = dist;
        expanded[insert_loc] = 0; // not expanded yet
        ++queue_size;
        return insert_loc;
      } else { // Queue is full, skip
        return -1;
      }
    }
    // Add into queue
    auto num = queue_size - insert_loc;
    memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
            reinterpret_cast<char *>(&vid_queue[insert_loc]),
            num * sizeof(T));
    vid_queue[insert_loc] = vid;
    memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
            reinterpret_cast<char *>(&distances[insert_loc]),
            num * sizeof(float));
    distances[insert_loc] = dist;
    memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
            reinterpret_cast<char *>(&expanded[insert_loc]),
            num * sizeof(uint8_t));
    expanded[insert_loc] = 0; // not expanded yet
    ++queue_size;
    if (insert_loc < next_idx) next_idx = insert_loc;
    return insert_loc;
  }

  void batch_push(const std::vector<std::pair<float,T> > &ins) {
    if (ins.empty()) return;
    if (ins.size() == 1) {
      push(ins[0]);
      return;
    }

    int i = 0,j = 0,k = 0;
    while ((i < queue_size) && (j < ins.size())) {
      if (vid_queue[i] == ins[j].second) j++;
      else if (distances[i] < ins[j].first) {
        vid_queue2[k] = vid_queue[i];
        distances2[k] = distances[i];
        expanded2[k] = expanded[i];
        i++,k++;
      }
      else {
        vid_queue2[k] = ins[j].second;
        distances2[k] = ins[j].first;
        expanded2[k] = 0;
        j++,k++;
      }
    }
    while (i < queue_size) {
      vid_queue2[k] = vid_queue[i];
      distances2[k] = distances[i];
      expanded2[k] = expanded[i];
      i++,k++;
    }
    while (j < ins.size()) {
      vid_queue2[k] = ins[j].second;
      distances2[k] = ins[j].first;
      expanded2[k] = 0;
      j++,k++;
    }

    queue_size = std::min(k,queue_capacity);
    /*
    memmove(reinterpret_cast<char *>(&vid_queue[0]),
            reinterpret_cast<char *>(&vid_queue2[0]),
            queue_size * sizeof(T));
    memmove(reinterpret_cast<char *>(&distances[0]),
            reinterpret_cast<char *>(&distances2[0]),
            queue_size * sizeof(float));
    memmove(reinterpret_cast<char *>(&expanded[0]),
            reinterpret_cast<char *>(&expanded2[0]),
            queue_size * sizeof(uint8_t));
    */
    swap(vid_queue,vid_queue2);
    swap(distances,distances2);
    swap(expanded,expanded2);

    next_idx = 0;
  }

  T* fetch_unexpanded_nodes(int P) {
    T* nodes = new T[P];
    int count = 0;
    for (int i = 0; i < queue_size; i++) {
      if (expanded[i]) nodes[count++] = i;
      if (count == P) break;
    }
    return nodes;
  }
};

template <typename T>
class new_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  // maintain invariant that next_idx and next_idx2 are always the two smallest
  // unexpanded positions. If they don't exist, then they equal queue_size.
  int next_idx, next_idx2;
  void _check_invariant() {
    assert(queue_size <= queue_capacity);
    assert(next_idx <= queue_size);
    assert((next_idx == queue_size) || !expanded[next_idx]);
    assert(next_idx2 <= queue_size);
    assert((next_idx2 == queue_size) || !expanded[next_idx2]);
    assert(next_idx <= next_idx2);
    assert((next_idx == queue_size) || (next_idx < next_idx2));
  }
  std::vector<T> vid_queue;
  std::vector<float> distances;
  std::vector<uint8_t> expanded;

public:
  new_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(L+1);
    distances.resize(L+1);
    expanded.resize(L+1);
    std::fill(expanded.begin(), expanded.end(), 0);
  }

  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  int get_next_index2() const { return next_idx2; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : distances[queue_size-1]; }
  float get_dist(int idx) const { return distances[idx]; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }
  
  bool has_unexpanded() { return next_idx < queue_size; }
  bool has_second_unexpanded() { return next_idx2 < queue_size; }
  void set_front_expanded() {
    expanded[next_idx] = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && expanded[next_idx2]) next_idx2++;
    //_check_invariant();
  }
  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > distances[queue_size-1]))
      return; // queue is full
    auto queue_start = &distances[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc < queue_size) {
      // if (vid_queue[insert_loc] == vid) return; // duplicate
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
              reinterpret_cast<char *>(&distances[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    distances[insert_loc] = dist;
    expanded[insert_loc] = 0;
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size:next_idx+1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
    //_check_invariant();
  }

  void push_update(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > distances[queue_size-1]))
      return;

    int existing_loc = -1;    // check for existing entry's location
    for (int i = 0; i < queue_size; i++) {
      if (vid_queue[i] == vid) { existing_loc = i; break; }
    }
    if (existing_loc < 0) return;

    if (distances[existing_loc] <= dist) return; // existing entry is already closer

    // find where the new entry should go
    auto queue_start = &distances[0];
    int insert_loc = std::lower_bound(queue_start, queue_start + existing_loc, dist) - queue_start;
    
    auto num = existing_loc - insert_loc;
    if (num > 0) {
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
              reinterpret_cast<char *>(&distances[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    distances[insert_loc] = dist;
    expanded[insert_loc] = 0;

    // Recompute next_idx/next_idx2 since the shift may have moved them
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int k = 0; k < queue_size; k++) {
      if (!expanded[k]) {
        if (next_idx == queue_size) next_idx = k;
        else { next_idx2 = k; break; }
      }
    }
    //_check_invariant();
  }
};

template <typename T>
struct pqueue_entry {
  T vid;
  float dist;
  uint8_t expanded;
};

// alternative: array of struct instead of struct of array
template <typename T>
class struct_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  int next_idx, next_idx2;

  void _check_invariant() {
    assert(queue_size <= queue_capacity);
    assert(next_idx <= queue_size);
    assert((next_idx == queue_size) || !entries[next_idx].expanded);
    assert(next_idx2 <= queue_size);
    assert((next_idx2 == queue_size) || !entries[next_idx2].expanded);
    assert(next_idx <= next_idx2);
    assert((next_idx == queue_size) || (next_idx < next_idx2));
  }

  std::vector<pqueue_entry<T>> entries;

public:
  struct_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    entries.resize(L + 1);
    for (auto& e : entries) e.expanded = 0;
  }

  T& operator[](size_t i) { return entries[i].vid; }
  const T& operator[](size_t i) const { return entries[i].vid; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  int get_next_index2() const { return next_idx2; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : entries[queue_size - 1].dist; }
  float get_dist(int idx) const { return entries[idx].dist; }
  bool is_expanded(int idx) const { return entries[idx].expanded == 1; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }

  bool has_unexpanded() { return next_idx < queue_size; }
  bool has_second_unexpanded() { return next_idx2 < queue_size; }

  void set_front_expanded() {
    entries[next_idx].expanded = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && entries[next_idx2].expanded) next_idx2++;
    //_check_invariant();
  }

  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > entries[queue_size - 1].dist))
      return;

    // binary search on dist field
    int lo = 0, hi = queue_size;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (entries[mid].dist < dist) lo = mid + 1;
      else hi = mid;
    }
    int insert_loc = lo;

    if (insert_loc < queue_size) {
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char*>(&entries[insert_loc + 1]),
              reinterpret_cast<char*>(&entries[insert_loc]),
              num * sizeof(pqueue_entry<T>));
    }
    entries[insert_loc] = {vid, dist, 0};
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size : next_idx + 1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
    //_check_invariant();
  }

  void push_update(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > entries[queue_size - 1].dist))
      return;

    int existing_loc = -1;
    for (int i = 0; i < queue_size; i++) {
      if (entries[i].vid == vid) { existing_loc = i; break; }
    }
    if (existing_loc < 0) return;

    if (entries[existing_loc].dist <= dist) return;

    // binary search up to existing_loc
    int lo = 0, hi = existing_loc;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (entries[mid].dist < dist) lo = mid + 1;
      else hi = mid;
    }
    int insert_loc = lo;

    auto num = existing_loc - insert_loc;
    if (num > 0) {
      memmove(reinterpret_cast<char*>(&entries[insert_loc + 1]),
              reinterpret_cast<char*>(&entries[insert_loc]),
              num * sizeof(pqueue_entry<T>));
    }
    entries[insert_loc] = {vid, dist, 0};

    // Recompute next_idx/next_idx2 since the shift may have moved them
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int k = 0; k < queue_size; k++) {
      if (!entries[k].expanded) {
        if (next_idx == queue_size) next_idx = k;
        else { next_idx2 = k; break; }
      }
    }
    //_check_invariant();
  }
};

template <typename T>
class both_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  // maintain invariant that next_idx and next_idx2 are always the two smallest
  // unexpanded positions. If they don't exist, then they equal queue_size.
  int next_idx, next_idx2;
  void _check_invariant() {
    assert(queue_size <= queue_capacity);
    assert(next_idx <= queue_size);
    assert((next_idx == queue_size) || !expanded[next_idx]);
    assert(next_idx2 <= queue_size);
    assert((next_idx2 == queue_size) || !expanded[next_idx2]);
    assert(next_idx <= next_idx2);
    assert((next_idx == queue_size) || (next_idx < next_idx2));
  }
  std::vector<T> vid_queue;
  std::vector<float> rabitq_dist;
  std::vector<float> l2_dist;
  std::vector<uint8_t> expanded;

public:
  both_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(L+1);
    rabitq_dist.resize(L+1);
    l2_dist.resize(L+1);
    expanded.resize(L+1);
    std::fill(l2_dist.begin(), l2_dist.end(), -1.0f);
    std::fill(expanded.begin(), expanded.end(), 0);
  }

  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  int get_next_index2() const { return next_idx2; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : rabitq_dist[queue_size-1]; }
  float get_dist(int idx) const { return rabitq_dist[idx]; }
  float get_l2_dist(int idx) const { return l2_dist[idx]; }
  void set_l2_dist(int idx, float dist) { l2_dist[idx] = dist; }
  bool has_l2_dist(int idx) const { return l2_dist[idx] >= 0.0f; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; next_idx2 = 0; }
  
  bool has_unexpanded() { return next_idx < queue_size; }
  bool has_second_unexpanded() { return next_idx2 < queue_size; }
  void recompute_front() {
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int i = 0; i < queue_size; i++) {
      if (!expanded[i]) {
        if (next_idx == queue_size) next_idx = i;
        else {
          next_idx2 = i;
          break;
        }
      }
    }
  }
  void set_front_expanded() {
    expanded[next_idx] = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && expanded[next_idx2]) next_idx2++;
    //_check_invariant();
  }
  void set_expanded(int idx) {
    if ((idx < 0) || (idx >= queue_size) || expanded[idx]) return;
    expanded[idx] = 1;
    recompute_front();
  }
  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > rabitq_dist[queue_size-1]))
      return; // queue is full
    auto queue_start = &rabitq_dist[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc < queue_size) {
      // if (vid_queue[insert_loc] == vid) return; // duplicate
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&rabitq_dist[insert_loc + 1]),
              reinterpret_cast<char *>(&rabitq_dist[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&l2_dist[insert_loc + 1]),
              reinterpret_cast<char *>(&l2_dist[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    rabitq_dist[insert_loc] = dist;
    l2_dist[insert_loc] = -1;
    expanded[insert_loc] = 0;
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size:next_idx+1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
    //_check_invariant();
  }
};


// ═══════════════════════════════════════════════════════════════════════
// ── include/timer.hpp (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

// Copyright (c) 2020 MIT
#include <sys/time.h>
#include <iostream>

class Timer {
public:
  Timer() : Timer("null") {}
  Timer(std::string name) : name_(name) {}
  void Start() {
    gettimeofday(&start_time_, NULL);
  }
  void Stop() {
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
  }
  double Seconds() const {
    return (double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6;
  }
  double Millisecs() const {
    return (double)1000*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/(double)1000;
  }
  double Microsecs() const {
    return 1e6*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec;
  }
  std::string get_name() { return name_; }
private:
  std::string name_;
  struct timeval start_time_;
  struct timeval elapsed_time_;
};

// Times op's execution using the timer t
#define TIME_OP(t, op) { t.Start(); (op); t.Stop(); }

template <typename F>
void timeThis(const F& f, const std::string name) {
  Timer t(name);
  t.Start();
  f();
  t.Stop();
  std::cout << "runtime[" << t.get_name() << "] = " << t.Seconds() << " sec\n";
}

inline uint64_t read_cycle() {
  volatile uint64_t t = 0;
#ifdef PROFILE_LATENCY
  asm __volatile__(
      "lfence\n"
      // Guaranteed to clear the high-order 32 bits of RAX and RDX.
      "rdtsc\n"
      "shlq $32, %%rdx\n"
      "orq %%rdx, %%rax\n"
      : "=a" (t)
      :
      : "%rdx"
      );
#endif
  return t;
}



// ═══════════════════════════════════════════════════════════════════════
// ── include/ctimer.h (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

/* -*- c -*- */

/**
 * [Include-only header library]
 * C/C++ timer utilities using POSIX `clock_gettime()`.
 *
 * @sa <https://github.com/sillycross/mlpds/blob/master/fasttime.h>
 *
 * @file        ctimer.h
 * @version     1.0.0
 * @author      Alexandros-Stavros Iliopoulos
 * @license     MIT
 * @copyright   Copyright (c) 2021 Supertech Research Group, CSAIL, MIT
 */


/******************************************************************************/
/* MIT License                                                                */
/*                                                                            */
/* Copyright (c) 2021 Supertech Research Group, CSAIL, MIT                    */
/*                                                                            */
/* Permission is hereby granted, free of charge, to any person obtaining      */
/* a copy of this software and associated documentation files (the            */
/* "Software"), to deal in the Software without restriction, including        */
/* without limitation the rights to use, copy, modify, merge, publish,        */
/* distribute, sublicense, and/or sell copies of the Software, and to         */
/* permit persons to whom the Software is furnished to do so, subject to      */
/* the following conditions:                                                  */
/*                                                                            */
/* The above copyright notice and this permission notice shall be             */
/* included in all copies or substantial portions of the Software.            */
/*                                                                            */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,            */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF         */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.     */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY       */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,       */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE          */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                     */
/******************************************************************************/


/**
 * @mainpage
 *
 * @section overview Overview
 *
 * CTimer is an include-only header library for timing C/C++ code.  CTimer uses
 * the [POSIX `clock_gettime()`][url-man-clock-gettime] function with a
 * `CLOCK_MONOTONIC` clock to get relative time stamps.
 *
 * [url-man-clock-gettime]: https://man7.org/linux/man-pages/man3/clock_gettime.3.html
 *
 * Stopwatch utilities:
 * - `ctimer_t`         :: type of CTimer stopwatch struct
 * - `ctimer_start()`   :: start stopwatch
 * - `ctimer_stop()`    :: stop stopwatch
 * - `ctimer_reset()`   :: reset elapsed time
 * - `ctimer_measure()` :: measure elapsed time between start & stop
 * - `ctimer_lap()`     :: accumulate elapsed time between start & stop
 * - `ctimer_print()`   :: print elapsed time in sec with fixed format
 *
 * Timespec struct utilities
 * - `timespec_sub()`   :: calculate difference between 2 timespecs
 * - `timespec_add()`   :: calculate sum of 2 timespecs
 * - `timespec_sec()`   :: timespec tv time in sec (double)
 * - `timespec_msec()`  :: timespec tv time in msec (long)
 * - `timespec_usec()`  :: timespec tv time in usec (long)
 * - `timespec_nsec()`  :: timespec tv time in nsec (long)
 *
 * @section usage Using CTimer
 *
 * @subsection c_std C standard
 *
 * C compilers may require standard `gnu99` or later.  Older compilers may also
 * require linking with `-lrt`.
 *
 * @subsection init Initialization
 *
 * There are no guarantees regarding the initial values of timespec fields in a
 * `ctimer_t` stopwatch.  Querying timespecs that haven't been initialized or
 * measured may return arbitrary results; this includes measuring the elapsed
 * time of an unstopped stopwatch.
 *
 * The `elapsed` timespec of a `ctimer_t` stopwatch can be reset to 0 using the
 * `ctimer_reset()` function.  This is not necessary if timings are only
 * measured using `ctimer_measure()`, but it *is* necessary before using
 * `ctimer_lap()` with an otherwise un-measured stopwatch.
 *
 * @subsection measure_on_stop Automatic elapsed-time measurement on stop
 *
 * If the preprocessor macro `CTIMER_MEASURE_ON_STOP` is defined, then
 * `ctimer_stop()` also calls `ctimer_measure()` internally to calculate and
 * store the elapsed time in the input `ctimer_t` object.
 *
 * @subsection example Example usage in C/C++
 *
 * @snippet ctimer_example.c ctimer_example
 */


#ifndef __H_CTIMER__
#define __H_CTIMER__


#include <time.h>
#include <unistd.h>
#include <stdio.h>


/**
 * @defgroup ctimer CTimer
 *
 * C/C++ timer utilities.
 *
 * @{
 */


/* prevent C++ compilers from mangling function names */
#ifdef __cplusplus
extern "C" {
#endif


/* ==================================================
 * CONSTANTS
 * ================================================== */


/* unit conversion constants */
#ifdef __cplusplus              /* C++ */
namespace {
    const long _MSEC_PER_SEC = 1000;
    const long _USEC_PER_SEC = 1000 * 1000;
    const long _NSEC_PER_SEC = 1000 * 1000 * 1000;
}
#else  /* C */
enum {
    _MSEC_PER_SEC = 1000,
    _USEC_PER_SEC = 1000 * 1000,
    _NSEC_PER_SEC = 1000 * 1000 * 1000
};
#endif  /* __cplusplus */


/* ==================================================
 * TIMESPEC API
 * ================================================== */


/**
 * @defgroup ctimer_timespec Timespec API
 *
 * Functions for manipulating and inspecting timespec structs.
 *
 * @{
 */


/**
 * Calculate the time difference between two `timespec` structs.  Store time in
 * sec and nsec in the `tv_sec` and `tv_nsec` field, respectively, of the output
 * `timespec`.
 *
 * @sa https://stackoverflow.com/a/53708448/1036677
 */
static inline
void timespec_sub(
    struct timespec       * t_diff, /**<[out] time difference */
    struct timespec const   t_end,  /**<[in]  end time */
    struct timespec const   t_start /**<[in]  start time */
) {
    t_diff->tv_nsec = t_end.tv_nsec - t_start.tv_nsec;
    t_diff->tv_sec  = t_end.tv_sec  - t_start.tv_sec;
    if ((t_diff->tv_sec > 0) && (t_diff->tv_nsec < 0)) {
        t_diff->tv_nsec += _NSEC_PER_SEC;
        t_diff->tv_sec--;
    } else if ((t_diff->tv_sec < 0) && (t_diff->tv_nsec > 0)) {
        t_diff->tv_nsec -= _NSEC_PER_SEC;
        t_diff->tv_sec++;
    }
    /* (s > 0 & ns > 0) : do nothing (t_start < t_end) */
    /* (s < 0 & ns < 0) : do nothing (t_start > t_end) */
}


/**
 * Calculate the time sum of two `timespec` structs.  Store time in sec and nsec
 * in the `tv_sec` and `tv_nsec` field, respectively, of the output `timespec`.
 */
static inline
void timespec_add(
    struct timespec       * t_sum, /**<[out] time sum */
    struct timespec const   t1,    /**<[in]  time 1 */
    struct timespec const   t2     /**<[in]  time 2 */
) {
    t_sum->tv_nsec = t1.tv_nsec + t2.tv_nsec;
    t_sum->tv_sec  = t1.tv_sec  + t2.tv_sec;
    if (t_sum->tv_nsec >= _NSEC_PER_SEC) {
        t_sum->tv_nsec -= _NSEC_PER_SEC;
        t_sum->tv_sec++;
    }
}


/**
 * Return `timespec` time in sec.
 *
 * @return (t.tv_sec + t.tv_nsec) in sec
 */
static inline
double timespec_sec(
    struct timespec const t
) {
    return (double)t.tv_sec
        + (double)t.tv_nsec / _NSEC_PER_SEC;
}


/**
 * Return `timespec` time in msec.
 *
 * @warning Sub-millisecond resolution (if supported by the system clock) is
 * lost with this function.
 *
 * @return (t.tv_sec + t.tv_nsec) in msec
 */
static inline
long timespec_msec(
    struct timespec const t
) {
    return t.tv_sec * _MSEC_PER_SEC
        + t.tv_nsec / _USEC_PER_SEC;
}


/**
 * Return `timespec` time in usec.
 *
 * @warning Sub-microsecond resolution (if supported by the system clock) is
 * lost with this function.
 *
 * @return (t.tv_sec + t.tv_nsec) in usec
 */
static inline
long timespec_usec(
    struct timespec const t
) {
    return t.tv_sec * _USEC_PER_SEC
        + t.tv_nsec / _MSEC_PER_SEC;
}


/**
 * Return `timespec` time in nsec.
 *
 * @return (t.tv_sec + t.tv_nsec) in nsec
 */
static inline
long timespec_nsec(
    struct timespec const t
) {
    return t.tv_sec * _NSEC_PER_SEC
        + t.tv_nsec;
}


/** @} */ /* end group ctimer_timespec */


/* ==================================================
 * STOPWATCH API
 * ================================================== */


/**
 * @defgroup ctimer_stopwatch Stopwatch API
 *
 * Functions for manipulating CTimer stopwatches.
 *
 * @{
 */


/**
 * Stopwatch timer struct using `clock_gettime()`.
 */
typedef struct {
    struct timespec start;      /**< Stopwatch start time  */
    struct timespec end;        /**< Stopwatch end time */
    struct timespec elapsed;    /**< Elapsed/measured time */
} ctimer_t;


/**
 * Measure elapsed time of `ctimer_t` stopwatch in s+ns and *store* it in the
 * `elapsed` timer.
 *
 * @warning The stopwatch must be started and stopped before the elapsed time is
 * measured.
 *
 * @note It is safe (albeit unnecessary) to measure the elapsed time of a
 * stopped timer multiple times.
 *
 * @note When measuring the cumulative execution time of non-contiguous chunks
 * of a program (e.g., the total time for a sub-computation of a loop body), use
 * `ctimer_lap()`.
 *
 * @sa ctimer_lap
 * @sa ctimer_start
 * @sa ctimer_stop
 */
static inline
void ctimer_measure(
    ctimer_t * t                /**<[in,out] stopwatch pointer */
) {
    timespec_sub(&t->elapsed, t->end, t->start);
}


/**
 * Measure elapsed time of `ctimer_t` stopwatch in s+ns and *add* it to the
 * `elapsed` timer.
 *
 * @warning The `elapsed` field of the input stopwatch must be properly
 * initialized (e.g. with `ctimer_reset()`) before calling `ctimer_lap()`.
 *
 * @warning The stopwatch must be started and stopped before the elapsed time is
 * measured.
 *
 * @note When measuring the execution time of a single, contiguous chunk of a
 * program, use `ctimer_measure()`.
 *
 * @sa ctimer_reset
 * @sa ctimer_measure
 * @sa ctimer_start
 * @sa ctimer_stop
 */
static inline
void ctimer_lap(
    ctimer_t * t                /**<[in,out] stopwatch pointer */
) {
    /* elapsed += end - start */
    timespec_add(&t->elapsed, t->elapsed, t->end);
    timespec_sub(&t->elapsed, t->elapsed, t->start);
}


/**
 * Zero out the `elapsed` timer of a `ctimer_t` stopwatch.
 *
 * @sa ctimer_lap
 */
static inline
void ctimer_reset(
    ctimer_t * t                /**<[in,out] stopwatch pointer */
) {
    t->elapsed = (struct timespec){0, 0};
}


/**
 * Start a `ctimer_t` stopwatch.  Sets the the `start` timer of the stopwatch.
 *
 * @sa ctimer_stop
 * @sa ctimer_measure
 * @sa ctimer_lap
 */
static inline
void ctimer_start(
    ctimer_t * t                /**<[in,out] stopwatch pointer */
) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}


/**
 * Stop a `ctimer_t` stopwatch.  Sets the `end` timer of the stopwatch.
 *
 * @note If the `CTIMER_MEASURE_ON_STOP` preprocessor macro is defined _before_
 * the `ctimer/ctimer.h` library is included in a program, then `ctimer_stop`
 * also calculates the elapsed time between `start` and `end` and stores it in
 * the `elapsed` field.
 *
 * @sa ctimer_start
 * @sa ctimer_measure
 * @sa ctimer_lap
 */
static inline
void ctimer_stop(
    ctimer_t * t                /**<[in,out] stopwatch pointer */
) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
#ifdef CTIMER_MEASURE_ON_STOP
    ctimer_measure(t);
#endif
}


/**
 * Print a line with the `elapsed` time of a `ctimer_t` stopwatch in seconds.
 *
 * The line is printed as:
 * ```
 * Time(<label>) = XX.XXXXXXXXX sec
 * ```
 *
 * If `label` is `NULL` or the empty string, the "(<label>)" tag is omitted
 * from the printed output.
 *
 * @note The elapsed time is always printed with 9 decimal digits, although the
 * system `CLOCK_MONOTONIC`, may have coarser resolution (e.g., microsecond
 * resolution).
 *
 * @sa ctimer_measure
 * @sa ctimer_lap
 */
static inline
void ctimer_print(
    ctimer_t const   t,         /**<[in] stopwatch pointer */
    char     const * label      /**<[in] label/description for printed time */
) {
    if ((label != NULL) && (label[0] != '\0'))
        printf("Time(%s) = ", label);
    else
        printf("Time = ");

    printf("%ld.%09ld sec\n", (long)t.elapsed.tv_sec, t.elapsed.tv_nsec);
}


/** @} */ /* end group ctimer_stopwatch */


#ifdef __cplusplus
} /* end extern "C" */
#endif


/** @} */ /* end group ctimer */


#endif  /* __H_CTIMER__ */


// ═══════════════════════════════════════════════════════════════════════
// ── include/index.hpp (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <string>
// [flattened] #include "utils.h"

typedef uint32_t vid_t;

enum MetricType { METRIC_L2, METRIC_IP };

template <typename T>
class IndexGraph {
public:
    size_t ntotal;           // number of data points
    int d;                   // vector dimension
    int maxDeg;              // maximum degree
    MetricType metric;       // distance metric type
    std::vector<vid_t> edges; // adjacency list (ntotal * maxDeg)
    std::vector<T> data;     // data vectors (ntotal * d)

    IndexGraph() : ntotal(0), d(0), maxDeg(0), metric(METRIC_L2) {}

    // Get data pointer
    T* get_data_ptr() { return data.data(); }

    // Load graph file (.bin built from ScaleANN)
    void load_graph_index(const char* gFile) {
        size_t nt = 0, deg = 0;
        vid_t* edge_data = read_bin<vid_t>(gFile, nt, deg);

        ntotal = nt;
        maxDeg = deg;
        edges.assign(edge_data, edge_data + ntotal * maxDeg);

        std::cout << "Loading graph index from " << gFile 
                  << ": ntotal = " << ntotal << ", maxDeg = " << maxDeg << std::endl;

        delete[] edge_data;
    }

    // Load graph file (.meta.txt + .edge.bin format)
    void load_graph(const std::string& prefix) {
        // Read meta file
        std::string meta_file = prefix + ".meta.txt";
        std::ifstream f_meta(meta_file);
        if (!f_meta.is_open()) {
            std::cerr << "Error: cannot open " << meta_file << std::endl;
            exit(1);
        }
        
        int64_t nv, ne;
        int vid_size, eid_size, vlabel_size, elabel_size, feat_len, nvc, nec;
        f_meta >> nv >> ne >> vid_size >> eid_size >> vlabel_size >> elabel_size
               >> maxDeg >> feat_len >> nvc >> nec;
        f_meta.close();
        
        ntotal = nv;
        d = feat_len;
        
        std::cout << "Loading graph: n=" << ntotal << ", maxDeg=" << maxDeg << std::endl;
        
        // Read edge file
        std::string edge_file = prefix + ".edge.bin";
        std::ifstream f_edge(edge_file, std::ios::binary);
        if (!f_edge.is_open()) {
            std::cerr << "Error: cannot open " << edge_file << std::endl;
            exit(1);
        }
        
        edges.resize(ntotal * maxDeg);
        f_edge.read(reinterpret_cast<char*>(edges.data()), 
                    sizeof(vid_t) * ntotal * maxDeg);
        f_edge.close();
        
        std::cout << "Graph loaded: " << edges.size() << " edges" << std::endl;
    }

    // Load data vectors (.fvecs format)
    void load_data(const std::string& data_file) {
        std::ifstream f(data_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << data_file << std::endl;
            exit(1);
        }
        
        // Read first dimension
        int dim_check;
        f.read(reinterpret_cast<char*>(&dim_check), sizeof(int));
        
        // Get file size to calculate number of vectors
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t n = file_size / (sizeof(int) + dim_check * sizeof(T));
        
        // Verify count matches if graph is already loaded
        if (ntotal > 0 && n != ntotal) {
            std::cerr << "Warning: data file has " << n << " vectors, but graph has " 
                      << ntotal << " vertices" << std::endl;
        }
        
        if (ntotal == 0) ntotal = n;
        if (d == 0) d = dim_check;
        
        std::cout << "Loading data: n=" << n << ", dim=" << d << std::endl;
        
        // Read all vectors
        data.resize(ntotal * d);
        f.seekg(0, std::ios::beg);
        
        for (size_t i = 0; i < ntotal; i++) {
            int dim_i;
            f.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
            assert(dim_i == d);
            f.read(reinterpret_cast<char*>(&data[i * d]), d * sizeof(T));
        }
        f.close();
        
    }

    // Load query vectors
    static std::vector<T> load_queries(const std::string& query_file, size_t& nq, int& dim) {
        std::ifstream f(query_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << query_file << std::endl;
            exit(1);
        }
        
        // Read dimension
        f.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
        // Get number of vectors
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + dim * sizeof(T));
        
        std::cout << "Loading queries: nq=" << nq << ", dim=" << dim << std::endl;
        
        // Read all queries
        std::vector<T> queries(nq * dim);
        f.seekg(0, std::ios::beg);
        
        for (size_t i = 0; i < nq; i++) {
            int dim_i;
            f.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
            assert(dim_i == dim);
            f.read(reinterpret_cast<char*>(&queries[i * dim]), dim * sizeof(T));
        }
        f.close();
        
        return queries;
    }

    // Load ground truth
    static std::vector<int> load_groundtruth(const std::string& gt_file, size_t& nq, int& k) {
        std::ifstream f(gt_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << gt_file << std::endl;
            exit(1);
        }
        
        // Read k
        f.read(reinterpret_cast<char*>(&k), sizeof(int));
        
        // Get number of queries
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + k * sizeof(int));
        
        // Read all ground truth
        std::vector<int> gt(nq * k);
        f.seekg(0, std::ios::beg);
        
        for (size_t i = 0; i < nq; i++) {
            int k_i;
            f.read(reinterpret_cast<char*>(&k_i), sizeof(int));
            f.read(reinterpret_cast<char*>(&gt[i * k]), k * sizeof(int));
        }
        f.close();

        std::cout << "Loading ground truth: nq=" << nq << ", k=" << k << std::endl;
        
        return gt;
    }

    // GPU search function declaration (implemented in beam_gpu_base.cu)
    void search(int nq, const T* queries, int K, T* result_vec,
                vid_t* result_idx, int beam_sz, double& elapsed,
                GpuRunStats* gpu_stats = nullptr,
                bool debug_mode = false);
};


// ═══════════════════════════════════════════════════════════════════════
// ── include/search_bitonic.cuh (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

// [flattened] #include "utils.cuh"
// [flattened] #include "distance.cuh"
#include <cuda_runtime.h>
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          GPU_HASH_BITLEN_FOR_TOPK(INTERNAL_TOPK)
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define USE_SEED        0
#define PRUNE_CONFIG    0
#define THRESHOLD_CONFIG    0
#define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
#define RESULT_BUFFER_SIZE (INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE)

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    size_t size = 0;    
    size += hashtable_getsize(BITLEN) * sizeof(INDEX_T);    // hash table
    size += RESULT_BUFFER_SIZE * sizeof(INDEX_T);           // top k and candidate index
    size += RESULT_BUFFER_SIZE * sizeof(DISTANCE_T);        // top k and candidate distance
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += 2 * 4;                                          // terminate flag and counter
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += GPU_TIMING_COUNT * sizeof(gpu_long_t);          // timing stats
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ void init_identity_map(size_t n, vid_t* map) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) map[idx] = idx;
}

static inline std::vector<uint32_t> load_sign_bit(const char* filename) {
    return std::vector<uint32_t>();
}

template <typename T>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void BeamSearch(
    int K, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_stats,
    gpu_long_t* d_total_timing,
    vid_t* d_starting_points,
    vid_t* d_seed_map,
    uint32_t* d_sign_bit,
    GraphGPU<vid_t> gg,
    bool debug_mode
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id > nq) return;
    const T* query = d_queries + query_id * dim;
    const uint32_t maxDegree = gg.get_max_degree();
    INDEX_T* d_graph_ptr = gg.N(0);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + INTERNAL_TOPK;

    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + RESULT_BUFFER_SIZE);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + INTERNAL_TOPK;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + RESULT_BUFFER_SIZE);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + hashtable_getsize(BITLEN));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);
    uint32_t* counter = (uint32_t*)(QUERY_BUFFER + dim);
    uint32_t* hops = (uint32_t*)(counter + 1);
    uint32_t* stats = hops + 1;
    gpu_long_t* timing = reinterpret_cast<gpu_long_t*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    if (tid == 0) {
        counter[0] = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
        for (int i = 0; i < GPU_TIMING_COUNT; i++) {
            timing[i] = 0;
        }
    }

    // initialize the hash table to record visited (computed) nodes
    if (HASH_TABLE_CONFIG) hashtable_init(HASH_TABLE, BITLEN);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }
    
    // load the top k list with dummy initial values
    for (int i = tid; i < INTERNAL_TOPK; i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }

    // load the candidate buffer with starting points
    for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
        CANDIDATE_INDEX[i] = d_starting_points[i];
        DISTANCE_T dist = FLT_MAX;
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0 && CANDIDATE_INDEX[i] != MAX_INDEX) l2_start = gpu_globaltimer();
        if (CANDIDATE_INDEX[i] != MAX_INDEX) {
            dist = warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim);
        }
        if (lane_id == 0) {     // warp computes, only lane 0 writes
            if (CANDIDATE_INDEX[i] != MAX_INDEX) {
                atomicAdd(&counter[0], 1);
                hashtable_insert(HASH_TABLE, BITLEN, CANDIDATE_INDEX[i]);
                if (GPU_DISTANCE_TIMING) {
                    atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
                    atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
                }
            }
            CANDIDATE_DISTANCE[i] = dist;
        }
    }
    __syncthreads();
    
    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // use warp level sort to find the actual closest top K elements and update the topK list
        topk_by_sort_candidates_and_merge<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
            CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, (iter == 0));
        __syncthreads();

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hashtable_init(HASH_TABLE, BITLEN);
                __syncthreads();
                hashtable_restore(HASH_TABLE, BITLEN, TOP_K_INDEX, INTERNAL_TOPK);
            }
            __syncthreads();
        }

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, INTERNAL_TOPK, TOP_K_INDEX);
        }
        __syncthreads();

        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;
        vid_t* neighbors = gg.N(current_node);
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        DISTANCE_T exact_dist = warp_l2_distance(dim, query, d_data + current_node * dim);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
        __syncthreads();

        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < maxDegree) ? neighbors[i] : MAX_INDEX;      // only in-bound threads take from the neighbor list

            if (child_id != MAX_INDEX) {    // check if visited already
                if (hashtable_insert(HASH_TABLE, BITLEN, child_id) == 0) {
                    child_id = MAX_INDEX;
                    CANDIDATE_DISTANCE[i] = FLT_MAX;
                } else {
                    CANDIDATE_DISTANCE[i] = 0;
                }
            }
            CANDIDATE_INDEX[i] = child_id;
        }
        __syncthreads();

        // all warps work together to compute candidate distance
        for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
            // visited candidates are marked as invalid
            if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

            // if current node is new in hash table, compute and store its distance
            gpu_long_t cand_l2_start = 0;
            if (GPU_DISTANCE_TIMING && lane_id == 0) cand_l2_start = gpu_globaltimer();
            DISTANCE_T dist = warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim);
            if (lane_id == 0) {
                atomicAdd(&counter[0], 1);
                if (GPU_DISTANCE_TIMING) {
                    atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
                    atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - cand_l2_start);
                }
                CANDIDATE_DISTANCE[i] = dist;
            }
        }
        __syncthreads();
    }

    // final sort before writeback
    topk_by_sort_candidates_and_merge<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, false);
    __syncthreads();

    // return the top K result
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[i] & ~0x80000000;
    }
    if (tid == 0) {
        if (GPU_SEARCH_STATS) {
            stats[GPU_SEARCH_STAT_DIST_CMPS] = counter[0];
            stats[GPU_SEARCH_STAT_DIST_EST] = 0;
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                atomicAdd(&d_total_stats[i], gpu_long_t(stats[i]));
            }
        }
        for (int i = 0; i < GPU_TIMING_COUNT; i++) {
            atomicAdd(&d_total_timing[i], timing[i]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// ── include/search_rabitq.cuh (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

// [flattened] #include "utils.cuh"
// [flattened] #include "distance.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          GPU_HASH_BITLEN_FOR_TOPK(INTERNAL_TOPK)
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
#define RESULT_BUFFER_SIZE (INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE)

#ifndef GPU_RABITQ_FASTSCAN
#define GPU_RABITQ_FASTSCAN 1
#endif

template <typename T>
static __device__ inline void staged_copy_async_begin_or_sync(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        __pipeline_memcpy_async(dst + i, src + i, sizeof(T));
    }
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
#endif
}

static __device__ inline void staged_copy_async_finish_or_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

static __host__ inline uint32_t calculate_shared_mem_size_rabitq(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    auto align_up = [](size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    };
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;
    size_t size = 0;    
    size += hashtable_getsize(BITLEN) * sizeof(INDEX_T);    // hash table
    size += RESULT_BUFFER_SIZE * sizeof(INDEX_T);           // top k and candidate index
    size += RESULT_BUFFER_SIZE * sizeof(DISTANCE_T);        // top k and candidate distance
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += padded_dim * sizeof(float);                     // rotated query
    size += padded_dim * sizeof(uint8_t);                   // quantized query
    size += (padded_dim << 2) * sizeof(uint8_t);            // LUT buffer
    size += 3 * sizeof(float);                              // low/high/width
    size += sizeof(int32_t);                                // sum_q
    size = align_up(size, alignof(uint64_t));               // align staged candidate codes
    size += max_deg * code_words * sizeof(uint64_t);        // staged candidate codes
    size += 3 * max_deg * sizeof(float);                    // staged candidate factors
    size += 2 * sizeof(uint32_t);                           // counter and hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += topk * sizeof(INDEX_T);                         // result pool index
    size += topk * sizeof(DISTANCE_T);                      // result pool distance
    size += 2 * sizeof(uint32_t);                           // result pool size and worst idx
    size += sizeof(DISTANCE_T);                             // result pool worst dist
#if GPU_DISTANCE_TIMING
    size = align_up(size, alignof(gpu_long_t));             // align timing stats
    size += GPU_TIMING_COUNT * sizeof(gpu_long_t);          // timing stats
#endif
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedBeamSearch(
    int K, int nq, int dim, int max_degree, size_t npoints,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_block_stats,
    gpu_long_t* d_block_timing,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset,
    bool debug_mode
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + INTERNAL_TOPK;

    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + RESULT_BUFFER_SIZE);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + INTERNAL_TOPK;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + RESULT_BUFFER_SIZE);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + hashtable_getsize(BITLEN));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* QUANTIZED_QUERY_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    uint8_t* LUT_BUFFER = QUANTIZED_QUERY_BUFFER + padded_dim;
    float* low_val = reinterpret_cast<float*>(LUT_BUFFER + (padded_dim << 2));
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* staged_base = reinterpret_cast<char*>(sum_q + 1);
    staged_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint64_t* STAGED_CODE_BUFFER = reinterpret_cast<uint64_t*>(staged_base);
    float* STAGED_FACTOR_BUFFER = reinterpret_cast<float*>(STAGED_CODE_BUFFER + max_degree * code_words);
    
    uint32_t* counter = reinterpret_cast<uint32_t*>(STAGED_FACTOR_BUFFER + 3 * max_degree);
    uint32_t* hops = counter + 1;
    uint32_t* stats = hops + 1;
    INDEX_T* RESULT_POOL_INDEX = reinterpret_cast<INDEX_T*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    DISTANCE_T* RESULT_POOL_DISTANCE = reinterpret_cast<DISTANCE_T*>(RESULT_POOL_INDEX + K);
    uint32_t* result_pool_size = reinterpret_cast<uint32_t*>(RESULT_POOL_DISTANCE + K);
    uint32_t* result_pool_worst_idx = result_pool_size + 1;
    DISTANCE_T* result_pool_worst_dist = reinterpret_cast<DISTANCE_T*>(result_pool_worst_idx + 1);
    gpu_long_t* timing = nullptr;
#if GPU_DISTANCE_TIMING
    char* timing_base = reinterpret_cast<char*>(result_pool_worst_dist + 1);
    timing_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(timing_base) + alignof(gpu_long_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(gpu_long_t)) - 1));
    timing = reinterpret_cast<gpu_long_t*>(timing_base);
#endif

    __shared__ QueryFactors qf;

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t current_expanded;
    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = QUANTIZED_QUERY_BUFFER;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        counter[0] = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
        result_pool_size[0] = 0;
        result_pool_worst_idx[0] = 0;
        result_pool_worst_dist[0] = -FLT_MAX;
        if (GPU_DISTANCE_TIMING) {
            for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                timing[i] = 0;
            }
        }
    }

    // initialize the hash table to record expanded nodes
    if (HASH_TABLE_CONFIG) hashtable_init(HASH_TABLE, BITLEN);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }
    
    // load the top k list and candidate list with dummy initial values
    for (int i = tid; i < RESULT_BUFFER_SIZE; i += blockDim.x) {
        ALL_INDEX[i] = MAX_INDEX;
        ALL_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        RESULT_POOL_INDEX[i] = MAX_INDEX;
        RESULT_POOL_DISTANCE[i] = FLT_MAX;
    }

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        CANDIDATE_INDEX[0] = entry_point;
        CANDIDATE_DISTANCE[0] = warp_l2_distance(dim, query, d_qg_data + static_cast<size_t>(entry_point) * row_offset);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
    }
    __syncthreads();

    // query preparation, quantize the query
    gpu_long_t query_prep_start = 0;
    if (GPU_DISTANCE_TIMING && tid == 0) query_prep_start = gpu_globaltimer();
    query_prepare_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    if constexpr (GPU_RABITQ_FASTSCAN) {
        pack_lut_gpu(static_cast<int>(padded_dim), qf.quantized_query, qf.lut);
    }
    __syncthreads();
    if (GPU_DISTANCE_TIMING && tid == 0) {
        atomicAdd(&timing[GPU_TIMING_QUERY_PREP_CYCLES], gpu_globaltimer() - query_prep_start);
    }
    
    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hashtable_init(HASH_TABLE, BITLEN);
                __syncthreads();
                hashtable_restore_expand_only(HASH_TABLE, BITLEN, TOP_K_INDEX, INTERNAL_TOPK);
            }
            __syncthreads();
        }

        // use warp level sort to find the actual closest top K elements and update the topK list
        gpu_long_t sort_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) sort_start = gpu_globaltimer();
        topk_by_sort_candidates_and_merge<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
            CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, (iter == 0));
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_SORT_CYCLES], gpu_globaltimer() - sort_start);
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            printf("[DEBUG] iteration %d\n", iter + 1);
            printf("  beam state:\n");
            for (int i = 0; i < INTERNAL_TOPK; ++i) {
                const INDEX_T raw = TOP_K_INDEX[i];
                const bool expanded = (raw & 0x80000000u) != 0;
                printf("    idx=%d, node=%u, dist=%.6f, expanded=%s\n",
                       i,
                       raw & 0x7fffffff,
                       TOP_K_DISTANCE[i],
                       expanded ? "true" : "false");
            }
            printf("  candidate buffer:\n");
            for (int i = 0; i < CANDIDATE_BUFFER_SIZE; ++i) {
                printf("    idx=%d, node=%u, dist=%.6f\n",
                       i,
                       CANDIDATE_INDEX[i] & 0x7fffffff,
                       CANDIDATE_DISTANCE[i]);
            }
        }

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        gpu_long_t pick_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) pick_start = gpu_globaltimer();
        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, INTERNAL_TOPK, TOP_K_INDEX);
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_PICK_CYCLES], gpu_globaltimer() - pick_start);
        }

        if (!keep_expanding) {
            if (debug_mode && query_id == 0 && tid == 0) {
                printf("  expanding node: none\n");
            }
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;

        if (tid == 0) {
            current_expanded = hashtable_insert(HASH_TABLE, BITLEN, current_node);
        }
        __syncthreads();
        if (!current_expanded) {
            continue;
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            const INDEX_T parent_slot = PARENT_LIST[0];
            const DISTANCE_T parent_dist =
                (parent_slot < INTERNAL_TOPK) ? TOP_K_DISTANCE[parent_slot] : FLT_MAX;
            printf("  expanding node: %u\n", current_node);
            printf("  expanding slot: %u, dist=%.6f\n", parent_slot, parent_dist);
        }
        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* current_neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
        const float* current_factors = current_row + factor_offset;

        gpu_long_t stage_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) stage_start = gpu_globaltimer();
        // initialize the async copy of packed code and factors as long as the current expander is decided
        staged_copy_async_begin_or_sync(
            STAGED_CODE_BUFFER,
            current_codes,
            static_cast<int>(max_degree * code_words));
        staged_copy_async_begin_or_sync(
            STAGED_FACTOR_BUFFER,
            current_factors,
            3 * max_degree);

        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        DISTANCE_T exact_dist = warp_l2_distance(dim, query, current_row);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
            stats[GPU_SEARCH_STAT_DIST_CMPS]++;
        }
        gpu_long_t result_pool_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) result_pool_start = gpu_globaltimer();
        if (tid == 0) {
            result_pool_push_unique_unsorted(
                RESULT_POOL_INDEX,
                RESULT_POOL_DISTANCE,
                result_pool_size,
                result_pool_worst_idx,
                result_pool_worst_dist,
                K,
                current_node,
                exact_dist);
        }
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_RESULT_POOL_CYCLES], gpu_globaltimer() - result_pool_start);
        }
        gpu_long_t visited_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) visited_start = gpu_globaltimer();
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < max_degree) ? current_neighbors[i] : MAX_INDEX;
            if (child_id != MAX_INDEX && hashtable_contains(HASH_TABLE, BITLEN, child_id)) {
                child_id = MAX_INDEX;
            }
            CANDIDATE_INDEX[i] = child_id;
            CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0;
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_VISITED_CYCLES], gpu_globaltimer() - visited_start);
        }

        // wait for the async copy to finish before move on to computation
        staged_copy_async_finish_or_sync();
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_STAGE_CYCLES], gpu_globaltimer() - stage_start);
        }

        // all warps work together to compute candidate distance
        gpu_long_t est_region_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) est_region_start = gpu_globaltimer();
        if constexpr (GPU_RABITQ_FASTSCAN) {
            // each warp handle a batch of neighbors
            const int num_batches = (max_degree + WARP_SIZE - 1) / WARP_SIZE;

            for (int batch = warp_id; batch < num_batches; batch += WARPS_PER_BLOCK) {
                const int batch_begin = batch * WARP_SIZE;
                const int active_neighbors = min(WARP_SIZE, max_degree - batch_begin);
                const uint8_t* packed_codes_block = reinterpret_cast<const uint8_t*>(STAGED_CODE_BUFFER) + batch * static_cast<int>(padded_dim) * 4;
                const float* triple_x = STAGED_FACTOR_BUFFER + batch_begin;
                const float* factor_dq = STAGED_FACTOR_BUFFER + max_degree + batch_begin;
                const float* factor_vq = STAGED_FACTOR_BUFFER + 2 * max_degree + batch_begin;

                gpu_long_t est_start = 0;
                if (GPU_DISTANCE_TIMING && lane_id == 0) est_start = gpu_globaltimer();
                scan_all_neighbors_32_gpu(qf, packed_codes_block, triple_x, factor_dq, factor_vq,
                    exact_dist, static_cast<int>(padded_dim), CANDIDATE_DISTANCE + batch_begin, active_neighbors);
                if (lane_id < active_neighbors) {
                    const int candidate_idx = batch_begin + lane_id;
                    const bool invalid_candidate = (CANDIDATE_INDEX[candidate_idx] == MAX_INDEX);
                    const bool invalid_factor = (triple_x[lane_id] == FLT_MAX);
                    const bool invalid_distance = !isfinite(CANDIDATE_DISTANCE[candidate_idx]);
                    if (invalid_candidate || invalid_factor || invalid_distance) {
                        CANDIDATE_DISTANCE[candidate_idx] = FLT_MAX;
                    }
                }
                if (GPU_DISTANCE_TIMING && lane_id == 0) {
                    atomicAdd(&timing[GPU_TIMING_EST_CALLS], gpu_long_t(active_neighbors));
                    atomicAdd(&timing[GPU_TIMING_EST_CYCLES], gpu_globaltimer() - est_start);
                }
                if (lane_id == 0) {
                    uint32_t active_count = 0;
                    for (int i = 0; i < active_neighbors; ++i) {
                        if (CANDIDATE_INDEX[batch_begin + i] != MAX_INDEX) {
                            active_count++;
                        }
                    }
                    atomicAdd(&counter[0], active_count);
                    if (GPU_SEARCH_STATS) {
                        atomicAdd(&stats[GPU_SEARCH_STAT_DIST_EST], active_count);
                    }
                }
            }
        } else {
            for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
                // visited candidates are marked as invalid
                if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

                // if current node is new in hash table, compute and store its distance
                const uint64_t* packed_code = STAGED_CODE_BUFFER + i * code_words;
                const float* factors = STAGED_FACTOR_BUFFER;
                float triple_x = 0.0f;
                float factor_dq = 0.0f;
                float factor_vq = 0.0f;
                if (lane_id == 0) {
                    triple_x = factors[i];
                    factor_dq = factors[max_degree + i];
                    factor_vq = factors[2 * max_degree + i];
                }
                triple_x = __shfl_sync(FULL_MASK, triple_x, 0);
                factor_dq = __shfl_sync(FULL_MASK, factor_dq, 0);
                factor_vq = __shfl_sync(FULL_MASK, factor_vq, 0);
                
                gpu_long_t est_start = 0;
                if (GPU_DISTANCE_TIMING && lane_id == 0) est_start = gpu_globaltimer();
                DISTANCE_T est_dist = estimated_distance_gpu(
                    qf, packed_code, triple_x, factor_dq, factor_vq, exact_dist, padded_dim);
                if (lane_id == 0) {
                    atomicAdd(&counter[0], 1);
                    if (GPU_SEARCH_STATS) {
                        atomicAdd(&stats[GPU_SEARCH_STAT_DIST_EST], 1u);
                    }
                    if (GPU_DISTANCE_TIMING) {
                        atomicAdd(&timing[GPU_TIMING_EST_CALLS], gpu_long_t(1));
                        atomicAdd(&timing[GPU_TIMING_EST_CYCLES], gpu_globaltimer() - est_start);
                    }
                    CANDIDATE_DISTANCE[i] = est_dist;
                }
            }
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            int active_count = 0;
            for (int i = 0; i < CANDIDATE_BUFFER_SIZE; ++i) {
                if (CANDIDATE_INDEX[i] != MAX_INDEX) {
                    active_count++;
                }
            }
            if (active_count > 0) {
                atomicAdd(&timing[GPU_TIMING_EST_REGION_CANDIDATES], gpu_long_t(active_count));
                atomicAdd(&timing[GPU_TIMING_EST_REGION_CYCLES], gpu_globaltimer() - est_region_start);
            }
        }
    }

    // final sort before writeback
    gpu_long_t final_sort_start = 0;
    if (GPU_DISTANCE_TIMING && tid == 0) final_sort_start = gpu_globaltimer();
    topk_by_sort_candidates_and_merge<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, false);
    __syncthreads();
    if (GPU_DISTANCE_TIMING && tid == 0) {
        atomicAdd(&timing[GPU_TIMING_FINAL_SORT_CYCLES], gpu_globaltimer() - final_sort_start);
    }

    if (tid == 0) {

        if (debug_mode && query_id == 0) {
            printf("[DEBUG] final result pool\n");
            for (uint32_t i = 0; i < result_pool_size[0]; ++i) {
                printf("  idx=%u, node=%u, dist=%.6f\n",
                       i,
                       RESULT_POOL_INDEX[i],
                       RESULT_POOL_DISTANCE[i]);
            }
        }

        const uint32_t pool_size = result_pool_size[0];
        for (int i = 0; i < K; ++i) {
            if (i < static_cast<int>(pool_size)) {
                d_results[query_id * K + i] = RESULT_POOL_INDEX[i];
            } else {
                // if there's not enough value in result pool, fill the rest with other top K
                d_results[query_id * K + i] = TOP_K_INDEX[i] & 0x7fffffff;
            }
        }
        if (GPU_SEARCH_STATS) {
            stats[GPU_SEARCH_STAT_DIST_EST] = counter[0];
            if (d_block_stats != nullptr) {
                gpu_long_t* dst = d_block_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
                for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                    dst[i] = static_cast<gpu_long_t>(stats[i]);
                }
            }
        }
        if (GPU_DISTANCE_TIMING) {
            if (d_block_timing != nullptr) {
                gpu_long_t* dst = d_block_timing + static_cast<size_t>(query_id) * GPU_TIMING_COUNT;
                for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                    dst[i] = timing[i];
                }
            }
        }
    }

    (void)d_qg_signs;
    (void)npoints;
    (void)code_offset;
    (void)factor_offset;
}


// ═══════════════════════════════════════════════════════════════════════
// ── include/qg.hpp (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <string>
#include <stdexcept>
#include <sys/mman.h>

// [flattened] #include "../include/utils.h"
#ifndef __CUDACC__
// [flattened] #include "../include/fastscan_impl.hpp"
#endif

static constexpr bool FASTSCAN_ENABLED = true;
static constexpr bool HUGE_PAGE_ENABLED = false;

class QuantizationGraph {
public:
    struct QueryBuffers {
        float* rotated_query = nullptr;
        uint8_t* quantized_query = nullptr;
        uint8_t* lut = nullptr;
    };

    struct QueryFactors {
        float* rotated_query = nullptr;
        uint8_t* quantized_query = nullptr;
        uint8_t* lut = nullptr;
        float low_val = 0.0f;
        float high_val = 0.0f;
        float width = 0.0f;
        int32_t sum_q = 0;
    };

    struct NeighborFactors {
        const uint64_t* bits;
        float triple_x;
        float factor_dq;
        float factor_vq;
    };

    // SymQG codebook file layout: 
    // entry point id
    // for each node: raw vector | packed codes | factors | neighbor IDs
    QuantizationGraph(size_t num_node, size_t dim, int degree, const std::string& codebook_path)
        : num_nodes_(num_node), dim_(dim), degree_(degree) {

        init_layout(/*bins_mode=*/false);

        // load from the codebook
        std::ifstream fin(codebook_path, std::ios::binary);
        if (!fin) throw std::runtime_error("Cannot open codebook file: " + codebook_path);

        // load the entry point
        fin.read(reinterpret_cast<char*>(&entry_point_), sizeof(uint32_t));

        // load the data table
        size_t total_floats = num_nodes_ * row_offset_;
        size_t total_bytes = total_floats * sizeof(float);

        if (HUGE_PAGE_ENABLED) {
            constexpr size_t kHugePageAlign = 1 << 22;  // 4 MB
            size_t alloc_bytes = (total_bytes + kHugePageAlign - 1) & ~(kHugePageAlign - 1);

            data_ = static_cast<float*>(std::aligned_alloc(kHugePageAlign, alloc_bytes));
            madvise(data_, alloc_bytes, MADV_HUGEPAGE);
        } else {
            size_t alloc_bytes = (total_bytes + 64 - 1) & ~(64 - 1);
            data_ = static_cast<float*>(std::aligned_alloc(64, alloc_bytes));
        }
        
        fin.read(reinterpret_cast<char*>(data_), total_floats * sizeof(float));

        // load the sign table
        signs_ptr_ = static_cast<float*>(std::aligned_alloc(64, padded_dim_ * sizeof(float)));
        fin.read(reinterpret_cast<char*>(signs_ptr_), padded_dim_ * sizeof(float));

        fin.close();

        // print to ensure loaded successfully
        printf("Loaded codebook: %zu nodes, dim=%zu, degree=%d, padded_dim=%zu, row_offset=%zu floats (entry_point=%u)\n",
               num_nodes_, dim_, degree_, padded_dim_, row_offset_, entry_point_);
    }

    QuantizationGraph(size_t num_node, size_t dim, int degree,
                      const std::string& graph_file,
                      const std::string& codes_file,
                      const std::string& signs_file,
                      const float* base_vectors)
        : num_nodes_(num_node), dim_(dim), degree_(degree),
          bins_mode_(true), base_vectors_(base_vectors) {

        init_layout(/*bins_mode=*/true);

        size_t graph_nodes = 0, graph_degree = 0;
        vidType* graph_tmp = read_bin<vidType>(graph_file.c_str(), graph_nodes, graph_degree);
        if (graph_nodes != num_nodes_ || static_cast<int>(graph_degree) != degree_) {
            release_bin_buffer(graph_tmp, graph_degree);
            throw std::runtime_error("Graph bin shape mismatch");
        }
        graph_edges_ = allocate_and_copy(graph_tmp, graph_nodes * graph_degree, graph_degree);
        release_bin_buffer(graph_tmp, graph_degree);

        size_t code_nodes = 0, code_width = 0;
        uint64_t* codes_tmp = read_bin<uint64_t>(codes_file.c_str(), code_nodes, code_width);
        if (code_nodes != num_nodes_ || code_width != codes_row_words_) {
            release_bin_buffer(codes_tmp, code_width);
            throw std::runtime_error("Packed-code bin shape mismatch");
        }
        codes_and_factors_ = allocate_and_copy(codes_tmp, code_nodes * code_width, code_width);
        release_bin_buffer(codes_tmp, code_width);

        size_t sign_rows = 0, sign_dim = 0;
        float* signs_tmp = read_bin<float>(signs_file.c_str(), sign_rows, sign_dim);
        if (!((sign_rows == 1 && sign_dim == padded_dim_) ||
              (sign_rows == padded_dim_ && sign_dim == 1))) {
            release_bin_buffer(signs_tmp, sign_dim);
            throw std::runtime_error("Sign-table bin shape mismatch");
        }
        signs_ptr_ = allocate_and_copy(signs_tmp, padded_dim_, padded_dim_);
        release_bin_buffer(signs_tmp, sign_dim);

        entry_point_ = UINT32_MAX;

        printf("Loaded bins-mode QG: %zu nodes, dim=%zu, degree=%d, padded_dim=%zu, codes_row_words=%zu u64\n",
               num_nodes_, dim_, degree_, padded_dim_, codes_row_words_);
    }

    ~QuantizationGraph() {
        if (bins_mode_) {
            std::free(codes_and_factors_);
            std::free(graph_edges_);
        } else {
            std::free(data_);
        }
        std::free(signs_ptr_);
    }

    #ifndef __CUDACC__
    QueryFactors query_prepare(const float* query_raw, const QueryBuffers& scratch) const {
        QueryFactors qf;
        qf.rotated_query = scratch.rotated_query;
        qf.quantized_query = scratch.quantized_query;
        qf.lut = scratch.lut;

        assert(qf.rotated_query != nullptr);
        assert(qf.quantized_query != nullptr);
        if constexpr (FASTSCAN_ENABLED) {
            assert(qf.lut != nullptr);
        }

        // rotate the query
        rotate_vector(query_raw, qf.rotated_query, signs_ptr_, dim_, padded_dim_);

        // data range: find the lowest and highest dimension in the rotated query
        for (size_t i = 0; i < padded_dim_; ++i) {
            float tmp = qf.rotated_query[i];
            qf.low_val = std::min(qf.low_val, tmp);
            qf.high_val = std::max(qf.high_val, tmp);
        }
        qf.width = (qf.high_val - qf.low_val) / ((1 << QG_BQUERY) - 1);

        // quantize the query 
        float inv_width = 1.0F / qf.width;
        int32_t sum = 0;
        for (size_t i = 0; i < padded_dim_; ++i) {
            uint8_t cur = static_cast<uint8_t>(std::lround(((qf.rotated_query[i] - qf.low_val) * inv_width) + 0.5));
            qf.quantized_query[i] = cur;
            sum += cur;
        }
        qf.sum_q = sum;

        if constexpr (FASTSCAN_ENABLED) {
            pack_lut_impl(padded_dim_, qf.quantized_query, qf.lut);
        }

        return qf;
    }

    NeighborFactors nbr_filter(uint32_t node_idx, int neighbor_rank) const {
        const uint64_t* codes = get_binary_codes(node_idx);
        const uint64_t* bits = codes + neighbor_rank * padded_dim_ / 64;
        const float* factors = get_factor(node_idx);
        return { bits, factors[neighbor_rank], factors[degree_ + neighbor_rank], factors[2 * degree_ + neighbor_rank] };
    }

    float estimated_distance(const QueryFactors& qf, const NeighborFactors& nf, float exact_dist) const {
        if (nf.triple_x == FLT_MAX) return 0.0f;
        
        int32_t raw_sum = 0;
        const uint8_t* q = qf.quantized_query;
        const uint64_t* bits = nf.bits;

    #if defined(__AVX512BW__) && defined(__AVX512VL__)
        // Process 64 query bytes + 1 bitcode word per iteration
        __m512i acc = _mm512_setzero_si512();

        for (size_t w = 0; w < bitcode_words_; ++w) {
            // Load 64 quantized query bytes
            __m512i qvec = _mm512_load_si512(reinterpret_cast<const __m512i*>(q + w * 64));

            // Broadcast the 64-bit mask and convert to byte mask
            __mmask64 mask = static_cast<__mmask64>(bits[w]);

            // Zero out bytes where bit is not set, then accumulate
            // _mm512_maskz_mov_epi8 zeros lanes where mask bit is 0
            __m512i masked = _mm512_maskz_mov_epi8(mask, qvec);

            // Horizontal sum of bytes via sad (sum of absolute differences against zero)
            // sad treats inputs as unsigned bytes, sums groups of 8 into uint16 in 64-bit lanes
            acc = _mm512_add_epi64(acc, _mm512_sad_epu8(masked, _mm512_setzero_si512()));
        }

        // Horizontal sum of the 8 x 64-bit accumulators
        raw_sum = static_cast<int32_t>(_mm512_reduce_add_epi64(acc));

    #elif defined(__AVX2__)
        __m256i acc_lo = _mm256_setzero_si256();
        __m256i acc_hi = _mm256_setzero_si256();

        for (size_t w = 0; w < bitcode_words_; ++w) {
            uint64_t mask = bits[w];
            uint32_t mask_lo = static_cast<uint32_t>(mask);
            uint32_t mask_hi = static_cast<uint32_t>(mask >> 32);

            const uint8_t* base = q + w * 64;

            // Expand each bit of the 32-bit mask into a full byte (0x00 or 0xFF)
            // _mm256_movemask_epi8 is the inverse; we need the forward direction.
            // Use a lookup: broadcast mask, AND with bit selectors, compare to produce byte mask.

            // --- Lower 32 bytes ---
            __m256i qvec_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(base));
            __m256i bitmask_lo = expand_bits_to_bytes_avx2(mask_lo);
            __m256i masked_lo = _mm256_and_si256(qvec_lo, bitmask_lo);
            acc_lo = _mm256_add_epi64(acc_lo, _mm256_sad_epu8(masked_lo, _mm256_setzero_si256()));

            // --- Upper 32 bytes ---
            __m256i qvec_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(base + 32));
            __m256i bitmask_hi = expand_bits_to_bytes_avx2(mask_hi);
            __m256i masked_hi = _mm256_and_si256(qvec_hi, bitmask_hi);
            acc_hi = _mm256_add_epi64(acc_hi, _mm256_sad_epu8(masked_hi, _mm256_setzero_si256()));
        }

        __m256i total = _mm256_add_epi64(acc_lo, acc_hi);
        // Horizontal sum of 4 x 64-bit lanes
        __m128i lo128 = _mm256_castsi256_si128(total);
        __m128i hi128 = _mm256_extracti128_si256(total, 1);
        __m128i sum128 = _mm_add_epi64(lo128, hi128);
        raw_sum = static_cast<int32_t>(_mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1));

    #else
        for (size_t d = 0; d < padded_dim_; ++d) {
            if ((bits[d / 64] >> (d % 64)) & 1ULL)
                raw_sum += static_cast<int32_t>(q[d]);
        }
    #endif

        float result_float = static_cast<float>((raw_sum << 1) - qf.sum_q);
        float est_dist = nf.triple_x + exact_dist;
        est_dist += (nf.factor_dq * qf.width * result_float);
        est_dist += (nf.factor_vq * qf.low_val);
        return est_dist;
    }

    void scan_all_neighbors(
        uint32_t node_idx,
        float exact_dist,
        float* appro_dist,
        const QueryFactors& qf,
        uint16_t* result_raw,
        float* result_float
    ) const {
        size_t num_blocks   = static_cast<size_t>(degree_) / kBatchSize;

        // Access packed codes through the mode-aware accessor so both layouts work.
        const uint8_t* packed_code = reinterpret_cast<const uint8_t*>(get_binary_codes(node_idx));
        const float* factors  = get_factor(node_idx);
        const float* triple_x = factors;
        const float* fac_dq   = triple_x + degree_;
        const float* fac_vq   = fac_dq   + degree_;

        // Step 1: accumulate raw uint16 sums for each block of 32 neighbors
        for (size_t b = 0; b < num_blocks; ++b) {
            accumulate_impl(padded_dim_, packed_code + b * padded_dim_ * 4, qf.lut, &result_raw[b * kBatchSize]);
        }

        // Step 2: convert uint16 raw sums to float: result = (raw * 2) - sum_q
    #if defined(__AVX512F__)
        {
            const __m512i qq = _mm512_set1_epi32(qf.sum_q);
            for (size_t i = 0; i < static_cast<size_t>(degree_); i += 32) {
                __m256i i16a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result_raw[i]));
                __m256i i16b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result_raw[i + 16]));
                __m512i i32a = _mm512_cvtepi16_epi32(i16a);
                __m512i i32b = _mm512_cvtepi16_epi32(i16b);
                i32a = _mm512_sub_epi32(_mm512_slli_epi32(i32a, 1), qq);
                i32b = _mm512_sub_epi32(_mm512_slli_epi32(i32b, 1), qq);
                _mm512_storeu_ps(&result_float[i],      _mm512_cvtepi32_ps(i32a));
                _mm512_storeu_ps(&result_float[i + 16], _mm512_cvtepi32_ps(i32b));
            }
        }
    #else
        for (size_t i = 0; i < degree_; ++i)
            result_float[i] = static_cast<float>((static_cast<int>(result_raw[i]) << 1) - qf.sum_q);
    #endif

        // Step 3: compute final approximate distances
        appro_dist_impl(degree_, exact_dist, qf.width, qf.low_val,
            result_float, triple_x, fac_dq, fac_vq, appro_dist);
    }

    void scan_all_neighbors_32(
        uint32_t node_idx,
        float exact_dist,
        float* appro_dist,
        const QueryFactors& qf,
        uint32_t* result_raw,
        float* result_float
    ) const {
        size_t num_blocks   = static_cast<size_t>(degree_) / kBatchSize;

        // Access packed codes through the mode-aware accessor so both layouts work.
        const uint8_t* packed_code = reinterpret_cast<const uint8_t*>(get_binary_codes(node_idx));
        const float* factors  = get_factor(node_idx);
        const float* triple_x = factors;
        const float* fac_dq   = triple_x + degree_;
        const float* fac_vq   = fac_dq   + degree_;

        // Step 1: accumulate raw uint32 sums for each block of 32 neighbors
        for (size_t b = 0; b < num_blocks; ++b) {
            accumulate_impl_32(padded_dim_, packed_code + b * padded_dim_ * 4, qf.lut, &result_raw[b * kBatchSize]);
        }

        // Step 2: convert uint32 raw sums to float: result = (raw * 2) - sum_q
    #if defined(__AVX512F__)
        {
            const __m512i qq = _mm512_set1_epi32(qf.sum_q);
            for (size_t i = 0; i < static_cast<size_t>(degree_); i += 16) {
                __m512i i32 = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(&result_raw[i]));
                i32 = _mm512_sub_epi32(_mm512_slli_epi32(i32, 1), qq);
                __m512 f32 = _mm512_cvtepi32_ps(i32);
                _mm512_storeu_ps(&result_float[i], f32);
            }
        }
    #else
        for (size_t i = 0; i < degree_; ++i)
            result_float[i] = static_cast<float>((static_cast<uint32_t>(result_raw[i]) << 1) - qf.sum_q);
    #endif

        // Step 3: compute final approximate distances
        appro_dist_impl(degree_, exact_dist, qf.width, qf.low_val,
            result_float, triple_x, fac_dq, fac_vq, appro_dist);
    }

    void prefetch_nbr(uint32_t node_idx) const {
        auto prefetch_span = [](const char* base, size_t bytes) {
            constexpr size_t kCacheLineBytes = 64;
            constexpr size_t kMaxPrefetchLines = 20;

            size_t num_lines = (bytes + kCacheLineBytes - 1) / kCacheLineBytes;
            while (num_lines > 0) {
                size_t chunk_lines = std::min(num_lines, kMaxPrefetchLines);
                mem_prefetch_l2(base, chunk_lines);
                base += chunk_lines * kCacheLineBytes;
                num_lines -= chunk_lines;
            }
        };

        if (bins_mode_) {
            const char* raw_base = reinterpret_cast<const char*>(get_row(node_idx));
            const char* codes_base = reinterpret_cast<const char*>(get_binary_codes(node_idx));
            const char* neighbors_base = reinterpret_cast<const char*>(get_neighbors(node_idx));

            prefetch_span(raw_base, dim_ * sizeof(float));
            prefetch_span(codes_base, codes_row_words_ * sizeof(uint64_t));
            prefetch_span(neighbors_base, static_cast<size_t>(degree_) * sizeof(uint32_t));
            return;
        }

        const char* row_base = reinterpret_cast<const char*>(get_row(node_idx));
        prefetch_span(row_base, row_offset_ * sizeof(float));
    }
    #endif

    inline const float* get_row(uint32_t node_idx) const {
        return bins_mode_
            ? base_vectors_ + static_cast<size_t>(node_idx) * dim_
            : data_ + static_cast<size_t>(node_idx) * row_offset_;
    }

    inline const uint64_t* get_binary_codes(uint32_t node_idx) const {
        return bins_mode_
            ? codes_and_factors_ + static_cast<size_t>(node_idx) * codes_row_words_
            : reinterpret_cast<const uint64_t*>(get_row(node_idx) + code_offset_);
    }

    inline const float* get_factor(uint32_t node_idx) const {
        return bins_mode_
            ? reinterpret_cast<const float*>(codes_and_factors_ + static_cast<size_t>(node_idx) * codes_row_words_ + bitcode_words_ * degree_)
            : get_row(node_idx) + factor_offset_;
    }
    
    inline const uint32_t* get_neighbors(uint32_t node_idx) const {
        return bins_mode_
            ? graph_edges_ + static_cast<size_t>(node_idx) * degree_
            : reinterpret_cast<const uint32_t*>(get_row(node_idx) + neighbor_offset_);
    }

    inline const float* get_data_ptr() const {
        return bins_mode_ ? base_vectors_ : data_;
    }
    inline const float* get_signs_ptr() const { return signs_ptr_; }
    inline size_t get_data_bytes() const {
        return bins_mode_
            ? num_nodes_ * dim_ * sizeof(float)
            : num_nodes_ * row_offset_ * sizeof(float);
    }
    inline size_t get_signs_bytes() const { return padded_dim_ * sizeof(float); }
    inline bool is_bins_mode() const { return bins_mode_; }

    inline size_t get_num_nodes() const { return num_nodes_; }
    inline size_t get_dim() const { return dim_; }
    inline int get_padded_dim() const { return padded_dim_; }
    inline int get_degree() const { return degree_; }

    inline size_t get_code_offset() const { return code_offset_; }
    inline size_t get_factor_offset() const { return factor_offset_; }
    inline size_t get_neighbor_offset() const { return neighbor_offset_; }
    inline size_t get_row_offset() const { return row_offset_; }
    inline size_t get_codes_row_words() const { return codes_row_words_; }

    inline vidType get_entry_point() const { return entry_point_; }
    inline void set_entry_point(vidType entry_point) { entry_point_ = entry_point; }

    void gpu_search(int nq, const float* queries, int K, float* result_vec,
                    vidType* result_idx, int beam_sz, double& elapsed,
                    GpuRunStats* gpu_stats = nullptr,
                    bool debug_mode = false);

private:
    void init_layout(bool bins_mode) {
        padded_dim_ = 1ULL << static_cast<size_t>(std::ceil(std::log2(dim_)));
        bitcode_words_ = padded_dim_ / 64;
        code_offset_ = bins_mode ? 0 : dim_;
        factor_offset_ = code_offset_ + bitcode_words_ * 2 * degree_;
        neighbor_offset_ = factor_offset_ + 3 * degree_;
        row_offset_ = neighbor_offset_ + degree_;
        codes_row_offset_ = factor_offset_ + 3 * degree_;
        codes_row_words_ = bitcode_words_ * degree_ + ((3 * degree_ * sizeof(float) + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    }

    template <typename T>
    static T* allocate_and_copy(const T* src, size_t count, size_t row_dim) {
        size_t row_bytes = row_dim * sizeof(T);
        T* dst = nullptr;
        if (row_bytes % 64 == 0) {
            dst = static_cast<T*>(std::aligned_alloc(64, sizeof(T) * count));
        } else if (row_bytes % 32 == 0) {
            dst = static_cast<T*>(std::aligned_alloc(32, sizeof(T) * count));
        } else {
            size_t bytes = sizeof(T) * count;
            size_t alloc_bytes = (bytes + 63) & ~size_t(63);
            dst = static_cast<T*>(std::aligned_alloc(64, alloc_bytes));
        }
        if (dst == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(dst, src, sizeof(T) * count);
        return dst;
    }

    template <typename T>
    static void release_bin_buffer(T* ptr, size_t row_dim) {
        if (ptr == nullptr) return;
        size_t row_bytes = row_dim * sizeof(T);
        if (row_bytes % 64 == 0 || row_bytes % 32 == 0) {
            std::free(ptr);
        } else {
            delete[] ptr;
        }
    }

    size_t num_nodes_;
    size_t dim_, padded_dim_;
    int degree_;
    size_t bitcode_words_;
    vidType entry_point_;
    bool bins_mode_ = false;

    // QG-style offsets (in units of float)
    size_t code_offset_;
    size_t factor_offset_;
    size_t neighbor_offset_;
    size_t row_offset_;
    size_t codes_row_offset_;
    size_t codes_row_words_;

    float* data_ = nullptr;
    const float* base_vectors_ = nullptr;
    uint64_t* codes_and_factors_ = nullptr;
    vidType* graph_edges_ = nullptr;
    float* signs_ptr_ = nullptr;
};


// ═══════════════════════════════════════════════════════════════════════
// ── Stubs for disabled tensor code paths ──
// ═══════════════════════════════════════════════════════════════════════


// search_tensor_bitonic.cuh does not exist on sophia branch (USE_TENSOR_BITONIC=0)
template <typename T>
__global__ void BeamSearchTensor(int, int, int, size_t, const T*, const T*,
    vid_t*, gpu_long_t*, gpu_long_t*, vid_t*, vid_t*, uint32_t*, GraphGPU<vid_t>, bool) {}

static __host__ inline uint32_t calculate_shared_mem_size_tensor_bitonic(int, int, int, int, int) { return 0; }

// search_tensor_rabitq.cuh: disabled (USE_TENSOR_RABITQ=0), has undefined symbols
__global__ void TensorCoreQuantizedBeamSearch(int, int, int, int, size_t, const float*,
    const float*, const float*, vidType*, gpu_long_t*, gpu_long_t*, vidType, size_t,
    size_t, size_t, size_t, bool) {}

static __host__ inline uint32_t calculate_shared_mem_size_tensor(int, int, int, int, int, int) { return 0; }

// ═══════════════════════════════════════════════════════════════════════
// ── src/beam_gpu_base.cu (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

#undef USE_TENSOR_BITONIC
#define USE_TENSOR_BITONIC 0
//#include <cuda_profiler_api.h>
//#define USE_RADIX // Use bitonic if not defined

// [flattened] #include "include/index.hpp"
#ifdef USE_RADIX
// [flattened] #include "include/search_radix.cuh"
#else
// [flattened] #include "include/search_bitonic.cuh"
// [flattened] #include "include/search_tensor_bitonic.cuh"
#endif

#include <chrono>
#include <array>
#include <cfloat>
#include <vector>

#ifndef USE_TENSOR_BITONIC
#define USE_TENSOR_BITONIC 0
#endif

static vid_t compute_gpu_base_entry_point(const float* data, size_t npoints, size_t dim) {
  std::vector<float> centroid(dim, 0.0f);
  const float inv_npoints = 1.0f / static_cast<float>(npoints);
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * dim;
    for (size_t d = 0; d < dim; ++d) {
      centroid[d] += row[d] * inv_npoints;
    }
  }

  vid_t best = 0;
  float best_dist = FLT_MAX;
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * dim;
    float dist = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      const float diff = row[d] - centroid[d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<vid_t>(i);
    }
  }
  return best;
}

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T *result_vec, vid_t *result_idx, int beam_sz, double& elapsed,
                           GpuRunStats* gpu_stats, bool debug_mode) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  printf("Beam search on GPU%s: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d, internal_topk=%d\n",
         USE_TENSOR_BITONIC ? " (tensor bitonic)" : "", K, nq, dim, npoints, beam_size, INTERNAL_TOPK);

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = GPU_DISTANCE_TIMING ? 1 : nq;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
  int numBlocksPerSM = 0;
  int blockSize = 256;   // threads per block

  // allocate device memory
  T* d_queries, *d_data, *h_data = this->get_data_ptr();
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_queries, nq * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, nq * dim * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, npoints * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, npoints * dim * sizeof(T), cudaMemcpyHostToDevice));

  vid_t *h_results = &result_idx[0];
  vid_t *d_results;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, nq * K * sizeof(vid_t)));

  // counters
  gpu_long_t *d_total_stats = nullptr;
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_stats, GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_total_stats, total_stats.data(), GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  }
  gpu_long_t *d_total_timing;
  std::array<gpu_long_t, GPU_TIMING_COUNT> total_timing{};
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_timing, GPU_TIMING_COUNT * sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_timing, total_timing.data(), GPU_TIMING_COUNT * sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(npoints, max_degree, this->edges.data()); // GPU graph 

  vid_t *starting_points = new vid_t[beam_size];
  vid_t *d_starting_points;
  const vid_t entry_point = compute_gpu_base_entry_point(h_data, npoints, dim);
  printf("GPU base entry point: %u\n", entry_point);
  for (int i = 0; i < beam_size; i++) {
    starting_points[i] = (i == 0) ? entry_point : MAX_INDEX;
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_starting_points, beam_size * sizeof(vid_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, beam_size * sizeof(vid_t), cudaMemcpyHostToDevice));

#ifdef USE_RADIX
  //vid_t *d_filters = nullptr;
  //printf("filter size = %d bytes\n", num_blocks * HASH_FILTER_SIZE * sizeof(vid_t));
  //CUDA_SAFE_CALL(cudaMalloc((void **)&d_filters, num_blocks * HASH_FILTER_SIZE * sizeof(vid_t)));
  BeamSearch<T><<<num_blocks, num_threads>>>(K, nq, dim, npoints, d_queries, d_data,
                                             d_results, d_total_stats, d_total_timing, d_starting_points, d_total_hops, gg);
#else
  uint32_t shm_size = USE_TENSOR_BITONIC
      ? calculate_shared_mem_size_tensor_bitonic(dim, INTERNAL_TOPK, max_degree, SEARCH_WIDTH, BITLEN)
      : calculate_shared_mem_size(dim, INTERNAL_TOPK, max_degree, SEARCH_WIDTH, BITLEN);
  vid_t *d_seed_map = nullptr;
  if (USE_SEED) {
    size_t ghost_scale_factor = 100;
    size_t ghost_num = npoints / ghost_scale_factor;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_seed_map, ghost_num * sizeof(vid_t)));
    init_identity_map<<<(ghost_num-1)/num_threads+1, num_threads>>>(ghost_num, d_seed_map);
  }
  uint32_t *d_sign_bit = nullptr;
  if (PRUNE_CONFIG) {
    auto sign_bit = load_sign_bit("sign_bit.bin");
    size_t sign_bit_size = sign_bit.size();
    uint32_t *h_sign_bit = sign_bit.data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sign_bit, sign_bit_size * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit, sign_bit_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  cudaError_t err = cudaSuccess;
  if (USE_TENSOR_BITONIC) {
    CUDA_SAFE_CALL(cudaFuncSetAttribute(
        BeamSearchTensor<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shm_size)));
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, BeamSearchTensor<T>, blockSize, shm_size);
  } else {
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, BeamSearch<T>, blockSize, shm_size);
  }
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  // Execute the search
  printf("\nStarting GPU %sbeam search...\n", USE_TENSOR_BITONIC ? "tensor " : "");
  auto start = std::chrono::high_resolution_clock::now();
  if (USE_TENSOR_BITONIC) {
    BeamSearchTensor<T><<<num_blocks, num_threads, shm_size>>>(
        K, nq, dim, npoints, d_queries, d_data, d_results,
        d_total_stats, d_total_timing, d_starting_points, d_seed_map, d_sign_bit, gg, debug_mode);
  } else {
    BeamSearch<T><<<num_blocks, num_threads, shm_size>>>(K, nq, dim, npoints, d_queries, d_data, d_results, 
                                                         d_total_stats, d_total_timing, d_starting_points, d_seed_map, d_sign_bit, gg, debug_mode);
  }
  CUDA_SAFE_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, nq * K * sizeof(int), cudaMemcpyDeviceToHost));
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMemcpy(total_stats.data(), d_total_stats, GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  }
  CUDA_SAFE_CALL(cudaMemcpy(total_timing.data(), d_total_timing, GPU_TIMING_COUNT * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaFree(d_total_stats));
  }
  CUDA_SAFE_CALL(cudaFree(d_total_timing));
  if (GPU_SEARCH_STATS) {
    std::cout << "total # exact distance computations: " << total_stats[GPU_SEARCH_STAT_DIST_CMPS] << "\n";
    std::cout << "average # exact distance computations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]) / nq << "\n";
    std::cout << "total # distance estimations: " << total_stats[GPU_SEARCH_STAT_DIST_EST] << "\n";
    std::cout << "average # distance estimations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_EST]) / nq << "\n";
    std::cout << "total # expansion steps: " << total_stats[GPU_SEARCH_STAT_ITERATIONS] << "\n";
    std::cout << "average # expansion steps: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_ITERATIONS]) / nq << "\n";
  }
  if (GPU_DISTANCE_TIMING) {
    const double avg_l2_cycles =
        total_timing[GPU_TIMING_L2_CALLS] ? static_cast<double>(total_timing[GPU_TIMING_L2_CYCLES]) / total_timing[GPU_TIMING_L2_CALLS] : 0.0;
    std::cout << "total # l2 timing calls: " << total_timing[GPU_TIMING_L2_CALLS] << "\n";
    std::cout << "average cycles per warp_l2_distance: " << avg_l2_cycles << "\n";
  }

  if (gpu_stats != nullptr) {
    gpu_stats->common.total_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_ds = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_EST]);
    gpu_stats->common.total_extra_dc = 0;
    gpu_stats->common.total_full_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_iter = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_ITERATIONS]);
    gpu_stats->common.total_pruned = 0;
    gpu_stats->common.time_search_total = elapsed;
    gpu_stats->gpu_l2_calls = static_cast<int64_t>(total_timing[GPU_TIMING_L2_CALLS]);
    gpu_stats->gpu_l2_cycles = static_cast<int64_t>(total_timing[GPU_TIMING_L2_CYCLES]);
    gpu_stats->gpu_est_calls = static_cast<int64_t>(total_timing[GPU_TIMING_EST_CALLS]);
    gpu_stats->gpu_est_cycles = static_cast<int64_t>(total_timing[GPU_TIMING_EST_CYCLES]);
  }
  (void)debug_mode;
}

template class IndexGraph<float>;


// ═══════════════════════════════════════════════════════════════════════
// ── src/beam_gpu_rabitq.cu (inlined) ──
// ═══════════════════════════════════════════════════════════════════════

#undef USE_TENSOR_RABITQ
#define USE_TENSOR_RABITQ 0
//#include <cuda_profiler_api.h>

// [flattened] #include "include/qg.hpp"
// [flattened] #include "include/search_rabitq.cuh"
// [flattened] #include "include/search_tensor_rabitq.cuh"

#include <chrono>
#include <cstddef>
#include <array>
#include <cfloat>
#include <vector>

#ifndef USE_TENSOR_RABITQ
#define USE_TENSOR_RABITQ 0
#endif

static vidType compute_gpu_rabitq_entry_point(const float* data, size_t npoints,
                                              size_t dim, size_t row_offset) {
  std::vector<float> centroid(dim, 0.0f);
  const float inv_npoints = 1.0f / static_cast<float>(npoints);
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * row_offset;
    for (size_t d = 0; d < dim; ++d) {
      centroid[d] += row[d] * inv_npoints;
    }
  }

  vidType best = 0;
  float best_dist = FLT_MAX;
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * row_offset;
    float dist = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      const float diff = row[d] - centroid[d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<vidType>(i);
    }
  }
  return best;
}

void QuantizationGraph::gpu_search(int nq, const float* queries, int K, float* result_vec,
                                   vidType* result_idx, int beam_sz, double& elapsed,
                                   GpuRunStats* gpu_stats, bool debug_mode) {
  if (this->is_bins_mode()) {
    throw std::runtime_error("gpu_search currently supports only codebook-mode QuantizationGraph");
  }

  auto dim = this->dim_;
  auto npoints = this->num_nodes_;
  auto max_degree = this->degree_;
  auto beam_size = beam_sz;
  const int active_nq = GPU_DISTANCE_TIMING ? 1 : nq;
  printf("\nBeam search on GPU: K=%d, nq=%d, dim=%ld, npoints=%ld, beam_size=%d, internal_topk=%d\n", K, active_nq, dim, npoints, beam_size, INTERNAL_TOPK);
  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = active_nq;
  std::cout << "\nnum_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  int numBlocksPerSM = 0;
  int blockSize = 256;

  float* d_queries = nullptr;
  float* d_qg_data = nullptr;
  float* d_qg_signs = nullptr;
  size_t free_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));

  const size_t query_bytes = static_cast<size_t>(active_nq) * dim * sizeof(float);
  const size_t results_bytes = static_cast<size_t>(active_nq) * K * sizeof(vidType);
  const size_t qg_data_bytes = this->get_data_bytes();
  const size_t qg_signs_bytes = this->get_signs_bytes();
  const size_t required_bytes = query_bytes + results_bytes + qg_data_bytes + qg_signs_bytes;

  constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
  printf("GPU memory: free=%.2f GiB, total=%.2f GiB, required=%.2f GiB (qg_data=%.2f GiB, qg_signs=%.4f GiB)\n",
         static_cast<double>(free_mem_bytes) / kBytesPerGiB,
         static_cast<double>(total_mem_bytes) / kBytesPerGiB,
         static_cast<double>(required_bytes) / kBytesPerGiB,
         static_cast<double>(qg_data_bytes) / kBytesPerGiB,
         static_cast<double>(qg_signs_bytes) / kBytesPerGiB);
  if (required_bytes > free_mem_bytes) {
    throw std::runtime_error("QG upload exceeds currently available GPU memory");
  }

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_queries, static_cast<size_t>(active_nq) * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, static_cast<size_t>(active_nq) * dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_data, qg_data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_data, this->get_data_ptr(), qg_data_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_signs, qg_signs_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_signs, this->get_signs_ptr(), qg_signs_bytes, cudaMemcpyHostToDevice));

  vidType* d_results = nullptr;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, static_cast<size_t>(active_nq) * K * sizeof(vidType)));

  gpu_long_t* d_total_stats = nullptr;
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_stats, num_blocks * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  }
  gpu_long_t* d_total_timing = nullptr;
  std::array<gpu_long_t, GPU_TIMING_COUNT> total_timing{};
  if (GPU_DISTANCE_TIMING) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_timing, num_blocks * GPU_TIMING_COUNT * sizeof(gpu_long_t)));
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  vidType entry_point =
      compute_gpu_rabitq_entry_point(this->get_data_ptr(), npoints, dim, this->get_row_offset());
  printf("GPU rabitq entry point: %u\n", entry_point);

  uint32_t shm_size = USE_TENSOR_RABITQ
      ? calculate_shared_mem_size_tensor(static_cast<int>(dim), INTERNAL_TOPK, max_degree, SEARCH_WIDTH, BITLEN, K)
      : calculate_shared_mem_size_rabitq(static_cast<int>(dim), INTERNAL_TOPK, max_degree, SEARCH_WIDTH, BITLEN);
  if (USE_TENSOR_RABITQ) {
    CUDA_SAFE_CALL(cudaFuncSetAttribute(
        TensorCoreQuantizedBeamSearch,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shm_size)));
  } else {
    CUDA_SAFE_CALL(cudaFuncSetAttribute(
        QuantizedBeamSearch,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shm_size)));
  }

  cudaError_t err = USE_TENSOR_RABITQ
      ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
            TensorCoreQuantizedBeamSearch, blockSize, shm_size)
      : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
            QuantizedBeamSearch, blockSize, shm_size);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  printf("\nStarting GPU rabitq search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  if (USE_TENSOR_RABITQ) {
    TensorCoreQuantizedBeamSearch<<<num_blocks, num_threads, shm_size>>>(
        K, active_nq, static_cast<int>(dim), max_degree, npoints, d_queries, d_qg_data, d_qg_signs, d_results,
        d_total_stats, d_total_timing, entry_point, this->get_row_offset(), this->get_neighbor_offset(),
        this->get_code_offset(), this->get_factor_offset(), debug_mode);
  } else {
    QuantizedBeamSearch<<<num_blocks, num_threads, shm_size>>>(
        K, active_nq, static_cast<int>(dim), max_degree, npoints, d_queries, d_qg_data, d_qg_signs, d_results,
        d_total_stats, d_total_timing, entry_point, this->get_row_offset(), this->get_neighbor_offset(),
        this->get_code_offset(), this->get_factor_offset(), debug_mode);
  }
  CUDA_SAFE_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, static_cast<size_t>(active_nq) * K * sizeof(int), cudaMemcpyDeviceToHost));
  if (GPU_SEARCH_STATS) {
    std::vector<gpu_long_t> h_block_stats(num_blocks * GPU_SEARCH_STAT_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_stats.data(), d_total_stats, h_block_stats.size() * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < num_blocks; ++b) {
      for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
        total_stats[i] += h_block_stats[b * GPU_SEARCH_STAT_COUNT + i];
      }
    }
  }
  if (GPU_DISTANCE_TIMING) {
    std::vector<gpu_long_t> h_block_timing(num_blocks * GPU_TIMING_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_timing.data(), d_total_timing, h_block_timing.size() * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < num_blocks; ++b) {
      for (int i = 0; i < GPU_TIMING_COUNT; ++i) {
        total_timing[i] += h_block_timing[b * GPU_TIMING_COUNT + i];
      }
    }
  }
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_qg_data));
  CUDA_SAFE_CALL(cudaFree(d_qg_signs));
  CUDA_SAFE_CALL(cudaFree(d_results));
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaFree(d_total_stats));
  }
  if (GPU_DISTANCE_TIMING) {
    CUDA_SAFE_CALL(cudaFree(d_total_timing));
  }

  if (GPU_SEARCH_STATS) {
    std::cout << "total # exact distance computations: " << total_stats[GPU_SEARCH_STAT_DIST_CMPS] << "\n";
    std::cout << "average # exact distance computations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]) / active_nq << "\n";
    std::cout << "total # distance estimations: " << total_stats[GPU_SEARCH_STAT_DIST_EST] << "\n";
    std::cout << "average # distance estimations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_EST]) / active_nq << "\n";
    std::cout << "total # expansion steps: " << total_stats[GPU_SEARCH_STAT_ITERATIONS] << "\n";
    std::cout << "average # expansion steps: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_ITERATIONS]) / active_nq << "\n";
  }
  if (GPU_DISTANCE_TIMING) {
    const double avg_l2_cycles =
        total_timing[GPU_TIMING_L2_CALLS] ? static_cast<double>(total_timing[GPU_TIMING_L2_CYCLES]) / total_timing[GPU_TIMING_L2_CALLS] : 0.0;
    const double avg_est_cycles =
        total_timing[GPU_TIMING_EST_CALLS] ? static_cast<double>(total_timing[GPU_TIMING_EST_CYCLES]) / total_timing[GPU_TIMING_EST_CALLS] : 0.0;
    const double avg_est_region_cycles =
        total_timing[GPU_TIMING_EST_REGION_CANDIDATES]
            ? static_cast<double>(total_timing[GPU_TIMING_EST_REGION_CYCLES]) / total_timing[GPU_TIMING_EST_REGION_CANDIDATES]
            : 0.0;
    std::cout << "total # l2 timing calls: " << total_timing[GPU_TIMING_L2_CALLS] << "\n";
    std::cout << "average cycles per warp_l2_distance: " << avg_l2_cycles << "\n";
    std::cout << "total # est timing calls: " << total_timing[GPU_TIMING_EST_CALLS] << "\n";
    std::cout << "average cycles per estimated_distance_gpu: " << avg_est_cycles << "\n";
    std::cout << "total # est region candidates: " << total_timing[GPU_TIMING_EST_REGION_CANDIDATES] << "\n";
    std::cout << "average cycles per estimation region candidate: " << avg_est_region_cycles << "\n";
    const struct {
      int idx;
      const char* label;
    } timing_sections[] = {
        {GPU_TIMING_QUERY_PREP_CYCLES, "query_prepare cycles"},
        {GPU_TIMING_SORT_CYCLES, "topk sort cycles"},
        {GPU_TIMING_PICK_CYCLES, "pick expander cycles"},
        {GPU_TIMING_VISITED_CYCLES, "visited check cycles"},
        {GPU_TIMING_STAGE_CYCLES, "stage code/factor cycles"},
        {GPU_TIMING_FINAL_SORT_CYCLES, "final sort cycles"},
        {GPU_TIMING_RESULT_POOL_CYCLES, "result pool update cycles"},
        {GPU_TIMING_TENSOR_QUERY_TILE_CYCLES, "tensor query tile cycles"},
        {GPU_TIMING_TENSOR_UNPACK_CYCLES, "tensor unpack cycles"},
        {GPU_TIMING_TENSOR_MMA_CYCLES, "tensor mma cycles"},
        {GPU_TIMING_TENSOR_FINAL_FACTOR_CYCLES, "tensor final factor cycles"},
    };
    for (const auto& section : timing_sections) {
      if (total_timing[section.idx] != 0) {
        std::cout << section.label << ": " << total_timing[section.idx] << "\n";
      }
    }
  }

  if (gpu_stats != nullptr) {
    gpu_stats->common.total_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_ds = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_EST]);
    gpu_stats->common.total_extra_dc = 0;
    gpu_stats->common.total_full_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_iter = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_ITERATIONS]);
    gpu_stats->common.total_pruned = 0;
    gpu_stats->common.time_search_total = elapsed;
    gpu_stats->gpu_l2_calls = static_cast<int64_t>(total_timing[GPU_TIMING_L2_CALLS]);
    gpu_stats->gpu_l2_cycles = static_cast<int64_t>(total_timing[GPU_TIMING_L2_CYCLES]);
    gpu_stats->gpu_est_calls = static_cast<int64_t>(total_timing[GPU_TIMING_EST_CALLS]);
    gpu_stats->gpu_est_cycles = static_cast<int64_t>(total_timing[GPU_TIMING_EST_CYCLES]);
  }

  (void)result_vec;
}


// ═══════════════════════════════════════════════════════════════════════
// ── run_sophia() entry point ──
// ═══════════════════════════════════════════════════════════════════════

// ── run_sophia(): entry point callable from merged gpu_cli ─────────────────

#include <nlohmann/json.hpp>

int run_sophia(const nlohmann::json& j) {
    auto& jd = j.at("data");
    std::string base_file  = jd.at("base");
    std::string query_file = jd.at("query");
    std::string gt_file    = jd.at("gt");
    std::string graph_file = jd.at("graph");

    auto& js = j.at("search");
    std::string mode    = js.at("mode");
    int K               = js.value("K", 100);
    int beam_size       = js.value("beam", 128);
    int degree          = js.value("degree", 32);
    std::string csv_file = js.value("csv", "");
    bool debug_mode     = js.value("debug", false);

    // strip "sophia_" prefix for internal dispatch
    if (mode.rfind("sophia_", 0) == 0) mode = mode.substr(7);

    std::cout << "search: mode=sophia_" << mode << " K=" << K << std::endl;

    srand(42);

    std::vector<vid_t> results;
    std::vector<float> distances;
    std::vector<float> queries;
    std::vector<int> groundtruth;
    size_t nq = 0;
    int gt_k = 0;
    double elapsed = 0.0;
    GpuRunStats gpu_stats;

    if (mode == "base") {
        IndexGraph<float> index;
        index.load_graph_index(graph_file.c_str());
        index.load_data(base_file.c_str());

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file.c_str(), nq, query_dim);
        if (debug_mode) { std::cout << "[DEBUG] running only query 0\n"; nq = std::min<size_t>(nq, 1); }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file.c_str(), gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        index.search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);

    } else if (mode == "rabitq") {
        IndexGraph<float> index;
        index.load_data(base_file.c_str());

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file.c_str(), nq, query_dim);
        if (debug_mode) { std::cout << "[DEBUG] running only query 0\n"; nq = std::min<size_t>(nq, 1); }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file.c_str(), gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, graph_file.c_str());
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);

    } else {
        std::cerr << "Unknown sophia mode: " << mode << "\n";
        return 1;
    }

    std::cout << "  time: " << elapsed * 1000.0 << " ms"
              << "  qps: " << nq / elapsed << std::endl;
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k);
    std::cout << "recall@" << K << ": " << recall << std::endl;

    return 0;
}

