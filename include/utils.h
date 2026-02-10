#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath> 
#include <chrono>
#include <cassert>
#include <cstring>
#include <immintrin.h>

using namespace std;
typedef uint32_t vidType;

template <typename T = float>
T* read_bin(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open bin file");
    int32_t num = 0, dim = 0;
    input.read(reinterpret_cast<char*>(&num), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (num <= 0 || dim <= 0)
        throw std::runtime_error("Invalid dimensions read from file");
    size_t total = static_cast<size_t>(num) * dim;
    T *data;
    if ((dim*sizeof(T)) % 32 == 0)
        data = (T*)aligned_alloc(32, sizeof(T) * total); // for SIMD
    else data = new T[total];
    input.read(reinterpret_cast<char*>(data), sizeof(T) * total);
    //for (size_t i = 0; i < num; i++) in.read((char*)(data+i*dim), sizeof(T)*dim);
    if (!input) throw std::runtime_error("Error reading bin vector data");
    n_out = size_t(num);
    d_out = size_t(dim);
    input.close();
    return data;
}

template <typename T>
void PREFETCH_VECTOR(int dim, T *vec) {
  char* a = (char*)vec;
  int l = dim * sizeof(T) / 64;
  for (int j = 0; j < l; j++) {
    __builtin_prefetch(a + 64 * j);
  }
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
    int64_t total_iter;
};

struct SearchResult {
    vector<int> neighbors;
    size_t dist_cmps;
    size_t iters;
};

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

inline int* read_ibin(const char* filename, size_t& n_out, size_t& d_out, size_t start_idx = 0, size_t chunk_size = 0) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open ibin file");

    int32_t n = 0, d = 0;
    input.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&d), sizeof(int32_t));

    if (start_idx >= size_t(n)) throw std::runtime_error("start_idx out of bounds");

    size_t read_n = (chunk_size == 0 || start_idx + chunk_size > size_t(n)) ?
                    size_t(n) - start_idx : chunk_size;

    int* data = new int[read_n * d];

    input.seekg(sizeof(int32_t) * d * start_idx, std::ios::cur);
    input.read(reinterpret_cast<char*>(data), sizeof(int32_t) * read_n * d);

    if (!input) throw std::runtime_error("Error reading ibin vector data");

    n_out = read_n;
    d_out = d;
    return data;
}

inline void write_ibin(const char* filename, const int* data, size_t n, size_t d) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot open output ibin file");
    }

    int32_t nn = static_cast<int32_t>(n);
    int32_t dd = static_cast<int32_t>(d);
    output.write(reinterpret_cast<const char*>(&nn), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(&dd), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(data), sizeof(int32_t) * n * d);

    if (!output) {
        throw std::runtime_error("Error writing to ibin file");
    }
}

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
    // a = (const float *)__builtin_assume_aligned(a, 32);
    // b = (const float *)__builtin_assume_aligned(b, 32);

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
        // load b_vec
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        // a_vec - b_vec
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    return _mm256_reduce_add_ps(sum);
}

// adapted from DiskANN
inline float compute_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    // a = (const float *)__builtin_assume_aligned(a, 32);
    // b = (const float *)__builtin_assume_aligned(b, 32);

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
    for (int i = 0; i < words; ++i)
        dist -= __builtin_popcount(a[i] ^ b[i]);
    return dist;
}

// compute recall@K for a single query
inline float recall_at_k(const vector<int>& ann_result, const int* gt_ptr, int K) {
    unordered_set<int> ann_set(ann_result.begin(), ann_result.begin() + K);

    int hit = 0;
    for (int i = 0; i < K; i++) {
        if (ann_set.count(gt_ptr[i])) hit++;
    }
    return float(hit) / K;
}

// compute average recall for all queries
inline float compute_avg_recall(const vector<std::vector<int>>& all_results,
                                const int* all_groundtruth, size_t n_query, int K) {
    float total_recall = 0.0f;
    for (size_t q = 0; q < n_query; q++) {
        const int* gt_ptr = all_groundtruth + q * K;
        total_recall += recall_at_k(all_results[q], gt_ptr, K);
    }
    return total_recall / n_query;
}