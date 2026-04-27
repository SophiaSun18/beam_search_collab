#pragma once
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

#include "../include/utils.h"
#ifndef __CUDACC__
#include "../include/fastscan_impl.hpp"
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
        constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);
        QueryFactors qf;
        qf.rotated_query = scratch.rotated_query;
        qf.quantized_query = scratch.quantized_query;
        qf.lut = scratch.lut;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;

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
        const float query_span = qf.high_val - qf.low_val;
        qf.width = query_span * kQueryLevelsInv;

        // quantize the query 
        const float inv_width = 1.0f / qf.width;
        int32_t sum = 0;
        for (size_t i = 0; i < padded_dim_; ++i) {
            const uint8_t cur = static_cast<uint8_t>(lroundf(((qf.rotated_query[i] - qf.low_val) * inv_width) + 0.5f));
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
