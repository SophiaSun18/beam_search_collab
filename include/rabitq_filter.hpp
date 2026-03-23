#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cassert>

#include "../include/utils.h"
#include "../include/fastscan_impl.hpp"

static constexpr bool FASTSCAN_ENABLED = false;  // set to true to use fastscan batch estimation

class RaBitQFilter {
public:
    struct QueryFactors {
        float* rotated_query;
        uint8_t* quantized_query;
        uint8_t* lut = nullptr;     // fastscan mode: packed LUT, size = padded_dim << 2
        float low_val = 0.0f;
        float high_val = 0.0f;
        float width = 0;
        int32_t sum_q = 0;
    };

    struct NeighborFactors {
        const uint64_t* bits;
        float triple_x;
        float factor_dq;
        float factor_vq;
    };

    RaBitQFilter(size_t dim, int degree, const std::string& table_path, const std::string& signs_path)
        : dim_(dim), degree_(degree) {

        padded_dim_ = 1ULL << (size_t)std::ceil(std::log2(dim_));
        bitcode_words_ = padded_dim_ / 64;
        code_u64_per_node_ = static_cast<size_t>(degree_) * bitcode_words_;

        size_t n_signs, d_signs;
        signs_ptr_ = read_bin<float>(signs_path.c_str(), n_signs, d_signs);
        assert(n_signs * d_signs >= padded_dim_);

        size_t n_nodes, d_node_data;
        table_ptr_ = read_bin<uint64_t>(table_path.c_str(), n_nodes, d_node_data);
        words_per_node_ = d_node_data;

        size_t factor_floats = static_cast<size_t>(degree_) * 3;
        size_t expected_words = code_u64_per_node_ + (factor_floats * sizeof(float) + 7) / sizeof(uint64_t);
        assert(words_per_node_ == expected_words);
    }

    QueryFactors query_prepare(const float* query_raw) const {
        QueryFactors qf;
        qf.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim_ * sizeof(float)));
        qf.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim_ * sizeof(uint8_t)));

        // rotate the query
        rotate_vector(query_raw, qf.rotated_query, signs_ptr_, dim_, padded_dim_);

        // data range: find the lowest and highest dimension in the rotated query
        for (size_t i = 0; i < padded_dim_; ++i) {
            float tmp = qf.rotated_query[i];
            qf.low_val = std::min(qf.low_val, tmp);
            qf.high_val = std::max(qf.high_val, tmp);
        }
        qf.width = (qf.high_val - qf.low_val) / ((1 << QG_BQUERY) - 1);
        // std::cout << std::fixed << std::setprecision(20) << "Query data range: [" << qf.low_val << ", " << qf.high_val << "], width: " << qf.width << " QG_BQUERY: " << QG_BQUERY << "\n";
        // std::cout << std::fixed << std::setprecision(20) << "high - low: " << (qf.high_val - qf.low_val) << " divisor: " << ((1 << QG_BQUERY) - 1) << "\n";

        // quantize the query 
        float inv_width = 1.0F / qf.width;
        int32_t sum = 0;
        // std::cout << "1/width: " << std::fixed << std::setprecision(20) << inv_width << "\n";
        for (size_t i = 0; i < padded_dim_; ++i) {
            uint8_t cur = static_cast<uint8_t>(std::lround(((qf.rotated_query[i] - qf.low_val) * inv_width) + 0.5));
            qf.quantized_query[i] = cur;
            sum += cur;
            // std::cout << "i: " << i << " rotated: " << std::fixed << std::setprecision(20) << qf.rotated_query[i] << " quantized: " << static_cast<uint32_t>(qf.quantized_query[i]) << " sum: " << sum << "\n";
        }
        qf.sum_q = sum;

        if constexpr (FASTSCAN_ENABLED) {
            qf.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim_ << 2));
            pack_lut_impl(padded_dim_, qf.quantized_query, qf.lut);
        }

        return qf;
    }

    NeighborFactors nbr_filter(uint32_t node_idx, int neighbor_rank) const {
        const uint64_t* node_ptr = table_ptr_ + (static_cast<size_t>(node_idx) * words_per_node_);
        const uint64_t* bits = node_ptr + neighbor_rank * bitcode_words_;
        const float* factors = reinterpret_cast<const float*>(node_ptr + code_u64_per_node_);
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
        for (size_t d = 0; d < dim_; ++d) {
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
        float* result_float,
        bool debug_print_
    ) const {
        size_t num_blocks   = static_cast<size_t>(degree_) / kBatchSize;
        size_t packed_bytes = num_blocks * padded_dim_ * 4;

        // locate the values of the current node in the codebook
        const uint8_t* node_data = reinterpret_cast<const uint8_t*>(
            table_ptr_ + static_cast<size_t>(node_idx) * words_per_node_);
        const uint8_t* packed_code = node_data;
        const float* factors  = reinterpret_cast<const float*>(node_data + packed_bytes);
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

        // Debug: Print per-neighbor factors and estimated distances (matches 1-by-1 path)
        if (debug_print_) {
            for (int i = 0; i < degree_; ++i) {
                std::cout << std::fixed << std::setprecision(6)
                          << "  [fastscan] Neighbor rank: " << i
                          << ", Estimated Distance: " << appro_dist[i]
                          << ", Factors: [tx: " << triple_x[i]
                          << ", dq: " << fac_dq[i]
                          << ", vq: " << fac_vq[i]
                          << ", result_float: " << result_float[i]
                          << ", sum_q: " << qf.sum_q
                          << ", width: " << qf.width
                          << ", low_val: " << qf.low_val
                          << "]\n";
            }
        }
    }

    void prefetch_nbr(uint32_t node_idx) const {
        const char* base = reinterpret_cast<const char*>(
            table_ptr_ + static_cast<size_t>(node_idx) * words_per_node_);
        size_t total_bytes = words_per_node_ * sizeof(uint64_t);
        for (size_t off = 0; off < total_bytes; off += 64)
            __builtin_prefetch(base + off, 0, 0);
    }

    float* get_signs_ptr() const {
        return signs_ptr_;
    }

private:
    size_t dim_, padded_dim_;
    int degree_, bitcode_words_;
    size_t code_u64_per_node_;
    size_t words_per_node_;

    // Pointers to data loaded via read_bin
    float* signs_ptr_ = nullptr;
    uint64_t* table_ptr_ = nullptr;
};