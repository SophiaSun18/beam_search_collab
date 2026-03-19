#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cassert>

#include "../include/utils.h"

class RaBitQFilter {
public:
    struct QueryFactors {
        float* rotated_query;
        uint8_t* quantized_query;
        float low_val = FLT_MAX;
        float high_val = FLT_MIN;
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
        bitcode_words_ = dim_ / 64;
        words_per_neighbor_ = bitcode_words_ + 2;
        words_per_node_ = degree_ * words_per_neighbor_;

        size_t n_signs, d_signs;
        signs_ptr_ = read_bin<float>(signs_path.c_str(), n_signs, d_signs);
        assert(n_signs * d_signs >= padded_dim_);

        size_t n_nodes, d_node_data;
        table_ptr_ = read_bin<uint64_t>(table_path.c_str(), n_nodes, d_node_data);
        assert(d_node_data == words_per_node_);
    }

    QueryFactors query_prepare(const float* query_raw) const {
        QueryFactors qf;
        qf.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim_ * sizeof(float)));
        qf.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim_ * sizeof(uint8_t)));

        // rotate the query
        rotate_vector(query_raw, qf.rotated_query, signs_ptr_, dim_, padded_dim_);

        // data range: find the lowest and highest dimension in the rotated query
        for (size_t i = 0; i < dim_; ++i) {
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
        for (size_t i = 0; i < dim_; ++i) {
            uint8_t cur = static_cast<uint8_t>(std::lround(((qf.rotated_query[i] - qf.low_val) * inv_width) + 0.5));
            qf.quantized_query[i] = cur;
            sum += cur;
            // std::cout << "i: " << i << " rotated: " << std::fixed << std::setprecision(20) << qf.rotated_query[i] << " quantized: " << static_cast<uint32_t>(qf.quantized_query[i]) << " sum: " << sum << "\n";
        }
        qf.sum_q = sum;

        return qf;
    }

    NeighborFactors nbr_filter(uint32_t node_idx, int neighbor_rank) const {
        const uint64_t* node_ptr = table_ptr_ + (static_cast<size_t>(node_idx) * words_per_node_);
        const uint64_t* neighbor_ptr = node_ptr + (neighbor_rank * words_per_neighbor_);
        
        const float* f_ptr = reinterpret_cast<const float*>(neighbor_ptr + bitcode_words_);
        return { neighbor_ptr, f_ptr[0], f_ptr[1], f_ptr[2] };
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
    int degree_, bitcode_words_, words_per_neighbor_;
    size_t words_per_node_;

    // Pointers to data loaded via read_bin
    float* signs_ptr_ = nullptr;
    uint64_t* table_ptr_ = nullptr;
};