#include <iostream>
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
        std::vector<float> rotated_query;
        std::vector<uint8_t> quantized_query;
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

    RaBitQFilter(size_t dim, int degree, const std::string& table_path, const std::string& signs_path, int qbits) 
        : dim_(dim), degree_(degree), QG_BQUERY(qbits) {
        
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
        qf.rotated_query.resize(padded_dim_);
        qf.quantized_query.resize(padded_dim_);

        // rotate the query
        rotate_vector(query_raw, qf.rotated_query.data(), signs_ptr_, dim_, padded_dim_);

        // data range: find the lowest and highest dimension in the rotated query
        for (size_t i = 0; i < dim_; ++i) {
            float tmp = qf.rotated_query[i];
            qf.low_val = std::min(qf.low_val, tmp);
            qf.high_val = std::max(qf.high_val, tmp);
        }
        qf.width = (qf.high_val - qf.low_val) / ((1 << QG_BQUERY) - 1);

        // quantize the query 
        float inv_width = 1.0F / qf.width;
        int32_t sum = 0;
        for (size_t i = 0; i < dim_; ++i) {
            uint8_t cur = static_cast<uint8_t>(std::lround(((qf.rotated_query[i] - qf.low_val) * inv_width) + 0.5));
            qf.quantized_query[i] = cur;
            sum += cur;
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
        int32_t raw_sum = 0;

        for (size_t d = 0; d < dim_; ++d) {
            bool n_sign_bit = (nf.bits[d / 64] >> (d % 64)) & 1ULL;
            if (n_sign_bit) {
                raw_sum += static_cast<int32_t>(qf.quantized_query[d]);
            }
        }

        float result_float = static_cast<float>((raw_sum << 1) - qf.sum_q);

        float est_dist = nf.triple_x + exact_dist;
        est_dist += (nf.factor_dq * qf.width * result_float);
        est_dist += (nf.factor_vq * qf.low_val);

        return est_dist;
    }

private:
    size_t dim_, padded_dim_;
    int degree_, bitcode_words_, words_per_neighbor_, QG_BQUERY;
    size_t words_per_node_;

    // Pointers to data loaded via read_bin
    float* signs_ptr_ = nullptr;
    uint64_t* table_ptr_ = nullptr;
};