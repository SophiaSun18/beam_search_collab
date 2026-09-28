#include <iostream>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <string>
#include <vector>
#include <random>
#include <new>
#include <omp.h>

#include "../include/qg.hpp"
#include "../third_party/Eigen/Dense"
#include "../third_party/ffht/fht_avx.hpp"

#ifndef PREPROCESS_RABITQ_SEQ_LUT_READY
#define PREPROCESS_RABITQ_SEQ_LUT_READY 0
#endif
static_assert(
    !(FASTSCAN_ENABLED && (PREPROCESS_RABITQ_SEQ_LUT_READY != 0)),
    "PREPROCESS_RABITQ_SEQ_LUT_READY requires BEAM_SEARCH_FASTSCAN_ENABLED=0");

// Match SymQG's row-major Eigen matrix layout for RaBitQ factor arithmetic.
template <typename T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Provide 64-byte aligned float storage for AVX loads/stores used by rotation.
struct AlignedFloatBuffer {
    explicit AlignedFloatBuffer(size_t count) : count_(count) {
        size_t bytes = count * sizeof(float);
        size_t alloc_bytes = (bytes + 64 - 1) & ~(size_t(64) - 1);
        ptr_ = static_cast<float*>(std::aligned_alloc(64, alloc_bytes));
        if (ptr_ == nullptr) throw std::bad_alloc();
        std::fill(ptr_, ptr_ + count_, 0.0f);
    }
    ~AlignedFloatBuffer() { std::free(ptr_); }
    AlignedFloatBuffer(const AlignedFloatBuffer&) = delete;
    AlignedFloatBuffer& operator=(const AlignedFloatBuffer&) = delete;
    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    float& operator[](size_t i) { return ptr_[i]; }
    const float& operator[](size_t i) const { return ptr_[i]; }

private:
    float* ptr_ = nullptr;
    size_t count_ = 0;
};

// Dispatch to the same specialized FFHT kernels used by SymQG's FHTRotator.
static inline void symqg_style_fht(float* dst, size_t padded_dim) {
    switch (padded_dim) {
        case 64:
            helper_float_6(dst);
            break;
        case 128:
            helper_float_7(dst);
            break;
        case 256:
            helper_float_8(dst);
            break;
        case 512:
            helper_float_9(dst);
            break;
        case 1024:
            helper_float_10(dst);
            break;
        case 2048:
            helper_float_11(dst);
            break;
        case 4096:
            helper_float_12(dst);
            break;
        default:
            std::cerr << "Unsupported padded dimension for SymQG-style FHT: "
                      << padded_dim << "\n";
            std::abort();
    }
}

// Convert raw sequential bytes into the neighbor-major sequential-LUT convention.
static inline uint8_t sequential_lut_ready_byte(uint8_t byte) {
    auto reverse_nibble = [](uint8_t value) -> uint8_t {
        return static_cast<uint8_t>(
            ((value & 0x1) << 3) |
            ((value & 0x2) << 1) |
            ((value & 0x4) >> 1) |
            ((value & 0x8) >> 3));
    };
    const uint8_t lo = reverse_nibble(byte & 0x0F);
    const uint8_t hi = reverse_nibble(byte >> 4);
    return static_cast<uint8_t>(lo | (hi << 4));
}

// Apply SymQG-compatible sign multiplication, zero padding, and FHT rotation.
static inline void rotate_vector_symqg_style(const float* __restrict__ src,
                                             float* __restrict__ dst,
                                             const float* __restrict__ signs,
                                             size_t dim,
                                             size_t padded_dim) {
    size_t iter = 0;
#if defined(__AVX512F__)
    const size_t remain = dim & 0b1111;
    iter = dim - remain;
    for (size_t idx = 0; idx < iter; idx += 16) {
        __m512 ss = _mm512_loadu_ps(&src[idx]);
        __m512 mm = _mm512_load_ps(&signs[idx]);
        ss = _mm512_mul_ps(ss, mm);
        _mm512_store_ps(&dst[idx], ss);
    }
#elif defined(__AVX2__)
    const size_t remain = dim & 0b111;
    iter = dim - remain;
    for (size_t idx = 0; idx < iter; idx += 8) {
        __m256 ss = _mm256_loadu_ps(&src[idx]);
        __m256 mm = _mm256_load_ps(&signs[idx]);
        ss = _mm256_mul_ps(ss, mm);
        _mm256_store_ps(&dst[idx], ss);
    }
#endif
    for (size_t idx = iter; idx < dim; ++idx) {
        dst[idx] = src[idx] * signs[idx];
    }
    std::fill(dst + dim, dst + padded_dim, 0.0f);
    symqg_style_fht(dst, padded_dim);
}

// Compute the three RaBitQ factor arrays with SymQG's Eigen expression order.
static inline void rabitq_factors_symqg_style(const RowMatrix<float>& rotated_data_residual,
                                              const RowMatrix<float>& rotated_centroid,
                                              const RowMatrix<int>& bin_x,
                                              float* triple_x,
                                              float* factor_dq,
                                              float* factor_vq) {
    long num_points = rotated_data_residual.rows();
    long dim = rotated_data_residual.cols();

    float fac_norm = 1.F / std::sqrt(static_cast<float>(dim));

    RowMatrix<float> ones(num_points, dim);
    ones.setOnes();
    RowMatrix<float> signed_x = 2 * bin_x.cast<float>() - ones;

    RowMatrix<float> fac_x0 =
        (rotated_data_residual.array() * signed_x.array() * fac_norm).rowwise().sum();
    RowMatrix<float> x_rotated_norm = rotated_data_residual.rowwise().norm();
    for (long i = 0; i < num_points; ++i) {
        float norm = x_rotated_norm(i, 0);
        float cur_x0 = fac_x0(i, 0);
        fac_x0(i, 0) = (norm == 0.0f) ? 0.0f : cur_x0 / norm;
    }

    RowMatrix<float> fac_x1(num_points, 1);
    for (long i = 0; i < num_points; ++i) {
        fac_x1(i, 0) = static_cast<float>(
            (rotated_centroid.array() * signed_x.row(i).array()).sum() * fac_norm
        );
    }

    for (long j = 0; j < num_points; ++j) {
        double cur_x = x_rotated_norm(j, 0);

        if (cur_x == 0.0) {
            triple_x[j] = FLT_MAX;
            factor_dq[j] = FLT_MAX;
            factor_vq[j] = FLT_MAX;
            continue;
        }

        double cur_x0 = fac_x0(j, 0);
        double cur_x1 = fac_x1(j, 0);
        long double x_x0 = static_cast<long double>(cur_x) / cur_x0;

        triple_x[j] = static_cast<float>((cur_x * cur_x) + (2 * x_x0 * cur_x1));
        factor_dq[j] = static_cast<float>(-2 * x_x0 * fac_norm);
        factor_vq[j] = static_cast<float>(
            -2 * x_x0 * fac_norm * (bin_x.row(j).sum() * 2 - static_cast<int>(dim))
        );
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " -base <base.fvecs> -graph <graph.bin> -out <output.bin> -sign <signs.bin> [-seed 0] [-sign_in <existing_signs.bin>]\n";
        return 1;
    }

    // Parse the input graph, base vectors, output paths, and optional sign source.
    std::string base_file, graph_file, out_file, signs_file, sign_in_file;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-base" && i + 1 < argc) base_file = argv[++i];
        else if (arg == "-graph" && i + 1 < argc) graph_file = argv[++i];
        else if (arg == "-out" && i + 1 < argc) out_file = argv[++i];
        else if (arg == "-sign" && i + 1 < argc) signs_file = argv[++i];
        else if (arg == "-seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "-sign_in" && i + 1 < argc) sign_in_file = argv[++i];
    }

    if (base_file.empty() || graph_file.empty() || out_file.empty() || signs_file.empty()) {
        std::cerr << "Error: -base, -graph, -out, and -sign are required.\n";
        return 1;
    }

    // Load base vectors and the external fixed-degree graph adjacency.
    size_t n_base, dim;
    float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim);
    std::cout << "Loaded " << n_base << " base vectors, dim=" << dim << "\n";

    Graph g;
    g.load_graph_index(graph_file.c_str());
    std::cout << "Loaded graph with " << g.n << " nodes, degree=" << g.degree << "\n";
    if (g.n != n_base) {
        std::cerr << "Graph node count " << g.n << " does not match base vectors " << n_base << "\n";
        delete[] base_vectors;
        return 1;
    }
    if constexpr (FASTSCAN_ENABLED) {
        if (g.degree % kBatchSize != 0) {
            std::cerr << "FastScan RaBitQ preprocessing requires degree to be a multiple of "
                      << kBatchSize << ", got " << g.degree << "\n";
            delete[] base_vectors;
            return 1;
        }
    }
    size_t padded_dim = 1ULL << (size_t)std::ceil(std::log2(dim));
    float scale = 1.0f / std::sqrt(static_cast<float>(padded_dim));

    // Reuse an existing sign table when provided; otherwise generate one deterministically.
    AlignedFloatBuffer signs(padded_dim);
    if (!sign_in_file.empty()) {
        size_t sign_rows = 0, sign_cols = 0;
        float* loaded_signs = read_bin<float>(sign_in_file.c_str(), sign_rows, sign_cols);
        if (!((sign_rows == padded_dim && sign_cols == 1) || (sign_rows == 1 && sign_cols == padded_dim))) {
            std::cerr << "Sign input shape mismatch: got " << sign_rows << "x" << sign_cols
                      << ", expected " << padded_dim << "x1 or 1x" << padded_dim << "\n";
            release_bin_buffer(loaded_signs, sign_cols);
            delete[] base_vectors;
            return 1;
        }
        std::copy(loaded_signs, loaded_signs + padded_dim, signs.data());
        release_bin_buffer(loaded_signs, sign_cols);
    } else {
        std::uniform_int_distribution<int> bernoulli(0, 1);
        std::mt19937_64 gen(seed);
        for (size_t i = 0; i < padded_dim; ++i) {
            signs[i] = static_cast<float>((2 * bernoulli(gen)) - 1) * scale;
        }
    }

    // Store each node as packed codes followed by triple_x, factor_dq, and factor_vq.
    const size_t bitcode_words = padded_dim / 64;
    const size_t code_u64_per_node = g.degree * bitcode_words;
    const size_t factor_floats = 3 * g.degree;
    const size_t factor_u64_words = (factor_floats * sizeof(float) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    const size_t words_per_node = code_u64_per_node + factor_u64_words;

    std::vector<uint64_t> table(g.n * words_per_node, 0);

    // Build one RaBitQ codebook row per graph node from its outgoing neighbors.
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t u = 0; u < g.n; u++) {
        AlignedFloatBuffer u_rotated(padded_dim);
        AlignedFloatBuffer v_rotated(padded_dim);
        RowMatrix<float> rotated_data_residual(g.degree, padded_dim);
        RowMatrix<float> rotated_centroid(1, padded_dim);
        RowMatrix<int> bin_x(g.degree, padded_dim);
        std::vector<uint64_t> binary(g.degree * bitcode_words, 0);
        std::vector<float> factors(factor_u64_words * 2, 0.0f);

        rotate_vector_symqg_style(base_vectors + u * dim, u_rotated.data(), signs.data(), dim, padded_dim);
        for (size_t d = 0; d < padded_dim; ++d) {
            rotated_centroid(0, d) = u_rotated[d];
        }

        const uint32_t* neighbors = g.get_neighbors(u);
        for (size_t nbr = 0; nbr < g.degree; nbr++) {
            uint32_t v = neighbors[nbr];
            if (v >= n_base) {
                std::cerr << "Neighbor id " << v << " out of range for node " << u << "\n";
                std::abort();
            }
            rotate_vector_symqg_style(base_vectors + static_cast<size_t>(v) * dim,
                                      v_rotated.data(), signs.data(), dim, padded_dim);

            for (size_t d = 0; d < padded_dim; d++) {
                rotated_data_residual(nbr, d) = v_rotated[d] - u_rotated[d];
            }

            uint64_t* binary_out = binary.data() + nbr * bitcode_words;
            for (size_t d = 0; d < padded_dim; d++) {
                if (rotated_data_residual(nbr, d) > 0.0f) {
                    bin_x(nbr, d) = 1;
                    const size_t bit = FASTSCAN_ENABLED ? (63 - (d % 64)) : (d % 64);
                    binary_out[d / 64] |= (1ULL << bit);
                } else {
                    bin_x(nbr, d) = 0;
                }
            }
        }

        float* triple_x = factors.data();
        float* factor_dq = triple_x + g.degree;
        float* factor_vq = factor_dq + g.degree;
        rabitq_factors_symqg_style(
            rotated_data_residual, rotated_centroid, bin_x, triple_x, factor_dq, factor_vq);

        uint64_t* row = table.data() + u * words_per_node;
        if constexpr (FASTSCAN_ENABLED) {
            pack_codes(padded_dim, binary.data(), g.degree, reinterpret_cast<uint8_t*>(row));
        } else {
            if constexpr (PREPROCESS_RABITQ_SEQ_LUT_READY != 0) {
                uint8_t* code_bytes = reinterpret_cast<uint8_t*>(binary.data());
                for (size_t byte = 0; byte < code_u64_per_node * sizeof(uint64_t); ++byte) {
                    code_bytes[byte] = sequential_lut_ready_byte(code_bytes[byte]);
                }
            }
            std::memcpy(row, binary.data(), code_u64_per_node * sizeof(uint64_t));
        }
        std::memcpy(row + code_u64_per_node, factors.data(), factor_floats * sizeof(float));
    }

    // Persist the generated RaBitQ sidecar and the sign table used to build it.
    write_bin<uint64_t>(out_file.c_str(), table.data(), g.n, words_per_node);
    write_bin<float>(signs_file.c_str(), signs.data(), padded_dim, 1);
    std::cout << "Wrote rabitQ table to " << out_file << "\n";
    std::cout << "Wrote signs to " << signs_file << "\n";

    delete[] base_vectors;
    return 0;
}
