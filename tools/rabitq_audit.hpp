#pragma once

#include "index.hpp"
#include "../code/beam_gpu_host_sidecars.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct RabitqAuditConfig {
    int query_samples = 1;
    int edge_samples = 32;
    int max_source_nodes = 32;
    int verbose_edges = 8;
};

static inline float routea_u32_to_float_audit(uint32_t bits) {
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline void rabitq_fwht_128_host_audit(float* x) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += (h << 1)) {
            for (int j = 0; j < h; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
}

static inline bool rabitq_load_proj_with_header_audit(
    const std::string& path,
    std::vector<float>& out,
    int& rows,
    int& cols) {
    rows = 0;
    cols = 0;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t r = 0;
    int32_t c = 0;
    f.read(reinterpret_cast<char*>(&r), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&c), sizeof(int32_t));
    if (!f || r <= 0 || c <= 0) return false;
    out.resize((size_t)r * (size_t)c);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!f) {
        out.clear();
        return false;
    }
    rows = (int)r;
    cols = (int)c;
    return true;
}

static inline bool rabitq_read_words_at(
    const std::string& path,
    std::streamoff byte_offset,
    size_t word_count,
    std::vector<uint32_t>& out) {
    out.assign(word_count, 0u);
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    const std::streamoff need = byte_offset +
        static_cast<std::streamoff>(word_count * sizeof(uint32_t));
    if (byte_offset < 0 || bytes < need) return false;
    f.seekg(byte_offset, std::ios::beg);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(word_count * sizeof(uint32_t)));
    return static_cast<bool>(f);
}

static inline bool rabitq_read_floats_at(
    const std::string& path,
    size_t elem_offset,
    size_t elem_count,
    std::vector<float>& out) {
    out.assign(elem_count, 0.0f);
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    const std::streamoff byte_offset =
        static_cast<std::streamoff>(elem_offset * sizeof(float));
    const std::streamoff need =
        byte_offset + static_cast<std::streamoff>(elem_count * sizeof(float));
    if (bytes < need) return false;
    f.seekg(byte_offset, std::ios::beg);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(elem_count * sizeof(float)));
    return static_cast<bool>(f);
}

static inline void rabitq_pack_sign_words_audit(
    const float* vec,
    int dim,
    std::vector<uint32_t>& out) {
    const int W = (dim + 31) >> 5;
    out.assign((size_t)W, 0u);
    for (int w = 0; w < W; ++w) {
        uint32_t word = 0u;
        const int base = w << 5;
        for (int b = 0; b < 32; ++b) {
            const int i = base + b;
            if (i >= dim) break;
            const uint32_t bit = (vec[i] >= 0.0f) ? 1u : 0u;
            word |= (bit << b);
        }
        out[(size_t)w] = word;
    }
}

template <typename T>
static inline void rabitq_pack_residual_sign_words_raw_audit(
    const T* uvec,
    const T* vvec,
    int dim,
    std::vector<uint32_t>& out) {
    const int W = (dim + 31) >> 5;
    out.assign((size_t)W, 0u);
    for (int w = 0; w < W; ++w) {
        uint32_t word = 0u;
        const int base = w << 5;
        for (int b = 0; b < 32; ++b) {
            const int i = base + b;
            if (i >= dim) break;
            const float diff = (float)vvec[i] - (float)uvec[i];
            const uint32_t bit = (diff >= 0.0f) ? 1u : 0u;
            word |= (bit << b);
        }
        out[(size_t)w] = word;
    }
}

static inline void rabitq_apply_proj_steps_audit(
    const float* src,
    int dim,
    const std::vector<float>& proj_steps,
    int rows,
    int cols,
    std::vector<float>& out) {
    out.assign((size_t)dim, 0.0f);
    if (dim != 128 || cols != 128 || rows <= 0 ||
        proj_steps.size() < (size_t)rows * (size_t)cols) {
        std::copy(src, src + dim, out.begin());
        return;
    }
    std::copy(src, src + 128, out.begin());
    for (int st = 0; st < rows; ++st) {
        const float* proj_row = proj_steps.data() + (size_t)st * 128u;
        for (int i = 0; i < 128; ++i) {
            out[(size_t)i] *= proj_row[i];
        }
        rabitq_fwht_128_host_audit(out.data());
    }
}

static inline double rabitq_sign_match_ratio_audit(
    const std::vector<uint32_t>& a,
    const std::vector<uint32_t>& b,
    int dim) {
    if (a.size() != b.size() || dim <= 0) return 0.0;
    int matched = 0;
    int total = dim;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t diff = a[i] ^ b[i];
        int bits = 32;
        const int used = std::min(32, std::max(0, dim - (int)(i * 32)));
        bits = used;
        if (used < 32) {
            const uint32_t mask = (used == 32) ? 0xFFFFFFFFu : ((1u << used) - 1u);
            diff &= mask;
        }
        matched += bits - __builtin_popcount(diff);
    }
    return (total > 0) ? (double)matched / (double)total : 0.0;
}

template <typename T>
static inline bool rabitq_compute_raw_factor_edge_audit(
    const T* uvec,
    const T* vvec,
    int dim,
    float& triple,
    float& factor_dq,
    float& factor_vq) {
    const float fac_norm = 1.0f / std::sqrt((float)dim);
    constexpr float kRabitqAuditFactorEps = 1e-8f;
    const float eps = kRabitqAuditFactorEps;
    float norm2 = 0.0f;
    float dot_x_obar = 0.0f;
    float dot_c_obar = 0.0f;
    int sign_sum = 0;
    for (int d = 0; d < dim; ++d) {
        const float r = (float)vvec[d] - (float)uvec[d];
        norm2 = fmaf(r, r, norm2);
        const float s = (r >= 0.0f) ? 1.0f : -1.0f;
        dot_x_obar += r * s;
        dot_c_obar += (float)uvec[d] * s;
        sign_sum += (r >= 0.0f) ? 1 : -1;
    }
    const float norm = std::sqrt(norm2);
    if (norm <= eps || dot_x_obar <= eps) {
        triple = norm2;
        factor_dq = 0.0f;
        factor_vq = 0.0f;
        return true;
    }
    const float fac_x0 = (dot_x_obar * fac_norm) / norm;
    if (std::fabs(fac_x0) <= eps) {
        triple = norm2;
        factor_dq = 0.0f;
        factor_vq = 0.0f;
        return true;
    }
    const float x_div_x0 = norm / fac_x0;
    const float fac_x1 = dot_c_obar * fac_norm;
    triple = norm2 + (2.0f * x_div_x0 * fac_x1);
    factor_dq = -2.0f * x_div_x0 * fac_norm;
    factor_vq = -2.0f * x_div_x0 * fac_norm * (float)sign_sum;
    return true;
}

static inline std::vector<std::string> rabitq_signbit_candidates_audit(const std::string& graph_prefix) {
    std::vector<std::string> out;
    const char* signbit_override_env = std::getenv("SIGNBIT_FILE");
    const char* signbit_override2_env = std::getenv("SIGNBIT_FILE_ALT");
    if (signbit_override_env && signbit_override_env[0] != '\0') {
        out.push_back(std::string(signbit_override_env));
    }
    if (signbit_override2_env && signbit_override2_env[0] != '\0') {
        const std::string p(signbit_override2_env);
        if (std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
    }
    const std::string primary = graph_prefix.empty() ? std::string("sign_bit.bin")
                                                     : (graph_prefix + ".signbit.bin");
    const std::string ref = graph_prefix.empty() ? std::string("graph-d32.ref.signbit.bin")
                                                 : (graph_prefix + ".ref.signbit.bin");
    const std::string fallback = "sign_bit.bin";
    const std::string fallback_ref = "graph-d32.ref.signbit.bin";
    for (const auto& p : {primary, ref, fallback, fallback_ref}) {
        if (std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
    }
    return out;
}

static inline std::vector<std::string> rabitq_factor_candidates_audit(
    const std::string& graph_prefix,
    const char* env_name,
    const char* basename) {
    std::vector<std::string> out;
    const char* override_path = std::getenv(env_name);
    if (override_path && override_path[0] != '\0') out.push_back(std::string(override_path));
    const std::string primary = graph_prefix.empty() ? std::string(basename)
                                                     : (graph_prefix + "." + basename);
    if (std::find(out.begin(), out.end(), primary) == out.end()) out.push_back(primary);
    if (std::find(out.begin(), out.end(), std::string(basename)) == out.end()) {
        out.push_back(std::string(basename));
    }
    return out;
}

static inline bool rabitq_read_signbit_edge_sample_audit(
    const std::vector<std::string>& candidates,
    size_t expected_words,
    size_t edge_pos,
    int words_per_edge,
    std::vector<uint32_t>& out,
    std::string& loaded_from,
    std::string& loaded_format) {
    const size_t word_offset = edge_pos * (size_t)words_per_edge;
    for (const auto& path : candidates) {
        std::ifstream f(path, std::ios::binary);
        if (!f) continue;
        f.seekg(0, std::ios::end);
        const std::streamoff bytes = f.tellg();
        f.close();
        const std::streamoff raw_bytes = static_cast<std::streamoff>(expected_words * sizeof(uint32_t));
        const std::streamoff read_bin_bytes =
            static_cast<std::streamoff>(sizeof(int32_t) * 2 + expected_words * sizeof(uint32_t));
        if (bytes == raw_bytes) {
            if (rabitq_read_words_at(path,
                                     static_cast<std::streamoff>(word_offset * sizeof(uint32_t)),
                                     (size_t)words_per_edge,
                                     out)) {
                loaded_from = path;
                loaded_format = "raw";
                return true;
            }
        }
        if (bytes >= read_bin_bytes) {
            if (rabitq_read_words_at(path,
                                     static_cast<std::streamoff>(sizeof(int32_t) * 2 + word_offset * sizeof(uint32_t)),
                                     (size_t)words_per_edge,
                                     out)) {
                loaded_from = path;
                loaded_format = "read_bin";
                return true;
            }
        }
    }
    loaded_from.clear();
    loaded_format.clear();
    out.clear();
    return false;
}

static inline bool rabitq_read_factor_edge_sample_audit(
    const std::vector<std::string>& candidates,
    size_t factor_count,
    size_t edge_pos,
    float& out,
    std::string& loaded_from) {
    std::vector<float> tmp;
    for (const auto& path : candidates) {
        std::ifstream f(path, std::ios::binary);
        if (!f) continue;
        f.seekg(0, std::ios::end);
        const std::streamoff bytes = f.tellg();
        f.close();
        if (bytes != static_cast<std::streamoff>(factor_count * sizeof(float))) continue;
        if (rabitq_read_floats_at(path, edge_pos, 1, tmp) && tmp.size() == 1) {
            out = tmp[0];
            loaded_from = path;
            return true;
        }
    }
    loaded_from.clear();
    return false;
}

static inline float routea_exact_l2_audit(
    const float* query,
    const float* vec,
    int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float dx = query[d] - vec[d];
        sum = fmaf(dx, dx, sum);
    }
    return sum;
}

static inline double routea_topk_overlap_ratio_audit(
    const std::vector<int>& exact_order,
    const std::vector<int>& approx_order,
    int k) {
    if (k <= 0 || exact_order.empty() || approx_order.empty()) return 0.0;
    k = std::min(k, std::min((int)exact_order.size(), (int)approx_order.size()));
    int overlap = 0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (exact_order[i] == approx_order[j]) {
                overlap++;
                break;
            }
        }
    }
    return (double)overlap / (double)k;
}

template <typename T>
static inline bool run_routea_rank_audit(
    const IndexGraph<T>& index,
    const T* queries,
    size_t nq,
    const RabitqAuditConfig& cfg) {
    std::cout << "========================================\n";
    std::cout << "RouteA Rank Audit\n";
    std::cout << "========================================\n";
    std::cout << "graph_prefix: " << index.graph_prefix << "\n";
    std::cout << "npoints: " << index.ntotal << ", dim: " << index.d
              << ", maxDeg: " << index.maxDeg << "\n";

    if (index.ntotal == 0 || index.d <= 0 || index.maxDeg <= 0 ||
        index.data.empty() || index.edges.empty()) {
        std::cerr << "RouteA audit failed: graph/data not loaded.\n";
        return false;
    }
    if (queries == nullptr || nq == 0) {
        std::cerr << "RouteA audit failed: queries not loaded.\n";
        return false;
    }

    const int words_per_edge = (index.d + 31) >> 5;
    const size_t edge_count = index.ntotal * static_cast<size_t>(index.maxDeg);
    const size_t expected_sign_words = edge_count * static_cast<size_t>(words_per_edge);
    const size_t factor_count = edge_count;
    const std::vector<std::string> sign_candidates =
        rabitq_signbit_candidates_audit(index.graph_prefix);
    const std::vector<std::string> triple_candidates =
        rabitq_factor_candidates_audit(
            index.graph_prefix, "RABITQ_TRIPLE_FILE", "rabitq.triple.f32.bin");
    const std::vector<std::string> dq_candidates =
        rabitq_factor_candidates_audit(
            index.graph_prefix, "RABITQ_FACTOR_DQ_FILE", "rabitq.factor_dq.f32.bin");

    std::cout << "routea_contract: per-edge raw residual\n";

    constexpr std::array<int, 8> kAuditTopKs{{1, 4, 8, 12, 16, 20, 24, 28}};
    std::array<double, kAuditTopKs.size()> overlap_sum{};
    std::array<int64_t, kAuditTopKs.size()> overlap_cnt{};
    double exact_best_rank_in_approx_sum = 0.0;
    double approx_best_rank_in_exact_sum = 0.0;
    int64_t pair_count = 0;

    const int query_samples = std::max(1, std::min((int)nq, cfg.query_samples));
    const size_t source_samples =
        std::min(index.ntotal, static_cast<size_t>(std::max(1, cfg.max_source_nodes)));
    const int64_t pair_limit = std::max<int64_t>(1, cfg.edge_samples);

    std::vector<float> exact_dist;
    std::vector<float> approx_dist;
    std::vector<int> valid_pos;
    std::vector<int> exact_order;
    std::vector<int> approx_order;

    for (int qi = 0; qi < query_samples; ++qi) {
        if (pair_count >= pair_limit) break;
        beam_gpu_host::RouteAQueryQ6Prepared prep;
        const float* query = reinterpret_cast<const float*>(queries + (size_t)qi * (size_t)index.d);
        const bool prep_ok =
            beam_gpu_host::routea_prepare_raw_query_q6_lut_host(query, index.d, prep);
        if (!prep_ok) {
            std::cerr << "RouteA audit failed: query prep failed for qi=" << qi << "\n";
            return false;
        }

        for (size_t si = 0; si < source_samples; ++si) {
            if (pair_count >= pair_limit) break;
            const size_t u = (si * index.ntotal) / source_samples;
            const size_t edge_base = u * static_cast<size_t>(index.maxDeg);
            const float* uvec =
                reinterpret_cast<const float*>(index.data.data()) +
                static_cast<size_t>(u) * static_cast<size_t>(index.d);
            const float sqr_y = routea_exact_l2_audit(query, uvec, index.d);
            valid_pos.clear();
            exact_dist.clear();
            approx_dist.clear();
            for (int j = 0; j < index.maxDeg; ++j) {
                const vid_t v = index.edges[edge_base + static_cast<size_t>(j)];
                if (static_cast<size_t>(v) >= index.ntotal) continue;
                std::vector<uint32_t> sign_words;
                std::string sign_from;
                std::string sign_format;
                const size_t edge_pos = edge_base + static_cast<size_t>(j);
                if (!rabitq_read_signbit_edge_sample_audit(
                        sign_candidates, expected_sign_words, edge_pos, words_per_edge,
                        sign_words, sign_from, sign_format)) {
                    continue;
                }
                float triple = 0.0f;
                float fdq = 0.0f;
                std::string triple_from;
                std::string dq_from;
                if (!rabitq_read_factor_edge_sample_audit(
                        triple_candidates, factor_count, edge_pos, triple, triple_from) ||
                    !rabitq_read_factor_edge_sample_audit(
                        dq_candidates, factor_count, edge_pos, fdq, dq_from)) {
                    continue;
                }
                int set_sum = 0;
                int pop_sum = 0;
                const int tail_bits = index.d & 31;
                const uint32_t tail_mask =
                    (tail_bits == 0) ? 0xFFFFFFFFu : ((1u << tail_bits) - 1u);
                for (int w = 0; w < words_per_edge; ++w) {
                    uint32_t sw = sign_words[(size_t)w];
                    if ((w == words_per_edge - 1) && (tail_bits != 0)) {
                        sw &= tail_mask;
                    }
                    pop_sum += __builtin_popcount(sw);
                    for (int nib = 0; nib < 8; ++nib) {
                        const int g = (w << 3) + nib;
                        if (g >= ((index.d + 3) >> 2)) break;
                        set_sum += (int)prep.lut[(size_t)g * 16u + ((sw >> (nib * 4)) & 0xFu)];
                    }
                }
                const int signed_sum = (set_sum << 1) - prep.sumq;
                const int sign_sum = (pop_sum << 1) - index.d;
                const float fvq = fdq * (float)sign_sum;
                float approx = sqr_y +
                               triple +
                               (fdq * (prep.width * (float)signed_sum)) +
                               (fvq * prep.lower);
                if (!(approx > 0.0f) || !std::isfinite(approx)) {
                    approx = std::numeric_limits<float>::infinity();
                }
                valid_pos.push_back(j);
                const float* vvec =
                    reinterpret_cast<const float*>(index.data.data()) +
                    static_cast<size_t>(v) * static_cast<size_t>(index.d);
                exact_dist.push_back(routea_exact_l2_audit(query, vvec, index.d));
                approx_dist.push_back(approx);
            }

            const int deg = static_cast<int>(valid_pos.size());
            if (deg <= 1) continue;
            exact_order.resize((size_t)deg);
            approx_order.resize((size_t)deg);
            for (int i = 0; i < deg; ++i) {
                exact_order[(size_t)i] = i;
                approx_order[(size_t)i] = i;
            }
            std::stable_sort(exact_order.begin(), exact_order.end(),
                             [&](int a, int b) {
                                 if (exact_dist[(size_t)a] != exact_dist[(size_t)b]) {
                                     return exact_dist[(size_t)a] < exact_dist[(size_t)b];
                                 }
                                 return a < b;
                             });
            std::stable_sort(approx_order.begin(), approx_order.end(),
                             [&](int a, int b) {
                                 if (approx_dist[(size_t)a] != approx_dist[(size_t)b]) {
                                     return approx_dist[(size_t)a] < approx_dist[(size_t)b];
                                 }
                                 return a < b;
                             });

            for (size_t ki = 0; ki < kAuditTopKs.size(); ++ki) {
                const int k = kAuditTopKs[ki];
                if (k > deg) continue;
                overlap_sum[ki] += routea_topk_overlap_ratio_audit(exact_order, approx_order, k);
                overlap_cnt[ki] += 1;
            }

            int exact_best_rank_in_approx = deg;
            int approx_best_rank_in_exact = deg;
            for (int i = 0; i < deg; ++i) {
                if (approx_order[(size_t)i] == exact_order[0]) {
                    exact_best_rank_in_approx = i + 1;
                    break;
                }
            }
            for (int i = 0; i < deg; ++i) {
                if (exact_order[(size_t)i] == approx_order[0]) {
                    approx_best_rank_in_exact = i + 1;
                    break;
                }
            }
            exact_best_rank_in_approx_sum += (double)exact_best_rank_in_approx;
            approx_best_rank_in_exact_sum += (double)approx_best_rank_in_exact;

            if (pair_count < (int64_t)std::max(0, cfg.verbose_edges)) {
                std::cout << std::fixed << std::setprecision(6)
                          << "routea_pair[" << pair_count << "] q=" << qi
                          << " src=" << u
                          << " overlap@8=" << routea_topk_overlap_ratio_audit(exact_order, approx_order, std::min(8, deg))
                          << " overlap@20=" << routea_topk_overlap_ratio_audit(exact_order, approx_order, std::min(20, deg))
                          << " exact_best_rank_in_approx=" << exact_best_rank_in_approx
                          << " approx_best_rank_in_exact=" << approx_best_rank_in_exact
                          << "\n";
            }
            pair_count++;
        }
    }

    if (pair_count == 0) {
        std::cerr << "RouteA audit failed: no valid query-source pairs sampled.\n";
        return false;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ROUTEA_AUDIT sampled_pairs=" << pair_count << "\n";
    for (size_t ki = 0; ki < kAuditTopKs.size(); ++ki) {
        const int64_t cnt = overlap_cnt[ki];
        if (cnt <= 0) continue;
        std::cout << "ROUTEA_AUDIT overlap@"
                  << kAuditTopKs[ki] << "="
                  << (overlap_sum[ki] / (double)cnt) << "\n";
    }
    std::cout << "ROUTEA_AUDIT exact_best_rank_in_approx_avg="
              << (exact_best_rank_in_approx_sum / (double)pair_count) << "\n";
    std::cout << "ROUTEA_AUDIT approx_best_rank_in_exact_avg="
              << (approx_best_rank_in_exact_sum / (double)pair_count) << "\n";
    std::cout << "========================================\n";
    return true;
}

template <typename T>
static inline bool run_rabitq_formula_rotation_audit(
    const IndexGraph<T>& index,
    const T* queries,
    size_t nq,
    const RabitqAuditConfig& cfg) {
    std::cout << "========================================\n";
    std::cout << "RaBitQ Formula / Rotation Audit\n";
    std::cout << "========================================\n";
    std::cout << "graph_prefix: " << index.graph_prefix << "\n";
    std::cout << "npoints: " << index.ntotal << ", dim: " << index.d
              << ", maxDeg: " << index.maxDeg << "\n";
    std::cout << "compile flags: SCORE_WORDS=" << kStableRabitqScoreWords
              << ", TOPN=" << REFLOW_TOPN << "\n";

    if (index.ntotal == 0 || index.d <= 0 || index.maxDeg <= 0 || index.data.empty() || index.edges.empty()) {
        std::cerr << "Audit failed: graph/data not loaded.\n";
        return false;
    }
    if (index.d != 128) {
        std::cerr << "Audit currently supports dim=128 only. current dim=" << index.d << "\n";
        return false;
    }

    const int W = (index.d + 31) >> 5;
    const size_t edge_count = index.ntotal * (size_t)index.maxDeg;
    const size_t expected_sign_words = edge_count * (size_t)W;
    const size_t factor_count = edge_count;

    std::vector<float> proj_steps;
    int proj_rows = 0;
    int proj_cols = 0;
    std::string proj_from;
    {
        std::vector<std::string> proj_candidates;
        const char* proj_override_env = std::getenv("RABITQ_PROJ_FILE");
        if (proj_override_env && proj_override_env[0] != '\0') {
            proj_candidates.push_back(std::string(proj_override_env));
        }
        const std::string primary = index.graph_prefix.empty()
                                      ? std::string("rabitq.proj.bin")
                                      : (index.graph_prefix + ".rabitq.proj.bin");
        const std::string fallback = "graph-d32.rabitq.proj.bin";
        for (const auto& p : {primary, fallback}) {
            if (std::find(proj_candidates.begin(), proj_candidates.end(), p) == proj_candidates.end()) {
                proj_candidates.push_back(p);
            }
        }
        for (const auto& p : proj_candidates) {
            if (rabitq_load_proj_with_header_audit(p, proj_steps, proj_rows, proj_cols)) {
                proj_from = p;
                break;
            }
        }
    }

    const bool proj_loaded = !proj_steps.empty() && proj_cols == 128 && proj_rows > 0;
    if (proj_loaded) {
        std::cout << "proj: loaded from " << proj_from
                  << " (rows=" << proj_rows << ", cols=" << proj_cols << ")\n";
    } else {
        std::cout << "proj: unavailable\n";
    }

    std::vector<std::string> sign_candidates = rabitq_signbit_candidates_audit(index.graph_prefix);
    std::vector<std::string> triple_candidates = rabitq_factor_candidates_audit(
        index.graph_prefix, "RABITQ_TRIPLE_FILE", "rabitq.triple.f32.bin");
    std::vector<std::string> dq_candidates = rabitq_factor_candidates_audit(
        index.graph_prefix, "RABITQ_FACTOR_DQ_FILE", "rabitq.factor_dq.f32.bin");
    std::vector<std::string> vq_candidates = rabitq_factor_candidates_audit(
        index.graph_prefix, "RABITQ_FACTOR_VQ_FILE", "rabitq.factor_vq.f32.bin");

    int query_samples = std::max(1, std::min((int)nq, cfg.query_samples));
    if (queries != nullptr && nq > 0 && proj_loaded) {
        for (int qi = 0; qi < query_samples; ++qi) {
            const float* q = reinterpret_cast<const float*>(queries + (size_t)qi * 128u);
            std::vector<float> qproj;
            rabitq_apply_proj_steps_audit(q, 128, proj_steps, proj_rows, proj_cols, qproj);
            float qmin = std::numeric_limits<float>::infinity();
            float qmax = -std::numeric_limits<float>::infinity();
            for (float v : qproj) {
                qmin = std::min(qmin, v);
                qmax = std::max(qmax, v);
            }
            std::cout << "query[" << qi << "] projected range: [" << qmin << ", " << qmax << "]\n";
        }
    }

    double raw_match_sum = 0.0;
    double proj_match_sum = 0.0;
    int raw_match_n = 0;
    int proj_match_n = 0;
    double triple_abs_err_sum = 0.0;
    double dq_abs_err_sum = 0.0;
    double vq_abs_err_sum = 0.0;
    double triple_abs_err_max = 0.0;
    double dq_abs_err_max = 0.0;
    double vq_abs_err_max = 0.0;
    int factor_n = 0;

    int sampled = 0;
    const size_t max_u = std::min(index.ntotal, (size_t)std::max(1, cfg.max_source_nodes));
    for (size_t u = 0; u < max_u && sampled < std::max(1, cfg.edge_samples); ++u) {
        const T* uvec = index.data.data() + u * (size_t)index.d;
        const size_t edge_base = u * (size_t)index.maxDeg;
        for (int j = 0; j < index.maxDeg && sampled < std::max(1, cfg.edge_samples); ++j) {
            const size_t edge_pos = edge_base + (size_t)j;
            const vid_t v = index.edges[edge_pos];
            if ((size_t)v >= index.ntotal) continue;
            const T* vvec = index.data.data() + (size_t)v * (size_t)index.d;

            std::vector<uint32_t> loaded_words;
            std::string sign_from;
            std::string sign_format;
            const bool have_loaded_sign = rabitq_read_signbit_edge_sample_audit(
                sign_candidates, expected_sign_words, edge_pos, W,
                loaded_words, sign_from, sign_format);
            if (!have_loaded_sign) continue;

            std::vector<uint32_t> raw_words;
            rabitq_pack_residual_sign_words_raw_audit(uvec, vvec, index.d, raw_words);
            const double raw_match = rabitq_sign_match_ratio_audit(loaded_words, raw_words, index.d);
            raw_match_sum += raw_match;
            raw_match_n++;

            double proj_match = -1.0;
            if (proj_loaded) {
                std::vector<float> residual(128, 0.0f);
                for (int d = 0; d < 128; ++d) {
                    residual[(size_t)d] = (float)vvec[d] - (float)uvec[d];
                }
                std::vector<float> residual_proj;
                rabitq_apply_proj_steps_audit(residual.data(), 128, proj_steps, proj_rows, proj_cols, residual_proj);
                std::vector<uint32_t> proj_words;
                rabitq_pack_sign_words_audit(residual_proj.data(), 128, proj_words);
                proj_match = rabitq_sign_match_ratio_audit(loaded_words, proj_words, index.d);
                proj_match_sum += proj_match;
                proj_match_n++;
            }

            float loaded_triple = 0.0f;
            float loaded_dq = 0.0f;
            float loaded_vq = 0.0f;
            std::string triple_from;
            std::string dq_from;
            std::string vq_from;
            const bool have_triple = rabitq_read_factor_edge_sample_audit(triple_candidates, factor_count, edge_pos, loaded_triple, triple_from);
            const bool have_dq = rabitq_read_factor_edge_sample_audit(dq_candidates, factor_count, edge_pos, loaded_dq, dq_from);
            const bool have_vq = rabitq_read_factor_edge_sample_audit(vq_candidates, factor_count, edge_pos, loaded_vq, vq_from);
            if (have_triple && have_dq && have_vq) {
                float raw_triple = 0.0f;
                float raw_dq = 0.0f;
                float raw_vq = 0.0f;
                rabitq_compute_raw_factor_edge_audit(uvec, vvec, index.d, raw_triple, raw_dq, raw_vq);
                const double triple_err = std::fabs((double)loaded_triple - (double)raw_triple);
                const double dq_err = std::fabs((double)loaded_dq - (double)raw_dq);
                const double vq_err = std::fabs((double)loaded_vq - (double)raw_vq);
                triple_abs_err_sum += triple_err;
                dq_abs_err_sum += dq_err;
                vq_abs_err_sum += vq_err;
                triple_abs_err_max = std::max(triple_abs_err_max, triple_err);
                dq_abs_err_max = std::max(dq_abs_err_max, dq_err);
                vq_abs_err_max = std::max(vq_abs_err_max, vq_err);
                factor_n++;
                if (sampled < std::max(0, cfg.verbose_edges)) {
                    std::cout << std::fixed << std::setprecision(6)
                              << "edge_sample[" << sampled << "] u=" << u
                              << " v=" << v
                              << " sign(raw=" << raw_match
                              << ", proj=" << proj_match << ")"
                              << " factor_err(triple=" << triple_err
                              << ", dq=" << dq_err
                              << ", vq=" << vq_err << ")\n";
                }
            } else if (sampled < std::max(0, cfg.verbose_edges)) {
                std::cout << std::fixed << std::setprecision(6)
                          << "edge_sample[" << sampled << "] u=" << u
                          << " v=" << v
                          << " sign(raw=" << raw_match
                          << ", proj=" << proj_match << ")"
                          << " factor_err(unavailable)\n";
            }

            if (sampled == 0) {
                std::cout << "signbit sample source: " << sign_from << " (format=" << sign_format << ")\n";
                if (have_triple) std::cout << "factor triple source: " << triple_from << "\n";
                if (have_dq) std::cout << "factor dq source: " << dq_from << "\n";
                if (have_vq) std::cout << "factor vq source: " << vq_from << "\n";
            }
            sampled++;
        }
    }

    if (sampled == 0) {
        std::cerr << "Audit failed: no valid sampled edges found.\n";
        return false;
    }

    const double raw_match_avg = (raw_match_n > 0) ? (raw_match_sum / (double)raw_match_n) : 0.0;
    const double proj_match_avg = (proj_match_n > 0) ? (proj_match_sum / (double)proj_match_n) : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "AUDIT sampled_edges=" << sampled << "\n";
    std::cout << "AUDIT sign_match_raw_avg=" << raw_match_avg << "\n";
    if (proj_match_n > 0) {
        std::cout << "AUDIT sign_match_proj_avg=" << proj_match_avg << "\n";
    } else {
        std::cout << "AUDIT sign_match_proj_avg=NA\n";
    }
    if (factor_n > 0) {
        std::cout << "AUDIT factor_raw_abs_err_avg=(triple="
                  << (triple_abs_err_sum / (double)factor_n)
                  << ", dq=" << (dq_abs_err_sum / (double)factor_n)
                  << ", vq=" << (vq_abs_err_sum / (double)factor_n) << ")\n";
        std::cout << "AUDIT factor_raw_abs_err_max=(triple="
                  << triple_abs_err_max
                  << ", dq=" << dq_abs_err_max
                  << ", vq=" << vq_abs_err_max << ")\n";
    } else {
        std::cout << "AUDIT factor_raw_abs_err=NA\n";
    }

    if (proj_loaded && raw_match_avg > 0.98 && proj_match_avg + 0.05 < raw_match_avg) {
        std::cout << "AUDIT verdict: loaded signbits look raw-residual-space, not projected-space.\n";
    } else if (proj_loaded && proj_match_avg > 0.98 && raw_match_avg + 0.05 < proj_match_avg) {
        std::cout << "AUDIT verdict: loaded signbits look projected-space.\n";
    } else if (proj_loaded) {
        std::cout << "AUDIT verdict: signbit space is ambiguous from current samples.\n";
    } else {
        std::cout << "AUDIT verdict: projection file unavailable; only raw-space consistency was checked.\n";
    }

    if (factor_n > 0 && triple_abs_err_max < 1e-3 && dq_abs_err_max < 1e-3 && vq_abs_err_max < 1e-3) {
        std::cout << "AUDIT factor verdict: loaded factor files match raw-space builder on sampled edges.\n";
    } else if (factor_n > 0) {
        std::cout << "AUDIT factor verdict: loaded factor files do not closely match the raw-space builder on sampled edges.\n";
    } else {
        std::cout << "AUDIT factor verdict: factor files unavailable for sampled audit.\n";
    }

    std::cout << "========================================\n";
    return true;
}
