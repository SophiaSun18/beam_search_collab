#pragma once

#include "../code/beam_gpu_host_sidecars.hpp"
#include "index.hpp"

#include <chrono>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct RabitqEdgeSidecarBuildConfig {
    std::string signbit_output_path;
    std::string triple_output_path;
    std::string factor_dq_output_path;
    std::string factor_vq_output_path;
    bool overwrite = true;
};

template <typename T>
static inline void build_rabitq_factors_host_build(
    const T* data,
    size_t npoints,
    int dim,
    const std::vector<vid_t>& edges,
    int max_degree,
    std::vector<float>& triple,
    std::vector<float>& factor_dq,
    std::vector<float>& factor_vq) {
    const size_t total_edges = npoints * static_cast<size_t>(max_degree);
    triple.assign(total_edges, FLT_MAX);
    factor_dq.assign(total_edges, 0.0f);
    factor_vq.assign(total_edges, 0.0f);
    if (data == nullptr || npoints == 0 || dim <= 0 || max_degree <= 0) {
        return;
    }

    const float fac_norm = 1.0f / std::sqrt(static_cast<float>(dim));
    const float eps = 1e-12f;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
#endif
    for (int64_t uu = 0; uu < static_cast<int64_t>(npoints); ++uu) {
        const size_t u = static_cast<size_t>(uu);
        const float* uvec = reinterpret_cast<const float*>(data) + u * static_cast<size_t>(dim);
        const size_t edge_base = u * static_cast<size_t>(max_degree);
        for (int j = 0; j < max_degree; ++j) {
            const size_t edge_pos = edge_base + static_cast<size_t>(j);
            const vid_t v = edges[edge_pos];
            if (static_cast<size_t>(v) >= npoints) {
                continue;
            }

            const float* vvec =
                reinterpret_cast<const float*>(data) + static_cast<size_t>(v) * static_cast<size_t>(dim);
            float norm2 = 0.0f;
            float dot_x_obar = 0.0f;
            float dot_u_obar = 0.0f;
            int sign_sum = 0;

            for (int d = 0; d < dim; ++d) {
                const float r = vvec[d] - uvec[d];
                norm2 = std::fmaf(r, r, norm2);
                const float s = (r >= 0.0f) ? 1.0f : -1.0f;
                dot_x_obar += r * s;
                dot_u_obar += uvec[d] * s;
                sign_sum += (r >= 0.0f) ? 1 : -1;
            }

            const float norm = std::sqrt(norm2);
            if (norm <= eps || dot_x_obar <= eps) {
                triple[edge_pos] = norm2;
                continue;
            }

            const float fac_x0 = (dot_x_obar * fac_norm) / norm;
            if (std::fabs(fac_x0) <= eps) {
                triple[edge_pos] = norm2;
                continue;
            }

            const float x_div_x0 = norm / fac_x0;
            const float fac_x1 = dot_u_obar * fac_norm;
            triple[edge_pos] = norm2 + (2.0f * x_div_x0 * fac_x1);
            factor_dq[edge_pos] = -2.0f * x_div_x0 * fac_norm;
            factor_vq[edge_pos] = -2.0f * x_div_x0 * fac_norm * static_cast<float>(sign_sum);
        }
    }
}

template <typename T>
static inline bool run_rabitq_edge_sidecar_build(
    const IndexGraph<T>& index,
    const RabitqEdgeSidecarBuildConfig& cfg) {
    std::cout << "========================================\n";
    std::cout << "RaBitQ Edge Sidecar Build\n";
    std::cout << "========================================\n";
    std::cout << "graph_prefix: " << index.graph_prefix << "\n";
    std::cout << "npoints: " << index.ntotal << ", dim: " << index.d
              << ", maxDeg: " << index.maxDeg << "\n";

    if (index.ntotal == 0 || index.d <= 0 || index.maxDeg <= 0 ||
        index.data.empty() || index.edges.empty()) {
        std::cerr << "Edge sidecar build failed: graph/data not loaded.\n";
        return false;
    }

    const std::string signbit_out = cfg.signbit_output_path.empty()
        ? (index.graph_prefix + ".signbit.bin")
        : cfg.signbit_output_path;
    const std::string triple_out = cfg.triple_output_path.empty()
        ? (index.graph_prefix + ".rabitq.triple.f32.bin")
        : cfg.triple_output_path;
    const std::string dq_out = cfg.factor_dq_output_path.empty()
        ? (index.graph_prefix + ".rabitq.factor_dq.f32.bin")
        : cfg.factor_dq_output_path;
    const std::string vq_out = cfg.factor_vq_output_path.empty()
        ? (index.graph_prefix + ".rabitq.factor_vq.f32.bin")
        : cfg.factor_vq_output_path;

    const auto must_not_exist = [&](const std::string& path) -> bool {
        if (cfg.overwrite) return false;
        std::ifstream probe(path, std::ios::binary);
        return probe.good();
    };
    if (must_not_exist(signbit_out) || must_not_exist(triple_out) ||
        must_not_exist(dq_out) || must_not_exist(vq_out)) {
        std::cerr << "Edge sidecar build refused to overwrite existing outputs.\n";
        return false;
    }

    std::cout << "signbit_out: " << signbit_out << "\n";
    std::cout << "triple_out: " << triple_out << "\n";
    std::cout << "factor_dq_out: " << dq_out << "\n";
    std::cout << "factor_vq_out: " << vq_out << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> signbit =
        beam_gpu_host::build_dir_sign_bits_host(
            reinterpret_cast<const float*>(index.data.data()),
            index.ntotal,
            index.d,
            index.edges,
            index.maxDeg);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> triple;
    std::vector<float> factor_dq;
    std::vector<float> factor_vq;
    build_rabitq_factors_host_build(
        index.data.data(),
        index.ntotal,
        index.d,
        index.edges,
        index.maxDeg,
        triple,
        factor_dq,
        factor_vq);
    auto t2 = std::chrono::high_resolution_clock::now();

    const bool ok_sign =
        beam_gpu_host::save_u32_vector_raw(signbit_out.c_str(), signbit.data(), signbit.size());
    const bool ok_triple =
        beam_gpu_host::save_float_vector_raw(triple_out.c_str(), triple.data(), triple.size());
    const bool ok_dq =
        beam_gpu_host::save_float_vector_raw(dq_out.c_str(), factor_dq.data(), factor_dq.size());
    const bool ok_vq =
        beam_gpu_host::save_float_vector_raw(vq_out.c_str(), factor_vq.data(), factor_vq.size());
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "signbit_words: " << signbit.size() << "\n";
    std::cout << "factor_count: " << triple.size() << "\n";
    std::cout << "signbit_build_sec: "
              << std::chrono::duration<double>(t1 - t0).count() << "\n";
    std::cout << "factor_build_sec: "
              << std::chrono::duration<double>(t2 - t1).count() << "\n";
    std::cout << "save_sec: "
              << std::chrono::duration<double>(t3 - t2).count() << "\n";
    std::cout << "save_signbit: " << (ok_sign ? "OK" : "FAILED") << "\n";
    std::cout << "save_triple: " << (ok_triple ? "OK" : "FAILED") << "\n";
    std::cout << "save_factor_dq: " << (ok_dq ? "OK" : "FAILED") << "\n";
    std::cout << "save_factor_vq: " << (ok_vq ? "OK" : "FAILED") << "\n";
    std::cout << "========================================\n";
    return ok_sign && ok_triple && ok_dq && ok_vq;
}
