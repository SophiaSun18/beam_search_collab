#pragma once

#include "index.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct RabitqRawSignbitBuildConfig {
    std::string output_path;
    bool overwrite = true;
};

static inline bool rabitq_save_sign_bit_raw_build(
    const std::string& path,
    const uint32_t* data,
    size_t words) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(words * sizeof(uint32_t)));
    return static_cast<bool>(f);
}

template <typename T>
static inline bool run_rabitq_raw_signbit_build(
    const IndexGraph<T>& index,
    const RabitqRawSignbitBuildConfig& cfg) {
    std::cout << "========================================\n";
    std::cout << "RaBitQ Raw Signbit Build\n";
    std::cout << "========================================\n";
    std::cout << "graph_prefix: " << index.graph_prefix << "\n";
    std::cout << "npoints: " << index.ntotal << ", dim: " << index.d
              << ", maxDeg: " << index.maxDeg << "\n";

    if (index.ntotal == 0 || index.d <= 0 || index.maxDeg <= 0 ||
        index.data.empty() || index.edges.empty()) {
        std::cerr << "Raw signbit build failed: graph/data not loaded.\n";
        return false;
    }

    const std::string out_path = cfg.output_path.empty()
        ? (index.graph_prefix + ".signbit.bin")
        : cfg.output_path;
    if (out_path.empty()) {
        std::cerr << "Raw signbit build failed: empty output path.\n";
        return false;
    }
    if (!cfg.overwrite) {
        std::ifstream probe(out_path, std::ios::binary);
        if (probe.good()) {
            std::cerr << "Raw signbit build refused to overwrite existing file: "
                      << out_path << "\n";
            return false;
        }
    }

    const int W = (index.d + 31) >> 5;
    const size_t edge_count = index.ntotal * (size_t)index.maxDeg;
    const size_t words = edge_count * (size_t)W;
    const size_t bytes = words * sizeof(uint32_t);
    std::cout << "output_path: " << out_path << "\n";
    std::cout << "words_per_edge: " << W << ", total_words: " << words
              << ", total_bytes: " << bytes << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> out(words, 0u);
    for (size_t u = 0; u < index.ntotal; ++u) {
        const T* uvec = index.data.data() + u * (size_t)index.d;
        const size_t edge_base = u * (size_t)index.maxDeg;
        for (int j = 0; j < index.maxDeg; ++j) {
            const size_t edge_pos = edge_base + (size_t)j;
            const vid_t v = index.edges[edge_pos];
            uint32_t* dst = out.data() + edge_pos * (size_t)W;
            if ((size_t)v >= index.ntotal) {
                std::memset(dst, 0, (size_t)W * sizeof(uint32_t));
                continue;
            }
            const T* vvec = index.data.data() + (size_t)v * (size_t)index.d;
            for (int w = 0; w < W; ++w) {
                uint32_t word = 0u;
                const int base_dim = w << 5;
                for (int b = 0; b < 32; ++b) {
                    const int i = base_dim + b;
                    if (i >= index.d) break;
                    const float diff = (float)vvec[i] - (float)uvec[i];
                    const uint32_t bit = (diff >= 0.0f) ? 1u : 0u;
                    word |= (bit << b);
                }
                dst[w] = word;
            }
        }
        if (((u + 1) % 1000000u) == 0u || (u + 1) == index.ntotal) {
            std::cout << "build progress: " << (u + 1) << "/" << index.ntotal << " source nodes\n";
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    const bool ok = rabitq_save_sign_bit_raw_build(out_path, out.data(), out.size());
    auto t2 = std::chrono::high_resolution_clock::now();
    const double build_sec = std::chrono::duration<double>(t1 - t0).count();
    const double save_sec = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "build_time_sec: " << build_sec << "\n";
    std::cout << "save_time_sec: " << save_sec << "\n";
    std::cout << "save_status: " << (ok ? "OK" : "FAILED") << "\n";
    std::cout << "========================================\n";
    return ok;
}
