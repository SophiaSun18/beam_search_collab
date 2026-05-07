#!/usr/bin/env python3
"""Flatten sophia's GPU search code from git branch into a single .cu file with JSON stdin CLI."""

import subprocess
import re
import sys

def git_show(path):
    r = subprocess.run(["git", "show", f"sophia:{path}"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARNING: cannot read sophia:{path}", file=sys.stderr)
        return None
    return r.stdout

HEADERS_IN_ORDER = [
    "include/utils.h",
    "include/utils.cuh",
    "include/topk.cuh",
    "include/distance.cuh",
    "include/hash_filter.hpp",
    "include/pqueue.hpp",
    "include/timer.hpp",
    "include/ctimer.h",
    "include/index.hpp",
    "include/search_bitonic.cuh",
    "include/search_rabitq.cuh",
    # search_tensor_rabitq.cuh: disabled (USE_TENSOR_RABITQ 0), has undefined symbols
    # search_tensor_bitonic.cuh: doesn't exist on sophia branch
    # fastscan_impl.hpp: only used in !__CUDACC__ path of qg.hpp
    "include/qg.hpp",
]

SOURCES_IN_ORDER = [
    "src/beam_gpu_base.cu",
    "src/beam_gpu_rabitq.cu",
]

LOCAL_INCLUDE_RE = re.compile(r'^\s*#include\s+"[^<].*\.(h|hpp|cuh)"')
PRAGMA_ONCE_RE = re.compile(r'^\s*#pragma\s+once')
SYSTEM_INCLUDE_RE = re.compile(r'^\s*#include\s+<(.+?)>')

# These includes must stay in their original guarded positions (not nvcc-safe at top level)
GUARDED_INCLUDES = {"immintrin.h", "omp.h"}

def process_header(text, path=""):
    """Strip #pragma once and local #includes. Keep system includes in place."""
    lines = text.split('\n')
    out = []
    # Rename duplicate calculate_shared_mem_size in search_rabitq.cuh
    rename_shm = (path == "include/search_rabitq.cuh")
    for line in lines:
        if PRAGMA_ONCE_RE.match(line):
            continue
        if LOCAL_INCLUDE_RE.match(line):
            out.append(f"// [flattened] {line.strip()}")
            continue
        if rename_shm and 'calculate_shared_mem_size' in line and 'calculate_shared_mem_size_tensor' not in line:
            line = line.replace('calculate_shared_mem_size', 'calculate_shared_mem_size_rabitq')
        out.append(line)
    return '\n'.join(out)

def process_source(text, path=""):
    """Strip local includes, missing headers, and tensor code paths."""
    lines = text.split('\n')
    out = []
    skip_tensor = False
    for line in lines:
        if LOCAL_INCLUDE_RE.match(line):
            out.append(f"// [flattened] {line.strip()}")
            continue
        if '#include "include/search_tensor_bitonic.cuh"' in line:
            out.append(f"// [flattened, file missing] {line.strip()}")
            continue
        # In beam_gpu_rabitq.cu, rename calculate_shared_mem_size to rabitq variant
        if path == "src/beam_gpu_rabitq.cu":
            if 'calculate_shared_mem_size' in line and 'calculate_shared_mem_size_tensor' not in line:
                line = line.replace('calculate_shared_mem_size', 'calculate_shared_mem_size_rabitq')
        out.append(line)
    # Strip USE_TENSOR_BITONIC / USE_TENSOR_RABITQ code paths by forcing them to 0
    result = '\n'.join(out)
    if path == "src/beam_gpu_base.cu":
        result = "#undef USE_TENSOR_BITONIC\n#define USE_TENSOR_BITONIC 0\n" + result
    if path == "src/beam_gpu_rabitq.cu":
        result = "#undef USE_TENSOR_RABITQ\n#define USE_TENSOR_RABITQ 0\n" + result
    return result

JSON_MAIN = r'''
// ── JSON stdin CLI (replaces original argv parser) ─────────────────────────

#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main() {
    json j;
    try {
        j = json::parse(std::cin, nullptr, true, true);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }

    if (!j.contains("data")) {
        std::cerr << "Missing \"data\" key" << std::endl;
        return 1;
    }
    auto& jd = j["data"];
    std::string base_file  = jd.at("base");
    std::string query_file = jd.at("query");
    std::string gt_file    = jd.at("gt");
    std::string graph_file = jd.at("graph");

    if (!j.contains("search")) {
        std::cerr << "Missing \"search\" key" << std::endl;
        return 1;
    }
    auto& js = j["search"];
    std::string mode    = js.at("mode");
    int K               = js.value("K", 100);
    int beam_size       = js.value("beam", 128);
    int degree          = js.value("degree", 32);
    std::string csv_file = js.value("csv", "");
    bool debug_mode     = js.value("debug", false);

    std::cout << "========================================\n"
         << "GPU Beam Search (Sophia)\n"
         << "========================================\n"
         << "Mode:    " << mode << "\n"
         << "Base:    " << base_file << "\n"
         << "Query:   " << query_file << "\n"
         << "GT:      " << gt_file << "\n"
         << "Graph:   " << graph_file << "\n"
         << "K:       " << K << "\n"
         << "Beam:    " << beam_size << "\n";
    if (mode == "rabitq") std::cout << "Degree:  " << degree << "\n";
    if (!csv_file.empty())    std::cout << "CSV:     " << csv_file << "\n";
    if (debug_mode)           std::cout << "Debug:   on\n";
    std::cout << "========================================\n\n";

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
        if (nq > 10000) { std::cout << "Limiting queries to 10000\n"; nq = 10000; }

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
        if (nq > 10000) { std::cout << "Limiting queries to 10000\n"; nq = 10000; }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file.c_str(), gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, graph_file.c_str());
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);

    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k);

    std::cout << "\n========================================\n"
         << "Results\n"
         << "========================================\n"
         << "Total time:  " << elapsed * 1000 << " ms\n"
         << "Throughput:  " << qps << " queries/sec\n"
         << "Avg latency: " << latency_ms << " ms/query\n"
         << "Recall@" << K << ":   " << recall << "\n"
         << "========================================\n";

    gpu_stats.common.runtime = elapsed;
    gpu_stats.common.latency = latency_ms;
    gpu_stats.common.throughput = qps;
    gpu_stats.common.recall = recall;

    if (!csv_file.empty()) {
        append_gpu_stats_to_csv(csv_file, K, beam_size, gpu_stats);
        std::cout << "Saved GPU stats CSV row to: " << csv_file << "\n";
    }

    return 0;
}
'''

def main():
    parts = []
    parts.append("// gpu_cli_sophia.cu — Flattened from sophia branch")
    parts.append("// All sophia headers and sources inlined into a single compilation unit.")
    parts.append("// CLI reads JSON config from stdin.\n")

    # Emit headers in dependency order — keep system includes in their original context
    for path in HEADERS_IN_ORDER:
        text = git_show(path)
        if text is None:
            parts.append(f"// WARNING: {path} not found on sophia branch\n")
            continue
        parts.append(f"// ═══════════════════════════════════════════════════════════════════════")
        parts.append(f"// ── {path} (inlined) ──")
        parts.append(f"// ═══════════════════════════════════════════════════════════════════════\n")
        parts.append(process_header(text, path))
        parts.append("")

    # Stubs for missing tensor code (USE_TENSOR_BITONIC=0, USE_TENSOR_RABITQ=0)
    parts.append("// ═══════════════════════════════════════════════════════════════════════")
    parts.append("// ── Stubs for disabled tensor code paths ──")
    parts.append("// ═══════════════════════════════════════════════════════════════════════\n")
    parts.append("""
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
""")

    # Emit source files
    for path in SOURCES_IN_ORDER:
        text = git_show(path)
        if text is None:
            parts.append(f"// WARNING: {path} not found on sophia branch\n")
            continue
        parts.append(f"// ═══════════════════════════════════════════════════════════════════════")
        parts.append(f"// ── {path} (inlined) ──")
        parts.append(f"// ═══════════════════════════════════════════════════════════════════════\n")
        parts.append(process_source(text, path))
        parts.append("")

    # Emit the JSON main
    parts.append("// ═══════════════════════════════════════════════════════════════════════")
    parts.append("// ── JSON stdin CLI ──")
    parts.append("// ═══════════════════════════════════════════════════════════════════════")
    parts.append(JSON_MAIN)

    print('\n'.join(parts))

if __name__ == "__main__":
    main()
