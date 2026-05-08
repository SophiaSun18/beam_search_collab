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

RUN_SOPHIA = r'''
// ── run_sophia(): entry point callable from merged gpu_cli ─────────────────

#include <nlohmann/json.hpp>

int run_sophia(const nlohmann::json& j) {
    auto& jd = j.at("data");
    std::string base_file  = jd.at("base");
    std::string query_file = jd.at("query");
    std::string gt_file    = jd.at("gt");
    std::string graph_file = jd.at("graph");

    auto& js = j.at("search");
    std::string mode    = js.at("mode");
    int K               = js.value("K", 100);
    int beam_size       = js.value("beam", 128);
    int degree          = js.value("degree", 32);
    std::string csv_file = js.value("csv", "");
    bool debug_mode     = js.value("debug", false);

    // strip "sophia_" prefix for internal dispatch
    if (mode.rfind("sophia_", 0) == 0) mode = mode.substr(7);

    std::cout << "search: mode=sophia_" << mode << " K=" << K << std::endl;

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


        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file.c_str(), gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, graph_file.c_str());
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);

    } else {
        std::cerr << "Unknown sophia mode: " << mode << "\n";
        return 1;
    }

    std::cout << "  time: " << elapsed * 1000.0 << " ms"
              << "  qps: " << nq / elapsed << std::endl;
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k);
    std::cout << "recall@" << K << ": " << recall << std::endl;

    return 0;
}
'''

def main():
    parts = []
    parts.append("// legacy_sophia.cu — Flattened from sophia branch")
    parts.append("// All sophia headers and sources inlined into a single compilation unit.")
    parts.append("// Exposes run_sophia(json) for use by the merged gpu_cli.\n")

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
    parts.append("// ── run_sophia() entry point ──")
    parts.append("// ═══════════════════════════════════════════════════════════════════════")
    parts.append(RUN_SOPHIA)

    print('\n'.join(parts))

if __name__ == "__main__":
    main()
