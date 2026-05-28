// Q-LEAF standalone driver. Two GPU modes:
//   gpu_base  -> exact GPU beam search (beam_gpu_base.cu)
//   gpu_rabitq_bitplane_navl2_prune_es_spec_auto -> full Q-LEAF (T6) GPU beam search
//
// Diagnostic modes from the qleaf_ablation_ladder (rabitq-audit,
// routea-audit, sidecar build) are intentionally not exposed here; the
// standalone tree only carries the formal benchmark surface.

#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>
#include <unordered_map>

#include "index.hpp"
#include "qleaf_config.cuh"
#include "qleaf_sidecars.hpp"

static bool parse_int_arg(int argc, char** argv, int& i, const char* key, int& out) {
    const std::string a(argv[i]);
    const std::string k(key);
    if (a == k) {
        if (i + 1 >= argc) return false;
        out = std::atoi(argv[i + 1]);
        i += 1;
        return true;
    }
    const std::string prefix = k + "=";
    if (a.rfind(prefix, 0) == 0) {
        out = std::atoi(a.c_str() + prefix.size());
        return true;
    }
    return false;
}

float compute_recall(const vid_t* results, const int* groundtruth,
                     int nq, int k, int gt_k) {
    if (results == nullptr || groundtruth == nullptr || nq <= 0 || k <= 0 || gt_k <= 0) {
        return 0.0f;
    }

    const int eval_k = std::min(k, gt_k);
    int64_t correct = 0;
    std::vector<uint8_t> gt_matched((size_t)eval_k, 0);

    for (int q = 0; q < nq; q++) {
        std::fill(gt_matched.begin(), gt_matched.end(), 0);
        for (int i = 0; i < k; i++) {
            int res_id = results[q * k + i];
            for (int j = 0; j < eval_k; j++) {
                if (!gt_matched[j] && res_id == groundtruth[q * gt_k + j]) {
                    gt_matched[j] = 1;
                    correct++;
                    break;
                }
            }
        }
    }

    return static_cast<float>(correct) / static_cast<float>((int64_t)nq * eval_k);
}

float compute_recall_legacy(const vid_t* results, const int* groundtruth,
                            int nq, int k, int gt_k) {
    if (results == nullptr || groundtruth == nullptr || nq <= 0 || k <= 0 || gt_k <= 0) {
        return 0.0f;
    }

    const int eval_k = std::min(k, gt_k);
    int64_t correct = 0;

    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            int res_id = static_cast<int>(results[q * k + i]);
            for (int j = 0; j < eval_k; j++) {
                if (res_id == groundtruth[q * gt_k + j]) {
                    correct++;
                    break;
                }
            }
        }
    }

    return static_cast<float>(correct) / static_cast<float>((int64_t)nq * eval_k);
}

double compute_avg_unique_at_k(const vid_t* results, int nq, int k) {
    if (results == nullptr || nq <= 0 || k <= 0) {
        return 0.0;
    }

    std::vector<int> row((size_t)k);
    int64_t total_unique = 0;
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            row[(size_t)i] = static_cast<int>(results[q * k + i]);
        }
        std::sort(row.begin(), row.end());
        const auto it = std::unique(row.begin(), row.end());
        total_unique += static_cast<int64_t>(std::distance(row.begin(), it));
    }

    return static_cast<double>(total_unique) / static_cast<double>(nq);
}

enum class GpuMode {
    Base,
    RabitqBitplaneNavL2PruneEsSpecAuto,
};

struct MainOptions {
    GpuMode mode = GpuMode::Base;
    int warmup = 0;
    int runs = 1;
    const char* data_file = nullptr;
    const char* query_file = nullptr;
    const char* gt_file = nullptr;
    const char* graph_prefix = nullptr;
    int K = 100;
    int beam_size = BEAM_SIZE;
};

struct LoadedQueries {
    std::vector<float> values;
    size_t nq = 0;
};

struct GroundtruthEval {
    std::vector<int> groundtruth;
    size_t recall_nq = 0;
    int gt_k = 0;
    int recall_k = 0;
};

struct BenchmarkStats {
    double elapsed = 0.0;
    double qps = 0.0;
    double latency_ms = 0.0;
    float recall = 0.0f;
    float recall_legacy = 0.0f;
    double avg_unique_at_k = 0.0;
    double duplicate_ratio = 0.0;
};

static int print_usage(const char* prog, int code) {
    std::cerr << "Usage: " << prog
              << " <mode:gpu_base|gpu_rabitq_bitplane_navl2_prune_es_spec_auto> <data.fvecs> <query.fvecs> <gt.ivecs> <graph_or_qg>"
              << " [K=100] [beam_size=" << BEAM_SIZE << "]"
              << " [--warmup N=0] [--runs N=1]\n";
    return code;
}

static bool parse_mode(const char* s, GpuMode& out) {
    const std::string v(s);
    if (v == "gpu_base") {
        out = GpuMode::Base;
        return true;
    }
    if (v == "gpu_rabitq_bitplane_navl2_prune_es_spec_auto") {
        out = GpuMode::RabitqBitplaneNavL2PruneEsSpecAuto;
        return true;
    }
    return false;
}

static int parse_main_options(int argc, char** argv, MainOptions& opts) {
    std::vector<const char*> pos;
    pos.reserve((size_t)argc);
    for (int i = 1; i < argc; i++) {
        if (parse_int_arg(argc, argv, i, "--warmup", opts.warmup)) continue;
        if (parse_int_arg(argc, argv, i, "--runs", opts.runs)) continue;
        const std::string a(argv[i]);
        if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n";
            return print_usage(argv[0], 1);
        }
        pos.push_back(argv[i]);
    }

    if (opts.warmup < 0) opts.warmup = 0;
    if (opts.runs < 1) opts.runs = 1;

    if (pos.size() < 5) {
        return print_usage(argv[0], 1);
    }
    if (!parse_mode(pos[0], opts.mode)) {
        std::cerr << "Unknown mode: " << pos[0]
                  << " (expected gpu_base or gpu_rabitq_bitplane_navl2_prune_es_spec_auto)\n";
        return print_usage(argv[0], 1);
    }
    opts.data_file = pos[1];
    opts.query_file = pos[2];
    opts.gt_file = pos[3];
    opts.graph_prefix = pos[4];
    opts.K = (pos.size() >= 6) ? std::atoi(pos[5]) : 100;
    opts.beam_size = (pos.size() >= 7) ? std::atoi(pos[6]) : BEAM_SIZE;

    if (opts.K <= 0) {
        std::cerr << "Invalid K: " << opts.K << " (must be > 0)\n";
        return print_usage(argv[0], 1);
    }
    if (opts.beam_size <= 0 || opts.beam_size > BEAM_SIZE) {
        std::cerr << "Invalid beam_size: " << opts.beam_size
                  << " (must be in [1, " << BEAM_SIZE << "])\n";
        return print_usage(argv[0], 1);
    }
    if (opts.K > opts.beam_size) {
        std::cerr << "Invalid config: K=" << opts.K
                  << " exceeds beam_size=" << opts.beam_size << "\n";
        return print_usage(argv[0], 1);
    }
    return 0;
}

static const char* mode_name(GpuMode m) {
    switch (m) {
        case GpuMode::Base: return "gpu_base";
        case GpuMode::RabitqBitplaneNavL2PruneEsSpecAuto: return "gpu_rabitq_bitplane_navl2_prune_es_spec_auto";
    }
    return "unknown";
}

static void print_run_banner(const MainOptions& opts) {
    std::cout << "========================================\n";
    std::cout << "Q-LEAF Standalone GPU Beam Search\n";
    std::cout << "========================================\n";
    std::cout << "Mode: " << mode_name(opts.mode) << "\n";
    std::cout << "Data file: " << opts.data_file << "\n";
    std::cout << "Query file: " << opts.query_file << "\n";
    std::cout << "Ground truth: " << opts.gt_file << "\n";
    std::cout << "Graph prefix: " << opts.graph_prefix << "\n";
    std::cout << "K: " << opts.K << "\n";
    std::cout << "Beam size: " << opts.beam_size
              << " (compile-time max: " << BEAM_SIZE << ")\n";
    std::cout << "Warmup runs: " << opts.warmup << "\n";
    std::cout << "Timed runs: " << opts.runs << "\n";
    std::cout << "========================================\n\n";
}

static bool validate_loaded_queries(
    const IndexGraph<float>& index,
    const char* query_file,
    size_t query_nq,
    int query_dim) {
    if (query_dim != index.d) {
        std::cerr << "Dimension mismatch: query dim=" << query_dim
                  << ", index dim=" << index.d << "\n";
        return false;
    }
    if (query_nq == 0) {
        std::cerr << "No queries loaded from file: " << query_file << "\n";
        return false;
    }
    return true;
}

static size_t resolve_query_cap() {
    size_t query_cap = 10000;
    if (const char* cap_env = std::getenv("MAX_QUERIES")) {
        const long long v = std::atoll(cap_env);
        if (v > 0) {
            query_cap = static_cast<size_t>(v);
        }
    }
    return query_cap;
}

static void apply_query_cap(LoadedQueries& loaded, size_t query_cap) {
    if (loaded.nq > query_cap) {
        std::cout << "Limiting queries to " << query_cap << "\n";
        loaded.nq = query_cap;
    }
}

static int load_runtime_queries(
    const IndexGraph<float>& index,
    const char* query_file,
    LoadedQueries& loaded) {
    int query_dim = 0;
    std::cout << "Loading queries...\n";
    loaded.values = IndexGraph<float>::load_queries(query_file, loaded.nq, query_dim);
    if (!validate_loaded_queries(index, query_file, loaded.nq, query_dim)) {
        return 1;
    }
    apply_query_cap(loaded, resolve_query_cap());
    return 0;
}

static int load_groundtruth_eval(
    const MainOptions& opts,
    size_t query_nq,
    GroundtruthEval& eval) {
    size_t gt_nq = 0;
    std::cout << "Loading ground truth...\n";
    eval.groundtruth = IndexGraph<float>::load_groundtruth(opts.gt_file, gt_nq, eval.gt_k);
    if (eval.gt_k <= 0 || gt_nq == 0) {
        std::cerr << "Invalid ground truth: nq=" << gt_nq << ", k=" << eval.gt_k << "\n";
        return 1;
    }

    eval.recall_nq = std::min(query_nq, gt_nq);
    if (eval.recall_nq < query_nq) {
        std::cout << "Warning: ground truth queries (" << gt_nq
                  << ") < search queries (" << query_nq
                  << "), recall will use first " << eval.recall_nq << " queries\n";
    }
    eval.recall_k = std::min(opts.K, eval.gt_k);
    if (eval.recall_k < opts.K) {
        std::cout << "Warning: ground truth top-k=" << eval.gt_k
                  << " < requested K=" << opts.K
                  << ", reporting Recall@" << eval.recall_k << "\n";
    }
    return 0;
}

static void print_benchmark_dataset_summary(
    const IndexGraph<float>& index,
    size_t query_nq,
    int gt_k) {
    std::cout << "\n";
    std::cout << "Data points: " << index.ntotal << "\n";
    std::cout << "Dimension: " << index.d << "\n";
    std::cout << "Max degree: " << index.maxDeg << "\n";
    std::cout << "Queries: " << query_nq << "\n";
    std::cout << "GT K: " << gt_k << "\n";
    std::cout << "\n";
}

static void invoke_search(IndexGraph<float>& index,
                          GpuMode mode,
                          int nq,
                          const float* queries,
                          int K,
                          vid_t* results,
                          int beam_sz) {
    switch (mode) {
        case GpuMode::Base:
            index.search_gpu_base(nq, queries, K, results, beam_sz);
            return;
        case GpuMode::RabitqBitplaneNavL2PruneEsSpecAuto:
            index.search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto(nq, queries, K, results, beam_sz);
            return;
    }
}

static double run_timed_benchmark_search(
    IndexGraph<float>& index,
    const MainOptions& opts,
    const LoadedQueries& loaded_queries,
    std::vector<vid_t>& results) {
    for (int w = 0; w < opts.warmup; w++) {
        std::cout << "Warmup run " << (w + 1) << "/" << opts.warmup << "...\n";
        invoke_search(index, opts.mode, (int)loaded_queries.nq,
                      loaded_queries.values.data(), opts.K,
                      results.data(), opts.beam_size);
    }

    std::cout << "Starting GPU beam search (timed)...\n";
    double elapsed = 0.0;
    for (int r = 0; r < opts.runs; r++) {
        const auto start = std::chrono::high_resolution_clock::now();
        invoke_search(index, opts.mode, (int)loaded_queries.nq,
                      loaded_queries.values.data(), opts.K,
                      results.data(), opts.beam_size);
        const auto end = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration<double>(end - start).count();
    }
    return elapsed / (double)opts.runs;
}

static BenchmarkStats compute_benchmark_stats(
    const MainOptions& opts,
    const std::vector<vid_t>& results,
    const GroundtruthEval& eval,
    size_t query_nq,
    double elapsed) {
    BenchmarkStats stats;
    stats.elapsed = elapsed;
    stats.qps = query_nq / elapsed;
    stats.latency_ms = elapsed / query_nq * 1000.0;
    stats.recall = compute_recall(
        results.data(), eval.groundtruth.data(), (int)eval.recall_nq, opts.K, eval.gt_k);
    stats.recall_legacy = compute_recall_legacy(
        results.data(), eval.groundtruth.data(), (int)eval.recall_nq, opts.K, eval.gt_k);
    stats.avg_unique_at_k = compute_avg_unique_at_k(
        results.data(), (int)eval.recall_nq, opts.K);
    stats.duplicate_ratio = 1.0 - (stats.avg_unique_at_k / (double)opts.K);
    return stats;
}

static void print_benchmark_results(
    const MainOptions& opts,
    const GroundtruthEval& eval,
    const BenchmarkStats& stats) {
    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    std::cout << "Total time: " << stats.elapsed * 1000 << " ms\n";
    std::cout << "Throughput: " << stats.qps << " queries/sec\n";
    std::cout << "Avg latency: " << stats.latency_ms << " ms/query\n";
    std::cout << "Recall@" << eval.recall_k
              << " (unique hit, on " << eval.recall_nq << " queries): " << stats.recall << "\n";
    std::cout << "Recall@" << eval.recall_k
              << " (legacy non-unique): " << stats.recall_legacy << "\n";
    std::cout << "Avg unique IDs@" << opts.K << ": " << stats.avg_unique_at_k << "\n";
    std::cout << "TopK duplicate ratio: " << stats.duplicate_ratio << "\n";
    std::cout << "========================================\n";
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;

    MainOptions opts;
    const int parse_rc = parse_main_options(argc, argv, opts);
    if (parse_rc != 0) {
        return parse_rc;
    }

    print_run_banner(opts);
    srand(42);

    IndexGraph<float> index;
    std::cout << "Loading graph...\n";
    index.load_graph(opts.graph_prefix);

    std::cout << "Loading data...\n";
    index.load_data(opts.data_file);
    // If ROUTEA_BITPLANE_QG_FILE is set, swap the graph adjacency for the
    // QG codebook's neighbour layout. The Q-LEAF formal mainline is run
    // against this QG-derived graph, not the original. Skipping this step
    // is the difference between the reference run's recall (~0.955) and
    // the original beam_search graph's recall (~0.78).
    if (const char* qg_path = beam_gpu_host::routea_bitplane_qg_file_env()) {
        if (!beam_gpu_host::routea_override_graph_from_bitplane_qg(index, qg_path)) {
            std::cerr << "Failed to override graph from ROUTEA_BITPLANE_QG_FILE="
                      << qg_path << "\n";
            return 2;
        }
    }

    LoadedQueries loaded_queries;
    const int load_queries_rc = load_runtime_queries(index, opts.query_file, loaded_queries);
    if (load_queries_rc != 0) {
        return load_queries_rc;
    }

    GroundtruthEval eval;
    const int load_gt_rc = load_groundtruth_eval(opts, loaded_queries.nq, eval);
    if (load_gt_rc != 0) {
        return load_gt_rc;
    }
    print_benchmark_dataset_summary(index, loaded_queries.nq, eval.gt_k);

    std::vector<vid_t> results(loaded_queries.nq * (size_t)opts.K);
    const double elapsed = run_timed_benchmark_search(index, opts, loaded_queries, results);
    const BenchmarkStats stats = compute_benchmark_stats(
        opts, results, eval, loaded_queries.nq, elapsed);
    print_benchmark_results(opts, eval, stats);
    return 0;
}
