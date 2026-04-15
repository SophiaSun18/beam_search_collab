#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>
#include <unordered_map>

#include "index.hpp"
#include "search_bitonic.cuh"
#include "../tools/rabitq_audit.hpp"
#include "../tools/rabitq_edge_sidecar_build.hpp"
#include "../tools/rabitq_signbit_build.hpp"

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

static bool parse_string_arg(int argc, char** argv, int& i, const char* key, std::string& out) {
    const std::string a(argv[i]);
    const std::string k(key);
    if (a == k) {
        if (i + 1 >= argc) return false;
        out = argv[i + 1];
        i += 1;
        return true;
    }
    const std::string prefix = k + "=";
    if (a.rfind(prefix, 0) == 0) {
        out = a.substr(prefix.size());
        return true;
    }
    return false;
}

static bool parse_flag_arg(const char* arg, const char* key) {
    return std::string(arg) == std::string(key);
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

struct MainOptions {
    int warmup = 0;
    int runs = 1;
    bool rabitq_audit = false;
    bool routea_audit = false;
    bool build_raw_signbit = false;
    bool build_rabitq_edge_sidecars = false;
    RabitqAuditConfig audit_cfg;
    RabitqRawSignbitBuildConfig signbit_cfg;
    RabitqEdgeSidecarBuildConfig edge_sidecar_cfg;
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
              << " <data.fvecs> <query.fvecs> <gt.ivecs> <graph_prefix> [K=100] [beam_size=" << BEAM_SIZE << "]"
              << " [--warmup N=0] [--runs N=1] [--rabitq-audit] [--routea-audit] [--audit-edges N=32] [--audit-queries N=1] [--audit-sources N=32] [--audit-verbose N=8] [--build-raw-signbit] [--signbit-out PATH] [--build-rabitq-edge-sidecars] [--rabitq-signbit-out PATH] [--rabitq-triple-out PATH] [--rabitq-factor-dq-out PATH] [--rabitq-factor-vq-out PATH]\n"
              << "Audit-only shortcut: " << prog
              << " <data.fvecs> <query.fvecs> <graph_prefix> --rabitq-audit [--audit-edges N] [--audit-queries N]\n"
              << "RouteA audit shortcut: " << prog
              << " <data.fvecs> <query.fvecs> <graph_prefix> --routea-audit [--audit-edges N] [--audit-queries N]\n"
              << "Build-only shortcut: " << prog
              << " <data.fvecs> <graph_prefix> --build-raw-signbit [--signbit-out PATH]\n"
              << "Build-only shortcut: " << prog
              << " <data.fvecs> <graph_prefix> --build-rabitq-edge-sidecars [--rabitq-signbit-out PATH] [--rabitq-triple-out PATH] [--rabitq-factor-dq-out PATH] [--rabitq-factor-vq-out PATH]\n";
    return code;
}

static bool parse_named_main_option(
    int argc,
    char** argv,
    int& i,
    MainOptions& opts) {
    if (parse_int_arg(argc, argv, i, "--warmup", opts.warmup)) return true;
    if (parse_int_arg(argc, argv, i, "--runs", opts.runs)) return true;
    if (parse_int_arg(argc, argv, i, "--audit-edges", opts.audit_cfg.edge_samples)) return true;
    if (parse_int_arg(argc, argv, i, "--audit-queries", opts.audit_cfg.query_samples)) return true;
    if (parse_int_arg(argc, argv, i, "--audit-sources", opts.audit_cfg.max_source_nodes)) return true;
    if (parse_int_arg(argc, argv, i, "--audit-verbose", opts.audit_cfg.verbose_edges)) return true;
    if (parse_string_arg(argc, argv, i, "--signbit-out", opts.signbit_cfg.output_path)) return true;
    if (parse_string_arg(argc, argv, i, "--rabitq-signbit-out", opts.edge_sidecar_cfg.signbit_output_path)) return true;
    if (parse_string_arg(argc, argv, i, "--rabitq-triple-out", opts.edge_sidecar_cfg.triple_output_path)) return true;
    if (parse_string_arg(argc, argv, i, "--rabitq-factor-dq-out", opts.edge_sidecar_cfg.factor_dq_output_path)) return true;
    if (parse_string_arg(argc, argv, i, "--rabitq-factor-vq-out", opts.edge_sidecar_cfg.factor_vq_output_path)) return true;
    if (parse_flag_arg(argv[i], "--rabitq-audit")) {
        opts.rabitq_audit = true;
        return true;
    }
    if (parse_flag_arg(argv[i], "--routea-audit")) {
        opts.routea_audit = true;
        return true;
    }
    if (parse_flag_arg(argv[i], "--build-raw-signbit")) {
        opts.build_raw_signbit = true;
        return true;
    }
    if (parse_flag_arg(argv[i], "--build-rabitq-edge-sidecars")) {
        opts.build_rabitq_edge_sidecars = true;
        return true;
    }
    return false;
}

static int required_main_positional_count(const MainOptions& opts) {
    if (opts.build_rabitq_edge_sidecars) return 2;
    if (opts.build_raw_signbit) return 2;
    if (opts.routea_audit) return 3;
    if (opts.rabitq_audit) return 3;
    return 4;
}

static void assign_main_positional_args(
    const std::vector<const char*>& pos,
    MainOptions& opts) {
    opts.data_file = pos[0];
    if (opts.build_rabitq_edge_sidecars) {
        opts.graph_prefix = pos[1];
        return;
    }
    if (opts.build_raw_signbit) {
        opts.graph_prefix = pos[1];
        return;
    }
    if (opts.routea_audit && static_cast<int>(pos.size()) == 3) {
        opts.query_file = pos[1];
        opts.graph_prefix = pos[2];
        return;
    }
    if (opts.rabitq_audit && static_cast<int>(pos.size()) == 3) {
        opts.query_file = pos[1];
        opts.graph_prefix = pos[2];
        return;
    }

    opts.query_file = pos[1];
    opts.gt_file = pos[2];
    opts.graph_prefix = pos[3];
    opts.K = (static_cast<int>(pos.size()) >= 5) ? std::atoi(pos[4]) : 100;
    opts.beam_size = (static_cast<int>(pos.size()) >= 6) ? std::atoi(pos[5]) : BEAM_SIZE;
}

static void normalize_main_options(MainOptions& opts) {
    if (opts.warmup < 0) opts.warmup = 0;
    if (opts.runs < 1) opts.runs = 1;
    opts.audit_cfg.query_samples = std::max(1, opts.audit_cfg.query_samples);
    opts.audit_cfg.edge_samples = std::max(1, opts.audit_cfg.edge_samples);
    opts.audit_cfg.max_source_nodes = std::max(1, opts.audit_cfg.max_source_nodes);
    opts.audit_cfg.verbose_edges = std::max(0, opts.audit_cfg.verbose_edges);
}

static int validate_main_options(
    const MainOptions& opts,
    const std::vector<const char*>& pos,
    const char* prog) {
    if (static_cast<int>(pos.size()) < required_main_positional_count(opts)) {
        return print_usage(prog, 1);
    }
    if (!opts.build_raw_signbit && !opts.rabitq_audit && !opts.routea_audit && opts.K <= 0) {
        std::cerr << "Invalid K: " << opts.K << " (must be > 0)\n";
        return print_usage(prog, 1);
    }
    if (!opts.build_raw_signbit && !opts.rabitq_audit && !opts.routea_audit &&
        (opts.beam_size <= 0 || opts.beam_size > BEAM_SIZE)) {
        std::cerr << "Invalid beam_size: " << opts.beam_size
                  << " (must be in [1, " << BEAM_SIZE << "])\n";
        return print_usage(prog, 1);
    }
    if (!opts.build_raw_signbit && !opts.rabitq_audit && !opts.routea_audit && opts.K > opts.beam_size) {
        std::cerr << "Invalid config: K=" << opts.K
                  << " exceeds beam_size=" << opts.beam_size << "\n";
        return print_usage(prog, 1);
    }
    return 0;
}

static int parse_main_options(int argc, char** argv, MainOptions& opts) {
    std::vector<const char*> pos;
    pos.reserve((size_t)argc);
    for (int i = 1; i < argc; i++) {
        const std::string a(argv[i]);
        if (parse_named_main_option(argc, argv, i, opts)) {
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n";
            return print_usage(argv[0], 1);
        }
        pos.push_back(argv[i]);
    }

    normalize_main_options(opts);
    if (static_cast<int>(pos.size()) < required_main_positional_count(opts)) {
        return print_usage(argv[0], 1);
    }
    assign_main_positional_args(pos, opts);
    return validate_main_options(opts, pos, argv[0]);
}

static void print_run_banner(const MainOptions& opts) {
    std::cout << "========================================\n";
    std::cout << "GPU Beam Search\n";
    std::cout << "========================================\n";
    std::cout << "Data file: " << opts.data_file << "\n";
    if (opts.query_file != nullptr) {
        std::cout << "Query file: " << opts.query_file << "\n";
    }
    if (opts.gt_file != nullptr) {
        std::cout << "Ground truth: " << opts.gt_file << "\n";
    }
    std::cout << "Graph prefix: " << opts.graph_prefix << "\n";
    if (!opts.build_raw_signbit && !opts.rabitq_audit && !opts.routea_audit) {
        std::cout << "K: " << opts.K << "\n";
        std::cout << "Beam size: " << opts.beam_size
                  << " (compile-time max: " << BEAM_SIZE << ")\n";
    }
    std::cout << "Warmup runs: " << opts.warmup << "\n";
    std::cout << "Timed runs: " << opts.runs << "\n";
    if (opts.rabitq_audit) {
        std::cout << "RaBitQ audit: enabled"
                  << " (queries=" << opts.audit_cfg.query_samples
                  << ", edges=" << opts.audit_cfg.edge_samples
                  << ", max_sources=" << opts.audit_cfg.max_source_nodes
                  << ", verbose_edges=" << opts.audit_cfg.verbose_edges << ")\n";
    } else if (opts.routea_audit) {
        std::cout << "RouteA audit: enabled"
                  << " (queries=" << opts.audit_cfg.query_samples
                  << ", edges=" << opts.audit_cfg.edge_samples
                  << ", max_sources=" << opts.audit_cfg.max_source_nodes
                  << ", verbose_edges=" << opts.audit_cfg.verbose_edges << ")\n";
    } else if (opts.build_raw_signbit) {
        std::cout << "Raw signbit build: enabled";
        if (!opts.signbit_cfg.output_path.empty()) {
            std::cout << " (output=" << opts.signbit_cfg.output_path << ")";
        }
        std::cout << "\n";
    } else if (opts.build_rabitq_edge_sidecars) {
        std::cout << "RaBitQ edge sidecar build: enabled\n";
    }
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

static int run_build_raw_signbit_mode(
    IndexGraph<float>& index,
    const MainOptions& opts) {
    const bool ok = run_rabitq_raw_signbit_build(index, opts.signbit_cfg);
    return ok ? 0 : 2;
}

static int run_build_rabitq_edge_sidecars_mode(
    IndexGraph<float>& index,
    const MainOptions& opts) {
    const bool ok = run_rabitq_edge_sidecar_build(index, opts.edge_sidecar_cfg);
    return ok ? 0 : 2;
}

static int run_rabitq_audit_mode(
    IndexGraph<float>& index,
    const MainOptions& opts,
    const LoadedQueries& queries) {
    const bool ok = run_rabitq_formula_rotation_audit(
        index, queries.values.data(), queries.nq, opts.audit_cfg);
    return ok ? 0 : 2;
}

static int run_routea_audit_mode(
    IndexGraph<float>& index,
    const MainOptions& opts,
    const LoadedQueries& queries) {
    const bool ok = run_routea_rank_audit(
        index, queries.values.data(), queries.nq, opts.audit_cfg);
    return ok ? 0 : 2;
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

static double run_timed_benchmark_search(
    IndexGraph<float>& index,
    const MainOptions& opts,
    const LoadedQueries& loaded_queries,
    std::vector<vid_t>& results) {
    for (int w = 0; w < opts.warmup; w++) {
        std::cout << "Warmup run " << (w + 1) << "/" << opts.warmup << "...\n";
        index.search((int)loaded_queries.nq, loaded_queries.values.data(),
                     opts.K, results.data(), opts.beam_size);
    }

    std::cout << "Starting GPU beam search (timed)...\n";
    double elapsed = 0.0;
    for (int r = 0; r < opts.runs; r++) {
        const auto start = std::chrono::high_resolution_clock::now();
        index.search((int)loaded_queries.nq, loaded_queries.values.data(),
                     opts.K, results.data(), opts.beam_size);
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

static bool should_dump_query0(size_t recall_nq) {
    const char* dump_query0_env = std::getenv("DUMP_QUERY0");
    return dump_query0_env && dump_query0_env[0] != '\0' && recall_nq > 0;
}

static void print_query0_top_lists(
    const std::vector<vid_t>& results,
    const std::vector<int>& groundtruth,
    int K,
    int gt_k) {
    const int show_n = std::min(K, 20);
    std::cout << "Query0 results(top" << show_n << "):";
    for (int i = 0; i < show_n; i++) {
        std::cout << " " << static_cast<int>(results[static_cast<size_t>(i)]);
    }
    std::cout << "\n";

    const int show_gt = std::min(gt_k, 20);
    std::cout << "Query0 gt(top" << show_gt << "):";
    for (int i = 0; i < show_gt; i++) {
        std::cout << " " << groundtruth[static_cast<size_t>(i)];
    }
    std::cout << "\n";
}

static void print_query0_duplicate_summary(
    const std::vector<vid_t>& results,
    const std::vector<int>& groundtruth,
    int K,
    int gt_k) {
    std::unordered_map<int, int> freq;
    freq.reserve(static_cast<size_t>(K) * 2);
    int q0_unique = 0;
    int q0_match_unique = 0;
    std::vector<int> gt_seen(static_cast<size_t>(gt_k), 0);
    for (int i = 0; i < K; i++) {
        const int rid = static_cast<int>(results[static_cast<size_t>(i)]);
        auto it = freq.find(rid);
        if (it == freq.end()) {
            freq.emplace(rid, 1);
            q0_unique++;
            for (int j = 0; j < gt_k; j++) {
                if (!gt_seen[static_cast<size_t>(j)] && rid == groundtruth[static_cast<size_t>(j)]) {
                    gt_seen[static_cast<size_t>(j)] = 1;
                    q0_match_unique++;
                    break;
                }
            }
            continue;
        }
        it->second += 1;
    }

    std::cout << "Query0 unique@" << K << ": " << q0_unique
              << ", unique_matches: " << q0_match_unique << "\n";
    std::cout << "Query0 repeated ids (freq>1):";
    int printed = 0;
    for (const auto& kv : freq) {
        if (kv.second > 1) {
            std::cout << " [" << kv.first << ":" << kv.second << "]";
            printed++;
            if (printed >= 12) break;
        }
    }
    if (printed == 0) {
        std::cout << " none";
    }
    std::cout << "\n";
}

static void maybe_dump_query0(
    const std::vector<vid_t>& results,
    const std::vector<int>& groundtruth,
    size_t recall_nq,
    int K,
    int gt_k) {
    if (!should_dump_query0(recall_nq)) {
        return;
    }

    print_query0_top_lists(results, groundtruth, K, gt_k);
    print_query0_duplicate_summary(results, groundtruth, K, gt_k);
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

static int run_stable_benchmark_mode(
    IndexGraph<float>& index,
    const MainOptions& opts,
    const LoadedQueries& loaded_queries) {
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

    maybe_dump_query0(results, eval.groundtruth, eval.recall_nq, opts.K, eval.gt_k);
    print_benchmark_results(opts, eval, stats);
    return 0;
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

    if (opts.build_rabitq_edge_sidecars) {
        return run_build_rabitq_edge_sidecars_mode(index, opts);
    }
    if (opts.build_raw_signbit) {
        return run_build_raw_signbit_mode(index, opts);
    }

    LoadedQueries loaded_queries;
    const int load_queries_rc = load_runtime_queries(index, opts.query_file, loaded_queries);
    if (load_queries_rc != 0) {
        return load_queries_rc;
    }
    if (opts.rabitq_audit) {
        return run_rabitq_audit_mode(index, opts, loaded_queries);
    }
    if (opts.routea_audit) {
        return run_routea_audit_mode(index, opts, loaded_queries);
    }

    return run_stable_benchmark_mode(index, opts, loaded_queries);
}
