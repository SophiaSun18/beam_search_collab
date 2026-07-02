#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <unordered_set>

#include "include/index.hpp"
#include "include/qg.hpp"
#include "include/metric.hpp"
#include "include/utils.h"
#include "include/search_bitonic.cuh"
using namespace std;

namespace {

bool parse_bool_arg(const string& value, bool& out) {
    if (value == "on" || value == "1" || value == "true") {
        out = true;
        return true;
    }
    if (value == "off" || value == "0" || value == "false") {
        out = false;
        return true;
    }
    return false;
}

float compute_query_recall(const vid_t* predicted, const int* gt_ptr, int K, int gt_k) {
    unordered_set<int> predicted_set;
    predicted_set.reserve(static_cast<size_t>(K) * 2);
    for (int i = 0; i < K; ++i) {
        predicted_set.insert(static_cast<int>(predicted[i]));
    }

    int hits = 0;
    const int eval_k = min(K, gt_k);
    for (int i = 0; i < eval_k; ++i) {
        if (predicted_set.count(gt_ptr[i])) {
            ++hits;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(K);
}

void print_query_results(const vid_t* predicted,
                         const int* gt_ptr,
                         int K,
                         int gt_k) {
    cout << "    predicted:";
    for (int j = 0; j < K; ++j) {
        cout << " " << predicted[j];
    }
    cout << "\n";
    cout << "    gt:";
    for (int j = 0; j < min(K, gt_k); ++j) {
        cout << " " << gt_ptr[j];
    }
    cout << "\n";
}

void dump_final_quality_trace(const string& filename,
                              const char* method,
                              const float* data,
                              size_t row_stride,
                              const float* queries,
                              size_t dim,
                              const vid_t* results,
                              const int* groundtruth,
                              size_t nq,
                              int K,
                              int gt_k) {
    if (filename.empty()) return;

    vector<QualityTraceItem> records;
    records.reserve(nq * static_cast<size_t>(K));
    for (size_t q = 0; q < nq; ++q) {
        vector<QualityTraceItem> query_records;
        QualityTraceContext ctx{
            method,
            static_cast<uint32_t>(q),
            data,
            row_stride,
            queries + q * dim,
            dim,
            groundtruth + q * static_cast<size_t>(gt_k),
            gt_k,
            K,
            &query_records,
        };
        vector<pair<vidType, float>> candidates;
        candidates.reserve(K);
        for (int i = 0; i < K; ++i) {
            const vidType node = static_cast<vidType>(results[q * static_cast<size_t>(K) + i]);
            const float exact = compute_distance(
                static_cast<int>(dim),
                queries + q * dim,
                data + static_cast<size_t>(node) * row_stride);
            candidates.emplace_back(node, exact);
        }
        QualityFrontierAudit audit;
        append_quality_trace_snapshot(
            ctx, 0, "final", 0, std::numeric_limits<double>::quiet_NaN(),
            audit, candidates);
        records.insert(records.end(), query_records.begin(), query_records.end());
    }
    write_quality_trace_csv(filename, records);
    cout << "Quality trace records: " << records.size() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0]
             << " <mode:gpu_base|gpu_pathw|gpu_rabitq|gpu_rabitq_pruned> <data.fvecs> <query.fvecs> <gt.ivecs> <graph_or_codebook>"
             << " [signbit.bin for gpu_pathw] [K=100] [beam_size=128] [degree=32]"
             << " [-p_ratio keep] [-a_ratio iter] [-csv output.csv]"
             << " [-expander_trace_file trace.csv] [-access_trace_file trace.csv]"
             << " [-query_steps_file steps.csv]"
             << " [-quality_trace_file trace.csv] [-quality_hit_trace_file trace.csv]"
             << " [-quality_beam_hit_trace_file trace.csv]"
             << " [-frontier_audit_file trace.csv] [-quality_trace_queries n]"
             << " [-query_limit n]\n";
        return 1;
    }

    string mode = argv[1];
    const char* data_file = argv[2];
    const char* query_file = argv[3];
    const char* gt_file = argv[4];

    cout << "========================================\n";
    cout << "GPU Beam Search Loading from: \n";
    cout << "========================================\n";
    cout << "Mode: " << mode << "\n";
    cout << "Data file: " << data_file << "\n";
    cout << "Query file: " << query_file << "\n";
    cout << "Ground truth: " << gt_file << "\n";
    g_metric_type = infer_metric_from_dataset_path(data_file);
    cout << "Metric: " << metric_name(g_metric_type) << "\n";

    // Initialize the random seed
    srand(42);

    vector<vid_t> results;
    vector<float> distances;
    vector<float> queries;
    vector<int> groundtruth;
    size_t nq = 0;
    int gt_k = 0;
    double elapsed = 0.0;
    int K = 100;
    int beam_size = 128;
    bool debug_mode = false;
    int debug_query_id = -1;
    int query_limit = -1;
    float prune_ratio = 0.0f;
    float pathw_iteration_prune_ratio = 0.7f;
    string csv_file;
    string expander_trace_file;
    string query_steps_file;
    string access_trace_file;
    string quality_trace_file;
    string quality_hit_trace_file;
    string quality_beam_hit_trace_file;
    string frontier_audit_file;
    int quality_trace_queries = 50;
    GpuRunStats gpu_stats;

    if (mode == "gpu_base") {
        if (argc < 6) {
            cerr << "gpu_base requires: <mode> <data> <query> <gt> <graph_index> [K] [beam_size] [-csv output.csv] [-expander_trace_file trace.csv] [-query_steps_file steps.csv] [-access_trace_file trace.csv] [-quality_trace_file trace.csv] [-quality_hit_trace_file trace.csv] [-quality_beam_hit_trace_file trace.csv] [-frontier_audit_file trace.csv] [-quality_trace_queries n] [-query_limit n]\n";
            return 1;
        }
        const char* graph_prefix = argv[5];
        int arg_idx = 6;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        while (arg_idx < argc) {
            string arg = argv[arg_idx];
            if (arg == "-csv" && arg_idx + 1 < argc) {
                csv_file = argv[++arg_idx];
            } else if (arg == "-expander_trace_file" && arg_idx + 1 < argc) {
                expander_trace_file = argv[++arg_idx];
            } else if (arg == "-query_steps_file" && arg_idx + 1 < argc) {
                query_steps_file = argv[++arg_idx];
            } else if (arg == "-access_trace_file" && arg_idx + 1 < argc) {
                access_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_trace_file" && arg_idx + 1 < argc) {
                quality_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_hit_trace_file" && arg_idx + 1 < argc) {
                quality_hit_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_beam_hit_trace_file" && arg_idx + 1 < argc) {
                quality_beam_hit_trace_file = argv[++arg_idx];
            } else if (arg == "-frontier_audit_file" && arg_idx + 1 < argc) {
                frontier_audit_file = argv[++arg_idx];
            } else if (arg == "-quality_trace_queries" && arg_idx + 1 < argc) {
                quality_trace_queries = atoi(argv[++arg_idx]);
            } else if (arg == "-debug" && arg_idx + 1 < argc) {
                string debug_arg = argv[++arg_idx];
                if (!parse_bool_arg(debug_arg, debug_mode)) {
                    cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if ((arg == "-kernel_debug" || arg == "-debug_kernel") && arg_idx + 1 < argc) {
                bool ignored_kernel_debug = false;
                string kernel_arg = argv[++arg_idx];
                if (!parse_bool_arg(kernel_arg, ignored_kernel_debug)) {
                    cerr << "Error: -kernel_debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if (arg == "-debug_query" && arg_idx + 1 < argc) {
                debug_query_id = atoi(argv[++arg_idx]);
                debug_mode = true;
            } else if (arg == "-query_limit" && arg_idx + 1 < argc) {
                query_limit = atoi(argv[++arg_idx]);
            } else {
                cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
            ++arg_idx;
        }
        cout << "Graph prefix: " << graph_prefix << "\n";
        if (!csv_file.empty()) cout << "CSV output: " << csv_file << "\n";
        if (!expander_trace_file.empty()) cout << "Expander trace output: " << expander_trace_file << "\n";
        if (!query_steps_file.empty()) cout << "Query steps output: " << query_steps_file << "\n";
        if (!access_trace_file.empty()) cout << "Access trace output: " << access_trace_file << "\n";
        if (!frontier_audit_file.empty()) cout << "Frontier audit output: " << frontier_audit_file << "\n";
        if (!quality_hit_trace_file.empty()) {
            cout << "Quality hit trace output: " << quality_hit_trace_file
                 << " (queries=" << quality_trace_queries << ")\n";
        }
        if (!quality_beam_hit_trace_file.empty()) {
            cout << "Quality beam hit trace output: " << quality_beam_hit_trace_file
                 << " (queries=" << quality_trace_queries << ", scan=beam)\n";
        }
        if (!quality_trace_file.empty()) {
            cout << "Quality trace output: " << quality_trace_file
                 << " (queries=" << quality_trace_queries << ")\n";
        }
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_graph_index(graph_prefix);
        index.load_data(data_file);
        index.metric = g_metric_type;

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        if (debug_query_id >= 0) {
            if (debug_query_id < 0 || debug_query_id >= static_cast<int>(nq)) {
                cerr << "Error: -debug_query " << debug_query_id << " out of range for nq=" << nq << "\n";
                return 1;
            }
            cout << "[DEBUG] tracing query " << debug_query_id << " only\n";
            queries = vector<float>(
                queries.begin() + static_cast<size_t>(debug_query_id) * query_dim,
                queries.begin() + static_cast<size_t>(debug_query_id + 1) * query_dim);
            nq = 1;
        } else if (debug_mode) {
            cout << "[DEBUG] debug mode enabled: running only query 0\n";
            nq = std::min<size_t>(nq, 1);
        } else if (query_limit > 0 && static_cast<size_t>(query_limit) < nq) {
            cout << "Query limit: " << query_limit << "\n";
            nq = static_cast<size_t>(query_limit);
            queries.resize(static_cast<size_t>(query_limit) * query_dim);
        } else if ((!quality_trace_file.empty() || !quality_hit_trace_file.empty() ||
                    !quality_beam_hit_trace_file.empty() || !frontier_audit_file.empty()) &&
                   quality_trace_queries > 0 &&
                   static_cast<size_t>(quality_trace_queries) < nq) {
            cout << "Quality/frontier trace query limit: " << quality_trace_queries << "\n";
            nq = static_cast<size_t>(quality_trace_queries);
            queries.resize(static_cast<size_t>(quality_trace_queries) * query_dim);
        }
        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        if (debug_query_id >= 0) {
            groundtruth = vector<int>(
                groundtruth.begin() + static_cast<size_t>(debug_query_id) * gt_k,
                groundtruth.begin() + static_cast<size_t>(debug_query_id + 1) * gt_k);
        } else if (debug_mode) {
            groundtruth.resize(static_cast<size_t>(nq) * gt_k);
        } else if (query_limit > 0 && static_cast<size_t>(query_limit) * gt_k < groundtruth.size()) {
            groundtruth.resize(static_cast<size_t>(query_limit) * gt_k);
        } else if ((!quality_trace_file.empty() || !quality_hit_trace_file.empty() ||
                    !quality_beam_hit_trace_file.empty() || !frontier_audit_file.empty()) &&
                   quality_trace_queries > 0 &&
                   static_cast<size_t>(quality_trace_queries) * gt_k < groundtruth.size()) {
            groundtruth.resize(static_cast<size_t>(quality_trace_queries) * gt_k);
        }
        results.resize(nq * K);
        distances.resize(nq * K);

        index.search(static_cast<int>(nq), queries.data(), K, distances.data(),
                     results.data(), beam_size, elapsed, &gpu_stats,
                     debug_mode,
                     expander_trace_file.empty() ? nullptr : expander_trace_file.c_str(),
                     query_steps_file.empty() ? nullptr : query_steps_file.c_str(),
                     access_trace_file.empty() ? nullptr : access_trace_file.c_str(),
                     frontier_audit_file.empty() ? nullptr : frontier_audit_file.c_str(),
                     groundtruth.data(),
                     gt_k,
                     quality_hit_trace_file.empty() ? nullptr : quality_hit_trace_file.c_str(),
                     quality_beam_hit_trace_file.empty() ? nullptr : quality_beam_hit_trace_file.c_str());
        if (!expander_trace_file.empty()) {
            accumulate_run_topk_efficiency(
                gpu_stats.common,
                index.get_data_ptr(),
                index.d,
                queries.data(),
                index.d,
                results.data(),
                groundtruth.data(),
                nq,
                K,
                gt_k);
        }
        if (!quality_trace_file.empty()) {
            dump_final_quality_trace(
                quality_trace_file, "gpu_base", index.get_data_ptr(), index.d,
                queries.data(), index.d, results.data(), groundtruth.data(),
                nq, K, gt_k);
        }
    }
#ifndef GPU_STEPEFF_ONLY
    else if (mode == "gpu_pathw") {
        if (argc < 7) {
            cerr << "gpu_pathw requires: <mode> <data> <query> <gt> <graph_index> <signbit.bin> [K] [beam_size] [-p_ratio keep_fraction] [-a_ratio iteration_prune_ratio] [-csv output.csv]\n";
            return 1;
        }
        const char* graph_prefix = argv[5];
        const char* signbit_file = argv[6];
        int arg_idx = 7;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        while (arg_idx < argc) {
            string arg = argv[arg_idx];
            if (arg == "-csv" && arg_idx + 1 < argc) {
                csv_file = argv[++arg_idx];
            } else if (arg == "-p_ratio" && arg_idx + 1 < argc) {
                prune_ratio = atof(argv[++arg_idx]);
            } else if (arg == "-a_ratio" && arg_idx + 1 < argc) {
                pathw_iteration_prune_ratio = atof(argv[++arg_idx]);
            } else {
                cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
            ++arg_idx;
        }
        if (prune_ratio < 0.0f || prune_ratio > 1.0f) {
            cerr << "Error: -p_ratio must be between 0 and 1.\n";
            return 1;
        }
        if (pathw_iteration_prune_ratio < 0.0f || pathw_iteration_prune_ratio > 1.0f) {
            cerr << "Error: -a_ratio must be between 0 and 1.\n";
            return 1;
        }
        cout << "Graph prefix: " << graph_prefix << "\n";
        cout << "Signbit file: " << signbit_file << "\n";
        cout << "PathW neighbor keep ratio: " << prune_ratio << "\n";
        cout << "PathW iteration prune ratio: " << pathw_iteration_prune_ratio << "\n";
        if (!csv_file.empty()) cout << "CSV output: " << csv_file << "\n";
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_graph_index(graph_prefix);
        index.load_data(data_file);
        index.metric = g_metric_type;

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        (void)query_dim;
        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        index.search_pathw(static_cast<int>(nq), queries.data(), K, distances.data(),
                           results.data(), beam_size, signbit_file, prune_ratio,
                           pathw_iteration_prune_ratio, elapsed, &gpu_stats, debug_mode);
    }
#endif
    else if (mode == "gpu_rabitq") {
        if (argc < 6) {
            cerr << "gpu_rabitq requires: <mode> <data> <query> <gt> <qg_codebook> [K] [beam_size] [degree] [-csv output.csv] [-expander_trace_file trace.csv] [-access_trace_file trace.csv] [-quality_trace_file trace.csv] [-quality_hit_trace_file trace.csv] [-quality_beam_hit_trace_file trace.csv] [-frontier_audit_file trace.csv] [-quality_trace_queries n] [-query_limit n]\n";
            return 1;
        }
        const char* qg_codebook = argv[5];
        int arg_idx = 6;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        int degree = 32;
        if (arg_idx < argc && argv[arg_idx][0] != '-') degree = atoi(argv[arg_idx++]);
        while (arg_idx < argc) {
            string arg = argv[arg_idx];
            if (arg == "-csv" && arg_idx + 1 < argc) {
                csv_file = argv[++arg_idx];
            } else if (arg == "-expander_trace_file" && arg_idx + 1 < argc) {
                expander_trace_file = argv[++arg_idx];
            } else if (arg == "-query_steps_file" && arg_idx + 1 < argc) {
                query_steps_file = argv[++arg_idx];
            } else if (arg == "-access_trace_file" && arg_idx + 1 < argc) {
                access_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_trace_file" && arg_idx + 1 < argc) {
                quality_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_hit_trace_file" && arg_idx + 1 < argc) {
                quality_hit_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_beam_hit_trace_file" && arg_idx + 1 < argc) {
                quality_beam_hit_trace_file = argv[++arg_idx];
            } else if (arg == "-frontier_audit_file" && arg_idx + 1 < argc) {
                frontier_audit_file = argv[++arg_idx];
            } else if (arg == "-quality_trace_queries" && arg_idx + 1 < argc) {
                quality_trace_queries = atoi(argv[++arg_idx]);
            } else if (arg == "-debug" && arg_idx + 1 < argc) {
                string debug_arg = argv[++arg_idx];
                if (!parse_bool_arg(debug_arg, debug_mode)) {
                    cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if ((arg == "-kernel_debug" || arg == "-debug_kernel") && arg_idx + 1 < argc) {
                bool ignored_kernel_debug = false;
                string kernel_arg = argv[++arg_idx];
                if (!parse_bool_arg(kernel_arg, ignored_kernel_debug)) {
                    cerr << "Error: -kernel_debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if (arg == "-debug_query" && arg_idx + 1 < argc) {
                debug_query_id = atoi(argv[++arg_idx]);
                debug_mode = true;
            } else if (arg == "-query_limit" && arg_idx + 1 < argc) {
                query_limit = atoi(argv[++arg_idx]);
            } else {
                cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
            ++arg_idx;
        }
        cout << "QG codebook: " << qg_codebook << "\n";
        if (!csv_file.empty()) cout << "CSV output: " << csv_file << "\n";
        if (!expander_trace_file.empty()) cout << "Expander trace output: " << expander_trace_file << "\n";
        if (!query_steps_file.empty()) cout << "Query steps output: " << query_steps_file << "\n";
        if (!access_trace_file.empty()) cout << "Access trace output: " << access_trace_file << "\n";
        if (!frontier_audit_file.empty()) cout << "Frontier audit output: " << frontier_audit_file << "\n";
        if (!quality_hit_trace_file.empty()) {
            cout << "Quality hit trace output: " << quality_hit_trace_file
                 << " (queries=" << quality_trace_queries << ")\n";
        }
        if (!quality_beam_hit_trace_file.empty()) {
            cout << "Quality beam hit trace output: " << quality_beam_hit_trace_file
                 << " (queries=" << quality_trace_queries << ", scan=beam)\n";
        }
        if (!quality_trace_file.empty()) {
            cout << "Quality trace output: " << quality_trace_file
                 << " (queries=" << quality_trace_queries << ")\n";
        }
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_data(data_file);

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        if (debug_query_id >= 0) {
            if (debug_query_id < 0 || debug_query_id >= static_cast<int>(nq)) {
                cerr << "Error: -debug_query " << debug_query_id << " out of range for nq=" << nq << "\n";
                return 1;
            }
            cout << "[DEBUG] tracing query " << debug_query_id << " only\n";
            queries = vector<float>(
                queries.begin() + static_cast<size_t>(debug_query_id) * query_dim,
                queries.begin() + static_cast<size_t>(debug_query_id + 1) * query_dim);
            nq = 1;
        } else if (debug_mode) {
            cout << "[DEBUG] debug mode enabled: running only query 0\n";
            nq = std::min<size_t>(nq, 1);
        } else if (query_limit > 0 && static_cast<size_t>(query_limit) < nq) {
            cout << "Query limit: " << query_limit << "\n";
            nq = static_cast<size_t>(query_limit);
            queries.resize(static_cast<size_t>(query_limit) * query_dim);
        } else if ((!quality_trace_file.empty() || !quality_hit_trace_file.empty() ||
                    !quality_beam_hit_trace_file.empty() || !frontier_audit_file.empty()) &&
                   quality_trace_queries > 0 &&
                   static_cast<size_t>(quality_trace_queries) < nq) {
            cout << "Quality/frontier trace query limit: " << quality_trace_queries << "\n";
            nq = static_cast<size_t>(quality_trace_queries);
            queries.resize(static_cast<size_t>(quality_trace_queries) * query_dim);
        }
        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        if (debug_query_id >= 0) {
            groundtruth = vector<int>(
                groundtruth.begin() + static_cast<size_t>(debug_query_id) * gt_k,
                groundtruth.begin() + static_cast<size_t>(debug_query_id + 1) * gt_k);
        } else if (debug_mode) {
            groundtruth.resize(static_cast<size_t>(nq) * gt_k);
        } else if (query_limit > 0 && static_cast<size_t>(query_limit) * gt_k < groundtruth.size()) {
            groundtruth.resize(static_cast<size_t>(query_limit) * gt_k);
        } else if ((!quality_trace_file.empty() || !quality_hit_trace_file.empty() ||
                    !quality_beam_hit_trace_file.empty() || !frontier_audit_file.empty()) &&
                   quality_trace_queries > 0 &&
                   static_cast<size_t>(quality_trace_queries) * gt_k < groundtruth.size()) {
            groundtruth.resize(static_cast<size_t>(quality_trace_queries) * gt_k);
        }
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, qg_codebook);
        qg.set_metric(g_metric_type);
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(),
                      results.data(), beam_size, elapsed, &gpu_stats,
                      debug_mode,
                      expander_trace_file.empty() ? nullptr : expander_trace_file.c_str(),
                      query_steps_file.empty() ? nullptr : query_steps_file.c_str(),
                      access_trace_file.empty() ? nullptr : access_trace_file.c_str(),
                      frontier_audit_file.empty() ? nullptr : frontier_audit_file.c_str(),
                      groundtruth.data(),
                      gt_k,
                      quality_hit_trace_file.empty() ? nullptr : quality_hit_trace_file.c_str(),
                      quality_beam_hit_trace_file.empty() ? nullptr : quality_beam_hit_trace_file.c_str());
        if (!expander_trace_file.empty()) {
            accumulate_run_topk_efficiency(
                gpu_stats.common,
                qg.get_data_ptr(),
                qg.is_bins_mode() ? qg.get_dim() : qg.get_row_offset(),
                queries.data(),
                qg.get_dim(),
                results.data(),
                groundtruth.data(),
                nq,
                K,
                gt_k);
        }
        if (!quality_trace_file.empty()) {
            dump_final_quality_trace(
                quality_trace_file, "gpu_rabitq",
                qg.get_data_ptr(),
                qg.is_bins_mode() ? qg.get_dim() : qg.get_row_offset(),
                queries.data(), qg.get_dim(), results.data(), groundtruth.data(),
                nq, K, gt_k);
        }
    }
#ifndef GPU_STEPEFF_ONLY
    else if (mode == "gpu_rabitq_pruned") {
        if (argc < 6) {
            cerr << "gpu_rabitq_pruned requires: <mode> <data> <query> <gt> <qg_codebook> [K] [beam_size] [degree] [-p_ratio prune_fraction] [-a_ratio prune_iteration_ratio] [-csv output.csv] [-quality_beam_hit_trace_file trace.csv] [-frontier_audit_file trace.csv] [-quality_trace_queries n] [-query_limit n]\n";
            return 1;
        }
        const char* qg_codebook = argv[5];
        int arg_idx = 6;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        int degree = 32;
        if (arg_idx < argc && argv[arg_idx][0] != '-') degree = atoi(argv[arg_idx++]);
        pathw_iteration_prune_ratio = 0.7f;
        while (arg_idx < argc) {
            string arg = argv[arg_idx];
            if (arg == "-csv" && arg_idx + 1 < argc) {
                csv_file = argv[++arg_idx];
            } else if (arg == "-frontier_audit_file" && arg_idx + 1 < argc) {
                frontier_audit_file = argv[++arg_idx];
            } else if (arg == "-quality_beam_hit_trace_file" && arg_idx + 1 < argc) {
                quality_beam_hit_trace_file = argv[++arg_idx];
            } else if (arg == "-quality_trace_queries" && arg_idx + 1 < argc) {
                quality_trace_queries = atoi(argv[++arg_idx]);
            } else if (arg == "-query_limit" && arg_idx + 1 < argc) {
                query_limit = atoi(argv[++arg_idx]);
            } else if (arg == "-p_ratio" && arg_idx + 1 < argc) {
                prune_ratio = atof(argv[++arg_idx]);
            } else if (arg == "-a_ratio" && arg_idx + 1 < argc) {
                pathw_iteration_prune_ratio = atof(argv[++arg_idx]);
            } else if (arg == "-debug" && arg_idx + 1 < argc) {
                string debug_arg = argv[++arg_idx];
                if (!parse_bool_arg(debug_arg, debug_mode)) {
                    cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if ((arg == "-kernel_debug" || arg == "-debug_kernel") && arg_idx + 1 < argc) {
                bool ignored_kernel_debug = false;
                string kernel_arg = argv[++arg_idx];
                if (!parse_bool_arg(kernel_arg, ignored_kernel_debug)) {
                    cerr << "Error: -kernel_debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if (arg == "-debug_query" && arg_idx + 1 < argc) {
                debug_query_id = atoi(argv[++arg_idx]);
                debug_mode = true;
            } else {
                cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
            ++arg_idx;
        }
        if (prune_ratio < 0.0f || prune_ratio > 1.0f) {
            cerr << "Error: -p_ratio must be between 0 and 1.\n";
            return 1;
        }
        if (pathw_iteration_prune_ratio < 0.0f || pathw_iteration_prune_ratio > 1.0f) {
            cerr << "Error: -a_ratio must be between 0 and 1.\n";
            return 1;
        }
        cout << "QG codebook: " << qg_codebook << "\n";
        cout << "RaBitQ-pruned prune ratio: " << prune_ratio << "\n";
        cout << "RaBitQ-pruned iteration prune ratio: " << pathw_iteration_prune_ratio << "\n";
        if (!csv_file.empty()) cout << "CSV output: " << csv_file << "\n";
        if (!frontier_audit_file.empty()) cout << "Frontier audit output: " << frontier_audit_file << "\n";
        if (!quality_beam_hit_trace_file.empty()) {
            cout << "Quality beam hit trace output: " << quality_beam_hit_trace_file
                 << " (queries=" << quality_trace_queries << ", scan=beam)\n";
        }
        if (!frontier_audit_file.empty() || !quality_beam_hit_trace_file.empty()) {
            cout << "Trace query limit: " << quality_trace_queries << "\n";
        }
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_data(data_file);

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        if (debug_query_id >= 0) {
            if (debug_query_id < 0 || debug_query_id >= static_cast<int>(nq)) {
                cerr << "Error: -debug_query " << debug_query_id << " out of range for nq=" << nq << "\n";
                return 1;
            }
            cout << "[DEBUG] tracing query " << debug_query_id << " only\n";
            queries = vector<float>(
                queries.begin() + static_cast<size_t>(debug_query_id) * query_dim,
                queries.begin() + static_cast<size_t>(debug_query_id + 1) * query_dim);
            nq = 1;
        } else if (debug_mode) {
            cout << "[DEBUG] debug mode enabled: running only query 0\n";
            nq = std::min<size_t>(nq, 1);
        } else if (query_limit > 0 && static_cast<size_t>(query_limit) < nq) {
            cout << "Query limit: " << query_limit << "\n";
            nq = static_cast<size_t>(query_limit);
            queries.resize(static_cast<size_t>(query_limit) * query_dim);
        } else if ((!frontier_audit_file.empty() || !quality_beam_hit_trace_file.empty()) &&
                   quality_trace_queries > 0 &&
                   static_cast<size_t>(quality_trace_queries) < nq) {
            cout << "Trace query limit: " << quality_trace_queries << "\n";
            nq = static_cast<size_t>(quality_trace_queries);
            queries.resize(static_cast<size_t>(quality_trace_queries) * query_dim);
        }
        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        if (debug_query_id >= 0) {
            groundtruth = vector<int>(
                groundtruth.begin() + static_cast<size_t>(debug_query_id) * gt_k,
                groundtruth.begin() + static_cast<size_t>(debug_query_id + 1) * gt_k);
        } else if (debug_mode) {
            groundtruth.resize(static_cast<size_t>(nq) * gt_k);
        } else if (query_limit > 0 && static_cast<size_t>(query_limit) * gt_k < groundtruth.size()) {
            groundtruth.resize(static_cast<size_t>(query_limit) * gt_k);
        } else if ((!frontier_audit_file.empty() || !quality_beam_hit_trace_file.empty()) &&
                   quality_trace_queries > 0 &&
                   static_cast<size_t>(quality_trace_queries) * gt_k < groundtruth.size()) {
            groundtruth.resize(static_cast<size_t>(quality_trace_queries) * gt_k);
        }
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, qg_codebook);
        qg.set_metric(g_metric_type);
        qg.gpu_search_pruned(static_cast<int>(nq), queries.data(), K, distances.data(),
                             results.data(), beam_size, prune_ratio,
                             pathw_iteration_prune_ratio, elapsed, &gpu_stats,
                             debug_mode,
                             frontier_audit_file.empty() ? nullptr : frontier_audit_file.c_str(),
                             groundtruth.data(),
                             gt_k,
                             quality_beam_hit_trace_file.empty() ? nullptr : quality_beam_hit_trace_file.c_str());
    }
#endif
    else {
        cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k) * 100.0f;

    // Output the results
    cout << "\n========================================\n";
    cout << "Results\n";
    cout << "========================================\n";
    cout << "Total time: " << elapsed * 1000 << " ms\n";
    cout << "Throughput: " << qps << " queries/sec\n";
    cout << "Avg latency: " << latency_ms << " ms/query\n";
    cout << "Recall@" << K << ": " << recall << "\n";
    cout << "========================================\n";

    if (debug_query_id >= 0 && !results.empty()) {
        const float query_recall = compute_query_recall(results.data(), groundtruth.data(), K, gt_k) * 100.0f;
        cout << "\nDebug query " << debug_query_id << " recall@" << K << ": " << query_recall << "\n";
        print_query_results(results.data(), groundtruth.data(), K, gt_k);
    }

    gpu_stats.common.runtime = elapsed;
    gpu_stats.common.latency = latency_ms;
    gpu_stats.common.throughput = qps;
    gpu_stats.common.recall = recall;
    if (!expander_trace_file.empty()) {
        dump_expander_trace_stats(expander_trace_file, gpu_stats.common);
    }

    if (!csv_file.empty()) {
        const float csv_prune_ratio = prune_ratio;
        const float csv_aux_ratio =
            (mode == "gpu_pathw" || mode == "gpu_rabitq_pruned") ? pathw_iteration_prune_ratio : 0.0f;
        append_gpu_stats_to_csv(csv_file, K, beam_size, gpu_stats, csv_prune_ratio, csv_aux_ratio);
        cout << "Saved GPU stats CSV row to: " << csv_file << "\n";
    }

    return 0;
}
