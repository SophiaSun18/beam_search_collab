#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#include "include/index.hpp"
#include "include/qg.hpp"
#include "include/metric.hpp"
#include "include/utils.h"
using namespace std;

namespace {

#define GPU_SEARCH_MODE_BASE 1
#define GPU_SEARCH_MODE_PATHW 2
#define GPU_SEARCH_MODE_RABITQ 3
#define GPU_SEARCH_MODE_RABITQ_PRUNED 4

#ifndef GPU_SEARCH_MODE
#error "GPU_SEARCH_MODE must be set by the GPU binary build target"
#endif

bool is_gpu_mode_arg(const string& value) {
    return value == "gpu_base" || value == "gpu_pathw" || value == "gpu_rabitq" ||
           value == "gpu_rabitq_pruned";
}

const char* gpu_search_mode_name() {
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_BASE
    return "gpu_base";
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    return "gpu_pathw";
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_RABITQ
    return "gpu_rabitq";
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_RABITQ_PRUNED
    return "gpu_rabitq_pruned";
#else
#error "Unknown GPU_SEARCH_MODE"
#endif
}

}  // namespace

int main(int argc, char** argv) {
    string mode = gpu_search_mode_name();
    int input_arg = 1;
    if (argc > 1 && is_gpu_mode_arg(argv[1])) {
        if (mode != argv[1]) {
            cerr << "This binary was built for mode " << mode << ", got " << argv[1] << "\n";
            return 1;
        }
        input_arg = 2;
    }
    if (argc < input_arg + 4) {
        cerr << "Usage: " << argv[0]
             << " [" << mode << "] <data.fvecs> <query.fvecs> <gt.ivecs> <graph_or_codebook>"
             << " [signbit.bin for gpu_pathw] [K=100] [beam_size=128] [degree=32]"
             << " [-p_ratio keep] [-a_ratio iter] [-csv output.csv]\n";
        return 1;
    }

    const char* data_file = argv[input_arg];
    const char* query_file = argv[input_arg + 1];
    const char* gt_file = argv[input_arg + 2];
    
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
    float prune_ratio = 0.0f;
    float pathw_iteration_prune_ratio = 0.7f;
    string csv_file;
    GpuRunStats gpu_stats;

#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_BASE
    {
        if (argc < input_arg + 4) {
            cerr << "gpu_base requires: <data> <query> <gt> <graph_index> [K] [beam_size] [-csv output.csv]\n";
            return 1;
        }
        const char* graph_prefix = argv[input_arg + 3];
        int arg_idx = input_arg + 4;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        while (arg_idx < argc) {
            string arg = argv[arg_idx];
            if (arg == "-csv" && arg_idx + 1 < argc) {
                csv_file = argv[++arg_idx];
            } else {
                cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
            ++arg_idx;
        }
        cout << "Graph prefix: " << graph_prefix << "\n";
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

        index.search(static_cast<int>(nq), queries.data(), K, distances.data(),
                     results.data(), beam_size, elapsed, &gpu_stats);
    }
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    {
        if (argc < input_arg + 5) {
            cerr << "gpu_pathw requires: <data> <query> <gt> <graph_index> <signbit.bin> [K] [beam_size] [-p_ratio keep_fraction] [-a_ratio iteration_prune_ratio] [-csv output.csv]\n";
            return 1;
        }
        const char* graph_prefix = argv[input_arg + 3];
        const char* signbit_file = argv[input_arg + 4];
        int arg_idx = input_arg + 5;
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
                           pathw_iteration_prune_ratio, elapsed, &gpu_stats);
    }
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_RABITQ
    {
        if (argc < input_arg + 4) {
            cerr << "gpu_rabitq requires: <data> <query> <gt> <qg_codebook> [K] [beam_size] [degree] [-csv output.csv]\n";
            return 1;
        }
        const char* qg_codebook = argv[input_arg + 3];
        int arg_idx = input_arg + 4;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        int degree = 32;
        if (arg_idx < argc && argv[arg_idx][0] != '-') degree = atoi(argv[arg_idx++]);
        while (arg_idx < argc) {
            string arg = argv[arg_idx];
            if (arg == "-csv" && arg_idx + 1 < argc) {
                csv_file = argv[++arg_idx];
            } else {
                cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
            ++arg_idx;
        }
        cout << "QG codebook: " << qg_codebook << "\n";
        if (!csv_file.empty()) cout << "CSV output: " << csv_file << "\n";
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_data(data_file);

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        (void)query_dim;
        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, qg_codebook);
        qg.set_metric(g_metric_type);
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(),
                      results.data(), beam_size, elapsed, &gpu_stats);
    }
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_RABITQ_PRUNED
    {
        if (argc < input_arg + 4) {
            cerr << "gpu_rabitq_pruned requires: <data> <query> <gt> <qg_codebook> [K] [beam_size] [degree] [-p_ratio prune_fraction] [-a_ratio prune_iteration_ratio] [-csv output.csv]\n";
            return 1;
        }
        const char* qg_codebook = argv[input_arg + 3];
        int arg_idx = input_arg + 4;
        if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
        if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
        int degree = 32;
        if (arg_idx < argc && argv[arg_idx][0] != '-') degree = atoi(argv[arg_idx++]);
        pathw_iteration_prune_ratio = 0.7f;
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
        cout << "QG codebook: " << qg_codebook << "\n";
        cout << "RaBitQ-pruned prune ratio: " << prune_ratio << "\n";
        cout << "RaBitQ-pruned iteration prune ratio: " << pathw_iteration_prune_ratio << "\n";
        if (!csv_file.empty()) cout << "CSV output: " << csv_file << "\n";
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_data(data_file);

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        (void)query_dim;
        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, qg_codebook);
        qg.set_metric(g_metric_type);
        qg.gpu_search_pruned(static_cast<int>(nq), queries.data(), K, distances.data(),
                             results.data(), beam_size, prune_ratio,
                             pathw_iteration_prune_ratio, elapsed, &gpu_stats);
    }
#endif

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

    gpu_stats.common.runtime = elapsed;
    gpu_stats.common.latency = latency_ms;
    gpu_stats.common.throughput = qps;
    gpu_stats.common.recall = recall;

    if (!csv_file.empty()) {
        const float csv_prune_ratio = prune_ratio;
        const float csv_aux_ratio =
            (mode == "gpu_pathw" || mode == "gpu_rabitq_pruned") ? pathw_iteration_prune_ratio : 0.0f;
        append_gpu_stats_to_csv(csv_file, K, beam_size, gpu_stats, csv_prune_ratio, csv_aux_ratio);
        cout << "Saved GPU stats CSV row to: " << csv_file << "\n";
    }
    
    return 0;
}
