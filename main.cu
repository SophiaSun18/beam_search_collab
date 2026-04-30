#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <unordered_set>

#include "include/index.hpp"
#include "include/qg.hpp"
#include "include/utils.h"
#include "include/search_bitonic.cuh"
using namespace std;

namespace {

float compute_query_recall(const vid_t* predicted, const int* gt_ptr, int K, int gt_k) {
    unordered_set<int> predicted_set;
    predicted_set.reserve(static_cast<size_t>(K) * 2);
    for (int i = 0; i < K; ++i) {
        predicted_set.insert(static_cast<int>(predicted[i]));
    }

    int hits = 0;
    for (int i = 0; i < gt_k; ++i) {
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] 
             << " <mode:gpu_base|gpu_rabitq> <data.fvecs> <query.fvecs> <gt.ivecs> <graph_or_codebook>"
             << " [K=100] [beam_size=128] [degree=32] [-csv output.csv]\n";
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
    int query_begin = 0;
    int query_count = -1;
    string csv_file;
    GpuRunStats gpu_stats;

    if (mode == "gpu_base") {
        if (argc < 6) {
            cerr << "gpu_base requires: <mode> <data> <query> <gt> <graph_index> [K] [beam_size] [-csv output.csv]\n";
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
            } else if (arg == "-debug" && arg_idx + 1 < argc) {
                string debug_arg = argv[++arg_idx];
                if (debug_arg == "on" || debug_arg == "1" || debug_arg == "true") {
                    debug_mode = true;
                } else if (debug_arg == "off" || debug_arg == "0" || debug_arg == "false") {
                    debug_mode = false;
                } else {
                    cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if (arg == "-debug_query" && arg_idx + 1 < argc) {
                debug_query_id = atoi(argv[++arg_idx]);
                debug_mode = true;
            } else if (arg == "-query_begin" && arg_idx + 1 < argc) {
                query_begin = atoi(argv[++arg_idx]);
            } else if (arg == "-query_count" && arg_idx + 1 < argc) {
                query_count = atoi(argv[++arg_idx]);
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

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        const size_t original_nq = nq;
        if (query_begin < 0 || query_begin >= static_cast<int>(original_nq)) {
            cerr << "Error: -query_begin " << query_begin << " out of range for nq=" << original_nq << "\n";
            return 1;
        }
        size_t sliced_nq = original_nq - static_cast<size_t>(query_begin);
        if (query_count >= 0) {
            sliced_nq = min(sliced_nq, static_cast<size_t>(query_count));
        }
        sliced_nq = max<size_t>(sliced_nq, 1);
        queries = vector<float>(
            queries.begin() + static_cast<size_t>(query_begin) * query_dim,
            queries.begin() + static_cast<size_t>(query_begin + sliced_nq) * query_dim);
        nq = sliced_nq;
        if (debug_query_id >= 0) {
            if (debug_query_id < query_begin ||
                debug_query_id >= query_begin + static_cast<int>(nq)) {
                cerr << "Error: -debug_query " << debug_query_id << " out of range for nq=" << original_nq << "\n";
                return 1;
            }
            cout << "[DEBUG] tracing query " << debug_query_id << " only\n";
            const int local_debug_query = debug_query_id - query_begin;
            queries = vector<float>(
                queries.begin() + static_cast<size_t>(local_debug_query) * query_dim,
                queries.begin() + static_cast<size_t>(local_debug_query + 1) * query_dim);
            nq = 1;
        } else if (debug_mode) {
            cout << "[DEBUG] debug mode enabled: running only query 0\n";
            nq = std::min<size_t>(nq, 1);
        }
        if (nq > 10000) {
            cout << "Limiting queries to 10000\n";
            nq = 10000;
        }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        groundtruth = vector<int>(
            groundtruth.begin() + static_cast<size_t>(query_begin) * gt_k,
            groundtruth.begin() + static_cast<size_t>(query_begin + sliced_nq) * gt_k);
        if (debug_query_id >= 0) {
            const int local_debug_query = debug_query_id - query_begin;
            groundtruth = vector<int>(
                groundtruth.begin() + static_cast<size_t>(local_debug_query) * gt_k,
                groundtruth.begin() + static_cast<size_t>(local_debug_query + 1) * gt_k);
        } else if (debug_mode) {
            groundtruth.resize(static_cast<size_t>(nq) * gt_k);
        }
        results.resize(nq * K);
        distances.resize(nq * K);

        index.search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);
    } else if (mode == "gpu_rabitq") {
        if (argc < 6) {
            cerr << "gpu_rabitq requires: <mode> <data> <query> <gt> <qg_codebook> [K] [beam_size] [degree] [-csv output.csv]\n";
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
            } else if (arg == "-debug" && arg_idx + 1 < argc) {
                string debug_arg = argv[++arg_idx];
                if (debug_arg == "on" || debug_arg == "1" || debug_arg == "true") {
                    debug_mode = true;
                } else if (debug_arg == "off" || debug_arg == "0" || debug_arg == "false") {
                    debug_mode = false;
                } else {
                    cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                    return 1;
                }
            } else if (arg == "-debug_query" && arg_idx + 1 < argc) {
                debug_query_id = atoi(argv[++arg_idx]);
                debug_mode = true;
            } else if (arg == "-query_begin" && arg_idx + 1 < argc) {
                query_begin = atoi(argv[++arg_idx]);
            } else if (arg == "-query_count" && arg_idx + 1 < argc) {
                query_count = atoi(argv[++arg_idx]);
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
        const size_t original_nq = nq;
        if (query_begin < 0 || query_begin >= static_cast<int>(original_nq)) {
            cerr << "Error: -query_begin " << query_begin << " out of range for nq=" << original_nq << "\n";
            return 1;
        }
        size_t sliced_nq = original_nq - static_cast<size_t>(query_begin);
        if (query_count >= 0) {
            sliced_nq = min(sliced_nq, static_cast<size_t>(query_count));
        }
        sliced_nq = max<size_t>(sliced_nq, 1);
        queries = vector<float>(
            queries.begin() + static_cast<size_t>(query_begin) * query_dim,
            queries.begin() + static_cast<size_t>(query_begin + sliced_nq) * query_dim);
        nq = sliced_nq;
        if (debug_query_id >= 0) {
            if (debug_query_id < query_begin ||
                debug_query_id >= query_begin + static_cast<int>(nq)) {
                cerr << "Error: -debug_query " << debug_query_id << " out of range for nq=" << original_nq << "\n";
                return 1;
            }
            cout << "[DEBUG] tracing query " << debug_query_id << " only\n";
            const int local_debug_query = debug_query_id - query_begin;
            queries = vector<float>(
                queries.begin() + static_cast<size_t>(local_debug_query) * query_dim,
                queries.begin() + static_cast<size_t>(local_debug_query + 1) * query_dim);
            nq = 1;
        } else if (debug_mode) {
            cout << "[DEBUG] debug mode enabled: running only query 0\n";
            nq = std::min<size_t>(nq, 1);
        }
        if (nq > 10000) {
            cout << "Limiting queries to 10000\n";
            nq = 10000;
        }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        groundtruth = vector<int>(
            groundtruth.begin() + static_cast<size_t>(query_begin) * gt_k,
            groundtruth.begin() + static_cast<size_t>(query_begin + sliced_nq) * gt_k);
        if (debug_query_id >= 0) {
            const int local_debug_query = debug_query_id - query_begin;
            groundtruth = vector<int>(
                groundtruth.begin() + static_cast<size_t>(local_debug_query) * gt_k,
                groundtruth.begin() + static_cast<size_t>(local_debug_query + 1) * gt_k);
        } else if (debug_mode) {
            groundtruth.resize(static_cast<size_t>(nq) * gt_k);
        }
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, qg_codebook);
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);
    } else {
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

    if (!csv_file.empty()) {
        append_gpu_stats_to_csv(csv_file, K, beam_size, gpu_stats);
        cout << "Saved GPU stats CSV row to: " << csv_file << "\n";
    }
    
    return 0;
}
