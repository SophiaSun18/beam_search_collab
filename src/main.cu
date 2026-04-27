#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "index.hpp"
#include "qg.hpp"
#include "utils.h"
#include "search_bitonic.cuh"
using namespace std;

static void usage(const char* prog) {
    cerr << "Usage: " << prog << " -mode <gpu_base|gpu_rabitq>\n"
         << "  -base   <base.fvecs>         base vectors\n"
         << "  -query  <query.fvecs>        query vectors\n"
         << "  -gt     <groundtruth.ivecs>  ground truth\n"
         << "  -graph  <graph_index>        graph index (gpu_base) or qg codebook (gpu_rabitq)\n"
         << "  [-k      <int>]              top-K results (default: 100)\n"
         << "  [-beam   <int>]              beam size (default: 128)\n"
         << "  [-degree <int>]              graph degree (gpu_rabitq, default: 32)\n"
         << "  [-csv    <output.csv>]       append stats to CSV\n"
         << "  [-debug  <on|off>]           debug mode (default: off)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    string mode, base_file, query_file, gt_file, graph_file, csv_file;
    int K = 100, beam_size = 128, degree = 32;
    bool debug_mode = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-mode" && i + 1 < argc)   { mode = argv[++i]; }
        else if (arg == "-base"  && i+1 < argc) { base_file  = argv[++i]; }
        else if (arg == "-query" && i+1 < argc) { query_file = argv[++i]; }
        else if (arg == "-gt"    && i+1 < argc) { gt_file    = argv[++i]; }
        else if (arg == "-graph" && i+1 < argc) { graph_file = argv[++i]; }
        else if (arg == "-k"     && i+1 < argc) { K          = atoi(argv[++i]); }
        else if (arg == "-beam"  && i+1 < argc) { beam_size  = atoi(argv[++i]); }
        else if (arg == "-degree"&& i+1 < argc) { degree     = atoi(argv[++i]); }
        else if (arg == "-csv"   && i+1 < argc) { csv_file   = argv[++i]; }
        else if (arg == "-debug" && i+1 < argc) {
            string v = argv[++i];
            debug_mode = (v == "on" || v == "1" || v == "true");
        }
        else { cerr << "Unknown argument: " << arg << "\n"; usage(argv[0]); return 1; }
    }

    if (mode.empty() || base_file.empty() || query_file.empty() || gt_file.empty() || graph_file.empty()) {
        cerr << "Error: -mode, -base, -query, -gt, -graph are required.\n";
        usage(argv[0]);
        return 1;
    }

    cout << "========================================\n"
         << "GPU Beam Search\n"
         << "========================================\n"
         << "Mode:    " << mode << "\n"
         << "Base:    " << base_file << "\n"
         << "Query:   " << query_file << "\n"
         << "GT:      " << gt_file << "\n"
         << "Graph:   " << graph_file << "\n"
         << "K:       " << K << "\n"
         << "Beam:    " << beam_size << "\n";
    if (mode == "gpu_rabitq") cout << "Degree:  " << degree << "\n";
    if (!csv_file.empty())    cout << "CSV:     " << csv_file << "\n";
    if (debug_mode)           cout << "Debug:   on\n";
    cout << "========================================\n\n";

    srand(42);

    vector<vid_t> results;
    vector<float> distances;
    vector<float> queries;
    vector<int> groundtruth;
    size_t nq = 0;
    int gt_k = 0;
    double elapsed = 0.0;
    GpuRunStats gpu_stats;

    if (mode == "gpu_base") {
        IndexGraph<float> index;
        index.load_graph_index(graph_file.c_str());
        index.load_data(base_file.c_str());

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file.c_str(), nq, query_dim);
        if (debug_mode) { cout << "[DEBUG] running only query 0\n"; nq = min<size_t>(nq, 1); }
        if (nq > 10000) { cout << "Limiting queries to 10000\n"; nq = 10000; }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file.c_str(), gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        index.search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);

    } else if (mode == "gpu_rabitq") {
        IndexGraph<float> index;
        index.load_data(base_file.c_str());

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file.c_str(), nq, query_dim);
        if (debug_mode) { cout << "[DEBUG] running only query 0\n"; nq = min<size_t>(nq, 1); }
        if (nq > 10000) { cout << "Limiting queries to 10000\n"; nq = 10000; }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file.c_str(), gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, graph_file.c_str());
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed, &gpu_stats, debug_mode);

    } else {
        cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k);

    cout << "\n========================================\n"
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
        cout << "Saved GPU stats CSV row to: " << csv_file << "\n";
    }

    return 0;
}
