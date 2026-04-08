#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "include/index.hpp"
#include "include/qg.hpp"
#include "include/utils.h"
#include "include/search_bitonic.cuh"
using namespace std;

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] 
             << " <mode:gpu_base|gpu_rabitq> <data.fvecs> <query.fvecs> <gt.ivecs> <graph_or_codebook>"
             << " [K=100] [beam_size=128] [degree=32]\n";
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

    if (mode == "gpu_base") {
        if (argc < 6) {
            cerr << "gpu_base requires: <mode> <data> <query> <gt> <graph_index> [K] [beam_size]\n";
            return 1;
        }
        const char* graph_prefix = argv[5];
        K = (argc >= 7) ? atoi(argv[6]) : 100;
        beam_size = (argc >= 8) ? atoi(argv[7]) : 128;
        cout << "Graph prefix: " << graph_prefix << "\n";
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_graph_index(graph_prefix);
        index.load_data(data_file);

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        if (nq > 10000) {
            cout << "Limiting queries to 10000\n";
            nq = 10000;
        }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);
        
        index.search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed);
    } else if (mode == "gpu_rabitq") {
        if (argc < 7) {
            cerr << "gpu_rabitq requires: <mode> <data> <query> <gt> <qg_codebook> [K] [beam_size] [degree]\n";
            return 1;
        }
        const char* qg_codebook = argv[5];
        K = (argc >= 7) ? atoi(argv[6]) : 100;
        beam_size = (argc >= 8) ? atoi(argv[7]) : 128;
        int degree = (argc >= 9) ? atoi(argv[8]) : 32;
        cout << "QG codebook: " << qg_codebook << "\n";
        cout << "========================================\n\n";

        IndexGraph<float> index;
        index.load_data(data_file);

        int query_dim = 0;
        queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
        if (nq > 10000) {
            cout << "Limiting queries to 10000\n";
            nq = 10000;
        }

        size_t gt_nq = 0;
        groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
        results.resize(nq * K);
        distances.resize(nq * K);

        QuantizationGraph qg(index.ntotal, index.d, degree, qg_codebook);
        qg.gpu_search(static_cast<int>(nq), queries.data(), K, distances.data(), results.data(), beam_size, elapsed);
    } else {
        cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k);
    
    // Output the results
    cout << "\n========================================\n";
    cout << "Results\n";
    cout << "========================================\n";
    cout << "Total time: " << elapsed * 1000 << " ms\n";
    cout << "Throughput: " << qps << " queries/sec\n";
    cout << "Avg latency: " << latency_ms << " ms/query\n";
    cout << "Recall@" << K << ": " << recall << "\n";
    cout << "========================================\n";
    
    return 0;
}
