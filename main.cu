#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <string>
#include <algorithm>

#include "index.hpp"
#include "search_bitonic.cuh"

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

int main(int argc, char** argv) {
    auto print_usage = [&](int code) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.fvecs> <query.fvecs> <gt.ivecs> <graph_prefix> [K=100] [beam_size=" << BEAM_SIZE << "]"
                  << " [--warmup N=0] [--runs N=1]\n";
        return code;
    };

    int warmup = 0;
    int runs = 1;
    std::vector<const char*> pos;
    pos.reserve((size_t)argc);
    for (int i = 1; i < argc; i++) {
        const std::string a(argv[i]);
        if (parse_int_arg(argc, argv, i, "--warmup", warmup)) continue;
        if (parse_int_arg(argc, argv, i, "--runs", runs)) continue;
        if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n";
            return print_usage(1);
        }
        pos.push_back(argv[i]);
    }
    if ((int)pos.size() < 4) {
        return print_usage(1);
    }

    const char* data_file = pos[0];
    const char* query_file = pos[1];
    const char* gt_file = pos[2];
    const char* graph_prefix = pos[3];
    int K = ((int)pos.size() >= 5) ? std::atoi(pos[4]) : 100;
    int beam_size = ((int)pos.size() >= 6) ? std::atoi(pos[5]) : BEAM_SIZE;

    if (warmup < 0) warmup = 0;
    if (runs < 1) runs = 1;
    if (K <= 0) {
        std::cerr << "Invalid K: " << K << " (must be > 0)\n";
        return print_usage(1);
    }
    if (beam_size <= 0 || beam_size > BEAM_SIZE) {
        std::cerr << "Invalid beam_size: " << beam_size
                  << " (must be in [1, " << BEAM_SIZE << "])\n";
        return print_usage(1);
    }
    if (K > beam_size) {
        std::cerr << "Invalid config: K=" << K << " exceeds beam_size=" << beam_size << "\n";
        return print_usage(1);
    }
    
    std::cout << "========================================\n";
    std::cout << "GPU Beam Search\n";
    std::cout << "========================================\n";
    std::cout << "Data file: " << data_file << "\n";
    std::cout << "Query file: " << query_file << "\n";
    std::cout << "Ground truth: " << gt_file << "\n";
    std::cout << "Graph prefix: " << graph_prefix << "\n";
    std::cout << "K: " << K << "\n";
    std::cout << "Beam size: " << beam_size << " (compile-time max: " << BEAM_SIZE << ")\n";
    std::cout << "Warmup runs: " << warmup << "\n";
    std::cout << "Timed runs: " << runs << "\n";
    std::cout << "========================================\n\n";
    
    // Initialize the random seed
    srand(42);
    
    // Load the index
    IndexGraph<float> index;
    
    std::cout << "Loading graph...\n";
    index.load_graph(graph_prefix);
    
    std::cout << "Loading data...\n";
    index.load_data(data_file);
    
    // Load the queries
    size_t nq;
    int query_dim;
    std::cout << "Loading queries...\n";
    std::vector<float> queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
    if (query_dim != index.d) {
        std::cerr << "Dimension mismatch: query dim=" << query_dim
                  << ", index dim=" << index.d << "\n";
        return 1;
    }
    if (nq == 0) {
        std::cerr << "No queries loaded from file: " << query_file << "\n";
        return 1;
    }
    
    // Limit the number of queries 
    if (nq > 10000) {
        std::cout << "Limiting queries to 10000\n";
        nq = 10000;
    }
    
    // Load the ground truth
    size_t gt_nq;
    int gt_k;
    std::cout << "Loading ground truth...\n";
    std::vector<int> groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
    if (gt_k <= 0 || gt_nq == 0) {
        std::cerr << "Invalid ground truth: nq=" << gt_nq << ", k=" << gt_k << "\n";
        return 1;
    }
    const size_t recall_nq = std::min(nq, gt_nq);
    if (recall_nq < nq) {
        std::cout << "Warning: ground truth queries (" << gt_nq
                  << ") < search queries (" << nq
                  << "), recall will use first " << recall_nq << " queries\n";
    }
    const int recall_k = std::min(K, gt_k);
    if (recall_k < K) {
        std::cout << "Warning: ground truth top-k=" << gt_k
                  << " < requested K=" << K
                  << ", reporting Recall@" << recall_k << "\n";
    }
    
    std::cout << "\n";
    std::cout << "Data points: " << index.ntotal << "\n";
    std::cout << "Dimension: " << index.d << "\n";
    std::cout << "Max degree: " << index.maxDeg << "\n";
    std::cout << "Queries: " << nq << "\n";
    std::cout << "GT K: " << gt_k << "\n";
    std::cout << "\n";
    
    // Allocate the result array
    std::vector<vid_t> results(nq * K);
    std::vector<float> distances(nq * K);
    
    // Warmup (excluded from timing)
    for (int w = 0; w < warmup; w++) {
        std::cout << "Warmup run " << (w + 1) << "/" << warmup << "...\n";
        index.search((int)nq, queries.data(), K, distances.data(), results.data(), beam_size);
    }

    // Timed runs
    std::cout << "Starting GPU beam search (timed)...\n";
    double elapsed = 0.0;
    for (int r = 0; r < runs; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        index.search((int)nq, queries.data(), K, distances.data(), results.data(), beam_size);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration<double>(end - start).count();
    }
    elapsed /= (double)runs;
    
    // Compute the performance metrics
    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    
    // Compute the Recall
    float recall = compute_recall(results.data(), groundtruth.data(), (int)recall_nq, K, gt_k);
    
    // Output the results
    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    std::cout << "Total time: " << elapsed * 1000 << " ms\n";
    std::cout << "Throughput: " << qps << " queries/sec\n";
    std::cout << "Avg latency: " << latency_ms << " ms/query\n";
    std::cout << "Recall@" << recall_k << " (on " << recall_nq << " queries): " << recall << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
