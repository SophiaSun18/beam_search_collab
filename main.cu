#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "index.hpp"
#include "search_bitonic.cuh"

float compute_recall(const vid_t* results, const int* groundtruth, 
                     int nq, int k, int gt_k) {
    int64_t correct = 0;
    
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            int res_id = results[q * k + i];
            for (int j = 0; j < k && j < gt_k; j++) {
                if (res_id == groundtruth[q * gt_k + j]) {
                    correct++;
                    break;
                }
            }
        }
    }
    
    return (float)correct / (nq * k);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <data.fvecs> <query.fvecs> <gt.ivecs> <graph_prefix> [K=100] [beam_size=64]\n";
        return 1;
    }
    
    const char* data_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = argv[3];
    const char* graph_prefix = argv[4];
    int K = (argc >= 6) ? std::atoi(argv[5]) : 100;
    int beam_size = (argc >= 7) ? std::atoi(argv[6]) : BEAM_SIZE;
    
    std::cout << "========================================\n";
    std::cout << "GPU Beam Search\n";
    std::cout << "========================================\n";
    std::cout << "Data file: " << data_file << "\n";
    std::cout << "Query file: " << query_file << "\n";
    std::cout << "Ground truth: " << gt_file << "\n";
    std::cout << "Graph prefix: " << graph_prefix << "\n";
    std::cout << "K: " << K << "\n";
    std::cout << "Beam size: " << beam_size << "\n";
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
    
    // Execute the search
    std::cout << "Starting GPU beam search...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    index.search(nq, queries.data(), K, distances.data(), results.data(), beam_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Compute the performance metrics
    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    
    // Compute the Recall
    float recall = compute_recall(results.data(), groundtruth.data(), nq, K, gt_k);
    
    // Output the results
    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    std::cout << "Total time: " << elapsed * 1000 << " ms\n";
    std::cout << "Throughput: " << qps << " queries/sec\n";
    std::cout << "Avg latency: " << latency_ms << " ms/query\n";
    std::cout << "Recall@" << K << ": " << recall << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
