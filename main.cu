#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "include/index.hpp"
#include "include/utils.hpp"
#include "include/search_bitonic.cuh"
using namespace std;

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] 
             << " <data.fvecs> <query.fvecs> <gt.ivecs> <graph_prefix> [K=100] [beam_size=128]\n";
        return 1;
    }
    
    const char* data_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = argv[3];
    const char* graph_prefix = argv[4];
    int K = (argc >= 6) ? atoi(argv[5]) : 100;
    int beam_size = (argc >= 7) ? atoi(argv[6]) : BEAM_SIZE;
    
    cout << "========================================\n";
    cout << "GPU Beam Search Loading from: \n";
    cout << "========================================\n";
    cout << "Data file: " << data_file << "\n";
    cout << "Query file: " << query_file << "\n";
    cout << "Ground truth: " << gt_file << "\n";
    cout << "Graph prefix: " << graph_prefix << "\n";
    cout << "========================================\n\n";
    
    // Initialize the random seed
    srand(42);
    
    // Load the index
    IndexGraph<float> index;
    // index.load_graph(graph_prefix);
    index.load_graph_index(graph_prefix);
    index.load_data(data_file);
    
    // Load the queries
    size_t nq;
    int query_dim;
    vector<float> queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
    
    // Limit the number of queries 
    if (nq > 10000) {
        cout << "Limiting queries to 10000\n";
        nq = 10000;
    }
    
    // Load the ground truth
    size_t gt_nq;
    int gt_k;
    vector<int> groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
    
    // Allocate the result array
    vector<vid_t> results(nq * K);
    vector<float> distances(nq * K);
    
    // Execute the search
    cout << "\nStarting GPU beam search...\n";
    auto start = chrono::high_resolution_clock::now();
    
    index.search(nq, queries.data(), K, distances.data(), results.data(), beam_size);
    
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();
    
    // Compute the performance metrics
    double qps = nq / elapsed;
    double latency_ms = elapsed / nq * 1000;
    
    // Compute the Recall
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