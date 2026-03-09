#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <string>
#include <algorithm>
#include <unordered_map>

#include "index.hpp"
#include "search_bitonic_reflow_entry.cuh"

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

int main(int argc, char** argv) {
    // Force line-buffer-like behavior for easier remote timeout diagnosis.
    std::cout << std::unitbuf;

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
    
    // Optional cap for fast smoke runs on unstable remote environments.
    size_t query_cap = 10000;
    if (const char* cap_env = std::getenv("MAX_QUERIES")) {
        long long v = std::atoll(cap_env);
        if (v > 0) query_cap = static_cast<size_t>(v);
    }
    if (nq > query_cap) {
        std::cout << "Limiting queries to " << query_cap << "\n";
        nq = query_cap;
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
    float recall_legacy = compute_recall_legacy(results.data(), groundtruth.data(), (int)recall_nq, K, gt_k);
    double avg_unique_at_k = compute_avg_unique_at_k(results.data(), (int)recall_nq, K);
    double duplicate_ratio = 1.0 - (avg_unique_at_k / (double)K);

    const char* dump_query0_env = std::getenv("DUMP_QUERY0");
    if (dump_query0_env && dump_query0_env[0] != '\0' && recall_nq > 0) {
        const int show_n = std::min(K, 20);
        std::cout << "Query0 results(top" << show_n << "):";
        for (int i = 0; i < show_n; i++) {
            std::cout << " " << (int)results[i];
        }
        std::cout << "\n";

        const int show_gt = std::min(gt_k, 20);
        std::cout << "Query0 gt(top" << show_gt << "):";
        for (int i = 0; i < show_gt; i++) {
            std::cout << " " << groundtruth[i];
        }
        std::cout << "\n";

        std::unordered_map<int, int> freq;
        freq.reserve((size_t)K * 2);
        int q0_unique = 0;
        int q0_match_unique = 0;
        std::vector<int> gt_seen((size_t)gt_k, 0);
        for (int i = 0; i < K; i++) {
            int rid = (int)results[i];
            auto it = freq.find(rid);
            if (it == freq.end()) {
                freq.emplace(rid, 1);
                q0_unique++;
                for (int j = 0; j < gt_k; j++) {
                    if (!gt_seen[(size_t)j] && rid == groundtruth[(size_t)j]) {
                        gt_seen[(size_t)j] = 1;
                        q0_match_unique++;
                        break;
                    }
                }
            } else {
                it->second += 1;
            }
        }
        std::cout << "Query0 unique@" << K << ": " << q0_unique << ", unique_matches: " << q0_match_unique << "\n";
        std::cout << "Query0 repeated ids (freq>1):";
        int printed = 0;
        for (const auto& kv : freq) {
            if (kv.second > 1) {
                std::cout << " [" << kv.first << ":" << kv.second << "]";
                printed++;
                if (printed >= 12) break;
            }
        }
        if (printed == 0) std::cout << " none";
        std::cout << "\n";
    }
    
    // Output the results
    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    std::cout << "Total time: " << elapsed * 1000 << " ms\n";
    std::cout << "Throughput: " << qps << " queries/sec\n";
    std::cout << "Avg latency: " << latency_ms << " ms/query\n";
    std::cout << "Recall@" << recall_k << " (unique hit, on " << recall_nq << " queries): " << recall << "\n";
    std::cout << "Recall@" << recall_k << " (legacy non-unique): " << recall_legacy << "\n";
    std::cout << "Avg unique IDs@" << K << ": " << avg_unique_at_k << "\n";
    std::cout << "TopK duplicate ratio: " << duplicate_ratio << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
