#include <chrono>
#include <cfloat>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "cpu_base_search.hpp"

using json = nlohmann::json;

static float* read_fvecs(const char* filename, size_t& n, size_t& d) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot open ") + filename);
    int dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    n = file_size / (dim * sizeof(float) + sizeof(int));
    d = dim;
    float* data = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        int dd;
        in.read(reinterpret_cast<char*>(&dd), sizeof(int));
        in.read(reinterpret_cast<char*>(data + i * d), d * sizeof(float));
    }
    return data;
}

static int* read_ivecs(const char* filename, size_t& n, size_t& d) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot open ") + filename);
    int dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    n = file_size / ((dim + 1) * sizeof(int));
    d = dim;
    int* data = new int[n * d];
    for (size_t i = 0; i < n; i++) {
        int dd;
        in.read(reinterpret_cast<char*>(&dd), sizeof(int));
        in.read(reinterpret_cast<char*>(data + i * d), d * sizeof(int));
    }
    return data;
}

static float compute_recall(const vid_t* predicted, const int* groundtruth,
                            size_t nq, int topk, int gt_k) {
    size_t correct = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<int> seen;
        seen.reserve(static_cast<size_t>(topk));
        for (int j = 0; j < topk; j++) {
            int pred = static_cast<int>(predicted[i * topk + j]);
            if (!seen.insert(pred).second) continue;
            for (int k = 0; k < gt_k; k++) {
                if (pred == groundtruth[i * gt_k + k]) { correct++; break; }
            }
        }
    }
    return static_cast<float>(correct) / (nq * topk);
}

static vid_t compute_entry_point(const CPUGraph& graph) {
    size_t n = graph.num_nodes;
    int dim = graph.dim;
    const float* data = graph.vectors.data();

    std::vector<float> centroid(static_cast<size_t>(dim), 0.0f);
    float inv = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; i++) {
        const float* row = data + i * dim;
        for (int d = 0; d < dim; d++) centroid[static_cast<size_t>(d)] += row[d] * inv;
    }

    vid_t best = 0;
    float best_dist = FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        const float* row = data + i * dim;
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = row[d] - centroid[static_cast<size_t>(d)];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best = static_cast<vid_t>(i);
        }
    }
    return best;
}

int main() {
    json j;
    try {
        j = json::parse(std::cin, nullptr, true, true);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }

    if (!j.contains("data")) {
        std::cerr << "Missing \"data\" key" << std::endl;
        return 1;
    }
    auto& jd = j["data"];
    std::string graph_file = jd.at("graph");
    std::string base_file  = jd.at("base");
    std::string query_file = jd.at("query");
    std::string gt_file    = jd.at("gt");

    CPUGraph cpu_graph;
    cpu_graph.load(graph_file, base_file);
    int dim = cpu_graph.dim;
    size_t npoints = cpu_graph.num_nodes;

    size_t nq = 0, query_dim = 0;
    float* h_queries = read_fvecs(query_file.c_str(), nq, query_dim);

    size_t gt_nq = 0, gt_k_sz = 0;
    int* h_gt = read_ivecs(gt_file.c_str(), gt_nq, gt_k_sz);
    int gt_k = static_cast<int>(gt_k_sz);

    vid_t entry_point = compute_entry_point(cpu_graph);

    std::cout << "data: " << npoints << " points, dim=" << dim
              << ", degree=" << cpu_graph.degree
              << ", queries=" << nq
              << ", entry=" << entry_point << std::endl;

    if (!j.contains("search")) {
        std::cerr << "Missing \"search\" key" << std::endl;
        return 1;
    }
    auto& js = j["search"];
    std::string mode = js.value("mode", "base");
    int K = js.value("K", 100);
    if (mode != "base") {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    CPUBaseSearchCfg cfg;
    cfg.topk_size = js.value("topk_size", 256u);
    cfg.entry_point = entry_point;
    cfg.max_iterations = js.value("max_iterations", 1024);
    cfg.trace_first_query = js.value("trace_first_query", false);
    cfg.trace_iterations = js.value("trace_iterations", 0);

    CPUBaseSearch searcher(cpu_graph, cfg);
    std::vector<vid_t> h_results(nq * static_cast<size_t>(K));

    std::cout << "search: mode=" << mode << " K=" << K << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t qid = 0; qid < nq; ++qid) {
        searcher.search(h_queries + qid * static_cast<size_t>(dim),
                        h_results.data() + qid * static_cast<size_t>(K), K,
                        cfg.trace_first_query && qid == 0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "  time: " << elapsed * 1000.0 << " ms"
              << "  qps: " << nq / elapsed << std::endl;

    float recall = compute_recall(h_results.data(), h_gt, nq, K, gt_k);
    std::cout << "recall@" << K << ": " << recall << std::endl;

    delete[] h_queries;
    delete[] h_gt;
    return 0;
}
