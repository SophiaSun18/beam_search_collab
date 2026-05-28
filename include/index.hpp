#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>

typedef uint32_t vid_t;

enum MetricType { METRIC_L2, METRIC_IP };

template <typename T>
class IndexGraph {
public:
    size_t ntotal;           // number of data points
    int d;                   // vector dimension
    int maxDeg;              // maximum degree
    MetricType metric;       // distance metric type
    std::string graph_prefix; // graph path prefix (for sidecar files)
    std::vector<vid_t> edges; // adjacency list (ntotal * maxDeg)
    std::vector<T> data;     // data vectors (ntotal * d)

    IndexGraph() : ntotal(0), d(0), maxDeg(0), metric(METRIC_L2) {}

    // Get data pointer
    T* get_data_ptr() { return data.data(); }

    // Load graph file (.meta.txt + .edge.bin format by default).
    // If GRAPH_INDEX_FORMAT=read_bin|header_bin, prefer <prefix> read_bin layout:
    // [int32 n][int32 degree][n*degree*vidType].
    void load_graph(const std::string& prefix) {
        graph_prefix = prefix;
        const char* graph_format_env = std::getenv("GRAPH_INDEX_FORMAT");
        const std::string graph_format = graph_format_env ? std::string(graph_format_env) : std::string();
        const bool prefer_read_bin = (graph_format == "read_bin" || graph_format == "header_bin");

        if (prefer_read_bin && try_load_graph_read_bin(prefix)) {
            return;
        }
        if (try_load_graph_meta_edge(prefix)) {
            return;
        }
        if (try_load_graph_read_bin(prefix)) {
            return;
        }

        std::cerr << "Error: cannot load graph from prefix " << prefix
                  << " (tried meta/edge and read_bin formats)" << std::endl;
        exit(1);
    }

private:
    bool try_load_graph_meta_edge(const std::string& prefix) {
        std::string meta_file = prefix + ".meta.txt";
        std::ifstream f_meta(meta_file);
        if (!f_meta.is_open()) {
            return false;
        }

        int64_t nv = 0, ne = 0;
        int vid_size = 0, eid_size = 0, vlabel_size = 0, elabel_size = 0, feat_len = 0, nvc = 0, nec = 0;
        f_meta >> nv >> ne >> vid_size >> eid_size >> vlabel_size >> elabel_size
               >> maxDeg >> feat_len >> nvc >> nec;
        f_meta.close();
        if (nv <= 0 || maxDeg <= 0 || feat_len <= 0) {
            std::cerr << "Warning: invalid graph meta file " << meta_file << std::endl;
            return false;
        }

        std::string edge_file = prefix + ".edge.bin";
        std::ifstream f_edge(edge_file, std::ios::binary);
        if (!f_edge.is_open()) {
            return false;
        }

        ntotal = static_cast<size_t>(nv);
        d = feat_len;
        std::cout << "Loading graph(meta+edge): n=" << ntotal << ", maxDeg=" << maxDeg << std::endl;

        edges.resize(ntotal * static_cast<size_t>(maxDeg));
        f_edge.read(reinterpret_cast<char*>(edges.data()),
                    static_cast<std::streamsize>(sizeof(vid_t) * ntotal * static_cast<size_t>(maxDeg)));
        const bool ok = static_cast<bool>(f_edge);
        f_edge.close();
        if (!ok) {
            std::cerr << "Warning: failed to read full edge payload from " << edge_file << std::endl;
            edges.clear();
            return false;
        }

        std::cout << "Graph loaded(meta+edge): " << edges.size() << " edges" << std::endl;
        return true;
    }

    bool try_load_graph_read_bin(const std::string& prefix) {
        std::ifstream f(prefix, std::ios::binary);
        if (!f.is_open()) {
            return false;
        }

        int32_t n = 0;
        int32_t degree = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&degree), sizeof(int32_t));
        if (!f || n <= 0 || degree <= 0) {
            return false;
        }

        f.seekg(0, std::ios::end);
        const std::streamoff file_size = f.tellg();
        const size_t edge_count = static_cast<size_t>(n) * static_cast<size_t>(degree);
        const std::streamoff expected_size =
            static_cast<std::streamoff>(sizeof(int32_t) * 2 + sizeof(vid_t) * edge_count);
        if (file_size != expected_size) {
            return false;
        }

        f.seekg(static_cast<std::streamoff>(sizeof(int32_t) * 2), std::ios::beg);
        edges.resize(edge_count);
        f.read(reinterpret_cast<char*>(edges.data()),
               static_cast<std::streamsize>(sizeof(vid_t) * edge_count));
        const bool ok = static_cast<bool>(f);
        f.close();
        if (!ok) {
            edges.clear();
            return false;
        }

        ntotal = static_cast<size_t>(n);
        maxDeg = degree;
        std::cout << "Loading graph(read_bin): n=" << ntotal << ", maxDeg=" << maxDeg << std::endl;
        std::cout << "Graph loaded(read_bin): " << edges.size() << " edges" << std::endl;
        return true;
    }

public:

    // Load data vectors (.fvecs format)
    void load_data(const std::string& data_file) {
        std::ifstream f(data_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << data_file << std::endl;
            exit(1);
        }

        // Read first dimension
        int dim_check;
        f.read(reinterpret_cast<char*>(&dim_check), sizeof(int));

        // Get file size to calculate number of vectors
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t n = file_size / (sizeof(int) + dim_check * sizeof(T));

        // Verify count matches if graph is already loaded
        if (ntotal > 0 && n != ntotal) {
            std::cerr << "Warning: data file has " << n << " vectors, but graph has "
                      << ntotal << " vertices" << std::endl;
        }

        if (ntotal == 0) ntotal = n;
        if (d == 0) d = dim_check;

        std::cout << "Loading data: n=" << n << ", dim=" << d << std::endl;

        // Read all vectors
        data.resize(ntotal * d);
        f.seekg(0, std::ios::beg);

        for (size_t i = 0; i < ntotal; i++) {
            int dim_i;
            f.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
            assert(dim_i == d);
            f.read(reinterpret_cast<char*>(&data[i * d]), d * sizeof(T));
        }
        f.close();

        std::cout << "Data loaded: " << ntotal << " vectors" << std::endl;
    }

    // Load query vectors
    static std::vector<T> load_queries(const std::string& query_file, size_t& nq, int& dim) {
        std::ifstream f(query_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << query_file << std::endl;
            exit(1);
        }

        // Read dimension
        f.read(reinterpret_cast<char*>(&dim), sizeof(int));

        // Get number of vectors
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + dim * sizeof(T));

        std::cout << "Loading queries: nq=" << nq << ", dim=" << dim << std::endl;

        // Read all queries
        std::vector<T> queries(nq * dim);
        f.seekg(0, std::ios::beg);

        for (size_t i = 0; i < nq; i++) {
            int dim_i;
            f.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
            assert(dim_i == dim);
            f.read(reinterpret_cast<char*>(&queries[i * dim]), dim * sizeof(T));
        }
        f.close();

        return queries;
    }

    // Load ground truth
    static std::vector<int> load_groundtruth(const std::string& gt_file, size_t& nq, int& k) {
        std::ifstream f(gt_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << gt_file << std::endl;
            exit(1);
        }

        // Read k
        f.read(reinterpret_cast<char*>(&k), sizeof(int));

        // Get number of queries
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + k * sizeof(int));

        // Read all ground truth
        std::vector<int> gt(nq * k);
        f.seekg(0, std::ios::beg);

        for (size_t i = 0; i < nq; i++) {
            int k_i;
            f.read(reinterpret_cast<char*>(&k_i), sizeof(int));
            f.read(reinterpret_cast<char*>(&gt[i * k]), k * sizeof(int));
        }
        f.close();

        return gt;
    }

    // GPU search entry points. Two parallel host paths share the same
    // graph/data state but launch different kernels:
    //   search_gpu_base  -> exact GPU beam search (beam_gpu_base.cu)
    //   search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto -> full Q-LEAF (T6) GPU beam search
    void search_gpu_base(int nq, const T* queries, int K, vid_t* result_idx, int beam_sz);
    void search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto(int nq, const T* queries, int K, vid_t* result_idx, int beam_sz);
};
