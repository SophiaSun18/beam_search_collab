// Part 1 microbench (CPU): cost of a single distance op.
//   - exact L2   : compute_distance(dim, query, row)
//   - rabitq est : qg.estimated_distance() (per neighbor) and
//                  qg.scan_all_neighbors_32() (batched, reported per neighbor)
// Reports ns/op. A volatile sink prevents the compiler from eliminating the loop.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../include/utils.h"
#include "../include/qg.hpp"
using namespace std;

int main(int argc, char** argv) {
    string base_file, query_file, qg_codebook;
    int degree = 32;
    long reps = 2000000;   // ops per measurement
    int nodes = 256;       // distinct nodes cycled to mimic realistic access

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-base" && i + 1 < argc) base_file = argv[++i];
        else if (a == "-query" && i + 1 < argc) query_file = argv[++i];
        else if (a == "-qg" && i + 1 < argc) qg_codebook = argv[++i];
        else if (a == "-degree" && i + 1 < argc) degree = atoi(argv[++i]);
        else if (a == "-reps" && i + 1 < argc) reps = atol(argv[++i]);
        else if (a == "-nodes" && i + 1 < argc) nodes = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (base_file.empty() || query_file.empty() || qg_codebook.empty()) {
        fprintf(stderr, "Usage: %s -base <base.fvecs> -query <query.fvecs> -qg <codebook>"
                        " [-degree 32] [-reps N] [-nodes M]\n", argv[0]);
        return 1;
    }

    g_metric_type = infer_metric_from_dataset_path(base_file.c_str());

    size_t n_base, dim_base, n_query, dim_query;
    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    (void)base_vectors;

    QuantizationGraph qg(n_base, dim_base, degree, qg_codebook);
    qg.set_metric(g_metric_type);
    const size_t dim = dim_base;
    if (nodes > (int)n_base) nodes = (int)n_base;

    // prepare one query (quantized) once, mirroring the search inner loop
    size_t padded_dim = (size_t)qg.get_padded_dim();
    QuantizationGraph::QueryBuffers qb;
    qb.rotated_query = (float*)aligned_alloc(64, padded_dim * sizeof(float));
    qb.quantized_query = (uint8_t*)aligned_alloc(64, padded_dim * sizeof(uint8_t));
    qb.lut = (uint8_t*)aligned_alloc(64, padded_dim << 2);
    auto qf = qg.query_prepare(query_vectors, qb);

    volatile float sink = 0.0f;

    // --- exact L2 ---
    {
        auto t0 = chrono::high_resolution_clock::now();
        float acc = 0.0f;
        for (long i = 0; i < reps; ++i) {
            uint32_t node = (uint32_t)(i % nodes);
            acc += compute_distance(dim, query_vectors, qg.get_row(node));
        }
        auto t1 = chrono::high_resolution_clock::now();
        sink += acc;
        double ns = chrono::duration<double, nano>(t1 - t0).count() / reps;
        printf("CPU,exact_l2,%.4f\n", ns);
    }

    // --- rabitq estimate, single neighbor ---
    {
        auto t0 = chrono::high_resolution_clock::now();
        float acc = 0.0f;
        for (long i = 0; i < reps; ++i) {
            uint32_t node = (uint32_t)(i % nodes);
            int rank = (int)(i % degree);
            float exact = qf.low_val;  // any base value; cost is invariant
            auto nf = qg.nbr_filter(node, rank);
            acc += qg.estimated_distance(qf, nf, exact);
        }
        auto t1 = chrono::high_resolution_clock::now();
        sink += acc;
        double ns = chrono::duration<double, nano>(t1 - t0).count() / reps;
        printf("CPU,rabitq_est_single,%.4f\n", ns);
    }

    // --- rabitq estimate, batched (fastscan) reported per neighbor ---
    {
        vector<float> appro(degree);
        vector<uint32_t> raw(degree);
        vector<float> rf(degree);
        long calls = reps / degree;
        if (calls < 1) calls = 1;
        auto t0 = chrono::high_resolution_clock::now();
        for (long i = 0; i < calls; ++i) {
            uint32_t node = (uint32_t)(i % nodes);
            qg.scan_all_neighbors_32(node, qf.low_val, appro.data(), qf, raw.data(), rf.data());
        }
        auto t1 = chrono::high_resolution_clock::now();
        for (int j = 0; j < degree; ++j) sink += appro[j];
        double ns = chrono::duration<double, nano>(t1 - t0).count() / (calls * (double)degree);
        printf("CPU,rabitq_est_batched_per_nbr,%.4f\n", ns);
    }

    free(qb.rotated_query); free(qb.quantized_query); free(qb.lut);
    (void)sink;
    return 0;
}
