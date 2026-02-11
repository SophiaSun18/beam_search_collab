#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath> 
#include <cfloat>
#include <chrono>
#include <omp.h>

#include "../include/utils.h"
#include "../include/timer.hpp"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
int MAX_DEGREE = 32;
int DIM = 128;

SearchResult beam_cpu_base(
    Graph& g, int K, int npoints,
    const float* d_data, const float* d_query, size_t dim,
    const vector<vidType>& starting_points)
{
    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);     // hash table to store visited nodes
    size_t dist_cmps = 0;                                       // number of distance computations
    new_pqueue_t<vidType> frontier(BEAM_SIZE);                  // current frontier to hold all candidates

    // initialize the current frontier with randomly picked starting points and compute their distances
    int start_node = -1;
    float start_dist = FLT_MAX;

    for (vidType p_id : starting_points) {
        float dist = compute_distance_vec512(dim, d_query, d_data + p_id * dim);
        dist_cmps++;
        if (dist < start_dist) {
            start_node = p_id;
            start_dist = dist;
        }
    }
    frontier.push(start_node, start_dist);
    is_visited.add(start_node);

    int iter = 0;
    while (frontier.has_unexpanded()) {    
        iter++;
        // get the closest unexpanded node in the frontier
        int idx = frontier.get_next_index();
        g.prefetch_node(frontier[idx]);
        frontier.set_front_expanded();

        // expand the current node
        const vidType* neighbors = g.get_neighbors(frontier[idx]);
        if (frontier.has_unexpanded()) {
            int idx2 = frontier.get_next_index();
            g.prefetch_node(frontier[idx2]);
        }

        // count the matching bits for each neighbor's direction
        for (size_t i = 0; i < g.degree; i++) {
            vidType n = neighbors[i];
            if (is_visited.contains(n)) continue;   // skip if already computed
            
            float dist = compute_distance_vec512(dim, d_query, d_data + n * dim);
            is_visited.add(n);
            dist_cmps++;
            if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                frontier.push(n, dist);
            }
        }
    }

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    int topK = min(K, (int)frontier.size());
    for (int i = 0; i < topK; i++)
        result.push_back(frontier[i]);

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.iters = iter;
    return res;
}

RunStats run_ann_experiment_base(
    Graph& g, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth
) {
    int actual_threads = 0;
    #pragma omp parallel
    {
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int64_t total_count_dc = 0;
    int64_t total_hop = 0;
    vector<vector<int>> all_results(n_query, vector<int>(K));

    Timer t;
    t.Start();

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_count_dc) reduction(+:total_hop)
    for (int q = 0; q < n_query; q++) {
        vector<vidType> starting_points(K);
        for (int i = 0; i < K; i++)
            starting_points[i] = rand() % n_base;
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_base(g, K, n_base, base_vectors, query_ptr, dim, starting_points);
            
        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_hop += result.iters;
    }

    t.Stop();

    RunStats stats;
    stats.runtime = t.Seconds();
    stats.latency = stats.runtime / n_query * 1000.0;
    stats.throughput = double(n_query) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, n_query, K);
    stats.total_dc = total_count_dc;
    stats.total_iter = total_hop;

    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # distance computation: " << stats.total_dc << "\n";
    cout << "average # distance computation: " << stats.total_dc / n_query << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / n_query << "\n";

    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -graph <graph_index>\n" << " [-beam <int>]\n";
        return 1;
    }

    // receive all required input files from arg
    string base_file, query_file, gt_file, graph_file;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "-base" && i + 1 < argc) {
            base_file = argv[++i];
        }
        else if (arg == "-query" && i + 1 < argc) {
            query_file = argv[++i];
        }
        else if (arg == "-gt" && i + 1 < argc) {
            gt_file = argv[++i];
        }
        else if (arg == "-graph" && i + 1 < argc) {
            graph_file = argv[++i];
        }
        else if (arg == "-beam" && i + 1 < argc) {
            BEAM_SIZE = atoi(argv[++i]);
        }
        else {
            cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    if (base_file.empty() || query_file.empty() || gt_file.empty()   || graph_file.empty()) {
        cerr << "Error: missing required input files.\n";
        return 1;
    }

    // load all input data
    size_t n_base, dim_base;
    size_t n_query, dim_query;
    size_t n_gt, dim_gt;

    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    const int* groundtruth = read_ivecs(gt_file.c_str(), n_gt, dim_gt);

    Graph g;
    g.load_graph_index(graph_file.c_str());

    // Run experiment
    RunStats stats = run_ann_experiment_base(g, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth);
    
    return 0;
}