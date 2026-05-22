#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <omp.h>

#include "../include/utils.h"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
float PRUNE_RATIO = 0.0;
float APPROX_RATIO = 0.0;
bool DEBUG_MODE = false;

SearchResult beam_cpu_base(
    Graph& g, int K, int npoints,
    const float* d_data, const float* d_query, size_t dim,
    vidType entry_point)
{
    (void)npoints;
    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);     // hash table to store visited nodes
    new_pqueue_t<vidType> frontier(BEAM_SIZE);                  // current frontier to hold all candidates
    vector<pair<vidType, float>> curr_candidate(g.degree);      // the distance of each neighbor, used in pruning, clear after each iteration

    size_t dist_cmps = 0;                                       // number of distance computations
    size_t dist_est = 0;
    size_t extra_dist_cmps = 0;
    size_t neighbor_pruned = 0;
    
    // push the fixed entry point to the frontier with max distance
    frontier.push(entry_point, FLT_MAX);

    int iter = 0;
    while (frontier.has_unexpanded()) {    
        iter++;
        curr_candidate.clear();

        // get the closest unexpanded node in the frontier
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        g.prefetch_node(frontier[idx]);
        frontier.set_front_expanded();

        // expand the current node
        const vidType* neighbors = g.get_neighbors(cur_node);
        if (frontier.has_unexpanded()) {
            int idx2 = frontier.get_next_index();
            g.prefetch_node(frontier[idx2]);
        }

        // count the exact distance between each neighbor and the query
        for (size_t i = 0; i < g.degree; i++) {
            vidType n = neighbors[i];
            if (is_visited.contains(n)) {
                continue;   // skip if already computed
            }

            float dist = compute_distance(dim, d_query, d_data + n * dim);
            dist_cmps++;

            is_visited.add(n);
            curr_candidate.emplace_back(n, dist);
        }

        // only do pruning when current iteration is in distance approximation stage
        // if (iter > BEAM_SIZE * APPROX_RATIO) {
        if (iter < 0) {
            // partial sort the neighbors by their distance to the query and prune by the ratio
            size_t num_kept = g.degree * (1 - PRUNE_RATIO);
            size_t iter_pruned = 0;
            if (num_kept > 0 && curr_candidate.size() > num_kept) {
                iter_pruned = curr_candidate.size() - num_kept;
                neighbor_pruned += iter_pruned;
                nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
                    [](const pair<vidType,float>& a, const pair<vidType,float>& b) {
                        return a.second < b.second;
                    });
                curr_candidate.resize(num_kept);
            }
        }

        // add the candidates after pruning into the frontier
        for (size_t i = 0; i < curr_candidate.size(); i++) {
            vidType n = curr_candidate[i].first;
            float dist = curr_candidate[i].second;
            if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                frontier.push(n, dist);
            }
        }

        if (DEBUG_MODE) {
            debug_print_candidates(iter, cur_node, curr_candidate);
            debug_print_beam_state(frontier);
        }
    }

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    int topK = min(K, (int)frontier.size());
    for (int i = 0; i < topK; i++)
        result.push_back(frontier[i]);

    if (DEBUG_MODE) {
        debug_print_final_results(frontier);
    }

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.dist_est = dist_est;
    res.extra_dist_cmps = extra_dist_cmps;
    res.iters = iter;
    res.neighbor_pruned = neighbor_pruned;
    return res;
}

RunStats run_ann_experiment_base(
    Graph& g, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt
) {
    int actual_threads = 0;
    #pragma omp parallel
    {
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int64_t total_count_dc = 0;
    int64_t total_count_ds = 0;
    int64_t total_extra_count_dc = 0;
    int64_t total_hop = 0;
    int64_t total_pruned = 0;
    const int query_count = DEBUG_MODE ? std::min<size_t>(1, n_query) : static_cast<int>(n_query);
    vector<vector<int>> all_results(query_count, vector<int>(K));

    // compute fixed entry point: centroid of base vectors -> nearest neighbor
    vector<float> centroid = compute_centroid(base_vectors, n_base, dim, actual_threads);
    vidType entry_point = exact_nn(base_vectors, centroid.data(), n_base, dim, actual_threads);
    cout << "Entry point: " << entry_point << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    if (DEBUG_MODE) {
        cout << "[DEBUG] debug mode enabled: running only query 0\n";
        for (int q = 0; q < query_count; q++) {
            const float* query_ptr = query_vectors + q * dim;
            SearchResult result = beam_cpu_base(g, K, n_base, base_vectors, query_ptr, dim, entry_point);

            all_results[q] = result.neighbors;
            total_count_dc += result.dist_cmps;
            total_count_ds += result.dist_est;
            total_extra_count_dc += result.extra_dist_cmps;
            total_hop += result.iters;
            total_pruned += result.neighbor_pruned;
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 1) \
            reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_extra_count_dc) \
            reduction(+:total_hop) reduction(+:total_pruned)
        for (int q = 0; q < query_count; q++) {
            const float* query_ptr = query_vectors + q * dim;

            SearchResult result = beam_cpu_base(g, K, n_base, base_vectors, query_ptr, dim, entry_point);

            all_results[q] = result.neighbors;
            total_count_dc += result.dist_cmps;
            total_count_ds += result.dist_est;
            total_extra_count_dc += result.extra_dist_cmps;
            total_hop += result.iters;
            total_pruned += result.neighbor_pruned;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    RunStats stats;
    stats.runtime = std::chrono::duration<double>(end - start).count();
    stats.latency = stats.runtime / query_count * 1000.0;
    stats.throughput = double(query_count) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, query_count, K, dim_gt);
    stats.total_dc = total_count_dc;
    stats.total_ds = total_count_ds;
    stats.total_extra_dc = total_extra_count_dc;
    stats.total_full_dc = total_count_dc + total_extra_count_dc;
    stats.total_iter = total_hop;
    stats.total_pruned = total_pruned;

    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # distance computation: " << stats.total_full_dc << "\n";
    cout << "average # distance computation: " << stats.total_full_dc / query_count << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / query_count << "\n";
    cout << "total # pruned neighbors: " << stats.total_pruned << "\n";
    cout << "average # pruned neighbors: " << stats.total_pruned / query_count << "\n";

    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -graph <graph_index>\n" << " [-k <int>]" << " [-beam <int>]" 
        << " [-p_ratio <float>]" << " [-a_ratio <float>]"
        << " [-debug <on|off|1|0>]" << " [-csv <output.csv>]\n";
        return 1;
    }

    // receive all required input files from arg
    string base_file, query_file, gt_file, graph_file;
    string csv_file;

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
        else if (arg == "-k" && i + 1 < argc) {
            K = atoi(argv[++i]);
        }
        else if (arg == "-beam" && i + 1 < argc) {
            BEAM_SIZE = atoi(argv[++i]);
        }
        else if (arg == "-p_ratio" && i + 1 < argc) {
            PRUNE_RATIO = atof(argv[++i]);
        }
        else if (arg == "-a_ratio" && i + 1 < argc) {
            APPROX_RATIO = atof(argv[++i]);
        }
        else if (arg == "-debug" && i + 1 < argc) {
            string debug_arg = argv[++i];
            if (debug_arg == "on" || debug_arg == "1" || debug_arg == "true") {
                DEBUG_MODE = true;
            }
            else if (debug_arg == "off" || debug_arg == "0" || debug_arg == "false") {
                DEBUG_MODE = false;
            }
            else {
                cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                return 1;
            }
        }
        else if (arg == "-csv" && i + 1 < argc) {
            csv_file = argv[++i];
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

    g_metric_type = infer_metric_from_dataset_path(base_file);
    cout << "Metric: " << metric_name(g_metric_type) << "\n";

    // load all input data
    size_t n_base, dim_base;
    size_t n_query, dim_query;
    size_t n_gt, dim_gt;

    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    const int* groundtruth = read_ivecs(gt_file.c_str(), n_gt, dim_gt);

    Graph g;
    g.load_graph_index(graph_file.c_str());

    cout << "Debug mode: " << (DEBUG_MODE ? "on" : "off") << "\n";

    // Run experiment
    RunStats stats = run_ann_experiment_base(g, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt);
    
    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }
    return 0;
}
