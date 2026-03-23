#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath> 
#include <cfloat>
#include <chrono>
#include <omp.h>

#include "../include/utils.h"
#include "../include/ctimer.h"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
float PRUNE_RATIO = 0.0;
float APPROX_RATIO = 0.0;

SearchResult beam_cpu_base(
    Graph& g, int K, int npoints,
    const float* d_data, const float* d_query, size_t dim,
    const vector<vidType>& starting_points)
{
    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);     // hash table to store visited nodes
    int count_expanded = 0;                                     // number of expanded nodes
    new_pqueue_t<vidType> frontier(BEAM_SIZE);                  // current frontier to hold all candidates
    vector<pair<vidType, float>> curr_candidate(g.degree);      // the distance of each neighbor, used in pruning, clear after each iteration

    size_t dist_cmps = 0;                                       // number of distance computations
    size_t dist_est = 0;
    size_t pruned = 0;                                          // number of pruned neighbor candidates
    size_t neighbor_pruned = 0;
    size_t visited_pruned = 0;

    // initialize the current frontier with randomly picked starting points and compute their distances
    int start_node = -1;
    float start_dist = FLT_MAX;

    for (vidType p_id : starting_points) {
        float dist = compute_distance(dim, d_query, d_data + p_id * dim);
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
        curr_candidate.clear();

        // get the closest unexpanded node in the frontier
        int idx = frontier.get_next_index();
        g.prefetch_node(frontier[idx]);
        frontier.set_front_expanded();
        count_expanded++;

        // expand the current node
        const vidType* neighbors = g.get_neighbors(frontier[idx]);
        if (frontier.has_unexpanded()) {
            int idx2 = frontier.get_next_index();
            g.prefetch_node(frontier[idx2]);
        }

        // count the exact distance between each neighbor and the query
        for (size_t i = 0; i < g.degree; i++) {
            vidType n = neighbors[i];
            if (is_visited.contains(n)) {
                visited_pruned++;
                pruned++;
                continue;   // skip if already computed
            }
            
            float dist = compute_distance(dim, d_query, d_data + n * dim);
            is_visited.add(n);
            dist_cmps++;
            curr_candidate.emplace_back(n, dist);
        }

        // only do pruning when current iteration is in distance approximation stage
        if (iter > BEAM_SIZE * APPROX_RATIO) {
            // partial sort the neighbors by their distance to the query and prune by the ratio
            size_t num_kept = g.degree * (1 - PRUNE_RATIO);
            size_t iter_pruned = 0;
            if (num_kept > 0 && curr_candidate.size() > num_kept) {
                iter_pruned = curr_candidate.size() - num_kept;
                pruned += iter_pruned;
                neighbor_pruned += iter_pruned;
                nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
                    [](const pair<vidType,float>& a, const pair<vidType,float>& b) {
                        return a.second < b.second;
                    });
                curr_candidate.resize(num_kept);
            }
        }

        // add the candidates after pruning into the frontier
        for (int i = 0; i < curr_candidate.size(); i++) {
            vidType n = curr_candidate[i].first;
            float dist = curr_candidate[i].second;
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
    res.dist_est = dist_est;
    res.iters = iter;
    res.pruned = pruned;
    res.neighbor_pruned = neighbor_pruned;
    res.visited_pruned = visited_pruned;
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
    int64_t total_hop = 0;
    int64_t total_pruned = 0;
    int64_t total_neighbor_pruned = 0;
    int64_t total_visited_pruned = 0;
    vector<vector<int>> all_results(n_query, vector<int>(K));

    ctimer_t t;
    ctimer_start(&t);

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_hop) reduction(+:total_pruned) reduction(+:total_neighbor_pruned) reduction(+:total_visited_pruned)
    for (int q = 0; q < n_query; q++) {
        vector<vidType> starting_points(K);
        for (int i = 0; i < K; i++)
            starting_points[i] = rand() % n_base;
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_base(g, K, n_base, base_vectors, query_ptr, dim, starting_points);
            
        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_hop += result.iters;
        total_pruned += result.pruned;
        total_neighbor_pruned += result.neighbor_pruned;
        total_visited_pruned += result.visited_pruned;
    }

    ctimer_stop(&t);
    ctimer_measure(&t);

    RunStats stats;
    stats.runtime = timespec_sec(t.elapsed);
    stats.latency = stats.runtime / n_query * 1000.0;
    stats.throughput = double(n_query) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, n_query, K, dim_gt);
    stats.total_dc = total_count_dc;
    stats.total_ds = total_count_ds;
    stats.total_iter = total_hop;
    stats.total_pruned = total_pruned;
    stats.total_neighbor_pruned = total_neighbor_pruned;
    stats.total_visited_pruned = total_visited_pruned;

    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # distance computation: " << stats.total_dc << "\n";
    cout << "average # distance computation: " << stats.total_dc / n_query << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / n_query << "\n";
    cout << "total # pruned neighbors: " << stats.total_pruned << "\n";
    cout << "average # pruned neighbors: " << stats.total_pruned / n_query << "\n";

    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -graph <graph_index>\n" << " [-beam <int>]" 
        << " [-p_ratio <float>]" << " [-a_ratio <float>]" << " [-csv <output.csv>]\n";
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
        else if (arg == "-beam" && i + 1 < argc) {
            BEAM_SIZE = atoi(argv[++i]);
        }
        else if (arg == "-p_ratio" && i + 1 < argc) {
            PRUNE_RATIO = atof(argv[++i]);
        }
        else if (arg == "-a_ratio" && i + 1 < argc) {
            APPROX_RATIO = atof(argv[++i]);
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
    RunStats stats = run_ann_experiment_base(g, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt);
    
    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }
    return 0;
}