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
float PRUNE_RATIO = 0.0;
float PRUNE_DELTA = 0.125;
int PRUNE_INTERVAL = 10;
float APPROX_RATIO = 0.0;

SearchResult beam_cpu_pathw(
    Graph& g, int K, int npoints,
    const float* d_data, const float* d_query, size_t dim,
    const vector<vidType>& starting_points, const uint32_t* signbit_table, 
    size_t packed_dim, size_t sign_bit_vector_size)
{
    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);     // hash table to store visited nodes
    vector<pair<vidType, int>> matching_score(g.degree);        // the matching bits of each neighbor, clear after each iteration
    vector<uint32_t> query_dir(packed_dim);                     // store the direction from current node to query vector
    int count_expanded = 0;                                     // number of expanded nodes
    new_pqueue_t<vidType> frontier(BEAM_SIZE);                  // current frontier to hold all candidates

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
        // get the closest unexpanded node in the frontier
        int idx = frontier.get_next_index();
        g.prefetch_node(frontier[idx]);
        frontier.set_front_expanded();
        count_expanded++;

        // get the direction between current node and query node
        vidType cur_node = frontier[idx];
        const uint32_t* cur_node_table = signbit_table + cur_node * sign_bit_vector_size;
        get_sign_bit_vec(d_data + cur_node * dim, d_query, query_dir.data(), dim);

        // expand the current node
        const vidType* neighbors = g.get_neighbors(cur_node);
        matching_score.clear();

        if (frontier.has_unexpanded()) {
            int idx2 = frontier.get_next_index();
            g.prefetch_node(frontier[idx2]);
        }

        if (iter > BEAM_SIZE * APPROX_RATIO) {
            // count the matching bits for each neighbor's direction
            for (size_t i = 0; i < g.degree; i++) {
                vidType n = neighbors[i];
                if (is_visited.contains(n)) {
                    visited_pruned++;
                    pruned++;
                    continue;
                }   // skip if already computed
                const uint32_t* neighbor_dir = cur_node_table + i * packed_dim;
                int matching_bits = get_matching_bits(query_dir.data(), neighbor_dir, packed_dim);
                dist_est++;
                matching_score.emplace_back(n, matching_bits);
            }

            size_t scored = matching_score.size();
            size_t iter_pruned = 0;
            size_t num_kept = g.degree * (1 - PRUNE_RATIO);
            if (num_kept > 0 && num_kept < matching_score.size()) {
                iter_pruned = matching_score.size() - num_kept;
                pruned += iter_pruned;
                neighbor_pruned += iter_pruned;
                nth_element(matching_score.begin(), matching_score.begin() + num_kept, matching_score.end(),
                    [](const pair<vidType,int>& a, const pair<vidType,int>& b) {
                        return a.second > b.second;
                    });
                matching_score.resize(num_kept);
            }

            // l2 distance computation for top 2 neighbors
            for (auto node : matching_score) {
                vidType n = node.first;
                float dist = compute_distance(dim, d_query, d_data + n * dim);
                is_visited.add(n);
                dist_cmps++;
                if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                    frontier.push(n, dist);
                }
            }
        } else {
            // count the exact l2 distance for each neighbor's direction
            for (size_t i = 0; i < g.degree; i++) {
                vidType n = neighbors[i];
                if (is_visited.contains(n)) {
                    visited_pruned++;
                    pruned++;
                    continue;
                }   // skip if already computed
                float dist = compute_distance(dim, d_query, d_data + n * dim);
                is_visited.add(n);
                dist_cmps++;
                if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                    frontier.push(n, dist);
                }
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

RunStats run_ann_experiment_pathw(
    Graph& g, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt, uint32_t* signbit_table
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

    size_t packed_dim = (dim + 31) / 32;
    size_t sign_bit_vector_size = g.degree * packed_dim;

    Timer t;
    t.Start();

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_hop) reduction(+:total_pruned) reduction(+:total_neighbor_pruned) reduction(+:total_visited_pruned)
    for (int q = 0; q < n_query; q++) {
        vector<vidType> starting_points(K);
        for (int i = 0; i < K; i++)
            starting_points[i] = rand() % n_base;
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_pathw(g, K, n_base, base_vectors, query_ptr, dim, starting_points, 
                                signbit_table, packed_dim, sign_bit_vector_size);
            
        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_hop += result.iters;
        total_pruned += result.pruned;
        total_neighbor_pruned += result.neighbor_pruned;
        total_visited_pruned += result.visited_pruned;
    }

    t.Stop();

    RunStats stats;
    stats.runtime = t.Seconds();
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
        << " -graph <graph_index>\n" << " [-signbit <signbit.bin>]"
        << " [-rabitq <prefix>]" << " [-straight]" << " [-qbits <2|4>]"
        << " [-beam <int>]" << " [-p_ratio <float>]" << " [-a_ratio <float>]" << " [-csv <output.csv>]\n";
        return 1;
    }

    // required input files
    string base_file, query_file, gt_file, graph_file, signbit_file;
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
        else if (arg == "-signbit" && i + 1 < argc) {
            signbit_file = argv[++i];
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

    size_t n_loaded, sign_bit_vector_size;
    uint32_t* signbit_table = read_bin<uint32_t>(signbit_file.c_str(), n_loaded, sign_bit_vector_size);

    // Run experiment
    RunStats stats = run_ann_experiment_pathw(g, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt, signbit_table);

    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }
    return 0;
}