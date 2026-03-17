#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath> 
#include <cfloat>
#include <cstring>
#include <chrono>
#include <omp.h>

#include "../include/utils.h"
#include "../include/timer.hpp"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
#include "../include/rabitq_filter.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
float PRUNE_RATIO = 0.0;
float APPROX_RATIO = 0.0;
bool DEBUG = false;
constexpr bool TIMING_ENABLED = true;  // set to false to disable all timing overhead at compile time

SearchResult beam_cpu_rabitq(
    Graph& g, int K, int npoints,
    const float* d_data, const float* d_query, size_t dim,
    vidType entry_point, RaBitQFilter& filter)
{
    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);
    vector<pair<vidType, float>> current_candidate(g.degree);
    new_pqueue_t<vidType> frontier(BEAM_SIZE);
    new_pqueue_t<vidType> result_pool(K);

    size_t dist_cmps = 0;
    size_t dist_est = 0;
    size_t pruned = 0;
    size_t neighbor_pruned = 0;
    size_t visited_pruned = 0;
    size_t extra_dist_cmps = 0;

    // timing accumulators
    using clock = std::chrono::high_resolution_clock;
    double t_dist_comp = 0, t_dist_est = 0, t_queue = 0, t_hash_filter = 0, t_other = 0;
    auto t_search_start = clock::now();
    auto t_lap = t_search_start;

    auto lap = [&]() -> double { if constexpr (!TIMING_ENABLED) return 0.0; auto now = clock::now(); auto d = std::chrono::duration<double>(now - t_lap).count(); t_lap = now; return d; };

    // push the starting point to the frontier with max distance
    frontier.push(entry_point, FLT_MAX);
    t_queue += lap();

    // quantize the query once to avoid redundant computations
    auto qf = filter.query_prepare(d_query);
    t_dist_est += lap();

    int iter = 0;
    while (frontier.has_unexpanded()) {    
        iter++;
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        frontier.set_front_expanded();
        t_queue += lap();
        
        g.prefetch_node(cur_node);
        t_other += lap();

        if (is_visited.contains(cur_node)) {
            visited_pruned++;
            pruned++;
            t_queue += lap();
            continue;
        }
        t_queue += lap();

        float exact_dist = compute_distance(dim, d_query, d_data + cur_node * dim);
        dist_cmps++;
        t_dist_comp += lap();

        result_pool.push(cur_node, exact_dist);
        is_visited.add(cur_node);
        t_queue += lap();

        if (DEBUG) {
            // Debug: Print current node being expanded
            std::cout << "\n=== Iteration " << iter << " ===\n";
            std::cout << "Expanding node: " << cur_node << " with exact distance: " << exact_dist << "\n";

            // Debug: Print the current sign table
            std::cout << "Sign table: ";
            for (size_t d = 0; d < dim; d++) {
                std::cout << filter.get_signs_ptr()[d] << " ";
            }
            std::cout << "\n";

            // Debug: Print the rotated query and quantized query
            std::cout << "Rotated query: ";
            for (size_t d = 0; d < dim; d++) {
                std::cout << qf.rotated_query[d] << " ";
            }
            std::cout << "\n";

            std::cout << "Quantized query: ";
            for (size_t d = 0; d < dim; d++) {
                std::cout << static_cast<int>(qf.quantized_query[d]) << " ";
            }
            std::cout << "\n";
        }

        // expand the current node
        const vidType* neighbors = g.get_neighbors(cur_node);
        filter.prefetch_nbr(cur_node);
        current_candidate.clear();

        if (frontier.has_unexpanded()) {
            int idx2 = frontier.get_next_index();
            g.prefetch_node(frontier[idx2]);
        }
        t_other += lap();

        // compute the estimated distance for each neighbor
        for (size_t i = 0; i < g.degree; i++) {
            auto nf = filter.nbr_filter(cur_node, i);
            dist_est++;
            float est = filter.estimated_distance(qf, nf, exact_dist);
            t_dist_est += lap();
            uint32_t est_bits;
            memcpy(&est_bits, &est, sizeof(est_bits));
            if ((est_bits & 0x7F800000u) == 0x7F800000u && (est_bits & 0x007FFFFFu) != 0) {
                // printf("Warning: estimated distance is NaN for neighbor %lu of node %lu\n", neighbors[i], cur_node);
                current_candidate.emplace_back(neighbors[i], -1);
            } else {
                // printf("current neighbor has estimated distance: %f\n", est);
                current_candidate.emplace_back(neighbors[i], est);
            }
            t_other += lap();

            // float gt = compute_distance(dim, d_query, d_data + neighbors[i] * dim);

            if (DEBUG) {
                // Debug: Print neighbor details
                std::cout << std::fixed << std::setprecision(6)
                        << "  Neighbor ID: " << neighbors[i]
                        << ", Estimated Distance: " << est
                        // << ", Exact Distance: " << gt
                        // << ", Error : " << (est - gt) / gt
                        // << ", Code[0]: " << nf.bits[0]
                        // << ", Code[1]: " << nf.bits[1]
                        << ", Factors: [tx: " << nf.triple_x
                        << ", dq: " << nf.factor_dq
                        << ", vq: " << nf.factor_vq
                        << ", sum q: " << qf.sum_q
                        << ", width: " << qf.width
                        << ", low_val: " << qf.low_val
                        // << ", raw_sum: " << raw_sum
                        // << ", result_float: " << result_float
                        << "]\n";
            }
        }

        // size_t num_kept = g.degree * (1 - PRUNE_RATIO);
        // if (num_kept > 0 && num_kept < current_candidate.size()) {
        //     size_t iter_pruned = current_candidate.size() - num_kept;
        //     pruned += iter_pruned;
        //     neighbor_pruned += iter_pruned;
        //     nth_element(current_candidate.begin(), current_candidate.begin() + num_kept, current_candidate.end(),
        //         [](const pair<vidType,float>& a, const pair<vidType,float>& b) {
        //             return a.second < b.second;
        //         });
        //     current_candidate.resize(num_kept);
        // }
        // t_other += lap();

        // push unvisited neighbors to the frontier with estimated distance
        for (auto node : current_candidate) {
            if (is_visited.contains(node.first)) {
                visited_pruned++;
                pruned++;
                t_queue += lap();
                continue;
            }   // skip if already computed
            if (frontier.size() < BEAM_SIZE || node.second < frontier.get_tail_dist()) {
                frontier.push(node.first, node.second);
            }
            t_queue += lap();   // frontier push, either push or not
        }

        if (DEBUG) {
            // Debug: Print current state of the frontier
            std::cout << "Frontier state (size = " << frontier.size() << "):\n";
            for (int i = 0; i < frontier.size(); i++) {
                std::cout << "  Node: " << frontier[i] << " Dist: " << frontier.get_dist(i) << " Visited: " << (is_visited.contains(frontier[i]) ? "Yes" : "No") << "\n";
            }

            // Debug: Print current state of the result pool
            std::cout << "Result pool state (size = " << result_pool.size() << "):\n";
            for (int i = 0; i < result_pool.size(); i++) {
                std::cout << "  Node: " << result_pool[i] << " Dist: " << result_pool.get_dist(i) << "\n";
            }
        }
    }
    t_queue += lap();   // final frontier has unexpanded check

    // update results: expand result pool by visiting neighbors of current results
    int pool_size = result_pool.size();
    if (pool_size != K) {
        for (int i = 0; i < pool_size; i++) {
            vidType data_id = result_pool[i];
            const vidType* neighbors = g.get_neighbors(data_id);
            for (size_t j = 0; j < g.degree; j++) {
                vidType cur_neighbor = neighbors[j];
                t_other += lap();
                if (!is_visited.contains(cur_neighbor)) {
                    is_visited.add(cur_neighbor);
                    t_queue += lap();
                    float dist = compute_distance(dim, d_query, d_data + cur_neighbor * dim);
                    extra_dist_cmps++;
                    t_dist_comp += lap();
                    result_pool.push(cur_neighbor, dist);
                    t_queue += lap();
                } else {
                    t_queue += lap();
                }
            }
            if (result_pool.size() == K) {
                t_queue += lap();
                break;
            }
        }
    }
    t_other += lap();

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    // int topK = min(K, (int)result_pool.size());
    for (int i = 0; i < K; i++)
        result.push_back(result_pool[i]);
    t_other += lap();

    if (DEBUG) {
        std::cout << "Final Results:\n";
        for (int i = 0; i < K; i++) {
            std::cout << "  Neighbor ID: " << result[i] << "\n";
        }
    }

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.dist_est = dist_est;
    res.extra_dist_cmps = extra_dist_cmps;
    res.iters = iter;
    res.pruned = pruned;
    res.neighbor_pruned = neighbor_pruned;
    res.visited_pruned = visited_pruned;
    res.time_dist_comp = t_dist_comp;
    res.time_dist_est = t_dist_est;
    res.time_queue = t_queue;
    res.time_hash_filter = t_hash_filter;
    res.time_other = t_other;
    if constexpr (TIMING_ENABLED)
        res.time_total = std::chrono::duration<double>(clock::now() - t_search_start).count();
    return res;
}

RunStats run_ann_experiment_rabitq(
    Graph& g, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt, RaBitQFilter& filter
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
    int64_t total_neighbor_pruned = 0;
    int64_t total_visited_pruned = 0;
    double total_time_dist_comp = 0, total_time_dist_est = 0, total_time_queue = 0, total_time_hash_filter = 0, total_time_other = 0, total_time_total = 0;
    vector<vector<int>> all_results(n_query, vector<int>(K));

    vector<float> centroid = compute_centroid(base_vectors, n_base, dim, actual_threads);
    vidType entry_point = exact_nn(base_vectors, centroid.data(), n_base, dim, actual_threads);

    int actual_query = (DEBUG) ? 1 : n_query;   // if in debug mode, only run 1 query for easy debugging

    Timer t;
    t.Start();

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_extra_count_dc) reduction(+:total_hop) reduction(+:total_pruned) reduction(+:total_neighbor_pruned) reduction(+:total_visited_pruned) reduction(+:total_time_dist_comp) reduction(+:total_time_dist_est) reduction(+:total_time_queue) reduction(+:total_time_hash_filter) reduction(+:total_time_other) reduction(+:total_time_total)
    for (int q = 0; q < actual_query; q++) {
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_rabitq(g, K, n_base, base_vectors, query_ptr, dim, entry_point, filter);

        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_extra_count_dc += result.extra_dist_cmps;
        total_hop += result.iters;
        total_pruned += result.pruned;
        total_neighbor_pruned += result.neighbor_pruned;
        total_visited_pruned += result.visited_pruned;
        total_time_dist_comp += result.time_dist_comp;
        total_time_dist_est += result.time_dist_est;
        total_time_queue += result.time_queue;
        total_time_hash_filter += result.time_hash_filter;
        total_time_other += result.time_other;
        total_time_total += result.time_total;
    }

    t.Stop();

    RunStats stats;
    stats.runtime = t.Seconds();
    stats.latency = stats.runtime / n_query * 1000.0;
    stats.throughput = double(n_query) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, n_query, K, dim_gt);
    stats.total_dc = total_count_dc;
    stats.total_ds = total_count_ds;
    stats.total_extra_dc = total_extra_count_dc;
    stats.total_full_dc = total_count_dc + total_extra_count_dc;
    stats.total_iter = total_hop;
    stats.total_pruned = total_pruned;
    stats.total_neighbor_pruned = total_neighbor_pruned;
    stats.total_visited_pruned = total_visited_pruned;
    stats.time_dist_comp = total_time_dist_comp;
    stats.time_dist_est = total_time_dist_est;
    stats.time_queue = total_time_queue;
    stats.time_hash_filter = total_time_hash_filter;
    stats.time_other = total_time_other;
    stats.time_total = total_time_total;

    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # distance computation: " << stats.total_full_dc << "\n";
    cout << "average # distance computation: " << stats.total_full_dc / n_query << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / n_query << "\n";
    cout << "total # pruned neighbors: " << stats.total_pruned << "\n";
    cout << "average # pruned neighbors: " << stats.total_pruned / n_query << "\n";

    double t_total = stats.time_dist_comp + stats.time_dist_est + stats.time_queue + stats.time_hash_filter + stats.time_other;
    printf("--- Timing breakdown (total across queries) ---\n");
    printf("  distance computation: %.6f sec (%.1f%%)\n", stats.time_dist_comp, stats.time_dist_comp / t_total * 100);
    printf("  distance estimation:  %.6f sec (%.1f%%)\n", stats.time_dist_est, stats.time_dist_est / t_total * 100);
    printf("  queue management:     %.6f sec (%.1f%%)\n", stats.time_queue, stats.time_queue / t_total * 100);
    printf("  hash filter:          %.6f sec (%.1f%%)\n", stats.time_hash_filter, stats.time_hash_filter / t_total * 100);
    printf("  other:                %.6f sec (%.1f%%)\n", stats.time_other, stats.time_other / t_total * 100);
    printf("  sum:                  %.6f sec\n", t_total);
    printf("  total (wall-clock):   %.6f sec\n", stats.time_total);

    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -graph <graph_index>\n" << " [-rabitq <rabitq_table>]" << "[-sign <sign_table>]"
        << " [-beam <int>]" << " [-p_ratio <float>]" << " [-a_ratio <float>]" << " [-csv <output.csv>]\n";
        return 1;
    }

    // required input files
    string base_file, query_file, gt_file, graph_file, rabitq_table, signs_table;
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
        else if (arg == "-rabitq" && i + 1 < argc) {
            rabitq_table = argv[++i];
        }
        else if (arg == "-sign" && i + 1 < argc) {
            signs_table = argv[++i];
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

    if (base_file.empty() || query_file.empty() || gt_file.empty() || graph_file.empty() || rabitq_table.empty()) {
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
    // g.load_qg_index(graph_file.c_str(), n_base, dim_base, 32);

    // initialize the filter
    RaBitQFilter filter(dim_base, g.degree, rabitq_table, signs_table);  // 6 bits for quantization as SymQG

    // Run experiment
    RunStats stats = run_ann_experiment_rabitq(g, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt, filter);

    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }
    return 0;
}