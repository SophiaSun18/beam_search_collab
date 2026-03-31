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
#include "../include/ctimer.h"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
#include "../include/qg.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
float PRUNE_RATIO = 0.0;
float APPROX_RATIO = 0.0;
bool DEBUG = false;
constexpr bool TIMING_ENABLED = false;  // set to false to disable all timing overhead at compile time
constexpr bool DUP_HANDLING_ENABLED = false;  // set to true to enable duplicate handling

SearchResult beam_cpu_rabitq(
    QuantizationGraph& qg, int K, const float* d_query, size_t dim, vidType entry_point)
{
    int degree = qg.get_degree();

    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);
    hash_filter<vidType> is_added(DUP_HANDLING_ENABLED ? BEAM_SIZE * BEAM_SIZE : 1);
    vector<pair<vidType, float>> curr_candidate(degree);
    new_pqueue_t<vidType> frontier(BEAM_SIZE);
    new_pqueue_t<vidType> result_pool(K);

    vector<float> appro_dist;
    vector<uint16_t> raw_buf;
    vector<float> result_f_buf;
    if constexpr (FASTSCAN_ENABLED) {
        appro_dist.resize(degree);
        raw_buf.resize(degree);
        result_f_buf.resize(degree);
    }

    size_t dist_cmps = 0;
    size_t dist_est = 0;
    size_t pruned = 0;
    size_t neighbor_pruned = 0;
    size_t visited_pruned = 0;
    size_t extra_dist_cmps = 0;
    size_t pushes = 0;
    size_t duplicate_pushes = 0;

    // pre-allocate buffers for query preparation
    size_t padded_dim = 1ULL << static_cast<size_t>(std::ceil(std::log2(dim)));
    QuantizationGraph::QueryBuffers query_buffers;
    query_buffers.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
    query_buffers.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
    if constexpr (FASTSCAN_ENABLED) {
        query_buffers.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));
    }

    // push the starting point to the frontier with max distance
    frontier.push(entry_point, FLT_MAX);
    pushes++;
    t_queue += lap();
    if constexpr (DUP_HANDLING_ENABLED) {
        is_added.add(entry_point);
        t_hash_filter += lap();
    }

    // quantize the query once to avoid redundant computations
    auto qf = qg.query_prepare(d_query, query_buffers);

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
            t_hash_filter += lap();
            continue;
        }
        t_hash_filter += lap();

        float exact_dist = compute_distance(dim, d_query, qg.get_row(cur_node));
        dist_cmps++;
        t_dist_comp += lap();

        result_pool.push(cur_node, exact_dist);
        t_queue += lap();
        is_visited.add(cur_node);
        t_hash_filter += lap();

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
        const vidType* neighbors = qg.get_neighbors(cur_node);

        // compute the estimated distance for each neighbor
        if constexpr (FASTSCAN_ENABLED) {
            filter.scan_all_neighbors(cur_node, exact_dist, appro_dist.data(), qf, raw_buf.data(), result_f_buf.data(), DEBUG);
            t_dist_est += lap();
            for (size_t i = 0; i < g.degree; i++) {
                float est = appro_dist[i];
                dist_est++;
                current_candidate.emplace_back(neighbors[i], est);
            }
            t_other += lap();
        } else {
            for (size_t i = 0; i < g.degree; i++) {
                auto nf = filter.nbr_filter(cur_node, i);
                dist_est++;
                float est = filter.estimated_distance(qf, nf, exact_dist);
                t_dist_est += lap();
                current_candidate.emplace_back(neighbors[i], est);
                t_other += lap();

                if (DEBUG) {
                    // Debug: Print neighbor details
                    std::cout << std::fixed << std::setprecision(6)
                            << "  Neighbor ID: " << neighbors[i]
                            << ", Estimated Distance: " << est
                            << ", Factors: [tx: " << nf.triple_x
                            << ", dq: " << nf.factor_dq
                            << ", vq: " << nf.factor_vq
                            << ", sum q: " << qf.sum_q
                            << ", width: " << qf.width
                            << ", low_val: " << qf.low_val
                            << "]\n";
                }
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
                t_hash_filter += lap();
                continue;
            }   // skip if already computed
            t_hash_filter += lap();
            if (frontier.size() < BEAM_SIZE || node.second < frontier.get_tail_dist()) {
                if constexpr (DUP_HANDLING_ENABLED) {
                    if (is_added.add(node.first)) {
                        frontier.push(node.first, node.second);
                        pushes++;
                    } else {
                        pushes++;
                        duplicate_pushes++;
                        frontier.push_update(node.first, node.second);
                    }
                } else {
                    frontier.push(node.first, node.second);
                    pushes++;
                }
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
            const vidType* neighbors = qg.get_neighbors(data_id);
            for (int j = 0; j < degree; j++) {
                vidType cur_neighbor = neighbors[j];
                t_other += lap();
                if (!is_visited.contains(cur_neighbor)) {
                    is_visited.add(cur_neighbor);
                    float dist = compute_distance(dim, d_query, qg.get_row(cur_neighbor));
                    extra_dist_cmps++;
                    t_dist_comp += lap();
                    result_pool.push(cur_neighbor, dist);
                    t_queue += lap();
                } else {
                    t_hash_filter += lap();
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
    res.hash_inserts = is_visited.get_total_inserts();
    res.hash_collisions = is_visited.get_total_collisions();
    res.pushes = pushes;
    res.duplicate_pushes = duplicate_pushes;
    if constexpr (TIMING_ENABLED)
        res.time_total = std::chrono::duration<double>(clock::now() - t_search_start).count();
    return res;
}

RunStats run_ann_experiment_rabitq(
    QuantizationGraph& qg, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt
) {
    int actual_threads = 0;
    #pragma omp parallel
    {
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int actual_query = (DEBUG) ? 1 : n_query;   // if in debug mode, only run 1 query for easy debugging

    int64_t total_count_dc = 0;
    int64_t total_count_ds = 0;
    int64_t total_extra_count_dc = 0;
    int64_t total_hop = 0;
    int64_t total_pruned = 0;
    int64_t total_neighbor_pruned = 0;
    int64_t total_visited_pruned = 0;
    double total_time_dist_comp = 0, total_time_dist_est = 0, total_time_queue = 0, total_time_hash_filter = 0, total_time_other = 0, total_time_total = 0;
    int64_t total_hash_inserts = 0, total_hash_collisions = 0;
    int64_t total_pushes = 0;
    int64_t total_duplicate_pushes = 0;
    vector<vector<int>> all_results(n_query, vector<int>(K));

    // vector<float> centroid = compute_centroid(base_vectors, n_base, dim, actual_threads);
    // vidType entry_point = exact_nn(base_vectors, centroid.data(), n_base, dim, actual_threads);
    vidType entry_point = qg.get_entry_point();

    ctimer_t t;
    ctimer_start(&t);

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_extra_count_dc) reduction(+:total_hop) reduction(+:total_pruned) reduction(+:total_neighbor_pruned) reduction(+:total_visited_pruned) reduction(+:total_time_dist_comp) reduction(+:total_time_dist_est) reduction(+:total_time_queue) reduction(+:total_time_hash_filter) reduction(+:total_time_other) reduction(+:total_time_total) reduction(+:total_hash_inserts) reduction(+:total_hash_collisions) reduction(+:total_pushes) reduction(+:total_duplicate_pushes)
    for (int q = 0; q < actual_query; q++) {
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_rabitq(qg, K, query_ptr, dim, entry_point);

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
        total_hash_inserts += result.hash_inserts;
        total_hash_collisions += result.hash_collisions;
        total_pushes += result.pushes;
        total_duplicate_pushes += result.duplicate_pushes;
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
    stats.total_hash_inserts = total_hash_inserts;
    stats.total_hash_collisions = total_hash_collisions;
    stats.total_pushes = total_pushes;
    stats.total_duplicate_pushes = total_duplicate_pushes;

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
    printf("--- Hash collision stats ---\n");
    printf("  total inserts:    %ld\n", stats.total_hash_inserts);
    printf("  total collisions: %ld\n", stats.total_hash_collisions);
    printf("  collision ratio:  %.4f%%\n", stats.total_hash_inserts > 0 ? (double)stats.total_hash_collisions / stats.total_hash_inserts * 100 : 0.0);
    printf("  avg collisions per query: %.1f\n", (double)stats.total_hash_collisions / n_query);

    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -degree <int>" << " [-mode <codebook|bins>]"
        << " [-qg <qg_codebook>]" << " [-graph <graph_index>]"
        << " [-rabitq <rabitq_table>]" << " [-sign <sign_table>]"
        << " [-beam <int>]" << " [-p_ratio <float>]" << " [-a_ratio <float>]" << " [-csv <output.csv>]\n";
        return 1;
    }

    // required input files
    string base_file, query_file, gt_file, graph_file, rabitq_table, signs_table, qg_codebook;
    string mode = "codebook";
    string csv_file;
    int degree = 32;

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
        else if (arg == "-mode" && i + 1 < argc) {
            mode = argv[++i];
        }
        else if (arg == "-graph" && i + 1 < argc) {
            graph_file = argv[++i];
        }
        else if (arg == "-degree" && i + 1 < argc) {
            degree = atoi(argv[++i]);
        }
        else if (arg == "-rabitq" && i + 1 < argc) {
            rabitq_table = argv[++i];
        }
        else if (arg == "-sign" && i + 1 < argc) {
            signs_table = argv[++i];
        }
        else if (arg == "-qg" && i + 1 < argc) {
            qg_codebook = argv[++i];
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

    if (mode != "codebook" && mode != "bins") {
        cerr << "Error: -mode must be either 'codebook' or 'bins'.\n";
        return 1;
    }

    if (base_file.empty() || query_file.empty() || gt_file.empty()) {
        cerr << "Error: missing required input files.\n";
        return 1;
    }

    if (mode == "codebook" && qg_codebook.empty()) {
        cerr << "Error: codebook mode requires -qg <qg_codebook>.\n";
        return 1;
    }

    if (mode == "bins" && (graph_file.empty() || rabitq_table.empty() || signs_table.empty())) {
        cerr << "Error: bins mode requires -graph <graph_index> -rabitq <rabitq_table> -sign <sign_table>.\n";
        return 1;
    }

    // load all input data
    size_t n_base, dim_base;
    size_t n_query, dim_query;
    size_t n_gt, dim_gt;

    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    const int* groundtruth = read_ivecs(gt_file.c_str(), n_gt, dim_gt);

    QuantizationGraph qg = (mode == "codebook")
        ? QuantizationGraph(n_base, dim_base, degree, qg_codebook)
        : QuantizationGraph(n_base, dim_base, degree, graph_file, rabitq_table, signs_table, base_vectors);

    if (mode == "bins") {
        int actual_threads = 0;
        #pragma omp parallel
        {
            actual_threads = omp_get_num_threads();
        }
        vector<float> centroid = compute_centroid(base_vectors, n_base, dim_base, actual_threads);
        vidType entry_point = exact_nn(base_vectors, centroid.data(), n_base, dim_base, actual_threads);
        qg.set_entry_point(entry_point);
        cout << "Computed bins-mode entry point: " << entry_point << "\n";
    }

    // Run experiment
    RunStats stats = run_ann_experiment_rabitq(qg, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt);

    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }
    return 0;
}
