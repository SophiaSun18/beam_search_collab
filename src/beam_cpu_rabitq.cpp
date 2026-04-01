#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath> 
#include <cfloat>
#include <cstring>
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

#define TIMING_ENABLED 0

#if TIMING_ENABLED
#define TSTART(tc) ctimer_start(&(tc))
#define TSTOP(tc)  do { ctimer_stop(&(tc)); ctimer_lap(&(tc)); } while(0)
#else
#define TSTART(tc) ((void)0)
#define TSTOP(tc)  ((void)0)
#endif

#define DUP_HANDLING_ENABLED 0

SearchResult beam_cpu_rabitq(
    QuantizationGraph& qg, int K, const float* d_query, size_t dim, vidType entry_point)
{
    struct Candidate {
        vidType vid;
        float dist;
    };

    int degree = qg.get_degree();

    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);
    hash_filter<vidType> is_added(DUP_HANDLING_ENABLED ? BEAM_SIZE * BEAM_SIZE : 1);
    vector<Candidate> curr_candidate;
    curr_candidate.reserve(degree);
    new_pqueue_t<vidType> frontier(BEAM_SIZE);
    new_pqueue_t<vidType> result_pool(K);

    vector<float> appro_dist;
    vector<uint32_t> raw_buf_32;
    vector<float> result_f_buf;
    if constexpr (FASTSCAN_ENABLED) {
        appro_dist.resize(degree);
        raw_buf_32.resize(degree);
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

    // fine-grained timing accumulators (ctimer-based, zero overhead when macros disabled)
    ctimer_t tc_dist_comp, tc_dist_est;
    ctimer_t tc_nbr_filter, tc_query_prepare;
    ctimer_t tc_queue_mgmt, tc_pruning;
    ctimer_t tc_hash_filter;
    ctimer_reset(&tc_dist_comp); ctimer_reset(&tc_dist_est);
    ctimer_reset(&tc_nbr_filter); ctimer_reset(&tc_query_prepare);
    ctimer_reset(&tc_queue_mgmt); ctimer_reset(&tc_pruning);
    ctimer_reset(&tc_hash_filter);

    // pre-allocate buffers for query preparation
    size_t padded_dim = 1ULL << static_cast<size_t>(std::ceil(std::log2(dim)));
    QuantizationGraph::QueryBuffers query_buffers;
    query_buffers.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
    query_buffers.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
    if constexpr (FASTSCAN_ENABLED) {
        query_buffers.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));
    }

    // push the starting point to the frontier with max distance
    TSTART(tc_queue_mgmt);
    frontier.push(entry_point, FLT_MAX);
    pushes++;
    TSTOP(tc_queue_mgmt);
    if constexpr (DUP_HANDLING_ENABLED) {
        TSTART(tc_hash_filter);
        is_added.add(entry_point);
        TSTOP(tc_hash_filter);
    }

    // quantize the query once to avoid redundant computations
    TSTART(tc_query_prepare);
    auto qf = qg.query_prepare(d_query, query_buffers);
    TSTOP(tc_query_prepare);

    int iter = 0;
    while (frontier.has_unexpanded()) {
        iter++;
        TSTART(tc_pruning);
        curr_candidate.clear();
        TSTOP(tc_pruning);

        TSTART(tc_queue_mgmt);
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        frontier.set_front_expanded();
        TSTOP(tc_queue_mgmt);

        TSTART(tc_hash_filter);
        if (is_visited.contains(cur_node)) {
            visited_pruned++;
            pruned++;
            TSTOP(tc_hash_filter);
            continue;
        }
        TSTOP(tc_hash_filter);

        TSTART(tc_dist_comp);
        float exact_dist = compute_distance(dim, d_query, qg.get_row(cur_node));
        dist_cmps++;
        TSTOP(tc_dist_comp);

        TSTART(tc_queue_mgmt);
        result_pool.push(cur_node, exact_dist);
        TSTOP(tc_queue_mgmt);
        
        TSTART(tc_hash_filter);
        is_visited.add(cur_node);
        TSTOP(tc_hash_filter);

        // expand the current node
        const vidType* neighbors = qg.get_neighbors(cur_node);

        // compute the estimated distance for each neighbor
        if constexpr (FASTSCAN_ENABLED) {
            TSTART(tc_dist_est);
            qg.scan_all_neighbors_32(cur_node, exact_dist, appro_dist.data(), qf, raw_buf_32.data(), result_f_buf.data());
            dist_est += degree;
            TSTOP(tc_dist_est);

            for (int i = 0; i < degree; i++) {
                TSTART(tc_hash_filter);
                if (is_visited.contains(neighbors[i])) {
                    visited_pruned++;
                    pruned++;
                    TSTOP(tc_hash_filter);
                    continue;
                }
                TSTOP(tc_hash_filter);
                TSTART(tc_pruning);
                curr_candidate.push_back({neighbors[i], appro_dist[i]});
                TSTOP(tc_pruning);
            }

            if (iter > BEAM_SIZE * APPROX_RATIO) {
                // partial sort the neighbors by their distance to the query and prune by the ratio
                TSTART(tc_pruning);
                size_t num_kept = degree * (1 - PRUNE_RATIO);
                size_t iter_pruned = 0;
                if (num_kept > 0 && curr_candidate.size() > num_kept) {
                    iter_pruned = curr_candidate.size() - num_kept;
                    pruned += iter_pruned;
                    neighbor_pruned += iter_pruned;
                    nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
                        [](const Candidate& a, const Candidate& b) {
                            return a.dist < b.dist;
                        });
                    curr_candidate.resize(num_kept);
                }
                TSTOP(tc_pruning);
            }

            for (size_t i = 0; i < curr_candidate.size(); i++) {
                vidType n = curr_candidate[i].vid;
                float dist = curr_candidate[i].dist;
                TSTART(tc_queue_mgmt);
                if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                    if constexpr (DUP_HANDLING_ENABLED) {
                        if (is_added.add(n)) {
                            frontier.push(n, dist);
                            pushes++;
                        } else {
                            pushes++;
                            duplicate_pushes++;
                            frontier.push_update(n, dist);
                        }
                    } else {
                        frontier.push(n, dist);
                        pushes++;
                    }
                }
                TSTOP(tc_queue_mgmt);
            }

            // prefetch next node
            // if (frontier.has_unexpanded()) {
            //     int new_idx = frontier.get_next_index();
            //     qg.prefetch_nbr(frontier[new_idx]);
            // }
        } else {
            for (size_t i = 0; i < degree; i++) {
                TSTART(tc_hash_filter);
                if (is_visited.contains(neighbors[i])) {
                    visited_pruned++;
                    pruned++;
                    TSTOP(tc_hash_filter);
                    continue;
                }
                TSTOP(tc_hash_filter);

                TSTART(tc_nbr_filter);
                auto nf = qg.nbr_filter(cur_node, i);
                TSTOP(tc_nbr_filter);
                dist_est++;
                TSTART(tc_dist_est);
                float est = qg.estimated_distance(qf, nf, exact_dist);
                TSTOP(tc_dist_est);

                TSTART(tc_pruning);
                curr_candidate.push_back({neighbors[i], est});
                TSTOP(tc_pruning);
            }

            if (iter > BEAM_SIZE * APPROX_RATIO) {
                // partial sort the neighbors by their distance to the query and prune by the ratio
                TSTART(tc_pruning);
                size_t num_kept = degree * (1 - PRUNE_RATIO);
                size_t iter_pruned = 0;
                if (num_kept > 0 && curr_candidate.size() > num_kept) {
                    iter_pruned = curr_candidate.size() - num_kept;
                    pruned += iter_pruned;
                    neighbor_pruned += iter_pruned;
                    nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
                        [](const Candidate& a, const Candidate& b) {
                            return a.dist < b.dist;
                        });
                    curr_candidate.resize(num_kept);
                }
                TSTOP(tc_pruning);
            }

            for (size_t i = 0; i < curr_candidate.size(); i++) {
                vidType n = curr_candidate[i].vid;
                float dist = curr_candidate[i].dist;
                TSTART(tc_queue_mgmt);
                if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                    if constexpr (DUP_HANDLING_ENABLED) {
                        if (is_added.add(n)) {
                            frontier.push(n, dist);
                            pushes++;
                        } else {
                            pushes++;
                            duplicate_pushes++;
                            frontier.push_update(n, dist);
                        }
                    } else {
                        frontier.push(n, dist);
                        pushes++;
                    }
                }
                TSTOP(tc_queue_mgmt);
            }

            // prefetch next node
            // if (frontier.has_unexpanded()) {
            //     int new_idx = frontier.get_next_index();
            //     qg.prefetch_nbr(frontier[new_idx]);
            // }
        }
    }

    // update results: expand result pool by visiting neighbors of current results
    int pool_size = result_pool.size();
    if (pool_size != K) {
        for (int i = 0; i < pool_size; i++) {
            vidType data_id = result_pool[i];
            const vidType* neighbors = qg.get_neighbors(data_id);
            for (int j = 0; j < degree; j++) {
                vidType cur_neighbor = neighbors[j];
                TSTART(tc_hash_filter);
                if (!is_visited.contains(cur_neighbor)) {
                    is_visited.add(cur_neighbor);
                    TSTOP(tc_hash_filter);
                    TSTART(tc_dist_comp);
                    float dist = compute_distance(dim, d_query, qg.get_row(cur_neighbor));
                    extra_dist_cmps++;
                    TSTOP(tc_dist_comp);
                    TSTART(tc_queue_mgmt);
                    result_pool.push(cur_neighbor, dist);
                    TSTOP(tc_queue_mgmt);
                } else {
                    TSTOP(tc_hash_filter);
                }
            }
            if (result_pool.size() == K) break;
        }
    }

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    for (int i = 0; i < K; i++)
        result.push_back(result_pool[i]);

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.dist_est = dist_est;
    res.extra_dist_cmps = extra_dist_cmps;
    res.iters = iter;
    res.pruned = pruned;
    res.neighbor_pruned = neighbor_pruned;
    res.visited_pruned = visited_pruned;
    res.time_dist_comp = timespec_sec(tc_dist_comp.elapsed);
    res.time_nbr_filter = timespec_sec(tc_nbr_filter.elapsed);
    res.time_dist_est = timespec_sec(tc_dist_est.elapsed);
    res.time_queue_mgmt = timespec_sec(tc_queue_mgmt.elapsed);
    res.time_pruning = timespec_sec(tc_pruning.elapsed);
    res.time_hash_filter = timespec_sec(tc_hash_filter.elapsed);
    res.time_query_prepare = timespec_sec(tc_query_prepare.elapsed);
    res.hash_inserts = is_visited.get_total_inserts();
    res.hash_collisions = is_visited.get_total_collisions();
    res.pushes = pushes;
    res.duplicate_pushes = duplicate_pushes;
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

    int64_t total_count_dc = 0;
    int64_t total_count_ds = 0;
    int64_t total_extra_count_dc = 0;
    int64_t total_hop = 0;
    int64_t total_pruned = 0;
    int64_t total_neighbor_pruned = 0;
    int64_t total_visited_pruned = 0;
    double total_time_dist_comp = 0, total_time_nbr_filter = 0, total_time_dist_est = 0;
    double total_time_queue_mgmt = 0, total_time_pruning = 0;
    double total_time_hash_filter = 0;
    double total_time_query_prepare = 0;
    int64_t total_hash_inserts = 0, total_hash_collisions = 0;
    int64_t total_pushes = 0;
    int64_t total_duplicate_pushes = 0;
    vector<vector<int>> all_results(n_query, vector<int>(K));

    vidType entry_point = qg.get_entry_point();

    ctimer_t t;
    ctimer_start(&t);

    #pragma omp parallel for schedule(dynamic, 1) \
        reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_extra_count_dc) \
        reduction(+:total_hop) reduction(+:total_pruned) reduction(+:total_neighbor_pruned) reduction(+:total_visited_pruned) \
        reduction(+:total_time_dist_comp) reduction(+:total_time_nbr_filter) reduction(+:total_time_dist_est) \
        reduction(+:total_time_queue_mgmt) reduction(+:total_time_pruning) \
        reduction(+:total_time_hash_filter) \
        reduction(+:total_time_query_prepare) \
        reduction(+:total_hash_inserts) reduction(+:total_hash_collisions) \
        reduction(+:total_pushes) reduction(+:total_duplicate_pushes)
    for (int q = 0; q < n_query; q++) {
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
        total_time_nbr_filter += result.time_nbr_filter;
        total_time_dist_est += result.time_dist_est;
        total_time_queue_mgmt += result.time_queue_mgmt;
        total_time_pruning += result.time_pruning;
        total_time_hash_filter += result.time_hash_filter;
        total_time_query_prepare += result.time_query_prepare;
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
    stats.time_nbr_filter = total_time_nbr_filter;
    stats.time_dist_est = total_time_dist_est;
    stats.time_queue_mgmt = total_time_queue_mgmt;
    stats.time_pruning = total_time_pruning;
    stats.time_hash_filter = total_time_hash_filter;
    stats.time_query_prepare = total_time_query_prepare;
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

    double t_sum = stats.time_dist_comp + stats.time_nbr_filter + stats.time_dist_est
                 + stats.time_queue_mgmt + stats.time_pruning
                 + stats.time_hash_filter + stats.time_query_prepare;
    printf("--- Timing breakdown (total across queries) ---\n");
    printf("  dist computation:  %.6f sec (%.1f%%)\n", stats.time_dist_comp, stats.time_dist_comp / t_sum * 100);
    printf("  nbr filter:        %.6f sec (%.1f%%)\n", stats.time_nbr_filter, stats.time_nbr_filter / t_sum * 100);
    printf("  dist estimation:   %.6f sec (%.1f%%)\n", stats.time_dist_est, stats.time_dist_est / t_sum * 100);
    printf("  queue management:  %.6f sec (%.1f%%)\n", stats.time_queue_mgmt, stats.time_queue_mgmt / t_sum * 100);
    printf("  pruning:           %.6f sec (%.1f%%)\n", stats.time_pruning, stats.time_pruning / t_sum * 100);
    printf("  hash filter:       %.6f sec (%.1f%%)\n", stats.time_hash_filter, stats.time_hash_filter / t_sum * 100);
    printf("  query prepare:     %.6f sec (%.1f%%)\n", stats.time_query_prepare, stats.time_query_prepare / t_sum * 100);
    printf("  sum:                  %.6f sec\n", t_sum);
    printf("  wall-clock runtime:   %.6f sec\n", stats.runtime);
    printf("  gap (unmeasured):     %.6f sec (%.1f%% of runtime)\n", stats.runtime - t_sum, (stats.runtime - t_sum) / stats.runtime * 100);
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
