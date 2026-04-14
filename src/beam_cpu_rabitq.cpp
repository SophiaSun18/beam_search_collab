#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath> 
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <omp.h>

#include "../include/utils.h"
#include "../include/ctimer.h"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
#include "../include/qg.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
/* Neighbor pruning */
bool PRUNE_ENABLED = false;
float PRUNE_RATIO = 0.0;
float APPROX_RATIO = 0.0;
/* L2 & RabitQ mixed search */
int L2_START_ITER_THRESHOLD = 0;
int L2_END_ITER_THRESHOLD = 0;
int L2_PERIOD = 0;
/* Duplicate handling */
bool DUP_HANDLING_ENABLED = false;
/* Stats collection */
bool DEBUG_MODE = false;
bool TIMING_ENABLED = false;
bool ESTIMATE_STATS_ENABLED = false;
bool STEP_EFFICIENCY_ENABLED = false;
string ESTIMATE_STATS_FILE;
string STEP_EFFICIENCY_STATS_FILE;

#define TSTART(tc) do { if (TIMING_ENABLED) ctimer_start(&(tc)); } while(0)
#define TSTOP(tc)  do { if (TIMING_ENABLED) { ctimer_stop(&(tc)); ctimer_lap(&(tc)); } } while(0)

SearchResult beam_cpu_rabitq(
    QuantizationGraph& qg, int K, const float* d_query, size_t dim, vidType entry_point)
{
    struct Candidate {
        vidType vid;
        float dist;
    };

    int degree = qg.get_degree();

    // initialize the hash table and frontiers for the search
    hash_filter<vidType> is_expanded(BEAM_SIZE * BEAM_SIZE);
    hash_filter<vidType> is_exact_scored(BEAM_SIZE * BEAM_SIZE);
    hash_filter<vidType> is_added(DUP_HANDLING_ENABLED ? BEAM_SIZE * BEAM_SIZE : 1);
    vector<Candidate> curr_candidate;
    curr_candidate.reserve(degree);
    new_pqueue_t<vidType> frontier(BEAM_SIZE);
    new_pqueue_t<vidType> result_pool(K);

    // initialize the buffers for fastscan distance estimation
    vector<float> appro_dist;
    vector<uint32_t> raw_buf_32;
    vector<float> result_f_buf;
    if constexpr (FASTSCAN_ENABLED) {
        appro_dist.resize(degree);
        raw_buf_32.resize(degree);
        result_f_buf.resize(degree);
    }

    // counters for various statistics
    size_t dist_cmps = 0, dist_est = 0, extra_dist_cmps = 0;
    size_t neighbor_pruned = 0;

    // fine-grained timing accumulators (ctimer-based, zero overhead when macros disabled)
    ctimer_t tc_search_total, tc_dist_comp, tc_dist_est;
    ctimer_t tc_nbr_filter, tc_hash_filter;
    ctimer_t tc_queue_mgmt, tc_candidate_buffer_mgmt, tc_nth_element;
    ctimer_reset(&tc_search_total); ctimer_reset(&tc_dist_comp); ctimer_reset(&tc_dist_est);
    ctimer_reset(&tc_nbr_filter); ctimer_reset(&tc_hash_filter);
    ctimer_reset(&tc_queue_mgmt); ctimer_reset(&tc_candidate_buffer_mgmt); ctimer_reset(&tc_nth_element); 

    // pre-allocate buffers for query preparation
    size_t padded_dim = 1ULL << static_cast<size_t>(std::ceil(std::log2(dim)));
    QuantizationGraph::QueryBuffers query_buffers;
    query_buffers.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
    query_buffers.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
    if constexpr (FASTSCAN_ENABLED) {
        query_buffers.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));
    }

    // data structures for estimation statistics
    unordered_map<vidType, size_t> estimated_count_by_node;
    if (ESTIMATE_STATS_ENABLED) estimated_count_by_node.reserve(BEAM_SIZE * degree);
    vector<double> step_efficiency_by_step;
    double prev_expanded_exact_dist = 0.0;
    bool has_prev_expanded_exact_dist = false;

    auto record_estimate = [&](vidType node) {
        if (ESTIMATE_STATS_ENABLED) {
            estimated_count_by_node[node]++;
        }
    };

    auto apply_ratio_pruning = [&](int iter) {
        if (!PRUNE_ENABLED || iter <= BEAM_SIZE * APPROX_RATIO) return;

        size_t num_kept = static_cast<size_t>(degree * (1.0f - PRUNE_RATIO));
        if (num_kept == 0 || curr_candidate.size() <= num_kept) return;

        size_t iter_pruned = curr_candidate.size() - num_kept;
        neighbor_pruned += iter_pruned;

        TSTART(tc_nth_element);
        nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.dist < b.dist;
            });
        TSTOP(tc_nth_element);

        TSTART(tc_candidate_buffer_mgmt);
        curr_candidate.resize(num_kept);
        TSTOP(tc_candidate_buffer_mgmt);
    };

    auto push_frontier_candidates = [&]() {
        for (size_t i = 0; i < curr_candidate.size(); i++) {
            if (frontier.size() < BEAM_SIZE || curr_candidate[i].dist < frontier.get_tail_dist()) {
                if (DUP_HANDLING_ENABLED) {
                    if (is_added.add(curr_candidate[i].vid)) {
                        frontier.push(curr_candidate[i].vid, curr_candidate[i].dist);
                    } else {
                        frontier.push_update(curr_candidate[i].vid, curr_candidate[i].dist);
                    }
                } else {
                    frontier.push(curr_candidate[i].vid, curr_candidate[i].dist);
                }
            }
        }
    };

    TSTART(tc_search_total);

    // push the starting point to the frontier with max distance
    TSTART(tc_queue_mgmt);
    frontier.push(entry_point, FLT_MAX);
    TSTOP(tc_queue_mgmt);
    if (DUP_HANDLING_ENABLED) {
        TSTART(tc_hash_filter);
        is_added.add(entry_point);
        TSTOP(tc_hash_filter);
    }

    // quantize the query once to avoid redundant computations
    auto qf = qg.query_prepare(d_query, query_buffers);

    int iter = 0;
    while (frontier.has_unexpanded()) {
        iter++;

        // determine whether to use exact L2 distance computation for this iteration
        const int late_exact_start_iter = std::max(1, BEAM_SIZE - L2_END_ITER_THRESHOLD);
        const bool use_periodic_exact_l2 = (L2_PERIOD > 0 && iter % L2_PERIOD == 0);
        const bool use_exact_l2 =
            (iter < L2_START_ITER_THRESHOLD) ||
            (L2_END_ITER_THRESHOLD > 0 && iter >= late_exact_start_iter) ||
            use_periodic_exact_l2;

        TSTART(tc_candidate_buffer_mgmt);
        curr_candidate.clear();
        TSTOP(tc_candidate_buffer_mgmt);

        TSTART(tc_queue_mgmt);
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        frontier.set_front_expanded();
        TSTOP(tc_queue_mgmt);

        TSTART(tc_hash_filter);
        if (is_expanded.contains(cur_node)) {
            TSTOP(tc_hash_filter);
            continue;
        }
        TSTOP(tc_hash_filter);

        TSTART(tc_dist_comp);
        float exact_dist = compute_distance(dim, d_query, qg.get_row(cur_node));
        TSTOP(tc_dist_comp);
        dist_cmps++;

        if (STEP_EFFICIENCY_ENABLED && has_prev_expanded_exact_dist) {
            step_efficiency_by_step.push_back(prev_expanded_exact_dist - exact_dist);
        }
        prev_expanded_exact_dist = exact_dist;
        has_prev_expanded_exact_dist = true;

        TSTART(tc_queue_mgmt);
        result_pool.push(cur_node, exact_dist);
        TSTOP(tc_queue_mgmt);
        
        TSTART(tc_hash_filter);
        is_expanded.add(cur_node);
        TSTOP(tc_hash_filter);

        // expand the current node
        const vidType* neighbors = qg.get_neighbors(cur_node);

        if (use_exact_l2) {
            for (size_t i = 0; i < static_cast<size_t>(degree); i++) {
                TSTART(tc_hash_filter);
                if (is_expanded.contains(neighbors[i]) || is_exact_scored.contains(neighbors[i])) {
                    TSTOP(tc_hash_filter);
                    continue;
                }

                TSTART(tc_dist_comp);
                float dist = compute_distance(dim, d_query, qg.get_row(neighbors[i]));
                TSTOP(tc_dist_comp);
                dist_cmps++;

                TSTART(tc_hash_filter);
                is_exact_scored.add(neighbors[i]);
                TSTOP(tc_hash_filter);

                TSTART(tc_candidate_buffer_mgmt);
                curr_candidate.push_back({neighbors[i], dist});
                TSTOP(tc_candidate_buffer_mgmt);
            }

            apply_ratio_pruning(iter);

            TSTART(tc_queue_mgmt);
            push_frontier_candidates();
            TSTOP(tc_queue_mgmt);
        } else if constexpr (FASTSCAN_ENABLED) {
            TSTART(tc_dist_est);
            qg.scan_all_neighbors_32(cur_node, exact_dist, appro_dist.data(), qf, raw_buf_32.data(), result_f_buf.data());
            TSTOP(tc_dist_est);
            dist_est += degree;

            for (int i = 0; i < degree; i++) {
                record_estimate(neighbors[i]);
                TSTART(tc_hash_filter);
                if (is_expanded.contains(neighbors[i])) {
                    TSTOP(tc_hash_filter);
                    continue;
                }
                TSTOP(tc_hash_filter);
                TSTART(tc_candidate_buffer_mgmt);
                curr_candidate.push_back({neighbors[i], appro_dist[i]});
                TSTOP(tc_candidate_buffer_mgmt);
            }
            
            apply_ratio_pruning(iter);

            TSTART(tc_queue_mgmt);
            push_frontier_candidates();
            TSTOP(tc_queue_mgmt);

            // prefetch next node
            // if (frontier.has_unexpanded()) {
            //     int new_idx = frontier.get_next_index();
            //     qg.prefetch_nbr(frontier[new_idx]);
            // }
        } else {
            for (size_t i = 0; i < degree; i++) {
                TSTART(tc_hash_filter);
                if (is_expanded.contains(neighbors[i])) {
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
                record_estimate(neighbors[i]);

                TSTART(tc_candidate_buffer_mgmt);
                curr_candidate.push_back({neighbors[i], est});
                TSTOP(tc_candidate_buffer_mgmt);
            }

            apply_ratio_pruning(iter);

            TSTART(tc_queue_mgmt);
            push_frontier_candidates();
            TSTOP(tc_queue_mgmt);

            // prefetch next node
            // if (frontier.has_unexpanded()) {
            //     int new_idx = frontier.get_next_index();
            //     qg.prefetch_nbr(frontier[new_idx]);
            // }
        }

        if (DEBUG_MODE) {
            debug_print_candidates(iter, cur_node, curr_candidate);
            debug_print_beam_state(frontier);
        }
    }
    // final fill up stage of the result pool if it's not full due to duplicates in the frontier
    // expand result pool by visiting neighbors of current results
    int pool_size = result_pool.size();
    if (pool_size != K) {
        for (int i = 0; i < pool_size; i++) {
            vidType data_id = result_pool[i];
            const vidType* neighbors = qg.get_neighbors(data_id);
            for (int j = 0; j < degree; j++) {
                vidType cur_neighbor = neighbors[j];
                TSTART(tc_hash_filter);
                if (!is_expanded.contains(cur_neighbor)) {
                    is_expanded.add(cur_neighbor);
                    TSTOP(tc_hash_filter);
                    TSTART(tc_dist_comp);
                    float dist = compute_distance(dim, d_query, qg.get_row(cur_neighbor));
                    TSTOP(tc_dist_comp);
                    extra_dist_cmps++;
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

    TSTOP(tc_search_total);

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    for (int i = 0; i < K; i++)
        result.push_back(result_pool[i]);

    free(query_buffers.rotated_query);
    free(query_buffers.quantized_query);
    if constexpr (FASTSCAN_ENABLED) {
        free(query_buffers.lut);
    }

    if (DEBUG_MODE) {
        debug_print_final_results(result_pool);
    }

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.dist_est = dist_est;
    res.extra_dist_cmps = extra_dist_cmps;
    res.iters = iter;
    res.neighbor_pruned = neighbor_pruned;
    res.time_search_total = timespec_sec(tc_search_total.elapsed);
    res.time_dist_comp = timespec_sec(tc_dist_comp.elapsed);
    res.time_nbr_filter = timespec_sec(tc_nbr_filter.elapsed);
    res.time_dist_est = timespec_sec(tc_dist_est.elapsed);
    res.time_queue_mgmt = timespec_sec(tc_queue_mgmt.elapsed);
    res.time_candidate_buffer_mgmt = timespec_sec(tc_candidate_buffer_mgmt.elapsed);
    res.time_nth_element = timespec_sec(tc_nth_element.elapsed);
    res.time_pruning = res.time_candidate_buffer_mgmt + res.time_nth_element;
    res.time_hash_filter = timespec_sec(tc_hash_filter.elapsed);
    res.total_estimate_events = dist_est;
    res.total_unique_estimated_nodes = ESTIMATE_STATS_ENABLED ? estimated_count_by_node.size() : 0;
    res.total_duplicate_estimates =
        ESTIMATE_STATS_ENABLED ? count_duplicate_estimates(estimated_count_by_node) : 0;
    if (ESTIMATE_STATS_ENABLED) {
        res.estimate_count_histogram = build_estimate_histogram(estimated_count_by_node);
        res.estimated_count_by_node = move(estimated_count_by_node);
    }
    if (STEP_EFFICIENCY_ENABLED) {
        res.total_step_efficiency_events = step_efficiency_by_step.size();
        res.total_step_efficiency =
            std::accumulate(step_efficiency_by_step.begin(), step_efficiency_by_step.end(), 0.0);
        res.step_efficiency_by_step = move(step_efficiency_by_step);
    }
    return res;
}

RunStats run_ann_experiment_rabitq(
    QuantizationGraph& qg, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt
) {
    (void)n_base;
    (void)base_vectors;
    int actual_threads = 0;
    #pragma omp parallel
    {
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int64_t total_count_dc = 0, total_count_ds = 0, total_extra_count_dc = 0, total_hop = 0;
    int64_t total_pruned = 0;
    double total_time_search_total = 0;
    double total_time_dist_comp = 0, total_time_nbr_filter = 0, total_time_dist_est = 0;
    double total_time_queue_mgmt = 0, total_time_pruning = 0;
    double total_time_candidate_buffer_mgmt = 0, total_time_nth_element = 0;
    double total_time_hash_filter = 0;
    int64_t total_estimate_events = 0, total_unique_estimated_nodes = 0, total_duplicate_estimates = 0;
    int64_t total_step_efficiency_events = 0;
    double total_step_efficiency = 0;
    unordered_map<size_t, size_t> estimate_count_histogram;
    vector<double> step_efficiency_sums_by_step;
    vector<size_t> step_efficiency_counts_by_step;
    const int query_count = DEBUG_MODE ? std::min<size_t>(1, n_query) : static_cast<int>(n_query);
    vector<vector<int>> all_results(query_count, vector<int>(K));

    vidType entry_point = qg.get_entry_point();

    ctimer_t t;
    ctimer_start(&t);

    if (DEBUG_MODE) {
        cout << "[DEBUG] debug mode enabled: running only query 0\n";
    }

    #pragma omp parallel for schedule(dynamic, 1) \
        reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_extra_count_dc) \
        reduction(+:total_hop) reduction(+:total_pruned) \
        reduction(+:total_time_search_total) \
        reduction(+:total_time_dist_comp) reduction(+:total_time_nbr_filter) reduction(+:total_time_dist_est) \
        reduction(+:total_time_queue_mgmt) reduction(+:total_time_pruning) \
        reduction(+:total_time_candidate_buffer_mgmt) reduction(+:total_time_nth_element) reduction(+:total_time_hash_filter) \
        reduction(+:total_estimate_events) reduction(+:total_step_efficiency_events) reduction(+:total_step_efficiency)
    for (int q = 0; q < query_count; q++) {
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_rabitq(qg, K, query_ptr, dim, entry_point);

        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_extra_count_dc += result.extra_dist_cmps;
        total_hop += result.iters;
        total_pruned += result.neighbor_pruned;
        total_time_search_total += result.time_search_total;
        total_time_dist_comp += result.time_dist_comp;
        total_time_nbr_filter += result.time_nbr_filter;
        total_time_dist_est += result.time_dist_est;
        total_time_queue_mgmt += result.time_queue_mgmt;
        total_time_pruning += result.time_pruning;
        total_time_candidate_buffer_mgmt += result.time_candidate_buffer_mgmt;
        total_time_nth_element += result.time_nth_element;
        total_time_hash_filter += result.time_hash_filter;
        total_estimate_events += result.total_estimate_events;
        total_step_efficiency_events += result.total_step_efficiency_events;
        total_step_efficiency += result.total_step_efficiency;
        if (ESTIMATE_STATS_ENABLED || STEP_EFFICIENCY_ENABLED) {
            #pragma omp critical(stats_merge)
            {
                if (ESTIMATE_STATS_ENABLED) {
                    total_unique_estimated_nodes += result.total_unique_estimated_nodes;
                    total_duplicate_estimates += result.total_duplicate_estimates;
                    merge_estimate_histograms(estimate_count_histogram, result.estimate_count_histogram);
                }
                if (STEP_EFFICIENCY_ENABLED) {
                    merge_step_efficiency_vectors(step_efficiency_sums_by_step,
                                                 step_efficiency_counts_by_step,
                                                 result.step_efficiency_by_step);
                }
            }
        }
    }

    ctimer_stop(&t);
    ctimer_measure(&t);

    RunStats stats;
    stats.runtime = timespec_sec(t.elapsed);
    stats.latency = stats.runtime / query_count * 1000.0;
    stats.throughput = double(query_count) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, query_count, K, dim_gt);
    stats.total_dc = total_count_dc;
    stats.total_ds = total_count_ds;
    stats.total_extra_dc = total_extra_count_dc;
    stats.total_full_dc = total_count_dc + total_extra_count_dc;
    stats.total_iter = total_hop;
    stats.total_pruned = total_pruned;
    stats.time_search_total = total_time_search_total;
    stats.time_dist_comp = total_time_dist_comp;
    stats.time_nbr_filter = total_time_nbr_filter;
    stats.time_dist_est = total_time_dist_est;
    stats.time_queue_mgmt = total_time_queue_mgmt;
    stats.time_pruning = total_time_pruning;
    stats.time_candidate_buffer_mgmt = total_time_candidate_buffer_mgmt;
    stats.time_nth_element = total_time_nth_element;
    stats.time_hash_filter = total_time_hash_filter;
    stats.total_estimate_events = total_estimate_events;
    stats.total_step_efficiency_events = total_step_efficiency_events;
    stats.total_step_efficiency = total_step_efficiency;
    if (ESTIMATE_STATS_ENABLED) {
        stats.total_unique_estimated_nodes = total_unique_estimated_nodes;
        stats.total_duplicate_estimates = total_duplicate_estimates;
        stats.estimate_count_histogram = move(estimate_count_histogram);
    }
    if (STEP_EFFICIENCY_ENABLED) {
        stats.step_efficiency_sums_by_step = move(step_efficiency_sums_by_step);
        stats.step_efficiency_counts_by_step = move(step_efficiency_counts_by_step);
    }

    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # distance computation: " << stats.total_full_dc << "\n";
    cout << "average # distance computation: " << stats.total_full_dc / query_count << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / query_count << "\n";
    cout << "total # pruned neighbors: " << stats.total_pruned << "\n";
    cout << "average # pruned neighbors: " << stats.total_pruned / query_count << "\n";

    double t_sum = stats.time_dist_comp + stats.time_nbr_filter + stats.time_dist_est
                 + stats.time_queue_mgmt + stats.time_candidate_buffer_mgmt
                 + stats.time_nth_element
                 + stats.time_hash_filter;
    double other_time = stats.time_search_total - t_sum;
    if (other_time < 0) other_time = 0;
    printf("--- Timing breakdown (total across queries) ---\n");
    printf("  search total:      %.6f sec\n", stats.time_search_total);
    printf("  dist computation:  %.6f sec (%.1f%%)\n", stats.time_dist_comp, stats.time_search_total > 0 ? stats.time_dist_comp / stats.time_search_total * 100 : 0.0);
    printf("  nbr filter:        %.6f sec (%.1f%%)\n", stats.time_nbr_filter, stats.time_search_total > 0 ? stats.time_nbr_filter / stats.time_search_total * 100 : 0.0);
    printf("  dist estimation:   %.6f sec (%.1f%%)\n", stats.time_dist_est, stats.time_search_total > 0 ? stats.time_dist_est / stats.time_search_total * 100 : 0.0);
    printf("  queue management:  %.6f sec (%.1f%%)\n", stats.time_queue_mgmt, stats.time_search_total > 0 ? stats.time_queue_mgmt / stats.time_search_total * 100 : 0.0);
    printf("  candidate buffer:  %.6f sec (%.1f%%)\n", stats.time_candidate_buffer_mgmt, stats.time_search_total > 0 ? stats.time_candidate_buffer_mgmt / stats.time_search_total * 100 : 0.0);
    printf("  nth_element:       %.6f sec (%.1f%%)\n", stats.time_nth_element, stats.time_search_total > 0 ? stats.time_nth_element / stats.time_search_total * 100 : 0.0);
    printf("  pruning total:     %.6f sec (%.1f%%)\n", stats.time_pruning, stats.time_search_total > 0 ? stats.time_pruning / stats.time_search_total * 100 : 0.0);
    printf("  hash filter:       %.6f sec (%.1f%%)\n", stats.time_hash_filter, stats.time_search_total > 0 ? stats.time_hash_filter / stats.time_search_total * 100 : 0.0);
    printf("  others:            %.6f sec (%.1f%%)\n", other_time, stats.time_search_total > 0 ? other_time / stats.time_search_total * 100 : 0.0);
    printf("  measured sum:      %.6f sec\n", t_sum);
    printf("  wall-clock runtime:   %.6f sec\n", stats.runtime);
    printf("  search total / runtime: %.2fx\n", stats.runtime > 0 ? stats.time_search_total / stats.runtime : 0.0);
    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -degree <int>" << " [-mode <codebook|bins>]"
        << " [-qg <qg_codebook>]" << " [-graph <graph_index>]"
        << " [-rabitq <rabitq_table>]" << " [-sign <sign_table>]"
        << " [-beam <int>]" << " [-p_ratio <float>]" << " [-a_ratio <float>]"
        << " [-l2_iter <int>]" << " [-l2_end_iter <int>]" << " [-l2_period <int>]"
        << " [-timer <on|off|1|0>]" << " [-debug <on|off|1|0>]"
        << " [-estimate_stats <on|off|1|0>]" << " [-estimate_stats_file <output.txt>]"
        << " [-step_efficiency <on|off|1|0>]" << " [-step_efficiency_file <output.txt>]"
        << " [-csv <output.csv>]\n";
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
            PRUNE_ENABLED = (PRUNE_RATIO != 0.0f && APPROX_RATIO != 0.0f);
        }
        else if (arg == "-a_ratio" && i + 1 < argc) {
            APPROX_RATIO = atof(argv[++i]);
            PRUNE_ENABLED = (PRUNE_RATIO != 0.0f && APPROX_RATIO != 0.0f);
        }
        else if (arg == "-l2_iter" && i + 1 < argc) {
            L2_START_ITER_THRESHOLD = atoi(argv[++i]);
            if (L2_START_ITER_THRESHOLD < 0) {
                cerr << "Error: -l2_iter must be >= 0.\n";
                return 1;
            }
        }
        else if (arg == "-l2_end_iter" && i + 1 < argc) {
            L2_END_ITER_THRESHOLD = atoi(argv[++i]);
            if (L2_END_ITER_THRESHOLD < 0) {
                cerr << "Error: -l2_end_iter must be >= 0.\n";
                return 1;
            }
        }
        else if (arg == "-l2_period" && i + 1 < argc) {
            L2_PERIOD = atoi(argv[++i]);
            if (L2_PERIOD < 0) {
                cerr << "Error: -l2_period must be >= 0.\n";
                return 1;
            }
        }
        else if (arg == "-timer" && i + 1 < argc) {
            string timer_arg = argv[++i];
            if (timer_arg == "on" || timer_arg == "1" || timer_arg == "true") {
                TIMING_ENABLED = true;
            }
            else if (timer_arg == "off" || timer_arg == "0" || timer_arg == "false") {
                TIMING_ENABLED = false;
            }
            else {
                cerr << "Error: -timer must be one of on, off, 1, 0, true, false.\n";
                return 1;
            }
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
        else if (arg == "-estimate_stats" && i + 1 < argc) {
            string estimate_stats_arg = argv[++i];
            if (estimate_stats_arg == "on" || estimate_stats_arg == "1" || estimate_stats_arg == "true") {
                ESTIMATE_STATS_ENABLED = true;
            }
            else if (estimate_stats_arg == "off" || estimate_stats_arg == "0" || estimate_stats_arg == "false") {
                ESTIMATE_STATS_ENABLED = false;
            }
            else {
                cerr << "Error: -estimate_stats must be one of on, off, 1, 0, true, false.\n";
                return 1;
            }
        }
        else if (arg == "-estimate_stats_file" && i + 1 < argc) {
            ESTIMATE_STATS_FILE = argv[++i];
        }
        else if (arg == "-step_efficiency" && i + 1 < argc) {
            string step_efficiency_arg = argv[++i];
            if (step_efficiency_arg == "on" || step_efficiency_arg == "1" ||
                step_efficiency_arg == "true") {
                STEP_EFFICIENCY_ENABLED = true;
            }
            else if (step_efficiency_arg == "off" || step_efficiency_arg == "0" ||
                     step_efficiency_arg == "false") {
                STEP_EFFICIENCY_ENABLED = false;
            }
            else {
                cerr << "Error: -step_efficiency must be one of on, off, 1, 0, true, false.\n";
                return 1;
            }
        }
        else if (arg == "-step_efficiency_file" && i + 1 < argc) {
            STEP_EFFICIENCY_STATS_FILE = argv[++i];
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

    cout << "Fine-grained timer: " << (TIMING_ENABLED ? "on" : "off") << "\n";
    cout << "Debug mode: " << (DEBUG_MODE ? "on" : "off") << "\n";
    cout << "Estimate stats: " << (ESTIMATE_STATS_ENABLED ? "on" : "off") << "\n";
    cout << "Step efficiency stats: " << (STEP_EFFICIENCY_ENABLED ? "on" : "off") << "\n";
    cout << "Exact L2 start iterations: iter < " << L2_START_ITER_THRESHOLD << "\n";
    if (L2_END_ITER_THRESHOLD > 0) {
        cout << "Exact L2 end iterations: iter >= " << std::max(1, BEAM_SIZE - L2_END_ITER_THRESHOLD)
             << " (beam - " << L2_END_ITER_THRESHOLD << ")\n";
    } else {
        cout << "Exact L2 end iterations: off\n";
    }
    if (L2_PERIOD > 0) {
        cout << "Exact L2 periodic adjustment: every " << L2_PERIOD << " iterations\n";
    } else {
        cout << "Exact L2 periodic adjustment: off\n";
    }
    if (ESTIMATE_STATS_ENABLED && !ESTIMATE_STATS_FILE.empty()) {
        cout << "Estimate stats file: " << ESTIMATE_STATS_FILE << "\n";
    }
    if (STEP_EFFICIENCY_ENABLED && !STEP_EFFICIENCY_STATS_FILE.empty()) {
        cout << "Step efficiency stats file: " << STEP_EFFICIENCY_STATS_FILE << "\n";
    }

    if (mode == "bins") {
        std::string dump_txt = rabitq_table + ".dump.txt";
        dump_rabitq_table(rabitq_table, dump_txt, static_cast<int>(dim_base), degree, 20);
    } else {
        cout << "Skipping dump_rabitq_table in codebook mode: helper expects standalone rabitq table format.\n";
    }

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
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats,
                            L2_START_ITER_THRESHOLD, L2_END_ITER_THRESHOLD, L2_PERIOD);
    }
    if (ESTIMATE_STATS_ENABLED && !ESTIMATE_STATS_FILE.empty()) {
        dump_estimate_stats(ESTIMATE_STATS_FILE, stats);
    }
    if (STEP_EFFICIENCY_ENABLED && !STEP_EFFICIENCY_STATS_FILE.empty()) {
        dump_step_efficiency_stats(STEP_EFFICIENCY_STATS_FILE, stats);
    }
    return 0;
}
