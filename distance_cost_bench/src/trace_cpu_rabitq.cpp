#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <limits>
#include <omp.h>

#include "../include/utils.h"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
#include "../include/qg.hpp"
#include "../include/rabitq_gpu_like.hpp"
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
bool DEBUG_MODE = false;
bool EXPANDER_TRACE_ENABLED = false;
string EXPANDER_TRACE_FILE;
bool ACCESS_TRACE_ENABLED = false;
string ACCESS_TRACE_FILE;
bool QUALITY_TRACE_ENABLED = false;
string QUALITY_TRACE_FILE;
bool QUALITY_HIT_TRACE_ENABLED = false;
string QUALITY_HIT_TRACE_FILE;
bool QUALITY_BEAM_HIT_TRACE_ENABLED = false;
string QUALITY_BEAM_HIT_TRACE_FILE;
int QUALITY_TRACE_QUERY_LIMIT = 50;
bool QUALITY_TRACE_STEPS = true;
int QUERY_LIMIT = -1;
bool GPU_LIKE_SEARCH = false;

SearchResult beam_cpu_rabitq(
    QuantizationGraph& qg, int K, const float* d_query, size_t dim, vidType entry_point,
    uint32_t query_id = 0,
    const QualityTraceContext* quality_ctx = nullptr)
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
    gpu_like_hash_filter<vidType> gpu_like_expanded(gpu_hash_bitlen_for_topk(BEAM_SIZE));
    vector<Candidate> curr_candidate;
    curr_candidate.reserve(degree);
    vector<pair<vidType, float>> gpu_like_candidates;
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

    // pre-allocate buffers for query preparation
    size_t padded_dim = static_cast<size_t>(qg.get_padded_dim());
    QuantizationGraph::QueryBuffers query_buffers;
    query_buffers.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
    query_buffers.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
    if constexpr (FASTSCAN_ENABLED) {
        query_buffers.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));
    }

    vector<ExpanderTraceStep> expander_trace_by_step;
    double prev_expander_exact_dist = 0.0;
    bool has_prev_expander_exact_dist = false;
    double best_expander_exact_dist = 0.0;
    vector<AccessTraceRecord> access_trace;

    auto apply_ratio_pruning = [&](int iter) {
        if (!PRUNE_ENABLED || iter <= BEAM_SIZE * APPROX_RATIO) return;

        size_t num_kept = static_cast<size_t>(degree * (1.0f - PRUNE_RATIO));
        if (num_kept == 0 || curr_candidate.size() <= num_kept) return;

        size_t iter_pruned = curr_candidate.size() - num_kept;
        neighbor_pruned += iter_pruned;
        nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.dist < b.dist;
            });
        curr_candidate.resize(num_kept);
    };

    auto push_frontier_candidates = [&]() {
        if (GPU_LIKE_SEARCH) {
            return;
        }
        for (size_t i = 0; i < curr_candidate.size(); i++) {
            if (frontier.size() < BEAM_SIZE || curr_candidate[i].dist < frontier.get_tail_dist()) {
                // if (DUP_HANDLING_ENABLED) {
                //     if (is_added.add(curr_candidate[i].vid)) {
                //         frontier.push(curr_candidate[i].vid, curr_candidate[i].dist);
                //     } else {
                //         frontier.push_update(curr_candidate[i].vid, curr_candidate[i].dist);
                //     }
                // } else {
                    frontier.push(curr_candidate[i].vid, curr_candidate[i].dist);
                // }
            }
        }
    };

    auto compute_exact_l2 = [&](vidType node) {
        float dist = GPU_LIKE_SEARCH
            ? compute_distance_gpu_warp_order(static_cast<int>(dim), d_query, qg.get_row(node))
            : compute_distance(static_cast<int>(dim), d_query, qg.get_row(node));
        dist_cmps++;
        return dist;
    };

    auto expanded_contains = [&](vidType node) {
        return GPU_LIKE_SEARCH ? gpu_like_expanded.contains(node)
                               : is_expanded.contains(node);
    };

    auto expanded_insert = [&](vidType node) {
        if (GPU_LIKE_SEARCH) {
            return gpu_like_expanded.insert(node);
        }
        if (is_expanded.contains(node)) {
            return false;
        }
        is_expanded.add(node);
        return true;
    };

    // push the starting point to the frontier with max distance
    if (GPU_LIKE_SEARCH) {
        gpu_like_candidates.assign(
            RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE,
            {std::numeric_limits<vidType>::max(), FLT_MAX});
        gpu_like_candidates[0] = {entry_point, compute_exact_l2(entry_point)};
    } else {
        frontier.push(entry_point, FLT_MAX);
    }
    if (DUP_HANDLING_ENABLED) {
        is_added.add(entry_point);
    }

    // quantize the query once to avoid redundant computations
    auto qf = qg.query_prepare(d_query, query_buffers);

    int expanded_steps = 0;
    while (frontier.has_unexpanded() || (GPU_LIKE_SEARCH && !gpu_like_candidates.empty())) {
        if (GPU_LIKE_SEARCH) {
            frontier.merge_candidates_gpu_like(gpu_like_candidates);
            gpu_like_candidates.clear();
            if (!frontier.has_unexpanded()) {
                break;
            }
        }
        curr_candidate.clear();
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        QualityFrontierAudit frontier_audit;
        if (quality_ctx != nullptr && quality_ctx->record_steps) {
            frontier_audit = audit_frontier_exact_rank(*quality_ctx, frontier, cur_node);
        }
        frontier.set_front_expanded();
        if (!expanded_insert(cur_node)) {
            continue;
        }

        const int iter = ++expanded_steps;
        const int late_exact_start_iter = std::max(1, BEAM_SIZE - L2_END_ITER_THRESHOLD);
        const bool use_periodic_exact_l2 = (L2_PERIOD > 0 && iter % L2_PERIOD == 0);
        const bool use_exact_l2 =
            (iter < L2_START_ITER_THRESHOLD) ||
            (L2_END_ITER_THRESHOLD > 0 && iter >= late_exact_start_iter) ||
            use_periodic_exact_l2;

        float exact_dist = compute_exact_l2(cur_node);

        if (EXPANDER_TRACE_ENABLED) {
            append_expander_trace_step(expander_trace_by_step, exact_dist,
                                       prev_expander_exact_dist,
                                       has_prev_expander_exact_dist,
                                       best_expander_exact_dist);
        }
        result_pool.push(cur_node, exact_dist);

        // expand the current node
        const vidType* neighbors = qg.get_neighbors(cur_node);

        if (use_exact_l2) {
            uint32_t active_mask = 0;
            for (size_t i = 0; i < static_cast<size_t>(degree); i++) {
                if (expanded_contains(neighbors[i]) || is_exact_scored.contains(neighbors[i])) {
                    continue;
                }
                float dist = compute_distance(dim, d_query, qg.get_row(neighbors[i]));
                dist_cmps++;
                if (i < 32) {
                    active_mask |= (1u << i);
                }
                is_exact_scored.add(neighbors[i]);
                curr_candidate.push_back({neighbors[i], dist});
            }
            if (ACCESS_TRACE_ENABLED) {
                access_trace.push_back(
                    {query_id, static_cast<uint32_t>(iter - 1),
                     ACCESS_TRACE_RQ_CHILD_EXACT, cur_node, active_mask});
            }

            apply_ratio_pruning(iter);
            if (GPU_LIKE_SEARCH) {
                gpu_like_candidates.assign(
                    RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE,
                    {std::numeric_limits<vidType>::max(), FLT_MAX});
                for (size_t i = 0;
                     i < curr_candidate.size() && i < gpu_like_candidates.size(); ++i) {
                    gpu_like_candidates[i] = {curr_candidate[i].vid, curr_candidate[i].dist};
                }
            } else {
                push_frontier_candidates();
            }
        } else if constexpr (FASTSCAN_ENABLED) {
            qg.scan_all_neighbors_32(cur_node, exact_dist, appro_dist.data(), qf, raw_buf_32.data(), result_f_buf.data());
            if (GPU_LIKE_SEARCH) {
                dist_est += rabitq_gpu_like_prepare_candidates(
                    neighbors, static_cast<size_t>(degree), gpu_like_expanded,
                    gpu_like_candidates,
                    [&](vidType, size_t idx) {
                        return appro_dist[idx];
                    });
            } else {
                dist_est += degree;
            }
            if (ACCESS_TRACE_ENABLED) {
                const uint32_t active_mask =
                    (degree >= 32) ? 0xffffffffu : ((1u << degree) - 1u);
                access_trace.push_back(
                    {query_id, static_cast<uint32_t>(iter - 1),
                     ACCESS_TRACE_RQ_EXPAND, cur_node, active_mask});
            }

            if (!GPU_LIKE_SEARCH) {
                for (int i = 0; i < degree; i++) {
                    if (expanded_contains(neighbors[i])) {
                        continue;
                    }
                    if (appro_dist[i] < 0.0f) {
                        appro_dist[i] = 0;
                    }
                    curr_candidate.push_back({neighbors[i], appro_dist[i]});
                }
            }

            apply_ratio_pruning(iter);
            push_frontier_candidates();

            // prefetch next node
            // if (frontier.has_unexpanded()) {
            //     int new_idx = frontier.get_next_index();
            //     qg.prefetch_nbr(frontier[new_idx]);
            // }
        } else {
            uint32_t active_mask = 0;
            if (GPU_LIKE_SEARCH) {
                dist_est += rabitq_gpu_like_prepare_candidates(
                    neighbors, static_cast<size_t>(degree), gpu_like_expanded,
                    gpu_like_candidates,
                    [&](vidType, size_t idx) {
                        auto nf = qg.nbr_filter(cur_node, static_cast<int>(idx));
                        if (idx < 32) {
                            active_mask |= (1u << idx);
                        }
                        return qg.estimated_distance(qf, nf, exact_dist);
                    });
            } else {
                for (size_t i = 0; i < degree; i++) {
                    if (expanded_contains(neighbors[i])) {
                        continue;
                    }
                    auto nf = qg.nbr_filter(cur_node, i);
                    dist_est++;
                    if (i < 32) {
                        active_mask |= (1u << i);
                    }
                    float est = qg.estimated_distance(qf, nf, exact_dist);
                    curr_candidate.push_back({neighbors[i], est});
                }
            }
            if (ACCESS_TRACE_ENABLED) {
                access_trace.push_back(
                    {query_id, static_cast<uint32_t>(iter - 1),
                     ACCESS_TRACE_RQ_EXPAND, cur_node, active_mask});
            }

            apply_ratio_pruning(iter);
            push_frontier_candidates();

            // prefetch next node
            // if (frontier.has_unexpanded()) {
            //     int new_idx = frontier.get_next_index();
            //     qg.prefetch_nbr(frontier[new_idx]);
            // }
        }

        if (quality_ctx != nullptr && quality_ctx->record_steps) {
            append_quality_trace_snapshot(
                *quality_ctx, iter, "expand", cur_node, exact_dist,
                frontier_audit, quality_queue_items(result_pool, K));
            append_quality_hit_trace_snapshot(
                *quality_ctx, iter, "expand", quality_queue_items(frontier, BEAM_SIZE),
                quality_ctx->beam_hit_records,
                quality_ctx->beam_previous_gt_present,
                quality_ctx->beam_seen_gt_present);
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
            uint32_t active_mask = 0;
            for (int j = 0; j < degree; j++) {
                vidType cur_neighbor = neighbors[j];
                if (!expanded_contains(cur_neighbor)) {
                    expanded_insert(cur_neighbor);
                    float dist = compute_exact_l2(cur_neighbor);
                    dist_cmps--;
                    extra_dist_cmps++;
                    if (j < 32) {
                        active_mask |= (1u << j);
                    }
                    result_pool.push(cur_neighbor, dist);
                }
            }
            if (ACCESS_TRACE_ENABLED && active_mask != 0) {
                access_trace.push_back(
                    {query_id, static_cast<uint32_t>(expanded_steps),
                     ACCESS_TRACE_RQ_EXTRA_EXACT, data_id, active_mask});
            }
            if (result_pool.size() == K) break;
        }
    }

    if (quality_ctx != nullptr) {
        QualityFrontierAudit final_audit;
        append_quality_trace_snapshot(
            *quality_ctx, expanded_steps + 1, "final", 0,
            std::numeric_limits<double>::quiet_NaN(), final_audit,
            quality_queue_items(result_pool, K));
        append_quality_hit_trace_snapshot(
            *quality_ctx, expanded_steps + 1, "final", quality_queue_items(frontier, BEAM_SIZE),
            quality_ctx->beam_hit_records,
            quality_ctx->beam_previous_gt_present,
            quality_ctx->beam_seen_gt_present);
    }

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
    res.iters = expanded_steps;
    res.neighbor_pruned = neighbor_pruned;
    if (ACCESS_TRACE_ENABLED) {
        res.access_trace = move(access_trace);
    }
    if (EXPANDER_TRACE_ENABLED) {
        for (const auto& step : expander_trace_by_step) {
            res.total_expander_positive_delta += step.positive_delta_sum;
            res.total_expander_trace_events += step.count;
        }
        if (!expander_trace_by_step.empty()) {
            res.total_expander_net_descent =
                expander_trace_by_step.front().expander_dist_sum - best_expander_exact_dist;
        }
        set_query_expander_efficiency(res, expander_trace_by_step, best_expander_exact_dist);
        res.expander_trace_by_step = move(expander_trace_by_step);
    }
    return res;
}

RunStats run_ann_experiment_rabitq(
    QuantizationGraph& qg, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt
) {
    (void)n_base;
    (void)base_vectors;
    const float* metric_data = qg.get_data_ptr();
    const size_t metric_row_stride = qg.is_bins_mode() ? dim : qg.get_row_offset();
    int actual_threads = 0;
    #pragma omp parallel
    {
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int64_t total_count_dc = 0, total_count_ds = 0, total_extra_count_dc = 0, total_hop = 0;
    int64_t total_pruned = 0;
    int64_t total_expander_trace_events = 0;
    double total_expander_positive_delta = 0.0;
    double total_expander_net_descent = 0.0;
    int64_t total_query_expander_steps = 0;
    int64_t total_query_expander_count = 0;
    double total_query_l2_progress = 0.0;
    double total_query_l2_progress_per_expander = 0.0;
    int64_t total_query_topk_metric_count = 0;
    int64_t total_query_topk_closed_gap_count = 0;
    int64_t total_query_topk_relative_count = 0;
    double total_query_topk_gt_l2 = 0.0;
    double total_query_topk_ret_l2 = 0.0;
    double total_query_topk_l2_progress = 0.0;
    double total_query_topk_l2_progress_per_expander = 0.0;
    double total_query_topk_closed_gap = 0.0;
    double total_query_topk_relative_excess = 0.0;
    double total_query_topk_completion = 0.0;
    vector<ExpanderTraceStep> expander_trace_by_step;
    vector<AccessTraceRecord> access_trace_records;
    int query_count = DEBUG_MODE
        ? std::min<size_t>(1, n_query)
        : (QUERY_LIMIT > 0
               ? static_cast<int>(std::min<size_t>(static_cast<size_t>(QUERY_LIMIT), n_query))
               : static_cast<int>(n_query));
    if ((QUALITY_TRACE_ENABLED || QUALITY_HIT_TRACE_ENABLED || QUALITY_BEAM_HIT_TRACE_ENABLED) &&
        QUALITY_TRACE_QUERY_LIMIT > 0) {
        query_count = std::min<int>(
            query_count,
            static_cast<int>(std::min<size_t>(static_cast<size_t>(QUALITY_TRACE_QUERY_LIMIT), n_query)));
    }
    vector<vector<int>> all_results(query_count, vector<int>(K));
    vector<QualityTraceItem> quality_trace_records;
    vector<QualityHitTraceItem> quality_hit_trace_records;
    vector<QualityHitTraceItem> quality_beam_hit_trace_records;

    vidType entry_point = qg.get_entry_point();

    auto start = std::chrono::high_resolution_clock::now();

    if (DEBUG_MODE) {
        cout << "[DEBUG] debug mode enabled: running only query 0\n";
    }

    #pragma omp parallel for schedule(dynamic, 1) \
        reduction(+:total_count_dc) reduction(+:total_count_ds) reduction(+:total_extra_count_dc) \
        reduction(+:total_hop) reduction(+:total_pruned) \
        reduction(+:total_expander_trace_events) reduction(+:total_expander_positive_delta) \
        reduction(+:total_expander_net_descent) reduction(+:total_query_expander_steps) \
        reduction(+:total_query_expander_count) reduction(+:total_query_l2_progress) \
        reduction(+:total_query_l2_progress_per_expander) \
        reduction(+:total_query_topk_metric_count) reduction(+:total_query_topk_closed_gap_count) \
        reduction(+:total_query_topk_relative_count) \
        reduction(+:total_query_topk_gt_l2) reduction(+:total_query_topk_ret_l2) \
        reduction(+:total_query_topk_l2_progress) reduction(+:total_query_topk_l2_progress_per_expander) \
        reduction(+:total_query_topk_closed_gap) reduction(+:total_query_topk_relative_excess) \
        reduction(+:total_query_topk_completion)
    for (int q = 0; q < query_count; q++) {
        const float* query_ptr = query_vectors + q * dim;
        vector<QualityTraceItem> query_quality_trace;
        vector<QualityHitTraceItem> query_quality_hit_trace;
        vector<QualityHitTraceItem> query_quality_beam_hit_trace;
        vector<uint8_t> quality_previous_gt(static_cast<size_t>(K + 1), 0);
        vector<uint8_t> quality_seen_gt(static_cast<size_t>(K + 1), 0);
        vector<uint8_t> quality_beam_previous_gt(static_cast<size_t>(K + 1), 0);
        vector<uint8_t> quality_beam_seen_gt(static_cast<size_t>(K + 1), 0);
        QualityTraceContext quality_ctx{
            "rabitq",
            static_cast<uint32_t>(q),
            metric_data,
            metric_row_stride,
            query_ptr,
            dim,
            groundtruth + static_cast<size_t>(q) * dim_gt,
            static_cast<int>(dim_gt),
            K,
            QUALITY_TRACE_ENABLED ? &query_quality_trace : nullptr,
        };
        quality_ctx.record_steps = QUALITY_TRACE_STEPS;
        if (QUALITY_HIT_TRACE_ENABLED) {
            quality_ctx.hit_records = &query_quality_hit_trace;
            quality_ctx.previous_gt_present = &quality_previous_gt;
            quality_ctx.seen_gt_present = &quality_seen_gt;
        }
        if (QUALITY_BEAM_HIT_TRACE_ENABLED) {
            quality_ctx.beam_hit_records = &query_quality_beam_hit_trace;
            quality_ctx.beam_previous_gt_present = &quality_beam_previous_gt;
            quality_ctx.beam_seen_gt_present = &quality_beam_seen_gt;
        }

        SearchResult result = beam_cpu_rabitq(qg, K, query_ptr, dim, entry_point,
                                              static_cast<uint32_t>(q),
                                              (QUALITY_TRACE_ENABLED || QUALITY_HIT_TRACE_ENABLED ||
                                               QUALITY_BEAM_HIT_TRACE_ENABLED) ? &quality_ctx : nullptr);
        if (result.has_query_expander_efficiency) {
            const int gt_count = std::min(K, static_cast<int>(dim_gt));
            const int ret_count = std::min(K, static_cast<int>(result.neighbors.size()));
            const double gt_l2 = mean_l2_for_ids(
                metric_data, metric_row_stride, query_ptr, dim,
                groundtruth + static_cast<size_t>(q) * dim_gt, gt_count);
            const double ret_l2 = mean_l2_for_ids(
                metric_data, metric_row_stride, query_ptr, dim,
                result.neighbors.data(), ret_count);
            const double relative_excess = topk_rank_relative_l2_excess_for_ids(
                metric_data, metric_row_stride, query_ptr, dim,
                result.neighbors.data(),
                groundtruth + static_cast<size_t>(q) * dim_gt,
                std::min(gt_count, ret_count));
            set_query_topk_efficiency(result, gt_l2, ret_l2, relative_excess);
        }

        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_extra_count_dc += result.extra_dist_cmps;
        total_hop += result.iters;
        total_pruned += result.neighbor_pruned;
        total_expander_trace_events += result.total_expander_trace_events;
        total_expander_positive_delta += result.total_expander_positive_delta;
        total_expander_net_descent += result.total_expander_net_descent;
        if (result.has_query_expander_efficiency) {
            total_query_expander_steps += result.query_expander_steps;
            total_query_expander_count += 1;
            total_query_l2_progress += result.query_l2_progress;
            total_query_l2_progress_per_expander += result.query_l2_progress_per_expander;
        }
        if (result.has_query_topk_efficiency) {
            total_query_topk_metric_count += 1;
            total_query_topk_gt_l2 += result.query_topk_gt_l2;
            total_query_topk_ret_l2 += result.query_topk_ret_l2;
            total_query_topk_l2_progress += result.query_topk_l2_progress;
            total_query_topk_l2_progress_per_expander += result.query_topk_l2_progress_per_expander;
            if (result.has_query_topk_closed_gap) {
                total_query_topk_closed_gap_count += 1;
                total_query_topk_closed_gap += result.query_topk_closed_gap;
            }
            if (result.has_query_topk_relative) {
                total_query_topk_relative_count += 1;
                total_query_topk_relative_excess += result.query_topk_relative_excess;
                total_query_topk_completion += result.query_topk_completion;
            }
        }
        if (EXPANDER_TRACE_ENABLED) {
            #pragma omp critical(expander_trace_merge)
            {
                merge_expander_trace(expander_trace_by_step, result.expander_trace_by_step);
            }
        }
        if (ACCESS_TRACE_ENABLED) {
            #pragma omp critical(access_trace_merge)
            {
                access_trace_records.insert(access_trace_records.end(),
                                            result.access_trace.begin(),
                                            result.access_trace.end());
            }
        }
        if (QUALITY_TRACE_ENABLED) {
            #pragma omp critical(quality_trace_merge)
            {
                quality_trace_records.insert(quality_trace_records.end(),
                                             query_quality_trace.begin(),
                                             query_quality_trace.end());
            }
        }
        if (QUALITY_HIT_TRACE_ENABLED) {
            #pragma omp critical(quality_hit_trace_merge)
            {
                quality_hit_trace_records.insert(quality_hit_trace_records.end(),
                                                 query_quality_hit_trace.begin(),
                                                 query_quality_hit_trace.end());
            }
        }
        if (QUALITY_BEAM_HIT_TRACE_ENABLED) {
            #pragma omp critical(quality_beam_hit_trace_merge)
            {
                quality_beam_hit_trace_records.insert(quality_beam_hit_trace_records.end(),
                                                      query_quality_beam_hit_trace.begin(),
                                                      query_quality_beam_hit_trace.end());
            }
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
    stats.total_expander_trace_events = total_expander_trace_events;
    stats.total_expander_positive_delta = total_expander_positive_delta;
    stats.total_expander_net_descent = total_expander_net_descent;
    stats.total_query_expander_steps = total_query_expander_steps;
    stats.total_query_expander_count = total_query_expander_count;
    stats.total_query_l2_progress = total_query_l2_progress;
    stats.total_query_l2_progress_per_expander = total_query_l2_progress_per_expander;
    stats.total_query_topk_metric_count = total_query_topk_metric_count;
    stats.total_query_topk_closed_gap_count = total_query_topk_closed_gap_count;
    stats.total_query_topk_relative_count = total_query_topk_relative_count;
    stats.total_query_topk_gt_l2 = total_query_topk_gt_l2;
    stats.total_query_topk_ret_l2 = total_query_topk_ret_l2;
    stats.total_query_topk_l2_progress = total_query_topk_l2_progress;
    stats.total_query_topk_l2_progress_per_expander = total_query_topk_l2_progress_per_expander;
    stats.total_query_topk_closed_gap = total_query_topk_closed_gap;
    stats.total_query_topk_relative_excess = total_query_topk_relative_excess;
    stats.total_query_topk_completion = total_query_topk_completion;
    if (EXPANDER_TRACE_ENABLED) {
        stats.expander_trace_by_step = move(expander_trace_by_step);
    }
    if (ACCESS_TRACE_ENABLED && !ACCESS_TRACE_FILE.empty()) {
        std::sort(access_trace_records.begin(), access_trace_records.end(),
                  [](const AccessTraceRecord& a, const AccessTraceRecord& b) {
                      if (a.query_id != b.query_id) return a.query_id < b.query_id;
                      if (a.step != b.step) return a.step < b.step;
                      return a.op < b.op;
                  });
        write_access_trace_csv(ACCESS_TRACE_FILE, access_trace_records);
        cout << "Access trace records: " << access_trace_records.size() << "\n";
    }
    if (QUALITY_TRACE_ENABLED && !QUALITY_TRACE_FILE.empty()) {
        std::sort(quality_trace_records.begin(), quality_trace_records.end(),
                  [](const QualityTraceItem& a, const QualityTraceItem& b) {
                      if (a.query_id != b.query_id) return a.query_id < b.query_id;
                      if (a.step != b.step) return a.step < b.step;
                      if (a.rank != b.rank) return a.rank < b.rank;
                      return a.method < b.method;
                  });
        write_quality_trace_csv(QUALITY_TRACE_FILE, quality_trace_records);
        cout << "Quality trace records: " << quality_trace_records.size() << "\n";
    }
    if (QUALITY_HIT_TRACE_ENABLED && !QUALITY_HIT_TRACE_FILE.empty()) {
        std::sort(quality_hit_trace_records.begin(), quality_hit_trace_records.end(),
                  [](const QualityHitTraceItem& a, const QualityHitTraceItem& b) {
                      if (a.query_id != b.query_id) return a.query_id < b.query_id;
                      if (a.step != b.step) return a.step < b.step;
                      if (a.stage != b.stage) return a.stage < b.stage;
                      return a.method < b.method;
                  });
        write_quality_hit_trace_csv(QUALITY_HIT_TRACE_FILE, quality_hit_trace_records);
        cout << "Quality hit trace records: " << quality_hit_trace_records.size() << "\n";
    }
    if (QUALITY_BEAM_HIT_TRACE_ENABLED && !QUALITY_BEAM_HIT_TRACE_FILE.empty()) {
        std::sort(quality_beam_hit_trace_records.begin(), quality_beam_hit_trace_records.end(),
                  [](const QualityHitTraceItem& a, const QualityHitTraceItem& b) {
                      if (a.query_id != b.query_id) return a.query_id < b.query_id;
                      if (a.step != b.step) return a.step < b.step;
                      if (a.stage != b.stage) return a.stage < b.stage;
                      return a.method < b.method;
                  });
        write_quality_hit_trace_csv(QUALITY_BEAM_HIT_TRACE_FILE, quality_beam_hit_trace_records);
        cout << "Quality beam hit trace records: " << quality_beam_hit_trace_records.size() << "\n";
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
    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -degree <int>" << " [-mode <codebook|bins>]"
        << " [-qg <qg_codebook>]" << " [-graph <graph_index>]"
        << " [-rabitq <rabitq_table>]" << " [-sign <sign_table>]"
        << " [-k <int>]" << " [-beam <int>]" << " [-p_ratio <float>]" << " [-a_ratio <float>]"
        << " [-l2_iter <int>]" << " [-l2_end_iter <int>]" << " [-l2_period <int>]"
        << " [-gpu_like_search]"
        << " [-debug <on|off|1|0>]"
        << " [-expander_trace_file <output.csv>]"
        << " [-access_trace_file <output.csv>]"
        << " [-quality_trace_file <output.csv>] [-quality_hit_trace_file <output.csv>]"
        << " [-quality_beam_hit_trace_file <output.csv>]"
        << " [-quality_trace_queries <int>] [-quality_trace_final_only]"
        << " [-query_limit <int>]"
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
        else if (arg == "-k" && i + 1 < argc) {
            K = atoi(argv[++i]);
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
        else if (arg == "-gpu_like_search") {
            GPU_LIKE_SEARCH = true;
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
        else if (arg == "-expander_trace_file" && i + 1 < argc) {
            EXPANDER_TRACE_FILE = argv[++i];
            EXPANDER_TRACE_ENABLED = true;
        }
        else if (arg == "-access_trace_file" && i + 1 < argc) {
            ACCESS_TRACE_FILE = argv[++i];
            ACCESS_TRACE_ENABLED = true;
        }
        else if (arg == "-quality_trace_file" && i + 1 < argc) {
            QUALITY_TRACE_FILE = argv[++i];
            QUALITY_TRACE_ENABLED = true;
        }
        else if (arg == "-quality_hit_trace_file" && i + 1 < argc) {
            QUALITY_HIT_TRACE_FILE = argv[++i];
            QUALITY_HIT_TRACE_ENABLED = true;
        }
        else if (arg == "-quality_beam_hit_trace_file" && i + 1 < argc) {
            QUALITY_BEAM_HIT_TRACE_FILE = argv[++i];
            QUALITY_BEAM_HIT_TRACE_ENABLED = true;
        }
        else if (arg == "-quality_trace_queries" && i + 1 < argc) {
            QUALITY_TRACE_QUERY_LIMIT = atoi(argv[++i]);
        }
        else if (arg == "-quality_trace_final_only") {
            QUALITY_TRACE_STEPS = false;
        }
        else if (arg == "-query_limit" && i + 1 < argc) {
            QUERY_LIMIT = atoi(argv[++i]);
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

    g_metric_type = infer_metric_from_dataset_path(base_file);
    cout << "Metric: " << metric_name(g_metric_type) << "\n";

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
    qg.set_metric(g_metric_type);

    cout << "Debug mode: " << (DEBUG_MODE ? "on" : "off") << "\n";
    cout << "GPU-like RaBitQ search: " << (GPU_LIKE_SEARCH ? "on" : "off");
    if (GPU_LIKE_SEARCH) {
        cout << " (hash_bitlen=" << gpu_hash_bitlen_for_topk(BEAM_SIZE)
             << ", candidate_buffer=" << RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE << ")";
    }
    cout << "\n";
    if (QUERY_LIMIT > 0) {
        cout << "Query limit: " << QUERY_LIMIT << "\n";
    }
    cout << "Expander trace: " << (EXPANDER_TRACE_ENABLED ? "on" : "off") << "\n";
    cout << "Access trace: " << (ACCESS_TRACE_ENABLED ? "on" : "off") << "\n";
    cout << "Quality trace: " << (QUALITY_TRACE_ENABLED ? "on" : "off");
    if (QUALITY_TRACE_ENABLED) {
        cout << " (" << QUALITY_TRACE_FILE << ", queries=" << QUALITY_TRACE_QUERY_LIMIT
             << ", mode=" << (QUALITY_TRACE_STEPS ? "steps+final" : "final-only") << ")";
    }
    cout << "\n";
    cout << "Quality hit trace: " << (QUALITY_HIT_TRACE_ENABLED ? "on" : "off");
    if (QUALITY_HIT_TRACE_ENABLED) {
        cout << " (" << QUALITY_HIT_TRACE_FILE << ", queries=" << QUALITY_TRACE_QUERY_LIMIT
             << ", mode=" << (QUALITY_TRACE_STEPS ? "steps+final" : "final-only") << ")";
    }
    cout << "\n";
    cout << "Quality beam hit trace: " << (QUALITY_BEAM_HIT_TRACE_ENABLED ? "on" : "off");
    if (QUALITY_BEAM_HIT_TRACE_ENABLED) {
        cout << " (" << QUALITY_BEAM_HIT_TRACE_FILE << ", queries=" << QUALITY_TRACE_QUERY_LIMIT
             << ", mode=" << (QUALITY_TRACE_STEPS ? "steps+final" : "final-only")
             << ", scan=beam" << ")";
    }
    cout << "\n";
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
    if (EXPANDER_TRACE_ENABLED && !EXPANDER_TRACE_FILE.empty()) {
        cout << "Expander trace file: " << EXPANDER_TRACE_FILE << "\n";
    }

    // if (mode == "bins") {
    //     std::string dump_txt = rabitq_table + ".dump.txt";
    //     dump_rabitq_table(rabitq_table, dump_txt, static_cast<int>(dim_base), degree, 20);
    // } else {
    //     cout << "Skipping dump_rabitq_table in codebook mode: helper expects standalone rabitq table format.\n";
    // }

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
    if (EXPANDER_TRACE_ENABLED && !EXPANDER_TRACE_FILE.empty()) {
        dump_expander_trace_stats(EXPANDER_TRACE_FILE, stats);
    }
    return 0;
}
