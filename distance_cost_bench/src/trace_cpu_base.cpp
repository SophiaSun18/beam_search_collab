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
bool GPU_LIKE_HASH = false;
constexpr int CPU_GPU_LIKE_HASH_RESET_INTERVAL = 16;
constexpr int CPU_GPU_LIKE_CANDIDATE_BUFFER_SIZE = 64;
constexpr vidType CPU_GPU_LIKE_INVALID_RESTORE_KEY = 0x7fffffffu;
constexpr vidType CPU_GPU_LIKE_INVALID_CANDIDATE = std::numeric_limits<vidType>::max();

SearchResult beam_cpu_base(
    Graph& g, int K, int npoints,
    const float* d_data, const float* d_query, size_t dim,
    vidType entry_point, uint32_t query_id = 0,
    const QualityTraceContext* quality_ctx = nullptr)
{
    (void)npoints;
    // initialize the search environment
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);     // exact CPU visited filter
    gpu_like_hash_filter<vidType> gpu_like_visited(gpu_hash_bitlen_for_topk(BEAM_SIZE));
    new_pqueue_t<vidType> frontier(BEAM_SIZE);                  // current frontier to hold all candidates
    vector<pair<vidType, float>> curr_candidate(g.degree);      // the distance of each neighbor, used in pruning, clear after each iteration

    size_t dist_cmps = 0;                                       // number of distance computations
    size_t dist_est = 0;
    size_t extra_dist_cmps = 0;
    size_t neighbor_pruned = 0;

    vector<ExpanderTraceStep> expander_trace_by_step;
    double prev_expander_exact_dist = 0.0;
    bool has_prev_expander_exact_dist = false;
    double best_expander_exact_dist = 0.0;
    vector<AccessTraceRecord> access_trace;

    auto exact_distance = [&](vidType node) {
        const float* row = d_data + static_cast<size_t>(node) * dim;
        if (GPU_LIKE_HASH) {
            return compute_distance_gpu_warp_order(static_cast<int>(dim), d_query, row);
        }
        return compute_distance(static_cast<int>(dim), d_query, row);
    };

    // push the fixed entry point to the frontier with max distance
    vector<pair<vidType, float>> pending_candidates;
    if (GPU_LIKE_HASH) {
        float entry_dist = exact_distance(entry_point);
        dist_cmps++;
        pending_candidates.emplace_back(entry_point, entry_dist);
    } else {
        frontier.push(entry_point, FLT_MAX);
    }
    if (GPU_LIKE_HASH) {
        gpu_like_visited.insert(entry_point);
    }

    auto restore_gpu_like_visited = [&]() {
        gpu_like_visited.clear();
        for (int i = 0; i < frontier.size(); i++) {
            gpu_like_visited.insert(frontier[i]);
        }
        for (int i = frontier.size(); i < BEAM_SIZE; i++) {
            gpu_like_visited.insert(CPU_GPU_LIKE_INVALID_RESTORE_KEY);
        }
    };

    auto accept_candidate = [&](vidType node) {
        if (GPU_LIKE_HASH) {
            return gpu_like_visited.insert(node);
        }
        if (is_visited.contains(node)) {
            return false;
        }
        is_visited.add(node);
        return true;
    };

    auto reset_gpu_like_pending_candidates = [&]() {
        pending_candidates.assign(CPU_GPU_LIKE_CANDIDATE_BUFFER_SIZE,
                                  {CPU_GPU_LIKE_INVALID_CANDIDATE, FLT_MAX});
    };

    int iter = 0;
    while (frontier.has_unexpanded() || (GPU_LIKE_HASH && !pending_candidates.empty())) {
        if (GPU_LIKE_HASH) {
            if ((iter + 1) % CPU_GPU_LIKE_HASH_RESET_INTERVAL == 0) {
                restore_gpu_like_visited();
            }
            frontier.merge_candidates_gpu_like(pending_candidates);
            pending_candidates.clear();
            if (!frontier.has_unexpanded()) {
                break;
            }
        }
        iter++;
        curr_candidate.clear();

        // get the closest unexpanded node in the frontier
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        float expander_exact_dist = frontier.get_dist(idx);
        QualityFrontierAudit frontier_audit;
        if (quality_ctx != nullptr && quality_ctx->record_steps) {
            frontier_audit = audit_frontier_exact_rank(*quality_ctx, frontier, cur_node);
        }
        g.prefetch_node(frontier[idx]);
        frontier.set_front_expanded();

        if (EXPANDER_TRACE_ENABLED) {
            if (!std::isfinite(expander_exact_dist) || expander_exact_dist > FLT_MAX * 0.5f) {
                expander_exact_dist = exact_distance(cur_node);
            }
            append_expander_trace_step(expander_trace_by_step, expander_exact_dist,
                                       prev_expander_exact_dist,
                                       has_prev_expander_exact_dist,
                                       best_expander_exact_dist);
        }

        // expand the current node
        const vidType* neighbors = g.get_neighbors(cur_node);
        if (frontier.has_unexpanded()) {
            int idx2 = frontier.get_next_index();
            g.prefetch_node(frontier[idx2]);
        }

        // count the exact distance between each neighbor and the query
        uint32_t active_mask = 0;
        if (GPU_LIKE_HASH) {
            reset_gpu_like_pending_candidates();
        }
        for (size_t i = 0; i < g.degree; i++) {
            vidType n = neighbors[i];
            if (!accept_candidate(n)) {
                continue;   // skip if already computed
            }

            float dist = exact_distance(n);
            dist_cmps++;
            if (i < 32) {
                active_mask |= (1u << i);
            }

            if (GPU_LIKE_HASH) {
                if (i < pending_candidates.size()) {
                    pending_candidates[i] = {n, dist};
                }
                continue;
            }
            curr_candidate.emplace_back(n, dist);
        }
        if (ACCESS_TRACE_ENABLED) {
            access_trace.push_back(
                {query_id, static_cast<uint32_t>(iter - 1),
                 ACCESS_TRACE_BASE_EXPAND_EXACT, cur_node, active_mask});
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

        if (quality_ctx != nullptr && quality_ctx->record_steps) {
            append_quality_trace_snapshot(
                *quality_ctx, iter, "expand", cur_node, expander_exact_dist,
                frontier_audit, quality_queue_items(frontier, K));
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

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    int topK = min(K, (int)frontier.size());
    for (int i = 0; i < topK; i++)
        result.push_back(frontier[i]);

    if (quality_ctx != nullptr) {
        QualityFrontierAudit final_audit;
        append_quality_trace_snapshot(
            *quality_ctx, iter + 1, "final", 0,
            std::numeric_limits<double>::quiet_NaN(), final_audit,
            quality_queue_items(frontier, K));
        append_quality_hit_trace_snapshot(
            *quality_ctx, iter + 1, "final", quality_queue_items(frontier, BEAM_SIZE),
            quality_ctx->beam_hit_records,
            quality_ctx->beam_previous_gt_present,
            quality_ctx->beam_seen_gt_present);
    }

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

    // compute fixed entry point: centroid of base vectors -> nearest neighbor
    vector<float> centroid = compute_centroid(base_vectors, n_base, dim, actual_threads);
    vidType entry_point = exact_nn(base_vectors, centroid.data(), n_base, dim, actual_threads);
    cout << "Entry point: " << entry_point << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    if (DEBUG_MODE) {
        cout << "[DEBUG] debug mode enabled: running only query 0\n";
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
                "base",
                static_cast<uint32_t>(q),
                base_vectors,
                dim,
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
            SearchResult result = beam_cpu_base(g, K, n_base, base_vectors, query_ptr, dim, entry_point,
                                                static_cast<uint32_t>(q),
                                                (QUALITY_TRACE_ENABLED || QUALITY_HIT_TRACE_ENABLED ||
                                                 QUALITY_BEAM_HIT_TRACE_ENABLED) ? &quality_ctx : nullptr);
            if (result.has_query_expander_efficiency) {
                const int gt_count = std::min(K, static_cast<int>(dim_gt));
                const int ret_count = std::min(K, static_cast<int>(result.neighbors.size()));
                const double gt_l2 = mean_l2_for_ids(
                    base_vectors, dim, query_ptr, dim, groundtruth + static_cast<size_t>(q) * dim_gt, gt_count);
                const double ret_l2 = mean_l2_for_ids(
                    base_vectors, dim, query_ptr, dim, result.neighbors.data(), ret_count);
                const double relative_excess = topk_rank_relative_l2_excess_for_ids(
                    base_vectors, dim, query_ptr, dim,
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
                merge_expander_trace(expander_trace_by_step, result.expander_trace_by_step);
            }
            if (ACCESS_TRACE_ENABLED) {
                access_trace_records.insert(access_trace_records.end(),
                                            result.access_trace.begin(),
                                            result.access_trace.end());
            }
            if (QUALITY_TRACE_ENABLED) {
                quality_trace_records.insert(quality_trace_records.end(),
                                             query_quality_trace.begin(),
                                             query_quality_trace.end());
            }
            if (QUALITY_HIT_TRACE_ENABLED) {
                quality_hit_trace_records.insert(quality_hit_trace_records.end(),
                                                 query_quality_hit_trace.begin(),
                                                 query_quality_hit_trace.end());
            }
            if (QUALITY_BEAM_HIT_TRACE_ENABLED) {
                quality_beam_hit_trace_records.insert(quality_beam_hit_trace_records.end(),
                                                      query_quality_beam_hit_trace.begin(),
                                                      query_quality_beam_hit_trace.end());
            }
        }
    } else {
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
                "base",
                static_cast<uint32_t>(q),
                base_vectors,
                dim,
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

            SearchResult result = beam_cpu_base(g, K, n_base, base_vectors, query_ptr, dim, entry_point,
                                                static_cast<uint32_t>(q),
                                                (QUALITY_TRACE_ENABLED || QUALITY_HIT_TRACE_ENABLED ||
                                                 QUALITY_BEAM_HIT_TRACE_ENABLED) ? &quality_ctx : nullptr);
            if (result.has_query_expander_efficiency) {
                const int gt_count = std::min(K, static_cast<int>(dim_gt));
                const int ret_count = std::min(K, static_cast<int>(result.neighbors.size()));
                const double gt_l2 = mean_l2_for_ids(
                    base_vectors, dim, query_ptr, dim, groundtruth + static_cast<size_t>(q) * dim_gt, gt_count);
                const double ret_l2 = mean_l2_for_ids(
                    base_vectors, dim, query_ptr, dim, result.neighbors.data(), ret_count);
                const double relative_excess = topk_rank_relative_l2_excess_for_ids(
                    base_vectors, dim, query_ptr, dim,
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
        << " -graph <graph_index>\n" << " [-k <int>]" << " [-beam <int>]"
        << " [-p_ratio <float>]" << " [-a_ratio <float>]"
        << " [-gpu_like_hash]"
        << " [-debug <on|off|1|0>]" << " [-expander_trace_file <output.csv>]"
        << " [-access_trace_file <output.csv>]"
        << " [-quality_trace_file <output.csv>] [-quality_hit_trace_file <output.csv>]"
        << " [-quality_beam_hit_trace_file <output.csv>]"
        << " [-quality_trace_queries <int>] [-quality_trace_final_only]"
        << " [-query_limit <int>]"
        << " [-csv <output.csv>]\n";
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
        else if (arg == "-gpu_like_hash") {
            GPU_LIKE_HASH = true;
        }
        else if (arg == "-csv" && i + 1 < argc) {
            csv_file = argv[++i];
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
    cout << "GPU-like CPU hash: " << (GPU_LIKE_HASH ? "on" : "off");
    if (GPU_LIKE_HASH) {
        cout << " (bitlen=" << gpu_hash_bitlen_for_topk(BEAM_SIZE)
             << ", reset_interval=" << CPU_GPU_LIKE_HASH_RESET_INTERVAL << ")";
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

    // Run experiment
    RunStats stats = run_ann_experiment_base(g, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt);

    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }
    if (EXPANDER_TRACE_ENABLED && !EXPANDER_TRACE_FILE.empty()) {
        dump_expander_trace_stats(EXPANDER_TRACE_FILE, stats);
    }
    return 0;
}
