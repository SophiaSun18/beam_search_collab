#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include "search_rabitq_pruned_spec_adaptive_config.cuh"
#include <cuda_runtime.h>

struct RabitqPrunedAdaptiveSearchState {
    int adaptive_spec_degree;       // Number of parent expanders AP will use in the next expansion iteration.
    float adaptive_prune_ratio;     // Prune ratio selected for the next local-gate pass.
    int policy_iters;               // Number of AP policy decisions already made.
    INDEX_T top1_node;             // Last observed top-1 node for warmup-stage detection.
    int top1_stall_iters;          // Consecutive policy checks where top-1 node did not change.
    bool warmup_done;              // Whether top-1 warmup has switched to aggressive rho.
    DISTANCE_T entry_distance;      // Exact distance from query to the entry point.
    DISTANCE_T last_expander_distance;     // Previous policy check's last selected expander distance.
    DISTANCE_T current_expander_distance;  // Current policy check's last selected expander distance.
};

static __device__ __forceinline__ DISTANCE_T rabitq_pruned_adaptive_current_kth_cutoff(
    int K, uint32_t logical_beam_size, const INDEX_T* topk_index, const DISTANCE_T* topk_distance) {
    if (topk_index[logical_beam_size - 1] == MAX_INDEX) return FLT_MAX;
    const uint32_t kth_pos = ((static_cast<uint32_t>(K) < logical_beam_size)
        ? static_cast<uint32_t>(K) : logical_beam_size) - 1;
    return (topk_index[kth_pos] != MAX_INDEX) ? topk_distance[kth_pos] : FLT_MAX;
}

static __device__ __forceinline__ int rabitq_pruned_adaptive_update_head_stall(
    const INDEX_T* topk_index, const RabitqPrunedAdaptiveSearchState& previous_state, INDEX_T* current_top1) {
    *current_top1 = (topk_index[0] != MAX_INDEX) ? (topk_index[0] & 0x7fffffffu) : MAX_INDEX;
    if (previous_state.policy_iters > 0 && *current_top1 != MAX_INDEX && *current_top1 == previous_state.top1_node) {
        return previous_state.top1_stall_iters + 1;
    }
    return 0;
}

static __device__ __forceinline__ bool rabitq_pruned_adaptive_phase_changed(
    const RabitqPrunedAdaptiveSearchState& previous_state, int head_stall_iters) {
    return previous_state.warmup_done || head_stall_iters >= static_cast<int>(RABITQ_ADAPTIVE_TOP1_WARMUP_STALL_ITERS);
}

static __device__ __forceinline__ float rabitq_pruned_adaptive_update_expander_progress(
    const RabitqPrunedAdaptiveSearchState& previous_state,
    DISTANCE_T* last_expander_distance, DISTANCE_T* current_expander_distance) {
    const DISTANCE_T entry_distance = previous_state.entry_distance;
    const bool have_last_expander =
        isfinite(previous_state.last_expander_distance) && previous_state.last_expander_distance < FLT_MAX;
    const bool have_current_expander =
        isfinite(previous_state.current_expander_distance) && previous_state.current_expander_distance < FLT_MAX;
    *last_expander_distance = have_last_expander ? previous_state.last_expander_distance : entry_distance;
    *current_expander_distance = have_current_expander
        ? previous_state.current_expander_distance : *last_expander_distance;

    const float denom = fabsf(static_cast<float>(entry_distance));
    if (!isfinite(*last_expander_distance) || !isfinite(*current_expander_distance) ||
        !isfinite(entry_distance) || denom <= 1.0e-20f) {
        return 0.0f;
    }

    const float progress = static_cast<float>(*last_expander_distance - *current_expander_distance) / denom;
    if (!isfinite(progress) || progress <= 0.0f) return 0.0f;
    return progress < 1.0f ? progress : 1.0f;
}

static __device__ __forceinline__ RabitqPrunedAdaptiveSearchState rabitq_pruned_apply_stage2_state(
    INDEX_T current_top1, int head_stall_iters, DISTANCE_T last_expander_distance, DISTANCE_T current_expander_distance,
    const RabitqPrunedAdaptiveSearchState& previous_state) {
    RabitqPrunedAdaptiveSearchState state = previous_state;

    state.top1_node = current_top1;
    state.top1_stall_iters = head_stall_iters;
    state.warmup_done = true;
    state.last_expander_distance = last_expander_distance;
    state.current_expander_distance = current_expander_distance;

    state.adaptive_spec_degree = static_cast<int>(RABITQ_ADAPTIVE_PHASE2_THETA);
    state.adaptive_prune_ratio = static_cast<float>(RABITQ_ADAPTIVE_PHASE2_RHO);
    state.policy_iters = previous_state.policy_iters + 1;
    return state;
}

static __device__ __forceinline__ RabitqPrunedAdaptiveSearchState rabitq_pruned_collect_adaptive_state_new(
    INDEX_T current_top1, int head_stall_iters, DISTANCE_T last_expander_distance,
    DISTANCE_T current_expander_distance, float distance_reduction_rate,
    const RabitqPrunedAdaptiveSearchState& previous_state) {
    RabitqPrunedAdaptiveSearchState state = previous_state;

    state.top1_node = current_top1;
    state.top1_stall_iters = head_stall_iters;
    state.warmup_done = false;
    state.last_expander_distance = last_expander_distance;
    state.current_expander_distance = current_expander_distance;

    const float tune_span = RABITQ_ADAPTIVE_PHASE2_RHO > RABITQ_ADAPTIVE_PHASE1_RHO
        ? RABITQ_ADAPTIVE_PHASE2_RHO - RABITQ_ADAPTIVE_PHASE1_RHO
        : RABITQ_ADAPTIVE_RHO_MAX - RABITQ_ADAPTIVE_RHO_MIN;
    const float base_prune = previous_state.policy_iters == 0
        ? RABITQ_ADAPTIVE_PHASE1_RHO
        : previous_state.adaptive_prune_ratio;
    const float rho_delta = tune_span * distance_reduction_rate;
    const float next_prune_cap = RABITQ_ADAPTIVE_PHASE2_RHO > RABITQ_ADAPTIVE_PHASE1_RHO
        ? RABITQ_ADAPTIVE_PHASE2_RHO
        : RABITQ_ADAPTIVE_RHO_MAX;
    const float next_prune = clamp_float(base_prune + rho_delta, RABITQ_ADAPTIVE_RHO_MIN, next_prune_cap);

    state.adaptive_spec_degree = static_cast<int>(RABITQ_ADAPTIVE_PHASE1_THETA);
    state.adaptive_prune_ratio = next_prune;
    state.policy_iters = previous_state.policy_iters + 1;
    return state;
}

// Select the number of neighbors to keep for one expander based on the current AP state.
// When Kth cutoff is enabled and valid, kth_near_count is the exact number of
// estimated candidates with est_dist <= current kth distance; it tightens the
// keep count without hard-invalidating candidates before the top-k selection.
static __device__ __forceinline__ int rabitq_pruned_adaptive_keep_count_for_expander(
    uint32_t active_neighbors, const RabitqPrunedAdaptiveSearchState& state,
    bool kth_cutoff_valid = false, uint32_t kth_near_count = 0) {
    const int active_count = static_cast<int>(active_neighbors);
    if (active_count <= 0) return 0;

    // Adjust the kept candidate count to be within the shared candidate buffer bound.
    const int prune_keep = static_cast<int>(keep_count_per_parent(active_neighbors, state.adaptive_prune_ratio));
    const uint32_t spec_degree = static_cast<uint32_t>(
        state.adaptive_spec_degree > 0 ? state.adaptive_spec_degree : 1);
    const uint32_t per_parent_bound = static_cast<uint32_t>(RABITQ_ADAPTIVE_BUFFER_BOUND) / spec_degree;
    const int budget_keep = clamp_int(static_cast<int>(per_parent_bound), 1, active_count);
    int max_keep = (prune_keep < budget_keep) ? prune_keep : budget_keep;
#if RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF
    if (kth_cutoff_valid) {
        max_keep = clamp_int(static_cast<int>(kth_near_count), 1, max_keep);
    }
#endif
    return clamp_int(max_keep, 1, active_count);
}

static __host__ __device__ inline uint32_t rabitq_pruned_adaptive_candidate_buffer_capacity(uint32_t max_degree) {
    const uint32_t max_capacity = static_cast<uint32_t>(RABITQ_ADAPTIVE_THETA_MAX) * max_degree;
    uint32_t capacity = static_cast<uint32_t>(RABITQ_ADAPTIVE_BUFFER_BOUND);
    if (capacity > max_capacity) capacity = max_capacity;
    return capacity;
}

static __host__ inline uint32_t calculate_pruned_shared_mem_size(int dim, int beam_sz, int result_k, int max_deg,
    int bitlen, float prune_ratio = 0.0f, float approx_ratio = 1.0f) {
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const size_t candidate_buffer_size = round_up_power2_u32(rabitq_pruned_adaptive_candidate_buffer_capacity(static_cast<uint32_t>(max_deg)));
    const uint32_t padded_beam_size = gpu_effective_sort_beam_size(static_cast<uint32_t>(beam_sz));
    const size_t result_buffer_size = static_cast<size_t>(padded_beam_size) + candidate_buffer_size;
    size_t size = 0;
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);       // visited-node hash table
    size += result_buffer_size * sizeof(INDEX_T);              // TOP_K_INDEX + CANDIDATE_INDEX
    size += result_buffer_size * sizeof(DISTANCE_T);           // TOP_K_DISTANCE + CANDIDATE_DISTANCE
    size += SEARCH_WIDTH * sizeof(INDEX_T);                    // PARENT_LIST: beam positions/ranks selected by pick_expanders
    size += SEARCH_WIDTH * sizeof(INDEX_T);                    // PARENT_NODE_LIST: graph node ids for selected parents
    size += SEARCH_WIDTH * sizeof(DISTANCE_T);                 // PARENT_DISTANCE_LIST: exact distances for selected parents
    size += dim * sizeof(DATA_T);                              // QUERY_BUFFER
    size += padded_dim * sizeof(float);                        // ROTATED_QUERY_BUFFER
    size += (padded_dim << 2) * sizeof(uint8_t);               // LUT_BUFFER
    size += 3 * sizeof(float);                                 // qf low/high/width
    size += sizeof(int32_t);                                   // qf sum_q
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint64_t)));
    size += sizeof(uint32_t);                                   // hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);       // search stats
    }
    size = static_cast<size_t>(align_up_uintptr(size, alignof(INDEX_T)));
    if (gpu_topk_external_merge_scratch_needed(padded_beam_size)) {
        size += padded_beam_size * sizeof(INDEX_T);             // MERGED_TOPK_INDEX scratch
        size = static_cast<size_t>(align_up_uintptr(size, alignof(DISTANCE_T)));
        size += padded_beam_size * sizeof(DISTANCE_T);          // MERGED_TOPK_DISTANCE scratch
    }
    return static_cast<uint32_t>(size);
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedPrunedBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, int max_degree, size_t npoints,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_block_stats,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset,
    bool use_ip,
    float prune_ratio,
    float approx_ratio
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    uint32_t local_exact_count = 0;
    uint32_t local_dist_est_count = 0;
    uint32_t local_pruned_count = 0;
    uint32_t local_loop_cycles = 0;
    uint32_t local_exact_cycles = 0;
    uint32_t local_est_cycles = 0;
    uint32_t local_rq_stage_cycles = 0;

    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const uint32_t candidate_buffer_size = round_up_power2_u32(rabitq_pruned_adaptive_candidate_buffer_capacity(static_cast<uint32_t>(max_degree)));
    uint32_t candidate_collect_capacity = static_cast<uint32_t>(RABITQ_ADAPTIVE_BUFFER_BOUND);
    if (candidate_collect_capacity > candidate_buffer_size) candidate_collect_capacity = candidate_buffer_size;
    const uint32_t logical_beam_size = static_cast<uint32_t>(beam_sz);
    const uint32_t padded_beam_size = gpu_effective_sort_beam_size(logical_beam_size);

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = reinterpret_cast<INDEX_T*>(shared_memory);
    INDEX_T* TOP_K_INDEX = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX = ALL_INDEX + padded_beam_size;

    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    DISTANCE_T* ALL_DISTANCE = reinterpret_cast<DISTANCE_T*>(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE = ALL_DISTANCE + padded_beam_size;

    INDEX_T* HASH_TABLE = reinterpret_cast<INDEX_T*>(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_LIST = HASH_TABLE + hashtable_getsize(bitlen);
    INDEX_T* PARENT_NODE_LIST = PARENT_LIST + SEARCH_WIDTH;
    DISTANCE_T* PARENT_DISTANCE_LIST = reinterpret_cast<DISTANCE_T*>(PARENT_NODE_LIST + SEARCH_WIDTH);
    DATA_T* QUERY_BUFFER = reinterpret_cast<DATA_T*>(PARENT_DISTANCE_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* LUT_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    char* query_factor_base = reinterpret_cast<char*>(LUT_BUFFER + (padded_dim << 2));
    float* low_val = reinterpret_cast<float*>(query_factor_base);
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* staged_base = reinterpret_cast<char*>(sum_q + 1);
    staged_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) & ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint32_t* hops = reinterpret_cast<uint32_t*>(staged_base);
    uint32_t* stats = hops + 1;
    char* tail_base = reinterpret_cast<char*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    tail_base = reinterpret_cast<char*>(align_up_uintptr(reinterpret_cast<uintptr_t>(tail_base), alignof(INDEX_T)));
    INDEX_T* MERGED_TOPK_INDEX = nullptr;
    DISTANCE_T* MERGED_TOPK_DISTANCE = nullptr;
    if (gpu_topk_external_merge_scratch_needed(padded_beam_size)) {
        MERGED_TOPK_INDEX = reinterpret_cast<INDEX_T*>(tail_base);
        tail_base = reinterpret_cast<char*>(MERGED_TOPK_INDEX + padded_beam_size);
        tail_base = reinterpret_cast<char*>(align_up_uintptr(reinterpret_cast<uintptr_t>(tail_base), alignof(DISTANCE_T)));
        MERGED_TOPK_DISTANCE = reinterpret_cast<DISTANCE_T*>(tail_base);
    }

    __shared__ QueryFactors qf;
    __shared__ uint32_t keep_expanding;
    __shared__ RabitqPrunedAdaptiveSearchState adaptive_state;
    __shared__ uint32_t adaptive_search_loop_iters;
    __shared__ uint32_t adaptive_search_expansion_steps;
    __shared__ uint32_t compact_candidate_count;
    __shared__ uint32_t shared_kth_near_count;
#if RABITQ_ADAPTIVE_TRACE
    __shared__ uint32_t adaptive_trace_gate_scored_count;
    __shared__ uint32_t adaptive_trace_gate_kept_count;
    __shared__ uint32_t adaptive_trace_gate_inserted_count;
    __shared__ uint32_t adaptive_trace_gate_overflow_count;
#endif
    __shared__ uint32_t warp_stat_counts[7 * WARPS_PER_BLOCK];

    // initialize the query and search state
    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = nullptr;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        adaptive_state.adaptive_spec_degree = static_cast<int>(RABITQ_ADAPTIVE_PHASE1_THETA);
        adaptive_state.adaptive_prune_ratio = static_cast<float>(RABITQ_ADAPTIVE_PHASE1_RHO);
        adaptive_state.policy_iters = 0;
        adaptive_state.top1_node = MAX_INDEX;
        adaptive_state.top1_stall_iters = 0;
        adaptive_state.warmup_done = false;
        adaptive_state.entry_distance = FLT_MAX;
        adaptive_state.last_expander_distance = FLT_MAX;
        adaptive_state.current_expander_distance = FLT_MAX;
        adaptive_search_loop_iters = 0;
        adaptive_search_expansion_steps = 0;
        compact_candidate_count = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
    }

    // initialize the hash table to record expanded nodes
    hashtable_init(HASH_TABLE, bitlen);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        QUERY_BUFFER[i] = query[i];
    }

    // load the top k list and candidate list with dummy initial values
    for (int i = tid; i < static_cast<int>(padded_beam_size); i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < candidate_buffer_size; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < SEARCH_WIDTH; i += blockDim.x) {
        PARENT_NODE_LIST[i] = MAX_INDEX;
        PARENT_DISTANCE_LIST[i] = FLT_MAX;
    }
    __syncthreads();

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        const unsigned long long entry_exact_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;
        DISTANCE_T entry_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(entry_point) * row_offset, use_ip);
        if (GPU_SEARCH_STATS && lane_id == 0) {
            local_exact_cycles += static_cast<uint32_t>(clock64() - entry_exact_start_clock);
        }
        if (lane_id == 0) {
            CANDIDATE_INDEX[0] = entry_point;
            CANDIDATE_DISTANCE[0] = entry_dist;
            adaptive_state.entry_distance = entry_dist;
            hashtable_insert(HASH_TABLE, bitlen, entry_point);
            local_exact_count++;
        }
    }
    __syncthreads();

    // query preparation, quantize the query and prepare the RaBitQ scoring state
    query_prepare_lut_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    __syncthreads();

    // loop end condition: either entire topK expanded, or reach max iteration
    const float prune_iteration_limit = static_cast<float>(beam_sz) * approx_ratio;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        const unsigned long long iter_loop_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;

        // forgettable hash table management
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(HASH_TABLE, bitlen);
            __syncthreads();
            hashtable_restore(HASH_TABLE, bitlen, TOP_K_INDEX, beam_sz);
        }
        __syncthreads();

        // sort and merge the existing candidates into the beam
        const uint32_t merge_candidate_count = candidate_buffer_size;
        dispatch_topk_by_sort_candidates_and_merge_runtime(
            ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
            result_buffer_size, merge_candidate_count, padded_beam_size, (iter == 0));
        __syncthreads();

        for (uint32_t i = logical_beam_size + tid; i < padded_beam_size; i += blockDim.x) {
            TOP_K_INDEX[i] = MAX_INDEX;
            TOP_K_DISTANCE[i] = FLT_MAX;
        }
        __syncthreads();

#if RABITQ_ADAPTIVE_BEAM_TRACE
        if (tid == 0 && query_id < RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT &&
            iter < RABITQ_ADAPTIVE_BEAM_TRACE_LIMIT) {
            const uint32_t beam_print =
                (logical_beam_size < RABITQ_ADAPTIVE_BEAM_TRACE_TOPN)
                    ? logical_beam_size
                    : static_cast<uint32_t>(RABITQ_ADAPTIVE_BEAM_TRACE_TOPN);
            printf("AP_BEAM_STATE q=%d iter=%d beam_print=%u beam_size=%u admitted_prev=%u\n",
                   query_id, iter + 1, beam_print, logical_beam_size, compact_candidate_count);
            for (uint32_t bi = 0; bi < beam_print; ++bi) {
                printf("AP_BEAM_ITEM q=%d iter=%d rank=%u node=%u dist=%.9g expanded=%u\n",
                       query_id, iter + 1, bi,
                       static_cast<uint32_t>(TOP_K_INDEX[bi] & 0x7fffffffu),
                       static_cast<double>(TOP_K_DISTANCE[bi]),
                       static_cast<uint32_t>((TOP_K_INDEX[bi] >> 31) & 1u));
            }
        }
#endif
        __syncthreads();

        // clear the candidate buffer before next iteration to avoid stale entries
        for (int i = tid; i < candidate_buffer_size; i += blockDim.x) {
            CANDIDATE_INDEX[i] = MAX_INDEX;
            CANDIDATE_DISTANCE[i] = FLT_MAX;
        }
        for (int i = tid; i < SEARCH_WIDTH; i += blockDim.x) {
            PARENT_NODE_LIST[i] = MAX_INDEX;
            PARENT_DISTANCE_LIST[i] = FLT_MAX;
        }
        if (tid == 0) {
            compact_candidate_count = 0;
#if RABITQ_ADAPTIVE_TRACE
            adaptive_trace_gate_scored_count = 0;
            adaptive_trace_gate_kept_count = 0;
            adaptive_trace_gate_inserted_count = 0;
            adaptive_trace_gate_overflow_count = 0;
#endif
        }
        __syncthreads();

        // Update AP parameters before selecting expanders for this iteration.
        const DISTANCE_T current_kth_cutoff = rabitq_pruned_adaptive_current_kth_cutoff(
            K, logical_beam_size, TOP_K_INDEX, TOP_K_DISTANCE);
        bool phase_changed_for_trace = false;
        float distance_reduction_rate_for_trace = 0.0f;
        if (tid == 0) {
            INDEX_T current_top1 = MAX_INDEX;
            const int head_stall_iters = rabitq_pruned_adaptive_update_head_stall(
                TOP_K_INDEX, adaptive_state, &current_top1);
            const bool phase_changed = rabitq_pruned_adaptive_phase_changed(
                adaptive_state, head_stall_iters);
            DISTANCE_T last_expander_distance = FLT_MAX;
            DISTANCE_T current_expander_distance = FLT_MAX;
            const float distance_reduction_rate = rabitq_pruned_adaptive_update_expander_progress(
                adaptive_state, &last_expander_distance, &current_expander_distance);
            phase_changed_for_trace = phase_changed;
            distance_reduction_rate_for_trace = distance_reduction_rate;

            if (phase_changed) {
                adaptive_state = rabitq_pruned_apply_stage2_state(
                    current_top1, head_stall_iters,
                    last_expander_distance, current_expander_distance,
                    adaptive_state);
            } else {
                const bool adaptive_should_check =
                    (iter == 0) || ((iter % static_cast<int>(RABITQ_ADAPTIVE_CHECK_INTERVAL)) == 0);
                if (adaptive_should_check) {
                    adaptive_state = rabitq_pruned_collect_adaptive_state_new(
                        current_top1, head_stall_iters,
                        last_expander_distance, current_expander_distance,
                        distance_reduction_rate, adaptive_state);
                } else {
                    adaptive_state.top1_node = current_top1;
                    adaptive_state.top1_stall_iters = head_stall_iters;
                    adaptive_state.last_expander_distance = last_expander_distance;
                    adaptive_state.current_expander_distance = current_expander_distance;
                }
            }
        }
        __syncthreads();

        // Pick expanders using the parameters selected above.
        if (warp_id == 0) {
            keep_expanding = pick_expanders(
                adaptive_state.adaptive_spec_degree, PARENT_LIST, logical_beam_size, TOP_K_INDEX);
        }
        __syncthreads();

        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        // collect the expanders into the parent list along with their exact distance
        for (int p = tid; p < SEARCH_WIDTH; p += blockDim.x) {
            if (p < static_cast<int>(keep_expanding)) {
                const uint32_t parent_pos = PARENT_LIST[p];
                const INDEX_T parent_node = TOP_K_INDEX[parent_pos] & 0x7fffffffu;
                PARENT_NODE_LIST[p] = (parent_node != MAX_INDEX && parent_node < npoints) ? parent_node : MAX_INDEX;
                PARENT_DISTANCE_LIST[p] = (parent_node != MAX_INDEX && parent_node < npoints) ? TOP_K_DISTANCE[parent_pos] : FLT_MAX;
            }
        }
        __syncthreads();

        if (tid == 0) {
            const DISTANCE_T selected_expander_distance =
                (keep_expanding > 0 && keep_expanding <= static_cast<uint32_t>(SEARCH_WIDTH))
                    ? PARENT_DISTANCE_LIST[keep_expanding - 1] : FLT_MAX;
            const bool have_previous_expander =
                isfinite(adaptive_state.current_expander_distance) && adaptive_state.current_expander_distance < FLT_MAX;
            adaptive_state.last_expander_distance = have_previous_expander
                ? adaptive_state.current_expander_distance : adaptive_state.entry_distance;
            adaptive_state.current_expander_distance = selected_expander_distance;
            adaptive_search_loop_iters++;
            adaptive_search_expansion_steps += keep_expanding;
#if RABITQ_ADAPTIVE_TRACE
            if (query_id < RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT &&
                iter < RABITQ_ADAPTIVE_TRACE_ITER_LIMIT) {
                printf("AP_TRACE q=%d iter=%d phase_changed=%d spec_degree=%d p=%.6f bound=%u kth=%.9g expanders=%u use_pruning=%d top1=%u top1_stall=%d warmup_done=%d entry_dist=%.9g last_expander_dist=%.9g current_expander_dist=%.9g dist_reduction_rate=%.6f\n",
                       query_id, iter + 1, static_cast<int>(phase_changed_for_trace), adaptive_state.adaptive_spec_degree,
                       adaptive_state.adaptive_prune_ratio, static_cast<uint32_t>(RABITQ_ADAPTIVE_BUFFER_BOUND),
                       static_cast<double>(current_kth_cutoff),
                       keep_expanding, static_cast<int>(prune_ratio > 0.0f &&
                           static_cast<float>(iter + 1) <= prune_iteration_limit),
                       static_cast<uint32_t>(adaptive_state.top1_node & 0x7fffffffu),
                       adaptive_state.top1_stall_iters, static_cast<int>(adaptive_state.warmup_done),
                       static_cast<double>(adaptive_state.entry_distance),
                       static_cast<double>(adaptive_state.last_expander_distance),
                       static_cast<double>(adaptive_state.current_expander_distance),
                       distance_reduction_rate_for_trace);
            }
#endif
        }
        __syncthreads();

#if RABITQ_ADAPTIVE_BEAM_TRACE
        if (tid == 0 && query_id < RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT &&
            iter < RABITQ_ADAPTIVE_BEAM_TRACE_LIMIT) {
            printf("AP_EXPAND_STATE q=%d iter=%d keep_expanding=%u spec_degree=%d p=%.6f\n",
                   query_id, iter + 1, keep_expanding, adaptive_state.adaptive_spec_degree,
                   adaptive_state.adaptive_prune_ratio);
            for (uint32_t pi = 0; pi < keep_expanding && pi < static_cast<uint32_t>(SEARCH_WIDTH); ++pi) {
                printf("AP_PARENT_ITEM q=%d iter=%d parent_slot=%u beam_pos=%u node=%u dist=%.9g\n",
                       query_id, iter + 1, pi, static_cast<uint32_t>(PARENT_LIST[pi]),
                       static_cast<uint32_t>(PARENT_NODE_LIST[pi]),
                       static_cast<double>(PARENT_DISTANCE_LIST[pi]));
            }
        }
#endif

        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS] += keep_expanding;
        }

        const bool use_pruning = prune_ratio > 0.0f && static_cast<float>(iter + 1) <= prune_iteration_limit;
        if (!use_pruning) {
            // if current iteration doesn't prune: collect and compute exact distance for all candidates
            uint32_t exact_candidate_limit = keep_expanding * static_cast<uint32_t>(max_degree);

            // each warp takes one child from the parent list and compute the exact distance for the child
            for (int i = warp_id; i < static_cast<int>(exact_candidate_limit); i += WARPS_PER_BLOCK) {
                const uint32_t parent_idx = static_cast<uint32_t>(i) / static_cast<uint32_t>(max_degree);
                const uint32_t neighbor_idx = static_cast<uint32_t>(i) - parent_idx * static_cast<uint32_t>(max_degree);
                const INDEX_T parent_node = PARENT_NODE_LIST[parent_idx];
                INDEX_T child_id = MAX_INDEX;
                uint32_t inserted = 0;
                if (lane_id == 0) {
                    if (parent_node != MAX_INDEX) {
                        const float* parent_row = d_qg_data + static_cast<size_t>(parent_node) * row_offset;
                        const vidType* parent_neighbors = reinterpret_cast<const vidType*>(parent_row + neighbor_offset);
                        child_id = parent_neighbors[neighbor_idx];
                        if (child_id != MAX_INDEX && child_id < npoints) {
                            inserted = hashtable_insert(HASH_TABLE, bitlen, child_id);
                        }
                    }
                    if (!inserted) {
                        child_id = MAX_INDEX;
                    }
                }
                child_id = SHFL(child_id, 0);
                inserted = SHFL(inserted, 0);

                DISTANCE_T child_dist = FLT_MAX;
                if (inserted && child_id != MAX_INDEX) {
                    const unsigned long long exact_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;
                    child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                    if (GPU_SEARCH_STATS && lane_id == 0) {
                        local_exact_cycles += static_cast<uint32_t>(clock64() - exact_start_clock);
                    }
                }
                if (lane_id == 0) {
                    CANDIDATE_INDEX[i] = child_id;
                    CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : child_dist;
                    if (child_id != MAX_INDEX) {
                        local_exact_count++;
                    }
                }
            }
        } else {
            const bool use_local_gate = adaptive_state.adaptive_spec_degree > 1;
            if (!use_local_gate) {
                const uint32_t candidate_work_count =
                    (static_cast<uint32_t>(max_degree) < candidate_buffer_size)
                        ? static_cast<uint32_t>(max_degree) : candidate_buffer_size;
                const INDEX_T parent_node = PARENT_NODE_LIST[0];
                const float* parent_row = (parent_node != MAX_INDEX)
                    ? d_qg_data + static_cast<size_t>(parent_node) * row_offset : nullptr;
                const vidType* parent_neighbors = (parent_row != nullptr)
                    ? reinterpret_cast<const vidType*>(parent_row + neighbor_offset) : nullptr;
                const float* parent_factors = (parent_row != nullptr) ? parent_row + factor_offset : nullptr;
                const uint8_t* parent_code_base = (parent_row != nullptr)
                    ? reinterpret_cast<const uint8_t*>(parent_row + code_offset) : nullptr;

                for (int i = tid; i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
                    INDEX_T child_id = MAX_INDEX;
                    if (i < static_cast<int>(candidate_work_count) && parent_neighbors != nullptr) {
                        child_id = parent_neighbors[i];
                        if (child_id >= npoints || hashtable_contains(HASH_TABLE, bitlen, child_id)) {
                            child_id = MAX_INDEX;
                        }
                    }
                    CANDIDATE_INDEX[i] = child_id;
                    CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0.0f;
                }
                __syncthreads();

                const int keep_count = rabitq_pruned_adaptive_keep_count_for_expander(
                    candidate_work_count, adaptive_state);
                const uint32_t valid_candidate_count = count_valid_candidates_warp_reduced(
                    CANDIDATE_INDEX, CANDIDATE_DISTANCE, static_cast<int>(candidate_work_count),
                    MAX_INDEX, warp_stat_counts);
                const bool keep_all_valid = keep_count >= static_cast<int>(valid_candidate_count);
                int effective_keep_count = keep_count;

                if (!keep_all_valid) {
                    const int bytes_per_neighbor = padded_dim >> 3;
                    uint32_t warp_dist_est_count = 0;

                    constexpr int lanes_per_neighbor = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
                    constexpr int groups_per_warp = WARP_SIZE / lanes_per_neighbor;
                    const int group_lane = lane_id & (lanes_per_neighbor - 1);
                    const int warp_group = lane_id / lanes_per_neighbor;
                    uint32_t group_dist_est_count = 0;

                    for (int base = warp_id * groups_per_warp;
                         base < static_cast<int>(candidate_work_count);
                         base += WARPS_PER_BLOCK * groups_per_warp) {
                        const int i = base + warp_group;
                        bool valid_candidate = i < static_cast<int>(candidate_work_count) &&
                            CANDIDATE_INDEX[i] != MAX_INDEX;
                        DISTANCE_T est_dist = FLT_MAX;
                        if (valid_candidate) {
                            const float* triple_x = parent_factors + i;
                            const float* factor_dq = parent_factors + max_degree + i;
                            const float* factor_vq = parent_factors + 2 * max_degree + i;
                            const unsigned long long est_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;
                            est_dist = scan_one_neighbor_lanes_gpu<lanes_per_neighbor>(
                                qf, parent_code_base, i, triple_x, factor_dq, factor_vq,
                                PARENT_DISTANCE_LIST[0], padded_dim, bytes_per_neighbor);
                            if (GPU_SEARCH_STATS) {
                                local_est_cycles += static_cast<uint32_t>(clock64() - est_start_clock);
                            }
                            valid_candidate = isfinite(est_dist);
                            if (group_lane == 0) {
                                CANDIDATE_DISTANCE[i] = valid_candidate ? est_dist : FLT_MAX;
                                if (!valid_candidate) {
                                    CANDIDATE_INDEX[i] = MAX_INDEX;
                                }
                                if (GPU_SEARCH_STATS) group_dist_est_count++;
                            }
                        }
                        __syncwarp();
                    }

                    if (GPU_SEARCH_STATS) {
                        warp_dist_est_count = group_dist_est_count;
                        for (int offset = lanes_per_neighbor; offset < WARP_SIZE; offset <<= 1) {
                            warp_dist_est_count += __shfl_down_sync(FULL_MASK, warp_dist_est_count, offset);
                        }
                    }
                    if (lane_id == 0 && GPU_SEARCH_STATS) {
                        local_dist_est_count += warp_dist_est_count;
                    }
                    __syncthreads();

#if RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF
                    const DISTANCE_T kth_cutoff = current_kth_cutoff;
                    const bool kth_cutoff_valid =
                        isfinite(static_cast<float>(kth_cutoff)) && kth_cutoff < FLT_MAX;
                    if (tid == 0) {
                        shared_kth_near_count = 0;
                    }
                    __syncthreads();
                    if (kth_cutoff_valid) {
                        uint32_t thread_near_count = 0;
                        for (int i = tid; i < static_cast<int>(candidate_work_count); i += blockDim.x) {
                            const DISTANCE_T dist = CANDIDATE_DISTANCE[i];
                            if (CANDIDATE_INDEX[i] != MAX_INDEX && isfinite(dist) &&
                                dist <= kth_cutoff) {
                                thread_near_count++;
                            }
                        }
                        if (thread_near_count > 0) {
                            atomicAdd(&shared_kth_near_count, thread_near_count);
                        }
                    }
                    __syncthreads();
                    effective_keep_count = rabitq_pruned_adaptive_keep_count_for_expander(
                        candidate_work_count, adaptive_state, kth_cutoff_valid, shared_kth_near_count);
#endif

                    candidate_by_bitonic_sort_runtime(
                        CANDIDATE_INDEX, CANDIDATE_DISTANCE, candidate_buffer_size);
                    __syncthreads();

                    for (int i = tid; i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
                        if (i >= effective_keep_count) {
                            if (CANDIDATE_INDEX[i] != MAX_INDEX) local_pruned_count++;
                            CANDIDATE_INDEX[i] = MAX_INDEX;
                            CANDIDATE_DISTANCE[i] = FLT_MAX;
                        }
                    }
                    __syncthreads();
                }

                const int admit_scan_count = keep_all_valid ? static_cast<int>(candidate_work_count) : effective_keep_count;
                for (int i = warp_id; i < admit_scan_count; i += WARPS_PER_BLOCK) {
                    INDEX_T child_id = MAX_INDEX;
                    uint32_t inserted = 0;
                    if (lane_id == 0) {
                        child_id = CANDIDATE_INDEX[i];
                        if (child_id != MAX_INDEX && CANDIDATE_DISTANCE[i] < FLT_MAX) {
                            inserted = hashtable_insert(HASH_TABLE, bitlen, child_id);
                        }
                        if (!inserted) {
                            child_id = MAX_INDEX;
                            CANDIDATE_INDEX[i] = MAX_INDEX;
                            CANDIDATE_DISTANCE[i] = FLT_MAX;
                        }
                    }
                    child_id = SHFL(child_id, 0);
                    inserted = SHFL(inserted, 0);

                    DISTANCE_T child_dist = FLT_MAX;
                    if (inserted && child_id != MAX_INDEX) {
                        const unsigned long long exact_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;
                        child_dist = warp_distance(dim, QUERY_BUFFER,
                            d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                        if (GPU_SEARCH_STATS && lane_id == 0) {
                            local_exact_cycles += static_cast<uint32_t>(clock64() - exact_start_clock);
                        }
                    }
                    if (lane_id == 0 && inserted && child_id != MAX_INDEX) {
                        CANDIDATE_DISTANCE[i] = child_dist;
                        local_exact_count++;
                    }
                }
            } else {
                // Local gate for speculative AP: each warp processes one parent tile,
                // admits only kept candidates, then exact-L2s the compact set.
                const int bytes_per_neighbor = padded_dim >> 3;
                const int tiles_per_parent = (max_degree + WARP_SIZE - 1) / WARP_SIZE;
                const int task_count = static_cast<int>(keep_expanding) * tiles_per_parent;

                for (int task = warp_id; task < task_count; task += WARPS_PER_BLOCK) {
                    const uint32_t parent_idx = static_cast<uint32_t>(task / tiles_per_parent);
                    const int tile_idx = task - static_cast<int>(parent_idx) * tiles_per_parent;
                    const uint32_t neighbor_idx = static_cast<uint32_t>(tile_idx * WARP_SIZE + lane_id);
                    const INDEX_T parent_node = PARENT_NODE_LIST[parent_idx];

                    INDEX_T child_id = MAX_INDEX;
                    float triple_x = FLT_MAX;
                    float factor_dq = 0.0f;
                    float factor_vq = 0.0f;
                    const uint8_t* parent_code_base = nullptr;
                    bool valid_candidate = false;

                    if (parent_node != MAX_INDEX && neighbor_idx < static_cast<uint32_t>(max_degree)) {
                        const float* parent_row = d_qg_data + static_cast<size_t>(parent_node) * row_offset;
                        const vidType* parent_neighbors = reinterpret_cast<const vidType*>(parent_row + neighbor_offset);
                        const float* parent_factors = parent_row + factor_offset;
                        parent_code_base = reinterpret_cast<const uint8_t*>(parent_row + code_offset);

                        child_id = parent_neighbors[neighbor_idx];
                        valid_candidate = child_id != MAX_INDEX && child_id < npoints && !hashtable_contains(HASH_TABLE, bitlen, child_id);
                        if (valid_candidate) {
                            triple_x = parent_factors[neighbor_idx];
                            factor_dq = parent_factors[max_degree + neighbor_idx];
                            factor_vq = parent_factors[2 * max_degree + neighbor_idx];
                        }
                    }

                    uint32_t active_neighbors = static_cast<uint32_t>(max_degree - tile_idx * WARP_SIZE);
                    if (active_neighbors > WARP_SIZE) active_neighbors = WARP_SIZE;
                    const int keep_count = rabitq_pruned_adaptive_keep_count_for_expander(
                        active_neighbors, adaptive_state);

                    const uint32_t raw_valid_mask = __ballot_sync(FULL_MASK, valid_candidate);
                    const uint32_t raw_valid_count = __popc(raw_valid_mask);
                    const bool keep_all_valid = keep_count >= static_cast<int>(raw_valid_count);

                    DISTANCE_T est_dist = FLT_MAX;
                    bool estimated_candidate = false;
                    if (!keep_all_valid && valid_candidate) {
                        const unsigned long long est_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;
                        est_dist = scan_one_neighbor_lane_seq_lut_gpu(
                            qf, parent_code_base, static_cast<int>(neighbor_idx),
                            triple_x, factor_dq, factor_vq, PARENT_DISTANCE_LIST[parent_idx],
                            padded_dim, bytes_per_neighbor);
                        if (GPU_SEARCH_STATS) {
                            local_est_cycles += static_cast<uint32_t>(clock64() - est_start_clock);
                        }
                        estimated_candidate = isfinite(est_dist);
                        valid_candidate = estimated_candidate;
                    }
                    int effective_keep_count = keep_count;
#if RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF
                    if (!keep_all_valid) {
                        const DISTANCE_T kth_cutoff = current_kth_cutoff;
                        const bool kth_cutoff_valid =
                            isfinite(static_cast<float>(kth_cutoff)) && kth_cutoff < FLT_MAX;
                        const uint32_t near_cutoff_mask = __ballot_sync(
                            FULL_MASK, valid_candidate && kth_cutoff_valid && est_dist <= kth_cutoff);
                        effective_keep_count = rabitq_pruned_adaptive_keep_count_for_expander(
                            active_neighbors, adaptive_state, kth_cutoff_valid, __popc(near_cutoff_mask));
                    }
#endif
                    const uint32_t scored_mask = __ballot_sync(FULL_MASK, valid_candidate);
                    const uint32_t estimated_mask = __ballot_sync(FULL_MASK, estimated_candidate);
                    const uint32_t scored_count = __popc(scored_mask);
                    const uint32_t estimated_count = __popc(estimated_mask);
                    const bool keep_lane = keep_all_valid
                        ? valid_candidate
                        : warp_keep_topk_smallest_f32(est_dist, valid_candidate, effective_keep_count);
                    uint32_t inserted = 0;
                    if (keep_lane) {
                        inserted = hashtable_insert(HASH_TABLE, bitlen, child_id);
                    }

                    const uint32_t keep_mask = __ballot_sync(FULL_MASK, keep_lane);
                    const uint32_t accepted_mask = __ballot_sync(FULL_MASK, inserted != 0);
                    const uint32_t kept_count = __popc(keep_mask);
                    const uint32_t accepted_count = __popc(accepted_mask);
#if RABITQ_ADAPTIVE_TRACE && RABITQ_ADAPTIVE_TRACE_VALID_PER_EXPANDER
                    if (lane_id == 0 && query_id < RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT &&
                        iter < RABITQ_ADAPTIVE_TRACE_ITER_LIMIT) {
                        printf("AP_VALID q=%d iter=%d task=%d parent_slot=%u tile=%d parent=%u spec_degree=%d p=%.6f keep_count=%d active_neighbors=%u valid_count=%u kept_count=%u inserted_count=%u\n",
                               query_id, iter + 1, task, parent_idx, tile_idx,
                               static_cast<uint32_t>(parent_node), adaptive_state.adaptive_spec_degree,
                               adaptive_state.adaptive_prune_ratio, keep_count, active_neighbors,
                               scored_count, kept_count, accepted_count);
                    }
#endif
                    uint32_t base_slot = 0;
                    if (lane_id == 0 && accepted_count > 0) {
                        base_slot = atomicAdd(&compact_candidate_count, accepted_count);
                    }
                    base_slot = SHFL(base_slot, 0);

                    const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                    const uint32_t lane_rank = __popc(accepted_mask & lane_mask);
                    if (inserted) {
                        const uint32_t slot = base_slot + lane_rank;
                        if (slot < candidate_collect_capacity) {
                            CANDIDATE_INDEX[slot] = child_id;
                            CANDIDATE_DISTANCE[slot] = FLT_MAX;
                        } else {
#if RABITQ_ADAPTIVE_TRACE
                            atomicAdd(&adaptive_trace_gate_overflow_count, 1u);
#endif
                            if (GPU_SEARCH_STATS) {
                                local_pruned_count++;
                            }
                        }
                    }

                    if (lane_id == 0) {
#if RABITQ_ADAPTIVE_TRACE
                        atomicAdd(&adaptive_trace_gate_scored_count, scored_count);
                        atomicAdd(&adaptive_trace_gate_kept_count, kept_count);
                        atomicAdd(&adaptive_trace_gate_inserted_count, accepted_count);
#endif
                        if (GPU_SEARCH_STATS) {
                            local_dist_est_count += estimated_count;
                            const uint32_t prune_base_count = keep_all_valid ? kept_count : estimated_count;
                            local_pruned_count += prune_base_count - accepted_count;
                        }
                    }
                }
                __syncthreads();

                if (tid == 0 && compact_candidate_count > candidate_collect_capacity) {
                    compact_candidate_count = candidate_collect_capacity;
                }
                __syncthreads();

                const uint32_t compact_count = compact_candidate_count;
                for (int i = warp_id; i < static_cast<int>(compact_count); i += WARPS_PER_BLOCK) {
                    const INDEX_T child_id = CANDIDATE_INDEX[i];
                    DISTANCE_T child_dist = FLT_MAX;
                    if (child_id != MAX_INDEX) {
                        const unsigned long long exact_start_clock = GPU_SEARCH_STATS ? clock64() : 0ull;
                        child_dist = warp_distance(dim, QUERY_BUFFER,
                            d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                        if (GPU_SEARCH_STATS && lane_id == 0) {
                            local_exact_cycles += static_cast<uint32_t>(clock64() - exact_start_clock);
                        }
                    }
                    if (lane_id == 0 && child_id != MAX_INDEX) {
                        CANDIDATE_DISTANCE[i] = child_dist;
                        local_exact_count++;
                    }
                }
                __syncthreads();
#if RABITQ_ADAPTIVE_TRACE
                if (tid == 0 && query_id < RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT &&
                    iter < RABITQ_ADAPTIVE_TRACE_ITER_LIMIT) {
                    printf("AP_GATE q=%d iter=%d spec_degree=%d p=%.6f expanders=%u gate_scored=%u gate_kept=%u gate_inserted=%u gate_overflow=%u\n",
                           query_id, iter + 1, adaptive_state.adaptive_spec_degree,
                           adaptive_state.adaptive_prune_ratio, keep_expanding,
                           adaptive_trace_gate_scored_count, adaptive_trace_gate_kept_count,
                           adaptive_trace_gate_inserted_count, adaptive_trace_gate_overflow_count);
                }
#endif
            }
        }
        __syncthreads();
        if (GPU_SEARCH_STATS && tid == 0) {
            local_loop_cycles += static_cast<uint32_t>(clock64() - iter_loop_start_clock);
        }
    }

    const uint32_t final_merge_candidate_count = candidate_buffer_size;
    dispatch_topk_by_sort_candidates_and_merge_runtime(
        ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
        result_buffer_size, final_merge_candidate_count, padded_beam_size, false);
    __syncthreads();

    if (tid == 0) {
        INDEX_T last_result = 0;
        for (int i = 0; i < K; ++i) {
            INDEX_T result = (i < beam_sz) ? (TOP_K_INDEX[i] & 0x7fffffffu) : MAX_INDEX;
            if (result != MAX_INDEX) {
                last_result = result;
            }
            d_results[query_id * K + i] = (result == MAX_INDEX) ? last_result : result;
        }
    }

    if (GPU_SEARCH_STATS) {
        uint32_t warp_exact_count = local_exact_count;
        uint32_t warp_est_count = local_dist_est_count;
        uint32_t warp_pruned_count = local_pruned_count;
        uint32_t warp_loop_cycles = local_loop_cycles;
        uint32_t warp_exact_cycles = local_exact_cycles;
        uint32_t warp_est_cycles = local_est_cycles;
        uint32_t warp_rq_stage_cycles = local_rq_stage_cycles;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warp_exact_count += __shfl_down_sync(FULL_MASK, warp_exact_count, offset);
            warp_est_count += __shfl_down_sync(FULL_MASK, warp_est_count, offset);
            warp_pruned_count += __shfl_down_sync(FULL_MASK, warp_pruned_count, offset);
            warp_loop_cycles += __shfl_down_sync(FULL_MASK, warp_loop_cycles, offset);
            warp_exact_cycles += __shfl_down_sync(FULL_MASK, warp_exact_cycles, offset);
            warp_est_cycles += __shfl_down_sync(FULL_MASK, warp_est_cycles, offset);
            warp_rq_stage_cycles += __shfl_down_sync(FULL_MASK, warp_rq_stage_cycles, offset);
        }
        if (lane_id == 0) {
            warp_stat_counts[warp_id] = warp_exact_count;
            warp_stat_counts[WARPS_PER_BLOCK + warp_id] = warp_est_count;
            warp_stat_counts[2 * WARPS_PER_BLOCK + warp_id] = warp_pruned_count;
            warp_stat_counts[3 * WARPS_PER_BLOCK + warp_id] = warp_loop_cycles;
            warp_stat_counts[4 * WARPS_PER_BLOCK + warp_id] = warp_exact_cycles;
            warp_stat_counts[5 * WARPS_PER_BLOCK + warp_id] = warp_est_cycles;
            warp_stat_counts[6 * WARPS_PER_BLOCK + warp_id] = warp_rq_stage_cycles;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t exact_count = 0;
            uint32_t est_count = 0;
            uint32_t pruned_count = 0;
            uint32_t loop_cycles = 0;
            uint32_t exact_cycles = 0;
            uint32_t est_cycles = 0;
            uint32_t rq_stage_cycles = 0;
            for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
                exact_count += warp_stat_counts[i];
                est_count += warp_stat_counts[WARPS_PER_BLOCK + i];
                pruned_count += warp_stat_counts[2 * WARPS_PER_BLOCK + i];
                loop_cycles += warp_stat_counts[3 * WARPS_PER_BLOCK + i];
                exact_cycles += warp_stat_counts[4 * WARPS_PER_BLOCK + i];
                est_cycles += warp_stat_counts[5 * WARPS_PER_BLOCK + i];
                rq_stage_cycles += warp_stat_counts[6 * WARPS_PER_BLOCK + i];
            }
            stats[GPU_SEARCH_STAT_DIST_CMPS] = exact_count;
            stats[GPU_SEARCH_STAT_DIST_EST] = est_count;
            stats[GPU_SEARCH_STAT_PRUNED] = pruned_count;
            stats[GPU_SEARCH_STAT_OUTER_EXPANSION_STEPS] = adaptive_search_expansion_steps;
            stats[GPU_SEARCH_STAT_LOOP_CYCLES] = loop_cycles;
            stats[GPU_SEARCH_STAT_EXACT_CYCLES] = exact_cycles;
            stats[GPU_SEARCH_STAT_EST_CYCLES] = est_cycles;
            stats[GPU_SEARCH_STAT_RQ_STAGE_CYCLES] = rq_stage_cycles;
            if (d_block_stats != nullptr) {
                gpu_long_t* dst = d_block_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
                for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                    dst[i] = static_cast<gpu_long_t>(stats[i]);
                }
            }
        }
    }
}
