#pragma once

#include "utils.cuh"
#include "distance.cuh"

#include <cuda_runtime.h>

#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK 128
#endif

#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH 1
#endif
#define TEAM_SIZE 8
#define INDEX_MSB_1_MASK 0x80000000u

#ifndef PATHW_BLOCK_SIZE
#define PATHW_BLOCK_SIZE 32
#endif

#define PATHW_WARPS_PER_BLOCK ((PATHW_BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE)

static __host__ inline int pathw_hash_bitlen_for_topk(int beam_sz) {
    if (beam_sz <= 128) return 10;
    return GPU_HASH_BITLEN_FOR_TOPK(beam_sz);
}

static __host__ inline uint32_t calculate_pathw_shared_mem_size(int dim, int beam_sz, int bitlen, uint32_t max_degree) {
    const size_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(SEARCH_WIDTH) * max_degree);
    const size_t result_buffer_size = static_cast<size_t>(beam_sz) + candidate_buffer_size;
    const size_t packed_dim = (static_cast<size_t>(dim) + 31) / 32;
    size_t size = 0;
    size += static_cast<size_t>(dim) * sizeof(DATA_T);
    size += result_buffer_size * sizeof(INDEX_T);
    size += result_buffer_size * sizeof(DISTANCE_T);
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);
    size += SEARCH_WIDTH * sizeof(INDEX_T);
    size += static_cast<size_t>(SEARCH_WIDTH) * packed_dim * sizeof(uint32_t);
    size += sizeof(uint32_t);  // hops
    size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);
    size += sizeof(uint32_t);  // selected expander count
    size += static_cast<size_t>(beam_sz) * sizeof(INDEX_T);     // scratch buffer for radix sort
    size += static_cast<size_t>(beam_sz) * sizeof(DISTANCE_T);
    printf("pathw shared memory size: %ld\n", size);
    return static_cast<uint32_t>(size);
}

template <typename T>
__device__ inline void pathw_build_query_sign_bits(int dim, const T* __restrict__ query, const T* __restrict__ parent,
                                                   uint32_t* __restrict__ query_sign_bits, uint32_t packed_dim) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id != 0) return;

    for (uint32_t word = lane_id; word < packed_dim; word += WARP_SIZE) {
        uint32_t bits = 0;
        const uint32_t base_dim = word * 32;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            const uint32_t d = base_dim + bit;
            if (d < static_cast<uint32_t>(dim) && query[d] > parent[d]) {
                bits |= (1u << (31 - bit));
            }
        }
        query_sign_bits[word] = bits;
    }
}

__device__ __forceinline__ bool pathw_is_in_internal_topk(INDEX_T child_id, const INDEX_T* internal_topk_list, int beam_sz) {
    for (int i = 0; i < beam_sz; ++i) {
        if (child_id == (internal_topk_list[i] & ~INDEX_MSB_1_MASK)) {
            return true;
        }
    }
    return false;
}

template <typename T, uint32_t TEAM_WIDTH>
__device__ __forceinline__ void pathw_compute_candidates_with_signbit_pruning(
    int dim,
    const INDEX_T* __restrict__ parent_list,
    const INDEX_T* __restrict__ internal_topk_list,
    INDEX_T* __restrict__ candidate_indices,
    DISTANCE_T* __restrict__ candidate_distances,
    const T* __restrict__ query,
    const T* __restrict__ data,
    const uint32_t* __restrict__ sign_bits,
    uint32_t* __restrict__ query_sign_bits,
    GraphGPU<vid_t> gg,
    uint32_t num_expanders,
    uint32_t graph_degree,
    uint32_t candidate_work_count,
    uint32_t candidate_buffer_size,
    uint32_t sign_bit_vector_size,
    uint32_t packed_dim,
    float prune_ratio,
    uint32_t runtime_internal_topk,
    uint32_t& local_exact_count,
    uint32_t& local_est_count,
    uint32_t& local_pruned_count,
    bool use_ip) {
    for (uint32_t parent_idx = 0; parent_idx < num_expanders; ++parent_idx) {
        const INDEX_T parent_slot = parent_list[parent_idx];
        const INDEX_T current_node = internal_topk_list[parent_slot] & ~INDEX_MSB_1_MASK;
        pathw_build_query_sign_bits<T>(
            dim, query, data + static_cast<size_t>(current_node) * dim,
            query_sign_bits + static_cast<size_t>(parent_idx) * packed_dim, packed_dim);
        __syncthreads();
    }

    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        const bool in_work_range = i < candidate_work_count;
        const uint32_t parent_idx = in_work_range ? i / graph_degree : 0;
        const uint32_t neighbor_idx = in_work_range ? i - parent_idx * graph_degree : 0;
        const bool active_parent = in_work_range && parent_idx < num_expanders;
        INDEX_T current_node = MAX_INDEX;
        INDEX_T child_id = MAX_INDEX;
        if (active_parent) {
            const INDEX_T parent_slot = parent_list[parent_idx];
            current_node = internal_topk_list[parent_slot] & ~INDEX_MSB_1_MASK;
            child_id = gg.N(current_node)[neighbor_idx];
        }
        bool valid_child = child_id != MAX_INDEX;
        if (valid_child && pathw_is_in_internal_topk(   // deduplication
            child_id, internal_topk_list, static_cast<int>(runtime_internal_topk))) {
            child_id = MAX_INDEX;
            valid_child = false;
        }

        DISTANCE_T score = -FLT_MAX;
        // load the precomputed sign bits of the child, and compute the matching score
        if (valid_child) {
            int direction = 0;
            const uint32_t* parent_query_sign = query_sign_bits + static_cast<size_t>(parent_idx) * packed_dim;
            const uint32_t* neighbor_sign = sign_bits + static_cast<size_t>(current_node) * sign_bit_vector_size + static_cast<size_t>(neighbor_idx) * packed_dim;
            for (uint32_t word = 0; word < packed_dim; ++word) {
                direction -= __popc(parent_query_sign[word] ^ neighbor_sign[word]);
            }
            score = static_cast<DISTANCE_T>(direction);
            local_est_count++;
        }

        candidate_indices[i] = child_id;
        candidate_distances[i] = score;
    }
    __syncthreads();

    candidate_by_bitonic_sort_inverse_runtime(
        candidate_indices, candidate_distances, candidate_buffer_size);
    __syncthreads();

    const uint32_t active_candidate_count = num_expanders * graph_degree;
    int keep_count = static_cast<int>(static_cast<float>(active_candidate_count) * prune_ratio);
    if (keep_count < 1) keep_count = 1;
    if (keep_count > static_cast<int>(candidate_buffer_size)) {
        keep_count = static_cast<int>(candidate_buffer_size);
    }

    // keep only the top candidates and mark the rest as pruned (invalid)
    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        if (i >= static_cast<uint32_t>(keep_count)) {
            if (candidate_indices[i] != MAX_INDEX) local_pruned_count++;
            candidate_distances[i] = FLT_MAX;
        }
    }
    __syncthreads();

    // compute the exact distances for the kept candidates
    const uint32_t distance_work_count = static_cast<uint32_t>(keep_count) * TEAM_WIDTH;
    const uint32_t padded_work_count = ((distance_work_count + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    for (uint32_t work = threadIdx.x; work < padded_work_count; work += blockDim.x) {
        const bool in_range = work < distance_work_count;
        const uint32_t i = in_range ? work / TEAM_WIDTH : 0;
        const INDEX_T child_id = in_range ? candidate_indices[i] : MAX_INDEX;
        const bool valid_child = in_range && child_id != MAX_INDEX;
        DISTANCE_T dist = compute_metric_vector_load_full<TEAM_WIDTH>(
            dim, data, query, valid_child ? child_id : 0, valid_child, use_ip);
        if (in_range && (work % TEAM_WIDTH) == 0) {
            candidate_distances[i] = valid_child ? dist : FLT_MAX;
            if (valid_child) local_exact_count++;
        }
    }
}

template <typename T, uint32_t TEAM_WIDTH>
__device__ __forceinline__ void pathw_compute_candidates_plain(
    int dim,
    const INDEX_T* __restrict__ parent_list,
    const INDEX_T* __restrict__ internal_topk_list,
    INDEX_T* __restrict__ candidate_indices,
    DISTANCE_T* __restrict__ candidate_distances,
    const T* __restrict__ query,
    const T* __restrict__ data,
    INDEX_T* __restrict__ visited_hash,
    int bitlen,
    GraphGPU<vid_t> gg,
    uint32_t num_expanders,
    uint32_t graph_degree,
    uint32_t candidate_work_count,
    uint32_t candidate_buffer_size,
    uint32_t& local_exact_count,
    bool use_ip) {
    // initialize the candidate buffer and filter out the visited ones
    for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
        const bool in_work_range = i < candidate_work_count;
        const uint32_t parent_idx = in_work_range ? i / graph_degree : 0;
        const uint32_t neighbor_idx = in_work_range ? i - parent_idx * graph_degree : 0;
        INDEX_T child_id = MAX_INDEX;
        if (in_work_range && parent_idx < num_expanders) {
            const INDEX_T parent_slot = parent_list[parent_idx];
            const INDEX_T current_node = internal_topk_list[parent_slot] & ~INDEX_MSB_1_MASK;
            child_id = gg.N(current_node)[neighbor_idx];
        }
        if (child_id != MAX_INDEX) {
            if (hashtable_insert(visited_hash, bitlen, child_id) == 0) {
                child_id = MAX_INDEX;
                candidate_distances[i] = FLT_MAX;
            } else {
                candidate_distances[i] = 0;
            }
        } else {
            candidate_distances[i] = FLT_MAX;
        }
        candidate_indices[i] = child_id;
    }
    __syncthreads();

    // compute the exact distances for the candidates that are not filtered out
    const uint32_t distance_work_count = candidate_buffer_size * TEAM_WIDTH;
    const uint32_t padded_work_count = ((distance_work_count + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    for (uint32_t work = threadIdx.x; work < padded_work_count; work += blockDim.x) {
        const bool in_range = work < distance_work_count;
        const uint32_t i = in_range ? work / TEAM_WIDTH : 0;
        const INDEX_T child_id = in_range ? candidate_indices[i] : MAX_INDEX;
        const bool valid_child = in_range && child_id != MAX_INDEX;
        DISTANCE_T dist = compute_metric_vector_load_full<TEAM_WIDTH>(
            dim, data, query, valid_child ? child_id : 0, valid_child, use_ip);
        if (in_range && (work % TEAM_WIDTH) == 0) {
            candidate_distances[i] = valid_child ? dist : FLT_MAX;
            if (valid_child) local_exact_count++;
        }
    }
}

template <typename T>
static __global__ GPU_LAUNCH_BOUNDS(PATHW_BLOCK_SIZE)
void PathWBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    const uint32_t* __restrict__ d_sign_bit,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_stats,
    vid_t entry_point, GraphGPU<vid_t> gg,
    float neighbor_keep_ratio, float iteration_prune_ratio, bool use_ip)
{
    const int query_id = blockIdx.x;
    if (query_id >= nq || beam_sz <= 0) return;

    const uint32_t logical_beam_size = static_cast<uint32_t>(beam_sz);
    const uint32_t padded_beam_size = gpu_effective_sort_beam_size(logical_beam_size);
    if (padded_beam_size > INTERNAL_TOPK) return;
    const uint32_t graph_degree = gg.get_max_degree();
    const uint32_t candidate_work_count = static_cast<uint32_t>(SEARCH_WIDTH) * graph_degree;
    const uint32_t candidate_buffer_size = round_up_power2_u32(candidate_work_count);
    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    const uint32_t packed_dim = (dim + 31) / 32;
    const uint32_t sign_bit_vector_size = graph_degree * packed_dim;
    const int iteration_limit = beam_sz;
    const float prune_iteration_limit = static_cast<float>(iteration_limit) * iteration_prune_ratio;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    uint32_t local_exact_count = 0;
    uint32_t local_est_count = 0;
    uint32_t local_pruned_count = 0;

    // initialize the shared memory
    extern __shared__ uint32_t smem[];
    DATA_T* query_buffer = reinterpret_cast<DATA_T*>(smem);
    INDEX_T* result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + dim);
    DISTANCE_T* result_distances_buffer = reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size);
    INDEX_T* visited_hash_buffer = reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size);
    INDEX_T* parent_list_buffer = visited_hash_buffer + hashtable_getsize(bitlen);
    uint32_t* query_sign_bits_buffer = reinterpret_cast<uint32_t*>(parent_list_buffer + SEARCH_WIDTH);
    uint32_t* hops = query_sign_bits_buffer + static_cast<size_t>(SEARCH_WIDTH) * packed_dim;
    uint32_t* stats = hops + 1;
    uint32_t* num_expanders = stats + GPU_SEARCH_STAT_COUNT;
    INDEX_T* merged_topk_index_buffer = reinterpret_cast<INDEX_T*>(num_expanders + 1);
    DISTANCE_T* merged_topk_distance_buffer = reinterpret_cast<DISTANCE_T*>(merged_topk_index_buffer + padded_beam_size);

    __shared__ uint32_t stat_scratch[3 * PATHW_WARPS_PER_BLOCK];
    if (tid == 0) {
        hops[0] = 0;
        num_expanders[0] = 0;
        for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) stats[i] = 0;
    }

    // initialize the hash table
    hashtable_init(visited_hash_buffer, bitlen);

    // load query and initialize result buffer
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = d_queries[static_cast<size_t>(query_id) * dim + i];
    }
    for (uint32_t i = tid; i < result_buffer_size; i += blockDim.x) {
        result_indices_buffer[i] = MAX_INDEX;
        result_distances_buffer[i] = FLT_MAX;
    }
    __syncthreads();

    // initialize the entry point
    const bool valid_entry = (entry_point != MAX_INDEX && entry_point < npoints);
    if (tid < WARP_SIZE) {
        const T* entry_ptr = valid_entry ? (d_data + static_cast<size_t>(entry_point) * dim) : d_data;
        DISTANCE_T entry_dist = valid_entry ? warp_distance(dim, query_buffer, entry_ptr, false) : FLT_MAX;
        if (tid == 0) {
            result_indices_buffer[padded_beam_size] = valid_entry ? entry_point : MAX_INDEX;
            result_distances_buffer[padded_beam_size] = entry_dist;
            if (valid_entry) {
                hashtable_insert(visited_hash_buffer, bitlen, entry_point);
                local_exact_count++;
            }
        }
    }
    __syncthreads();

    // start the main search loop
    for (int iter = 0; iter < iteration_limit; ++iter) {
        // hash table periodical reset
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(visited_hash_buffer, bitlen);
        }
        __syncthreads();

        // sort the candidates and select the top-k
        dispatch_topk_by_sort_candidates_and_merge_runtime(
            result_indices_buffer, result_distances_buffer,
            merged_topk_index_buffer, merged_topk_distance_buffer,
            result_buffer_size, candidate_buffer_size, padded_beam_size, iter == 0);
        __syncthreads();

        for (uint32_t i = logical_beam_size + tid; i < padded_beam_size; i += blockDim.x) {
            result_indices_buffer[i] = MAX_INDEX;
            result_distances_buffer[i] = FLT_MAX;
        }
        __syncthreads();

        // check termination condition
        if (iter + 1 == iteration_limit) {
            if (tid == 0) hops[0] = iter + 1;
            break;
        }

        // pick the next expander
        if (tid < WARP_SIZE) {
            const uint32_t picked = pick_expanders(
                SEARCH_WIDTH, parent_list_buffer, logical_beam_size, result_indices_buffer);
            if (tid == 0) {
                num_expanders[0] = picked;
            }
        }
        __syncthreads();

        // restore the hash table
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            const unsigned first_tid = ((blockDim.x <= WARP_SIZE) ? 0 : WARP_SIZE);
            hashtable_restore(
                visited_hash_buffer,
                bitlen,
                result_indices_buffer,
                logical_beam_size,
                first_tid);
        }
        __syncthreads();

        if (num_expanders[0] == 0) {
            if (tid == 0) hops[0] = iter;
            break;
        }
        if (tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS] += num_expanders[0];
        }

        if (d_sign_bit != nullptr && neighbor_keep_ratio > 0.0f &&
            neighbor_keep_ratio < 1.0f && iter < prune_iteration_limit) {
            pathw_compute_candidates_with_signbit_pruning<T, TEAM_SIZE>(
                dim, parent_list_buffer, result_indices_buffer,
                result_indices_buffer + padded_beam_size,
                result_distances_buffer + padded_beam_size,
                query_buffer, d_data, d_sign_bit, query_sign_bits_buffer,
                gg, num_expanders[0], graph_degree, candidate_work_count,
                candidate_buffer_size, sign_bit_vector_size, packed_dim,
                neighbor_keep_ratio, logical_beam_size,
                local_exact_count, local_est_count, local_pruned_count, use_ip);
        } else {
            pathw_compute_candidates_plain<T, TEAM_SIZE>(
                dim, parent_list_buffer, result_indices_buffer,
                result_indices_buffer + padded_beam_size,
                result_distances_buffer + padded_beam_size, query_buffer, d_data,
                visited_hash_buffer, bitlen, gg, num_expanders[0], graph_degree,
                candidate_work_count, candidate_buffer_size,
                local_exact_count, use_ip);
        }
        __syncthreads();
    }

    // final sorting and write back the result
    dispatch_topk_by_sort_candidates_and_merge_runtime(
        result_indices_buffer, result_distances_buffer,
        merged_topk_index_buffer, merged_topk_distance_buffer,
        result_buffer_size, candidate_buffer_size, padded_beam_size, false);
    __syncthreads();

    for (uint32_t i = tid; i < static_cast<uint32_t>(K); i += blockDim.x) {
        d_results[static_cast<size_t>(query_id) * K + i] = result_indices_buffer[i] & ~INDEX_MSB_1_MASK;
    }

    {
        uint32_t warp_exact_count = local_exact_count;
        uint32_t warp_est_count = local_est_count;
        uint32_t warp_pruned_count = local_pruned_count;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warp_exact_count += __shfl_down_sync(FULL_MASK, warp_exact_count, offset);
            warp_est_count += __shfl_down_sync(FULL_MASK, warp_est_count, offset);
            warp_pruned_count += __shfl_down_sync(FULL_MASK, warp_pruned_count, offset);
        }
        if (lane_id == 0) {
            stat_scratch[warp_id] = warp_exact_count;
            stat_scratch[PATHW_WARPS_PER_BLOCK + warp_id] = warp_est_count;
            stat_scratch[2 * PATHW_WARPS_PER_BLOCK + warp_id] = warp_pruned_count;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t exact_count = 0;
            uint32_t est_count = 0;
            uint32_t pruned_count = 0;
            for (int i = 0; i < PATHW_WARPS_PER_BLOCK; ++i) {
                exact_count += stat_scratch[i];
                est_count += stat_scratch[PATHW_WARPS_PER_BLOCK + i];
                pruned_count += stat_scratch[2 * PATHW_WARPS_PER_BLOCK + i];
            }
            stats[GPU_SEARCH_STAT_DIST_CMPS] = exact_count;
            stats[GPU_SEARCH_STAT_DIST_EST] = est_count;
            stats[GPU_SEARCH_STAT_PRUNED] = pruned_count;
            gpu_long_t* dst = d_total_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
                dst[i] = static_cast<gpu_long_t>(stats[i]);
            }
        }
    }
}
