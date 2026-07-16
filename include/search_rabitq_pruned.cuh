#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif

#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK 128
#endif

#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH 1
#endif

#define MAX_ITERATIONS (1 << 15)

#ifndef CANDIDATE_BUFFER_SIZE
#define CANDIDATE_BUFFER_SIZE 64
#endif


#ifndef GPU_RABITQ_FASTSCAN_SUBWARP_LANES
#define GPU_RABITQ_FASTSCAN_SUBWARP_LANES 32
#endif

#if !GPU_RABITQ_USE_LUT && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_USE_LUT=0 currently supports only GPU_RABITQ_FASTSCAN_SUBWARP_LANES=32"
#endif

#if GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 2 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 4 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 8 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 16 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_FASTSCAN_SUBWARP_LANES must be one of 2, 4, 8, 16, or 32"
#endif

template <typename T>
static __device__ inline void staged_copy_async_begin_or_sync(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        __pipeline_memcpy_async(dst + i, src + i, sizeof(T));
    }
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
#endif
}

static __device__ inline void staged_copy_async_finish_or_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

static __host__ __device__ inline uint32_t rabitq_pruned_candidate_buffer_size(
    uint32_t, float = 0.0f, float = 1.0f) {
    return CANDIDATE_BUFFER_SIZE;
}

static __host__ inline uint32_t calculate_pruned_shared_mem_size(int dim, int beam_sz, int result_k,
                                                                 int max_deg, int bitlen,
                                                                 float = 0.0f,
                                                                 float = 1.0f) {
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;
    const uint32_t padded_beam_size = gpu_effective_sort_beam_size(static_cast<uint32_t>(beam_sz));
    const size_t result_buffer_size = static_cast<size_t>(padded_beam_size) + CANDIDATE_BUFFER_SIZE;
    size_t size = 0;
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);       // HASH_TABLE: visited-node table
    size += result_buffer_size * sizeof(INDEX_T);              // TOP_K_INDEX + CANDIDATE_INDEX
    size += result_buffer_size * sizeof(DISTANCE_T);           // TOP_K_DISTANCE + CANDIDATE_DISTANCE
    size += SEARCH_WIDTH * sizeof(INDEX_T);                    // PARENT_LIST: beam positions/ranks selected by pick_expanders
    size += dim * sizeof(DATA_T);                              // QUERY_BUFFER: query staged in shared memory
    size += padded_dim * sizeof(float);                        // ROTATED_QUERY_BUFFER
    size += (padded_dim << 2) * sizeof(uint8_t);               // LUT_BUFFER: 4-byte lookup table per padded dimension
    size += 3 * sizeof(float);                                 // qf low/high/width
    size += sizeof(int32_t);                                   // qf sum_q
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint64_t)));                  // align staged candidate codes
    size += static_cast<size_t>(max_deg) * code_words * sizeof(uint64_t); // STAGED_CODE_BUFFER for one parent neighborhood
    size += 3 * static_cast<size_t>(max_deg) * sizeof(float);  // STAGED_FACTOR_BUFFER for one parent neighborhood
    size += sizeof(uint32_t);                                  // hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);      // search stats
    }
    size = static_cast<size_t>(align_up_uintptr(size, alignof(INDEX_T)));
    if (gpu_topk_external_merge_scratch_needed(padded_beam_size)) {
        size += padded_beam_size * sizeof(INDEX_T);            // MERGED_TOPK_INDEX scratch
        size = static_cast<size_t>(align_up_uintptr(size, alignof(DISTANCE_T)));
        size += padded_beam_size * sizeof(DISTANCE_T);         // MERGED_TOPK_DISTANCE scratch
    }
    return static_cast<uint32_t>(size);
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedPrunedBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, int max_degree, size_t,
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

    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;
    const uint32_t logical_beam_size = static_cast<uint32_t>(beam_sz);
    const uint32_t padded_beam_size = gpu_effective_sort_beam_size(logical_beam_size);

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = reinterpret_cast<INDEX_T*>(shared_memory);
    INDEX_T* TOP_K_INDEX = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX = ALL_INDEX + padded_beam_size;

    const uint32_t result_buffer_size = padded_beam_size + CANDIDATE_BUFFER_SIZE;
    DISTANCE_T* ALL_DISTANCE = reinterpret_cast<DISTANCE_T*>(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE = ALL_DISTANCE + padded_beam_size;

    INDEX_T* HASH_TABLE = reinterpret_cast<INDEX_T*>(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_LIST = HASH_TABLE + hashtable_getsize(bitlen);
    DATA_T* QUERY_BUFFER = reinterpret_cast<DATA_T*>(PARENT_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* LUT_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    char* query_factor_base = reinterpret_cast<char*>(LUT_BUFFER);
    query_factor_base += (padded_dim << 2) * sizeof(uint8_t);
    float* low_val = reinterpret_cast<float*>(query_factor_base);
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* staged_base = reinterpret_cast<char*>(sum_q + 1);
    staged_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) & ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint64_t* STAGED_CODE_BUFFER = reinterpret_cast<uint64_t*>(staged_base);
    float* STAGED_FACTOR_BUFFER = reinterpret_cast<float*>(STAGED_CODE_BUFFER + max_degree * code_words);

    uint32_t* hops = reinterpret_cast<uint32_t*>(STAGED_FACTOR_BUFFER + 3 * max_degree);
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
    __shared__ uint32_t valid_candidate_counts_by_warp[WARPS_PER_BLOCK];
    __shared__ uint32_t warp_stat_counts[3 * WARPS_PER_BLOCK];

    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = nullptr;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
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
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    __syncthreads();

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        DISTANCE_T entry_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(entry_point) * row_offset, use_ip);
        if (lane_id == 0) {
            CANDIDATE_INDEX[0] = entry_point;
            CANDIDATE_DISTANCE[0] = entry_dist;
            hashtable_insert(HASH_TABLE, bitlen, entry_point);
            local_exact_count++;
        }
    }
    __syncthreads();

    // query preparation, quantize the query and prepare the LUT for RaBitQ distance estimation
    query_prepare_lut_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    __syncthreads();

    // loop end condition: either entire topK expanded, or reach max iteration
    const float prune_iteration_limit = static_cast<float>(beam_sz) * approx_ratio;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(HASH_TABLE, bitlen);
            __syncthreads();
            hashtable_restore(HASH_TABLE, bitlen, TOP_K_INDEX, beam_sz);
        }
        __syncthreads();

        dispatch_topk_by_sort_candidates_and_merge<2>(
            ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
            result_buffer_size, CANDIDATE_BUFFER_SIZE, padded_beam_size, (iter == 0));
        __syncthreads();

        for (uint32_t i = logical_beam_size + tid; i < padded_beam_size; i += blockDim.x) {
            TOP_K_INDEX[i] = MAX_INDEX;
            TOP_K_DISTANCE[i] = FLT_MAX;
        }
        __syncthreads();

        // clear the candidate buffer before next iteration to avoid stale entries
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            CANDIDATE_INDEX[i] = MAX_INDEX;
            CANDIDATE_DISTANCE[i] = FLT_MAX;
        }
        __syncthreads();

        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, logical_beam_size, TOP_K_INDEX);
        }
        __syncthreads();

        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffffu;
        DISTANCE_T exact_dist = TOP_K_DISTANCE[PARENT_LIST[0]];
        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* current_neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
        const float* current_factors = current_row + factor_offset;

        staged_copy_async_begin_or_sync(STAGED_CODE_BUFFER, current_codes, static_cast<int>(max_degree * code_words));
        staged_copy_async_begin_or_sync(STAGED_FACTOR_BUFFER, current_factors, 3 * max_degree);

        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
        }

        const bool use_pruning = prune_ratio > 0.0f && static_cast<float>(iter + 1) <= prune_iteration_limit;
        if (!use_pruning) {
            for (int i = warp_id; i < max_degree; i += WARPS_PER_BLOCK) {
                INDEX_T child_id = MAX_INDEX;
                uint32_t inserted = 0;
                if (lane_id == 0) {
                    child_id = current_neighbors[i];
                    if (child_id != MAX_INDEX) {
                        inserted = hashtable_insert(HASH_TABLE, bitlen, child_id);
                    }
                    if (!inserted) {
                        child_id = MAX_INDEX;
                    }
                }
                child_id = SHFL(child_id, 0);
                inserted = SHFL(inserted, 0);

                DISTANCE_T child_dist = FLT_MAX;
                if (inserted && child_id != MAX_INDEX) {
                    child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
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
            staged_copy_async_finish_or_sync();
            __syncthreads();

            for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
                INDEX_T child_id = (i < max_degree) ? current_neighbors[i] : MAX_INDEX;
                if (child_id != MAX_INDEX && hashtable_contains(HASH_TABLE, bitlen, child_id)) {
                    child_id = MAX_INDEX;
                }
                CANDIDATE_INDEX[i] = child_id;
                CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0.0f;
            }
            __syncthreads();

            int keep_count = static_cast<int>(static_cast<float>(max_degree) * (1.0f - prune_ratio));
            if (keep_count < 1) keep_count = 1;
            if (keep_count > max_degree) keep_count = max_degree;

            const uint32_t valid_candidate_count = count_valid_candidates_warp_reduced(
                CANDIDATE_INDEX, CANDIDATE_DISTANCE, max_degree, MAX_INDEX,
                valid_candidate_counts_by_warp);
            const bool keep_all_valid = keep_count >= static_cast<int>(valid_candidate_count);
            if (!keep_all_valid) {
                const int bytes_per_neighbor = padded_dim >> 3;
                uint32_t warp_dist_est_count = 0;

                constexpr int lanes_per_neighbor = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
                constexpr int groups_per_warp = WARP_SIZE / lanes_per_neighbor;
                const int group_lane = lane_id & (lanes_per_neighbor - 1);
                const int warp_group = lane_id / lanes_per_neighbor;
                uint32_t group_dist_est_count = 0;

                for (int base = warp_id * groups_per_warp; base < max_degree; base += WARPS_PER_BLOCK * groups_per_warp) {
                    const int i = base + warp_group;
                    const bool valid_candidate = i < max_degree && CANDIDATE_INDEX[i] != MAX_INDEX;
                    if (valid_candidate) {
                        const int batch = i / WARP_SIZE;
                        const int neighbor_idx = i & (WARP_SIZE - 1);
                        const uint8_t* packed_codes_block =
                            reinterpret_cast<const uint8_t*>(STAGED_CODE_BUFFER) + batch * padded_dim * 4;
                        const float* triple_x = STAGED_FACTOR_BUFFER + i;
                        const float* factor_dq = STAGED_FACTOR_BUFFER + max_degree + i;
                        const float* factor_vq = STAGED_FACTOR_BUFFER + 2 * max_degree + i;

                        const DISTANCE_T est_dist = scan_one_neighbor_lanes_gpu<lanes_per_neighbor>(
                            qf, packed_codes_block, neighbor_idx, triple_x, factor_dq, factor_vq,
                            exact_dist, padded_dim, bytes_per_neighbor);
                        if (group_lane == 0) {
                            CANDIDATE_DISTANCE[i] = isfinite(est_dist) ? est_dist : FLT_MAX;
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

                candidate_by_bitonic_sort<2, 0>(
                    CANDIDATE_INDEX, CANDIDATE_DISTANCE, CANDIDATE_BUFFER_SIZE);
                __syncthreads();

                for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
                    if (i >= keep_count) {
                        if (CANDIDATE_INDEX[i] != MAX_INDEX) local_pruned_count++;
                        CANDIDATE_INDEX[i] = MAX_INDEX;
                        CANDIDATE_DISTANCE[i] = FLT_MAX;
                    }
                }
                __syncthreads();
            }

            const int admit_scan_count = keep_all_valid ? max_degree : keep_count;
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
                    child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                }
                if (lane_id == 0 && inserted && child_id != MAX_INDEX) {
                    CANDIDATE_DISTANCE[i] = child_dist;
                    local_exact_count++;
                }
            }
        }
        __syncthreads();
    }

    dispatch_topk_by_sort_candidates_and_merge<2>(
        ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
        result_buffer_size, CANDIDATE_BUFFER_SIZE, padded_beam_size, false);
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
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warp_exact_count += __shfl_down_sync(FULL_MASK, warp_exact_count, offset);
            warp_est_count += __shfl_down_sync(FULL_MASK, warp_est_count, offset);
            warp_pruned_count += __shfl_down_sync(FULL_MASK, warp_pruned_count, offset);
        }
        if (lane_id == 0) {
            warp_stat_counts[warp_id] = warp_exact_count;
            warp_stat_counts[WARPS_PER_BLOCK + warp_id] = warp_est_count;
            warp_stat_counts[2 * WARPS_PER_BLOCK + warp_id] = warp_pruned_count;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t exact_count = 0;
            uint32_t est_count = 0;
            uint32_t pruned_count = 0;
            for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
                exact_count += warp_stat_counts[i];
                est_count += warp_stat_counts[WARPS_PER_BLOCK + i];
                pruned_count += warp_stat_counts[2 * WARPS_PER_BLOCK + i];
            }
            stats[GPU_SEARCH_STAT_DIST_CMPS] = exact_count;
            stats[GPU_SEARCH_STAT_DIST_EST] = est_count;
            stats[GPU_SEARCH_STAT_PRUNED] = pruned_count;
            if (d_block_stats != nullptr) {
                gpu_long_t* dst = d_block_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
                for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                    dst[i] = static_cast<gpu_long_t>(stats[i]);
                }
            }
        }
    }
}
