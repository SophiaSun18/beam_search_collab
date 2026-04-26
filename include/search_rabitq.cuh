#pragma once
#include "utils.cuh"
#include "hash_filter.cuh"
#include "topk.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          10
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
#define TOPK_BUFFER_SIZE  BLOCK_SIZE

#ifndef GPU_RABITQ_FASTSCAN
#define GPU_RABITQ_FASTSCAN 1
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

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg,
                                                           int search_width, int bitlen) {
    auto align_up = [](size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    };
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;
    size_t size = 0;
    size += TOPK_BUFFER_SIZE * sizeof(INDEX_T);             // top k index
    size += TOPK_BUFFER_SIZE * sizeof(DISTANCE_T);          // top k distance
    size += CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T);        // candidate index
    size += CANDIDATE_BUFFER_SIZE * sizeof(DISTANCE_T);     // candidate distance
    size += HashFilter::shared_mem_bytes(BITLEN);            // hash table
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += padded_dim * sizeof(float);                     // rotated query
    size += padded_dim * sizeof(uint8_t);                   // quantized query
    size += (padded_dim << 2) * sizeof(uint8_t);            // LUT buffer
    size += 3 * sizeof(float);                              // low/high/width
    size += sizeof(int32_t);                                // sum_q
    size = align_up(size, alignof(uint64_t));               // align staged candidate codes
    size += max_deg * code_words * sizeof(uint64_t);        // staged candidate codes
    size += 3 * max_deg * sizeof(float);                    // staged candidate factors
    size += 2 * sizeof(uint32_t);                           // counter and hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += topk * sizeof(INDEX_T);                         // result pool index
    size += topk * sizeof(DISTANCE_T);                      // result pool distance
    size += 2 * sizeof(uint32_t);                           // result pool size and worst idx
    size += sizeof(DISTANCE_T);                             // result pool worst dist
#if GPU_DISTANCE_TIMING
    size = align_up(size, alignof(gpu_long_t));             // align timing stats
    size += GPU_TIMING_COUNT * sizeof(gpu_long_t);          // timing stats
#endif
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedBeamSearch(
    int K, int nq, int dim, int max_degree, size_t npoints,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_block_stats,
    gpu_long_t* d_block_timing,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset,
    bool debug_mode
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* TOP_K_INDEX = (INDEX_T*)shared_memory;
    DISTANCE_T* TOP_K_DISTANCE = (DISTANCE_T*)(TOP_K_INDEX + TOPK_BUFFER_SIZE);
    INDEX_T* CANDIDATE_INDEX = (INDEX_T*)(TOP_K_DISTANCE + TOPK_BUFFER_SIZE);
    DISTANCE_T* CANDIDATE_DISTANCE = (DISTANCE_T*)(CANDIDATE_INDEX + CANDIDATE_BUFFER_SIZE);
    INDEX_T* HASH_TABLE = (INDEX_T*)(CANDIDATE_DISTANCE + CANDIDATE_BUFFER_SIZE);
    HashFilter hf(HASH_TABLE, BITLEN);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + HashFilter::shared_mem_bytes(BITLEN) / sizeof(INDEX_T));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* QUANTIZED_QUERY_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    uint8_t* LUT_BUFFER = QUANTIZED_QUERY_BUFFER + padded_dim;
    float* low_val = reinterpret_cast<float*>(LUT_BUFFER + (padded_dim << 2));
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* staged_base = reinterpret_cast<char*>(sum_q + 1);
    staged_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint64_t* STAGED_CODE_BUFFER = reinterpret_cast<uint64_t*>(staged_base);
    float* STAGED_FACTOR_BUFFER = reinterpret_cast<float*>(STAGED_CODE_BUFFER + max_degree * code_words);

    uint32_t* counter = reinterpret_cast<uint32_t*>(STAGED_FACTOR_BUFFER + 3 * max_degree);
    uint32_t* hops = counter + 1;
    uint32_t* stats = hops + 1;
    INDEX_T* RESULT_POOL_INDEX = reinterpret_cast<INDEX_T*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    DISTANCE_T* RESULT_POOL_DISTANCE = reinterpret_cast<DISTANCE_T*>(RESULT_POOL_INDEX + K);
    uint32_t* result_pool_size = reinterpret_cast<uint32_t*>(RESULT_POOL_DISTANCE + K);
    uint32_t* result_pool_worst_idx = result_pool_size + 1;
    DISTANCE_T* result_pool_worst_dist = reinterpret_cast<DISTANCE_T*>(result_pool_worst_idx + 1);
    gpu_long_t* timing = nullptr;
#if GPU_DISTANCE_TIMING
    char* timing_base = reinterpret_cast<char*>(result_pool_worst_dist + 1);
    timing_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(timing_base) + alignof(gpu_long_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(gpu_long_t)) - 1));
    timing = reinterpret_cast<gpu_long_t*>(timing_base);
#endif

    SharedBitonicTopK<BLOCK_SIZE, 1> topk(TOPK_BUFFER_SIZE, TOP_K_DISTANCE, TOP_K_INDEX);

    __shared__ QueryFactors qf;

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t current_expanded;
    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = QUANTIZED_QUERY_BUFFER;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        counter[0] = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
        result_pool_size[0] = 0;
        result_pool_worst_idx[0] = 0;
        result_pool_worst_dist[0] = -FLT_MAX;
        if (GPU_DISTANCE_TIMING) {
            for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                timing[i] = 0;
            }
        }
    }

    // initialize the hash table to record expanded nodes
    if (HASH_TABLE_CONFIG) hf.init();

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }

    // load the top k list and candidate list with dummy initial values
    for (int i = tid; i < TOPK_BUFFER_SIZE; i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        RESULT_POOL_INDEX[i] = MAX_INDEX;
        RESULT_POOL_DISTANCE[i] = FLT_MAX;
    }

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        CANDIDATE_INDEX[0] = entry_point;
        CANDIDATE_DISTANCE[0] = warp_l2_distance(dim, query, d_qg_data + static_cast<size_t>(entry_point) * row_offset);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
    }
    __syncthreads();

    // query preparation, quantize the query
    gpu_long_t query_prep_start = 0;
    if (GPU_DISTANCE_TIMING && tid == 0) query_prep_start = gpu_globaltimer();
    query_prepare_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    if constexpr (GPU_RABITQ_FASTSCAN) {
        pack_lut_gpu(static_cast<int>(padded_dim), qf.quantized_query, qf.lut);
    }
    __syncthreads();
    if (GPU_DISTANCE_TIMING && tid == 0) {
        atomicAdd(&timing[GPU_TIMING_QUERY_PREP_CYCLES], gpu_globaltimer() - query_prep_start);
    }

    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hf.init();
            }
            __syncthreads();
        }

        // merge candidates into topk buffer
        gpu_long_t sort_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) sort_start = gpu_globaltimer();
        {
            float    dist[1];
            uint32_t idx[1];
            dist[0] = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_DISTANCE[tid] : FLT_MAX;
            idx[0]  = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_INDEX[tid]    : MAX_INDEX;
            topk.merge(dist, idx);
        }
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_SORT_CYCLES], gpu_globaltimer() - sort_start);
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            printf("[DEBUG] iteration %d\n", iter + 1);
            printf("  beam state:\n");
            for (int i = 0; i < TOPK_BUFFER_SIZE; ++i) {
                const INDEX_T raw = TOP_K_INDEX[i];
                if (raw == MAX_INDEX) continue;
                const bool expanded = (raw & 0x80000000u) != 0;
                printf("    idx=%d, node=%u, dist=%.6f, expanded=%s\n",
                       i,
                       raw & 0x7fffffff,
                       TOP_K_DISTANCE[i],
                       expanded ? "true" : "false");
            }
            printf("  candidate buffer:\n");
            for (int i = 0; i < CANDIDATE_BUFFER_SIZE; ++i) {
                printf("    idx=%d, node=%u, dist=%.6f\n",
                       i,
                       CANDIDATE_INDEX[i] & 0x7fffffff,
                       CANDIDATE_DISTANCE[i]);
            }
        }

        // pick the top-N unexpanded nodes
        gpu_long_t pick_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) pick_start = gpu_globaltimer();
        {
            uint32_t parent_buf[BLOCK_SIZE];
            unsigned found = topk.pick(parent_buf, SEARCH_WIDTH);
            if (warp_id == 0 && lane_id == 0) {
                keep_expanding = found;
                if (found > 0) PARENT_LIST[0] = parent_buf[0];
            }
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_PICK_CYCLES], gpu_globaltimer() - pick_start);
        }

        if (!keep_expanding) {
            if (debug_mode && query_id == 0 && tid == 0) {
                printf("  expanding node: none\n");
            }
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;

        if (tid == 0) {
            current_expanded = hf.insert(current_node);
        }
        __syncthreads();
        if (!current_expanded) {
            continue;
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            const INDEX_T parent_slot = PARENT_LIST[0];
            const DISTANCE_T parent_dist =
                (parent_slot < TOPK_BUFFER_SIZE) ? TOP_K_DISTANCE[parent_slot] : FLT_MAX;
            printf("  expanding node: %u\n", current_node);
            printf("  expanding slot: %u, dist=%.6f\n", parent_slot, parent_dist);
        }
        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* current_neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
        const float* current_factors = current_row + factor_offset;

        gpu_long_t stage_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) stage_start = gpu_globaltimer();
        // initialize the async copy of packed code and factors as long as the current expander is decided
        staged_copy_async_begin_or_sync(
            STAGED_CODE_BUFFER,
            current_codes,
            static_cast<int>(max_degree * code_words));
        staged_copy_async_begin_or_sync(
            STAGED_FACTOR_BUFFER,
            current_factors,
            3 * max_degree);

        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        DISTANCE_T exact_dist = warp_l2_distance(dim, query, current_row);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
            stats[GPU_SEARCH_STAT_DIST_CMPS]++;
        }
        gpu_long_t result_pool_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) result_pool_start = gpu_globaltimer();
        if (tid == 0) {
            result_pool_push_unique_unsorted(
                RESULT_POOL_INDEX,
                RESULT_POOL_DISTANCE,
                result_pool_size,
                result_pool_worst_idx,
                result_pool_worst_dist,
                K,
                current_node,
                exact_dist);
        }
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_RESULT_POOL_CYCLES], gpu_globaltimer() - result_pool_start);
        }
        gpu_long_t visited_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) visited_start = gpu_globaltimer();
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < max_degree) ? current_neighbors[i] : MAX_INDEX;
            if (child_id != MAX_INDEX && hf.contains(child_id)) {
                child_id = MAX_INDEX;
            }
            CANDIDATE_INDEX[i] = child_id;
            CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0;
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_VISITED_CYCLES], gpu_globaltimer() - visited_start);
        }

        // wait for the async copy to finish before move on to computation
        staged_copy_async_finish_or_sync();
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_STAGE_CYCLES], gpu_globaltimer() - stage_start);
        }

        // all warps work together to compute candidate distance
        gpu_long_t est_region_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) est_region_start = gpu_globaltimer();
        if constexpr (GPU_RABITQ_FASTSCAN) {
            // each warp handle a batch of neighbors
            const int num_batches = (max_degree + WARP_SIZE - 1) / WARP_SIZE;

            for (int batch = warp_id; batch < num_batches; batch += WARPS_PER_BLOCK) {
                const int batch_begin = batch * WARP_SIZE;
                const int active_neighbors = min(WARP_SIZE, max_degree - batch_begin);
                const uint8_t* packed_codes_block = reinterpret_cast<const uint8_t*>(STAGED_CODE_BUFFER) + batch * static_cast<int>(padded_dim) * 4;
                const float* triple_x = STAGED_FACTOR_BUFFER + batch_begin;
                const float* factor_dq = STAGED_FACTOR_BUFFER + max_degree + batch_begin;
                const float* factor_vq = STAGED_FACTOR_BUFFER + 2 * max_degree + batch_begin;

                gpu_long_t est_start = 0;
                if (GPU_DISTANCE_TIMING && lane_id == 0) est_start = gpu_globaltimer();
                scan_all_neighbors_32_gpu(qf, packed_codes_block, triple_x, factor_dq, factor_vq,
                    exact_dist, static_cast<int>(padded_dim), CANDIDATE_DISTANCE + batch_begin, active_neighbors);
                if (lane_id < active_neighbors) {
                    const int candidate_idx = batch_begin + lane_id;
                    const bool invalid_candidate = (CANDIDATE_INDEX[candidate_idx] == MAX_INDEX);
                    const bool invalid_factor = (triple_x[lane_id] == FLT_MAX);
                    const bool invalid_distance = !isfinite(CANDIDATE_DISTANCE[candidate_idx]);
                    if (invalid_candidate || invalid_factor || invalid_distance) {
                        CANDIDATE_DISTANCE[candidate_idx] = FLT_MAX;
                    }
                }
                if (GPU_DISTANCE_TIMING && lane_id == 0) {
                    atomicAdd(&timing[GPU_TIMING_EST_CALLS], gpu_long_t(active_neighbors));
                    atomicAdd(&timing[GPU_TIMING_EST_CYCLES], gpu_globaltimer() - est_start);
                }
                if (lane_id == 0) {
                    uint32_t active_count = 0;
                    for (int i = 0; i < active_neighbors; ++i) {
                        if (CANDIDATE_INDEX[batch_begin + i] != MAX_INDEX) {
                            active_count++;
                        }
                    }
                    atomicAdd(&counter[0], active_count);
                    if (GPU_SEARCH_STATS) {
                        atomicAdd(&stats[GPU_SEARCH_STAT_DIST_EST], active_count);
                    }
                }
            }
        } else {
            for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
                // visited candidates are marked as invalid
                if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

                // if current node is new in hash table, compute and store its distance
                const uint64_t* packed_code = STAGED_CODE_BUFFER + i * code_words;
                const float* factors = STAGED_FACTOR_BUFFER;
                float triple_x = 0.0f;
                float factor_dq = 0.0f;
                float factor_vq = 0.0f;
                if (lane_id == 0) {
                    triple_x = factors[i];
                    factor_dq = factors[max_degree + i];
                    factor_vq = factors[2 * max_degree + i];
                }
                triple_x = __shfl_sync(FULL_MASK, triple_x, 0);
                factor_dq = __shfl_sync(FULL_MASK, factor_dq, 0);
                factor_vq = __shfl_sync(FULL_MASK, factor_vq, 0);

                gpu_long_t est_start = 0;
                if (GPU_DISTANCE_TIMING && lane_id == 0) est_start = gpu_globaltimer();
                DISTANCE_T est_dist = estimated_distance_gpu(
                    qf, packed_code, triple_x, factor_dq, factor_vq, exact_dist, padded_dim);
                if (lane_id == 0) {
                    atomicAdd(&counter[0], 1);
                    if (GPU_SEARCH_STATS) {
                        atomicAdd(&stats[GPU_SEARCH_STAT_DIST_EST], 1u);
                    }
                    if (GPU_DISTANCE_TIMING) {
                        atomicAdd(&timing[GPU_TIMING_EST_CALLS], gpu_long_t(1));
                        atomicAdd(&timing[GPU_TIMING_EST_CYCLES], gpu_globaltimer() - est_start);
                    }
                    CANDIDATE_DISTANCE[i] = est_dist;
                }
            }
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            int active_count = 0;
            for (int i = 0; i < CANDIDATE_BUFFER_SIZE; ++i) {
                if (CANDIDATE_INDEX[i] != MAX_INDEX) {
                    active_count++;
                }
            }
            if (active_count > 0) {
                atomicAdd(&timing[GPU_TIMING_EST_REGION_CANDIDATES], gpu_long_t(active_count));
                atomicAdd(&timing[GPU_TIMING_EST_REGION_CYCLES], gpu_globaltimer() - est_region_start);
            }
        }
    }

    // final merge before writeback
    gpu_long_t final_sort_start = 0;
    if (GPU_DISTANCE_TIMING && tid == 0) final_sort_start = gpu_globaltimer();
    {
        float    dist[1];
        uint32_t idx[1];
        dist[0] = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_DISTANCE[tid] : FLT_MAX;
        idx[0]  = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_INDEX[tid]    : MAX_INDEX;
        topk.merge(dist, idx);
    }
    if (GPU_DISTANCE_TIMING && tid == 0) {
        atomicAdd(&timing[GPU_TIMING_FINAL_SORT_CYCLES], gpu_globaltimer() - final_sort_start);
    }

    if (tid == 0) {

        if (debug_mode && query_id == 0) {
            printf("[DEBUG] final result pool\n");
            for (uint32_t i = 0; i < result_pool_size[0]; ++i) {
                printf("  idx=%u, node=%u, dist=%.6f\n",
                       i,
                       RESULT_POOL_INDEX[i],
                       RESULT_POOL_DISTANCE[i]);
            }
        }

        const uint32_t pool_size = result_pool_size[0];
        for (int i = 0; i < K; ++i) {
            if (i < static_cast<int>(pool_size)) {
                d_results[query_id * K + i] = RESULT_POOL_INDEX[i];
            } else {
                // if there's not enough value in result pool, fill the rest with other top K
                d_results[query_id * K + i] = TOP_K_INDEX[TOPK_BUFFER_SIZE - K + i] & 0x7fffffff;
            }
        }
        if (GPU_SEARCH_STATS) {
            stats[GPU_SEARCH_STAT_DIST_EST] = counter[0];
            if (d_block_stats != nullptr) {
                gpu_long_t* dst = d_block_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
                for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                    dst[i] = static_cast<gpu_long_t>(stats[i]);
                }
            }
        }
        if (GPU_DISTANCE_TIMING) {
            if (d_block_timing != nullptr) {
                gpu_long_t* dst = d_block_timing + static_cast<size_t>(query_id) * GPU_TIMING_COUNT;
                for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                    dst[i] = timing[i];
                }
            }
        }
    }

    (void)d_qg_signs;
    (void)npoints;
    (void)code_offset;
    (void)factor_offset;
}
