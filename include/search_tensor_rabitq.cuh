#pragma once
#include "utils.cuh"
#include "hash_filter.cuh"
#include "topk.cuh"
#include <cstdint>
#include <mma.h>
#include <cuda_runtime.h>
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          10
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define HASH_TABLE_CONFIG   1
#define INTERNAL_TOPK   128
#define CANDIDATE_BUFFER_SIZE 64
#define TOPK_BUFFER_SIZE  BLOCK_SIZE

static constexpr int TENSOR_RABITQ_BATCH = 16;
static constexpr int TENSOR_RABITQ_TILE = 16;

// fill up the query matrix for tensor-core distance estimation
// each row is the same quantized query value replicated for all dimensions
static __device__ inline void prepare_tensor_query_matrix_gpu(
    const QueryFactors& qf, int padded_dim, int8_t* __restrict__ query_matrix) {
    for (int row = threadIdx.x; row < padded_dim; row += blockDim.x) {
        const int8_t value = static_cast<int8_t>(qf.quantized_query[row]);
        int8_t* row_ptr = query_matrix + row * TENSOR_RABITQ_TILE;
        for (int col = 0; col < TENSOR_RABITQ_TILE; ++col) {
            row_ptr[col] = value;
        }
    }
}

// unpack the candidate codes from uint64_t to int8_t for tensor-core distance estimation
static __device__ inline void unpack_candidate_tile_gpu(
    const INDEX_T* __restrict__ candidate_index, const uint64_t* __restrict__ staged_code_buffer,
    int code_words, int padded_dim, int batch_base, int8_t* __restrict__ dense_codes) {
    const int total = TENSOR_RABITQ_BATCH * padded_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int row = idx / padded_dim;
        const int dim_idx = idx - row * padded_dim;
        const int candidate_slot = batch_base + row;

        // skip invalid candidates
        if (candidate_slot >= CANDIDATE_BUFFER_SIZE || candidate_index[candidate_slot] == MAX_INDEX) {
            dense_codes[idx] = int8_t(0);
            continue;
        }

        const uint64_t* packed_code = staged_code_buffer + candidate_slot * code_words;
        const uint64_t bits = packed_code[dim_idx >> 6];
        dense_codes[idx] = static_cast<int8_t>((bits >> (dim_idx & 63)) & 1ULL);
    }
}

static __device__ inline void estimate_distance_tensor_tile_gpu(
    const int8_t* __restrict__ dense_codes, const int8_t* __restrict__ query_matrix,
    int padded_dim, int* __restrict__ mma_output) {
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (int k = 0; k < padded_dim; k += TENSOR_RABITQ_TILE) {
        wmma::fragment<wmma::matrix_a, TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE, int8_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE, int8_t, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, dense_codes + k, padded_dim);
        wmma::load_matrix_sync(b_frag, query_matrix + k * TENSOR_RABITQ_TILE, TENSOR_RABITQ_TILE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(mma_output, c_frag, TENSOR_RABITQ_TILE, wmma::mem_row_major);
}

static __host__ inline uint32_t calculate_shared_mem_size_tensor(int dim, int topk, int max_deg, int search_width, int bitlen) {

    auto align_up = [](size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1);};

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
    size += TENSOR_RABITQ_BATCH * padded_dim * sizeof(int8_t);        // unpacked candidate tile
    size += padded_dim * TENSOR_RABITQ_TILE * sizeof(int8_t);         // replicated query tile
    size = align_up(size, alignof(int));                               // align MMA output tile
    size += TENSOR_RABITQ_BATCH * TENSOR_RABITQ_TILE * sizeof(int);   // MMA output tile
    size += 2 * sizeof(uint32_t);                           // counter and hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size = align_up(size, alignof(gpu_long_t));             // align timing stats
    size += GPU_TIMING_COUNT * sizeof(gpu_long_t);          // timing stats
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void TensorCoreQuantizedBeamSearch(
    int K, int nq, int dim, int max_degree, size_t npoints,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_total_stats,
    gpu_long_t* d_total_timing,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset
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

    int8_t* MMA_INPUT_TILE = reinterpret_cast<int8_t*>(STAGED_FACTOR_BUFFER + 3 * max_degree);
    int8_t* QUERY_MATRIX_TILE = MMA_INPUT_TILE + TENSOR_RABITQ_BATCH * padded_dim;
    char* mma_output_base = reinterpret_cast<char*>(QUERY_MATRIX_TILE + padded_dim * TENSOR_RABITQ_TILE);
    mma_output_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(mma_output_base) + alignof(int) - 1) &
                                              ~(static_cast<uintptr_t>(alignof(int)) - 1));
    int* MMA_OUTPUT_TILE = reinterpret_cast<int*>(mma_output_base);

    uint32_t* counter = reinterpret_cast<uint32_t*>(MMA_OUTPUT_TILE + TENSOR_RABITQ_BATCH * TENSOR_RABITQ_TILE);
    uint32_t* hops = counter + 1;
    uint32_t* stats = hops + 1;
    char* timing_base = reinterpret_cast<char*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    timing_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(timing_base) + alignof(gpu_long_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(gpu_long_t)) - 1));
    gpu_long_t* timing = reinterpret_cast<gpu_long_t*>(timing_base);

    SharedBitonicTopK<BLOCK_SIZE, 1> topk(TOPK_BUFFER_SIZE, TOP_K_DISTANCE, TOP_K_INDEX);

    __shared__ QueryFactors qf;

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
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
        if (GPU_DISTANCE_TIMING) {
            for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                timing[i] = 0;
            }
        }
        for (int i = 0; i < TENSOR_RABITQ_BATCH * TENSOR_RABITQ_TILE; i++) {
            MMA_OUTPUT_TILE[i] = 0;
        }
    }

    // initialize the hash table to record visited (computed) nodes
    if (HASH_TABLE_CONFIG) hf.init();

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }

    // init candidate list with dummy values
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
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
        if (lane_id == 0) hf.insert(entry_point);
    }
    __syncthreads();

    // query preparation, quantize the query
    query_prepare_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    prepare_tensor_query_matrix_gpu(qf, static_cast<int>(padded_dim), QUERY_MATRIX_TILE);
    __syncthreads();

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
        {
            float    dist[1];
            uint32_t idx[1];
            dist[0] = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_DISTANCE[tid] : FLT_MAX;
            idx[0]  = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_INDEX[tid]    : MAX_INDEX;
            topk.merge(dist, idx);
        }

        // pick the top-N unexpanded nodes
        {
            uint32_t parent_buf[BLOCK_SIZE];
            unsigned found = topk.pick(parent_buf, SEARCH_WIDTH);
            if (warp_id == 0 && lane_id == 0) {
                keep_expanding = found;
                if (found > 0) PARENT_LIST[0] = parent_buf[0];
            }
        }
        __syncthreads();
        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = PARENT_LIST[0];
        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* current_neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
        const float* current_factors = current_row + factor_offset;

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
        __syncthreads();

        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < max_degree) ? current_neighbors[i] : MAX_INDEX;      // only in-bound threads take from the neighbor list

            if (child_id != MAX_INDEX) {    // check if visited already
                if (hf.insert(child_id) == 0) {
                    child_id = MAX_INDEX;
                    CANDIDATE_DISTANCE[i] = FLT_MAX;
                } else {
                    CANDIDATE_DISTANCE[i] = 0;
                }
            }
            CANDIDATE_INDEX[i] = child_id;
        }
        __syncthreads();

        // load the packed code and factors for the current row once so the candidate loop can
        // read it from shared memory instead of issuing many global loads
        for (int j = tid; j < max_degree * code_words; j += blockDim.x) {
            STAGED_CODE_BUFFER[j] = current_codes[j];
        }
        for (int j = tid; j < 3 * max_degree; j += blockDim.x) {
            STAGED_FACTOR_BUFFER[j] = current_factors[j];
        }
        __syncthreads();

        for (int batch_base = 0; batch_base < CANDIDATE_BUFFER_SIZE; batch_base += TENSOR_RABITQ_BATCH) {
            unpack_candidate_tile_gpu(
                CANDIDATE_INDEX,
                STAGED_CODE_BUFFER,
                static_cast<int>(code_words),
                static_cast<int>(padded_dim),
                batch_base,
                MMA_INPUT_TILE);
            __syncthreads();

            gpu_long_t est_start = 0;
            if (GPU_DISTANCE_TIMING && tid == 0) est_start = gpu_globaltimer();
            if (warp_id == 0) {
                estimate_distance_tensor_tile_gpu(
                    MMA_INPUT_TILE,
                    QUERY_MATRIX_TILE,
                    static_cast<int>(padded_dim),
                    MMA_OUTPUT_TILE);
            }
            __syncthreads();
            if (GPU_DISTANCE_TIMING && tid == 0) {
                int active_count = 0;
                for (int row = 0; row < TENSOR_RABITQ_BATCH; ++row) {
                    const int candidate_slot = batch_base + row;
                    if (candidate_slot < CANDIDATE_BUFFER_SIZE && CANDIDATE_INDEX[candidate_slot] != MAX_INDEX) {
                        active_count++;
                    }
                }
                if (active_count > 0) {
                    atomicAdd(&timing[GPU_TIMING_EST_CALLS], gpu_long_t(active_count));
                    atomicAdd(&timing[GPU_TIMING_EST_CYCLES], gpu_globaltimer() - est_start);
                }
            }

            for (int row = tid; row < TENSOR_RABITQ_BATCH; row += blockDim.x) {
                const int candidate_slot = batch_base + row;
                if (candidate_slot >= CANDIDATE_BUFFER_SIZE) continue;
                if (CANDIDATE_INDEX[candidate_slot] == MAX_INDEX) {
                    CANDIDATE_DISTANCE[candidate_slot] = FLT_MAX;
                    continue;
                }

                const float result_float =
                    2.0f * static_cast<float>(MMA_OUTPUT_TILE[row * TENSOR_RABITQ_TILE]) -
                    static_cast<float>(qf.sum_q);
                const float triple_x = STAGED_FACTOR_BUFFER[candidate_slot];
                const float factor_dq = STAGED_FACTOR_BUFFER[max_degree + candidate_slot];
                const float factor_vq = STAGED_FACTOR_BUFFER[2 * max_degree + candidate_slot];
                DISTANCE_T est_dist = triple_x + exact_dist;
                est_dist += factor_dq * qf.width * result_float;
                est_dist += factor_vq * qf.low_val;
                CANDIDATE_DISTANCE[candidate_slot] = est_dist;
            }
            if (GPU_SEARCH_STATS && tid == 0) {
                for (int row = 0; row < TENSOR_RABITQ_BATCH; ++row) {
                    const int candidate_slot = batch_base + row;
                    if (candidate_slot < CANDIDATE_BUFFER_SIZE && CANDIDATE_INDEX[candidate_slot] != MAX_INDEX) {
                        counter[0]++;
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // final merge before writeback
    {
        float    dist[1];
        uint32_t idx[1];
        dist[0] = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_DISTANCE[tid] : FLT_MAX;
        idx[0]  = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_INDEX[tid]    : MAX_INDEX;
        topk.merge(dist, idx);
    }

    // return the top K result (smallest K at tail of descending buffer)
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[TOPK_BUFFER_SIZE - K + i] & ~0x80000000u;
    }
    if (tid == 0) {
        if (GPU_SEARCH_STATS) {
            stats[GPU_SEARCH_STAT_DIST_EST] = counter[0];
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                atomicAdd(&d_total_stats[i], gpu_long_t(stats[i]));
            }
        }
        if (GPU_DISTANCE_TIMING) {
            for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                atomicAdd(&d_total_timing[i], timing[i]);
            }
        }
    }

    (void)d_qg_signs;
    (void)npoints;
    (void)code_offset;
    (void)factor_offset;
}
