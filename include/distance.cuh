#pragma once

#include "utils.cuh"
#define MAX_DIM 128

__device__ __forceinline__ uint32_t gpu_fast_rand_u32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_full(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    constexpr unsigned reg_nelem = (MAX_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff[reg_nelem];
    unsigned int full_mask = 0xffffffff;
    DISTANCE_T norm2 = 0;
    if (valid_child) {
        // Load
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;
            dl_buff[e] = reinterpret_cast<const float4*>(child_data_ptr)[k / 4];
        }
        // Compute
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;
            DISTANCE_T d = query_ptr[k];
            norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
            d = query_ptr[k + 1];
            norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
            d = query_ptr[k + 2];
            norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
            d = query_ptr[k + 3];
            norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

        }
    }
    unsigned team_mask = __ballot_sync(0xffffffff, valid_child);
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE, uint32_t RATIO = 2>
__device__ DISTANCE_T compute_similarity_vector_load_partial(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child,
    DISTANCE_T threshold = 9999999999
) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;

    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte, vlen == 4
    constexpr unsigned NELEM = ((MAX_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen) + RATIO - 1) / RATIO * (RATIO - 1);
    constexpr unsigned reg_nelem = NELEM > 1U ? NELEM : 1U;
    float4 dl_buff[reg_nelem];
    unsigned int full_mask = 0xffffffff;
    DISTANCE_T norm2 = 0;
    if (valid_child) {
        // Load
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;
            dl_buff[e] = reinterpret_cast<const float4*>(child_data_ptr)[k / 4];
        }
        
        // Compute
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= dim) break;

            DISTANCE_T d = query_ptr[k];
            norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
            d = query_ptr[k + 1];
            norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
            d = query_ptr[k + 2];
            norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
            d = query_ptr[k + 3];
            norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

        }
    }
    unsigned team_mask = __ballot_sync(0xffffffff, valid_child);
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team(
    uint32_t dim,
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    unsigned team_id  = threadIdx.x / TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    constexpr unsigned reg_nelem = (MAX_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff[reg_nelem];
    DISTANCE_T norm2 = 0;
    // Load
    #pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
        const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dim) break;
        dl_buff[e] = reinterpret_cast<float4*>(child_data_ptr)[k / 4];
    }
    // Compute
    #pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
        const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dim) break;
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
        d = query_ptr[k + 2];
        norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
        d = query_ptr[k + 3];
        norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);
    }

    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team_partial_float4(
    uint32_t dim,
    const DATA_T* d_dataset_ptr,
    const DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask,
    uint32_t e) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    float4 dl_buff;
    DISTANCE_T norm2 = 0;
    
    const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
    if (k < dim) {
        // Load
        dl_buff = reinterpret_cast<const float4*>(child_data_ptr)[k / 4];
        // Compute
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff.x) * (d - dl_buff.x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff.y) * (d - dl_buff.y);
        d = query_ptr[k + 2];
        norm2 += (d - dl_buff.z) * (d - dl_buff.z);
        d = query_ptr[k + 3];
        norm2 += (d - dl_buff.w) * (d - dl_buff.w);
    }
    // Reduce
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }
    return norm2;
}

template<uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team_partial_float2(
    uint32_t dim,
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask,
    uint32_t e) {
    auto child_data_ptr  = d_dataset_ptr + child_id * dim;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    unsigned team_id  = threadIdx.x / TEAM_SIZE;
    float2 dl_buff;
    DISTANCE_T norm2 = 0;
    const uint32_t k = (lane_id + (TEAM_SIZE * e)) * 2;
    if (k < dim) {
        // Load
        dl_buff = reinterpret_cast<float2*>(child_data_ptr)[k / 2];
        // Compute
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff.x) * (d - dl_buff.x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff.y) * (d - dl_buff.y);
    }
    // Reduce
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }
    return norm2;
}

// map top10 nodes and fetch neighbors and calculate distance
template<uint32_t TEAM_SIZE>
__device__ void compute_distance_to_maped_top10_nodes(
    uint32_t dim,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    const DATA_T* query_ptr,
    const DATA_T* dataset_ptr,
    INDEX_T* graph_ptr,
    INDEX_T* top10_ptr,
    INDEX_T* seed_map_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t DATASET_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t SEED_CONFIG,
    uint32_t HASH_TABLE_CONFIG,
    uint32_t SEED_TOPK_SIZE) {
    const INDEX_T invalid_index        = MAX_INDEX;
    // Load top N nodes
    INDEX_T top1 = top10_ptr[blockIdx.y * SEED_TOPK_SIZE];
    // Map top1 nodes to dataset
    INDEX_T parent_id = seed_map_ptr[top1];
    // fetch neighbors and insert to hash table
    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        INDEX_T child_id = graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
        if (HASH_TABLE_CONFIG) {
            if (child_id != invalid_index) {
                if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0) {
                    child_id = invalid_index;
                    candidate_distances_ptr[i] = FLT_MAX;
                } else {
                    candidate_distances_ptr[i] = 0;
                }
            }
            candidate_indices_ptr[i] = child_id;
        }
    }
    __syncthreads();

    // calculate distance
    for (uint32_t tid = threadIdx.x; tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE; tid += blockDim.x) {
        const auto i       = tid / TEAM_SIZE;
        const bool valid_i = (i < (CANDIDATE_BUFFER_SIZE));
        INDEX_T child_id   = invalid_index;
        if (valid_i) { child_id = candidate_indices_ptr[i]; }
        DISTANCE_T norm2 = compute_similarity_vector_load_full<TEAM_SIZE>(dim, dataset_ptr, 
                                    query_ptr, child_id, child_id != invalid_index);
        //if (tid % TEAM_SIZE == 0 && child_id != invalid_index) atomicAdd(counter, 1);
        // Store the distance
        const unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (valid_i && lane_id == 0) {
            if (child_id != invalid_index) {
                candidate_distances_ptr[i] = norm2;
            } else {
                candidate_distances_ptr[i] = FLT_MAX;
            }
        }
    }    
}

#define NUM_DIST 1

template <uint32_t TEAM_SIZE>
__device__ void compute_distance_to_random_nodes(
    uint32_t dim,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    const DATA_T* query_ptr,
    const DATA_T* dataset_ptr,
    uint32_t DATASET_SIZE,
    uint32_t max_i, // number of starting nodes to pick
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t HASH_TABLE_CONFIG) {
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }
  for (uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += blockDim.x / TEAM_SIZE) {
    const bool valid_i = (i < max_i);
    INDEX_T best_index_team_local;
    DISTANCE_T best_norm2_team_local = FLT_MAX;
    for (uint32_t j = 0; j < NUM_DIST; j++) {
      // Select a node randomly and compute the distance to it
      INDEX_T seed_index;
      if (valid_i) {
        uint32_t gid = threadIdx.x + (blockDim.x * (i + (max_i * j)));
        uint32_t seed = gid ^ static_cast<uint32_t>(clock64());
        seed = gpu_fast_rand_u32(seed);
        seed_index = seed % DATASET_SIZE;
      }
      auto norm2 = compute_similarity_vector_load_full<TEAM_SIZE>(dim, dataset_ptr,
          query_ptr, seed_index, valid_i);
      //if (valid_i) atomicAdd(counter, 1);
      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }
    const unsigned lane_id = threadIdx.x % TEAM_SIZE;
    if (valid_i && lane_id == 0) {
      if (HASH_TABLE_CONFIG) {
        if (hashtable_insert(visited_hash_ptr, BITLEN, best_index_team_local)) {
          candidate_distances_ptr[i] = best_norm2_team_local;
          candidate_indices_ptr[i]   = best_index_team_local;
        } else {
          candidate_distances_ptr[i] = FLT_MAX;
          candidate_indices_ptr[i]   = MAX_INDEX;
        }
      } else {
        candidate_distances_ptr[i] = best_norm2_team_local;
        candidate_indices_ptr[i]   = best_index_team_local;
      }
    }
  }
}

template <uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
__device__ void compute_distance_to_child_nodes(
    uint32_t dim,
    INDEX_T* parent_buffer,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    INDEX_T* internal_topk_list,
    const DATA_T* query_ptr,
    const DATA_T* d_dataset_ptr,
    INDEX_T* d_graph_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t SEARCH_WIDTH,
    uint32_t CANDIDATE_BUFFER_SIZE,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t* counter,
    uint32_t THRESHOLD_CONFIG,
    uint32_t HASH_TABLE_CONFIG) {

  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  const INDEX_T invalid_index        = MAX_INDEX;
  // Read child indices of parents from knn graph and check if the distance computaiton is necessary.
  for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
    const INDEX_T smem_parent_id = parent_buffer[0];
    const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
    INDEX_T child_id = d_graph_ptr[(int64_t(GRAPH_DEGREE) * parent_id) + i];
    //if (blockIdx.x == 0 && threadIdx.x < 4) printf("block %d, thread %d expanding node %d 's %d th child node %d \n", blockIdx.x, i, parent_id, i, child_id);
    if (!HASH_TABLE_CONFIG) {
      // check with internal topk
      for (int32_t internal_topk_ptr = 0; internal_topk_ptr < INTERNAL_TOPK; internal_topk_ptr++) {
        INDEX_T internal_topk_id = internal_topk_list[internal_topk_ptr] & ~index_msb_1_mask;
        if (child_id == internal_topk_id) {
          child_id = invalid_index;
          break;
        }
      }
      if (child_id != invalid_index) {
        candidate_distances_ptr[i] = 0;
      }
      candidate_indices_ptr[i] = child_id;
    } else {
      if (child_id != invalid_index) {
        if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0) {
          child_id = invalid_index;
          candidate_distances_ptr[i] = FLT_MAX;
        } else {
          candidate_distances_ptr[i] = 0;
        }
      }
      candidate_indices_ptr[i] = child_id;
    }
  }
  // Compute half of the distance to child nodes
  __syncthreads();
  if (THRESHOLD_CONFIG == 0) {
    std::uint32_t max_i = GRAPH_DEGREE * SEARCH_WIDTH;
    for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x) {
      const auto i       = tid / TEAM_SIZE;
      const bool valid_i = (i < (GRAPH_DEGREE * SEARCH_WIDTH));
      INDEX_T child_id   = invalid_index;
      if (valid_i) { child_id = candidate_indices_ptr[i]; }
      // each team of threads do a distance computation
      DISTANCE_T norm2 = compute_similarity_vector_load_full<TEAM_SIZE>(dim, 
                         d_dataset_ptr, query_ptr, child_id, child_id != invalid_index);
      if (tid % TEAM_SIZE == 0 && child_id != invalid_index) atomicAdd(counter, 1);
      //if (blockIdx.x == 0 && tid % TEAM_SIZE == 0) printf("query 0: tid %d, team %d: child id %d 's distance %f\n", tid, i, child_id, norm2);
      // Store the distance
      const unsigned lane_id = threadIdx.x % TEAM_SIZE;
      if (valid_i && lane_id == 0) {
        if (child_id != invalid_index) {
          candidate_distances_ptr[i] = norm2;
        } else {
          candidate_distances_ptr[i] = FLT_MAX;
        }
      }
    }
  } else {
    DISTANCE_T threshold = candidate_distances_ptr[-1];
    std::uint32_t max_i = GRAPH_DEGREE * SEARCH_WIDTH;
    for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x) {
      const auto i       = tid / TEAM_SIZE;
      const bool valid_i = (i < (GRAPH_DEGREE * SEARCH_WIDTH));
      INDEX_T child_id   = invalid_index;
      if (valid_i) { child_id = candidate_indices_ptr[i]; }
      DISTANCE_T norm2 = compute_similarity_vector_load_partial<TEAM_SIZE, 4/*compute 75%*/>
        (dim, d_dataset_ptr, query_ptr, child_id, child_id != invalid_index);
      // Store the distance
      const unsigned lane_id = threadIdx.x % TEAM_SIZE;
      if (valid_i && lane_id == 0) {
        if (child_id != invalid_index && norm2 < threshold) {
          candidate_distances_ptr[i] = norm2;
        } else {
          candidate_distances_ptr[i] = FLT_MAX;
        }
      }
    }

    // atomic add
    __syncthreads();
    const uint32_t _TEAM_SIZE = TEAM_SIZE;
    uint8_t _lane_id = threadIdx.x % _TEAM_SIZE;
    uint8_t _team_id = threadIdx.x / _TEAM_SIZE;
    unsigned _team_mask = 0xff << (_team_id * _TEAM_SIZE);
    uint32_t _offset_e_maxi = ((dim + _TEAM_SIZE * 4 - 1) / (_TEAM_SIZE * 4) + 4 -1) / 4;
    uint32_t _max_i = _offset_e_maxi * CANDIDATE_BUFFER_SIZE;
    while(true) {   
      uint32_t i = 0;
      if(_lane_id == 0) {
        while(true) {
          i =  atomicAdd(counter, 1);
          if(i >= _max_i) {
            break;
          }
          if(candidate_distances_ptr[i % CANDIDATE_BUFFER_SIZE] != FLT_MAX) {
            break;
          }
        }
      }
      i = __shfl_sync(_team_mask, i, _team_id * _TEAM_SIZE);
      if(i >= _max_i) {
        break;
      }
      uint32_t vector_offset = i / CANDIDATE_BUFFER_SIZE;
      i = i % CANDIDATE_BUFFER_SIZE;
      INDEX_T child_id = candidate_indices_ptr[i];
      // Distance calculation
      uint32_t e = _offset_e_maxi * (4 - 1);
      DISTANCE_T partial_norm2 = compute_similarity_vector_load_by_team_partial_float4<_TEAM_SIZE>(dim,
          d_dataset_ptr, query_ptr, child_id, _team_mask, e + vector_offset); // vector offset
      if (_lane_id == 0) {
        DISTANCE_T saved_norm2 = candidate_distances_ptr[i];
        if(saved_norm2 + partial_norm2 > threshold) {
          candidate_distances_ptr[i] = FLT_MAX;
        } else {
          candidate_distances_ptr[i] = saved_norm2 + partial_norm2;
        }
      }  
    }
  }
}

// template <uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
// __device__ inline void compute_distance_to_child_nodes_team_with_direction(
//     uint32_t dim,
//     INDEX_T* parent_buffer,
//     INDEX_T* internal_topk_list,
//     INDEX_T* candidate_indices_ptr,
//     DISTANCE_T* candidate_distances_ptr,
//     const DATA_T* query_ptr,
//     const DATA_T* d_dataset_ptr,
//     INDEX_T* d_graph_ptr,
//     uint32_t* d_sign_bit_ptr,
//     uint32_t GRAPH_DEGREE,
//     uint32_t CANDIDATE_BUFFER_SIZE,
//     uint32_t PRUNE_CONFIG,
//     float PRUNE_RATIO) {
//     constexpr INDEX_T index_msb_1_mask = 0x80000000;
//     const INDEX_T invalid_index        = MAX_INDEX;
//     const INDEX_T smem_parent_id = parent_buffer[0];
//     const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
//     int8_t lane_id = threadIdx.x % 32;
//     uint32_t sign_bit_vector_size = dim * GRAPH_DEGREE / 32; // 256 for sift-128
//     constexpr uint32_t packed_vector_dim_size = (MAX_DIM + 31) / 32;
//     int32_t query_sign_bit[packed_vector_dim_size];

//     uint32_t VECTOR_DIM_32 = (dim + 31) & ~31;
//     for (uint32_t i = threadIdx.x; i < VECTOR_DIM_32; i += blockDim.x) {
//         const bool valid_i = i < dim;
//         uint32_t sign_bit = query_ptr[i] > d_dataset_ptr[parent_id * dim + i];

//         sign_bit <<= (31 - lane_id);
//         unsigned active_threads = __ballot_sync(0xffffffff, valid_i);
//         __syncwarp(active_threads);

//         for (int offset = 16; offset > 0; offset /= 2) {
//             sign_bit |= __shfl_xor_sync(0xffffffff, sign_bit, offset);
//         }
//         query_sign_bit[i / 32] = sign_bit;
//         __syncwarp(active_threads);
//     }

//     for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
//         // const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
//         INDEX_T child_id = d_graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
//         // check with internal topk
//         for (int32_t internal_topk_ptr = 0; internal_topk_ptr < INTERNAL_TOPK; internal_topk_ptr++) {
//             INDEX_T internal_topk_id = internal_topk_list[internal_topk_ptr] & ~index_msb_1_mask;
//             if(child_id == internal_topk_id) {
//                 child_id = invalid_index;
//                 break;
//             }
//         }
//         if (child_id != invalid_index) {
//             if(PRUNE_CONFIG == RANDOM_PRUNE) {
//                 uint32_t gid = threadIdx.x + blockDim.x * i;
//                 curandState state;
//                 curand_init(clock64(), gid, 0, &state);
//                 DISTANCE_T direction = (curand_uniform(&state) - 0.5) * dim;
//                 candidate_distances_ptr[i] = direction;
//             } else if(PRUNE_CONFIG == SIGN_BIT_PRUNE) {
//                 int direction = 0; 
//                 for (int j = 0; j < packed_vector_dim_size; j++) {
//                     direction -= __popcll(query_sign_bit[j] ^ d_sign_bit_ptr[parent_id * sign_bit_vector_size + i * packed_vector_dim_size + j]);
//                 }
//                 candidate_distances_ptr[i] = static_cast<float>(direction);
//             }
//         } else {
//             candidate_distances_ptr[i] = -99999999.0;
//         }
//         candidate_indices_ptr[i] = child_id;
//     }
//     __syncwarp(0xffffffff);
//     // sort
//     candidate_by_bitonic_sort_inverse<2, INTERNAL_TOPK / 32>(
//             candidate_indices_ptr,
//             candidate_distances_ptr,
//             CANDIDATE_BUFFER_SIZE
//         );

//     __syncwarp(0xffffffff);

//     uint32_t max_tid = CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO + CANDIDATE_BUFFER_SIZE - CANDIDATE_BUFFER_SIZE * PRUNE_RATIO;
//     for (uint32_t tid = threadIdx.x; tid < max_tid ; tid += blockDim.x) {
//         if (tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO) {
//             uint32_t i = tid / TEAM_SIZE;
//             INDEX_T child_id = candidate_indices_ptr[i];
//             // Distance calculation
//             DISTANCE_T norm2 = compute_similarity_vector_load_full<TEAM_SIZE>
//                                (dim, d_dataset_ptr, query_ptr, child_id, child_id != invalid_index);
//             //if (tid % TEAM_SIZE == 0 && child_id != invalid_index) atomicAdd(counter, 1);
//             uint8_t lane_id = threadIdx.x % TEAM_SIZE;
//             if (lane_id == 0) {
//                 if(child_id != invalid_index) {
//                     candidate_distances_ptr[i] = norm2;
//                 } else {
//                     candidate_distances_ptr[i] = FLT_MAX;
//                 }
//             }
//         } else {
//             uint32_t i = tid - CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO + CANDIDATE_BUFFER_SIZE * PRUNE_RATIO;
//             candidate_distances_ptr[i] = FLT_MAX;
//         }
//     }
// }
