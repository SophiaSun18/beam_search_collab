// Stable mainline kernel main file.
// Keep the stable path readable from top to bottom in this one file:
// helper routines first, then the BeamSearch kernel body.

#pragma once

#include "qleaf_config.cuh"

template <typename T>
static __device__ __forceinline__ void rabitq_build_query_residual_words(
    const T* query,
    const T* center,
    int dim,
    int score_words,
    uint32_t* q_words) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    uint32_t word = 0u;
    if (lane_id < score_words) {
        const int base_dim = lane_id << 5;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            const int i = base_dim + b;
            if (i >= dim) {
                break;
            }
            const float diff = (float)query[i] - (float)center[i];
            word |= ((diff >= 0.0f) ? 1u : 0u) << b;
        }
    }
    #pragma unroll
    for (int w = 0; w < kStableRabitqScoreWords; ++w) {
        q_words[w] = (w < score_words) ? __shfl_sync(FULL_MASK, word, w) : 0u;
    }
}

static __device__ __forceinline__ int rabitq_diff_count_words(
    const uint32_t* v_words,
    const uint32_t* q_words,
    int score_words,
    int score_dims) {
    int diff_cnt = 0;
    const int tail_bits = score_dims & 31;
    const uint32_t tail_mask = (tail_bits == 0) ? 0xFFFFFFFFu : ((1u << tail_bits) - 1u);
    #pragma unroll
    for (int w = 0; w < kStableRabitqScoreWords; ++w) {
        if (w >= score_words) {
            break;
        }
        uint32_t diff = v_words[w] ^ q_words[w];
        if ((w == score_words - 1) && (tail_bits != 0)) {
            diff &= tail_mask;
        }
        diff_cnt += __popc(diff);
    }
    return diff_cnt;
}

static __device__ __forceinline__ float routea_warp_min_f32(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(FULL_MASK, value, offset));
    }
    return __shfl_sync(FULL_MASK, value, 0);
}

template <typename T>
static __device__ __forceinline__ DISTANCE_T routea_lane_l2_distance_cached(
    int dim,
    const float* __restrict__ s_query_cache,
    const T* __restrict__ d_data,
    vid_t neighbor) {
    DISTANCE_T acc = 0.0f;
    const T* vec = d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim);
    #pragma unroll 1
    for (int i = 0; i < dim && i < kStableRabitqMaxScoreDims; ++i) {
        const DISTANCE_T diff = s_query_cache[i] - static_cast<DISTANCE_T>(vec[i]);
        acc += diff * diff;
    }
    return acc;
}

static __device__ __forceinline__ int block_pick_new_parents_to_expand(
    INDEX_T* frontier_idx,
    int frontier_size,
    int expand_width,
    INDEX_T* expand_pos,
    int* pick_counts,
    int* pick_num,
    int* pick_round_base,
    const INDEX_T index_msb_mask) {
    const int tid = threadIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (tid == 0) {
        *pick_num = 0;
    }
    __syncthreads();

    int padded_frontier = frontier_size;
    if (padded_frontier % blockDim.x) {
        padded_frontier += blockDim.x - (padded_frontier % blockDim.x);
    }

    for (int base = 0; base < padded_frontier; base += blockDim.x) {
        const int j = base + tid;
        INDEX_T index = 0;
        bool new_parent = false;
        if (j < frontier_size) {
            index = frontier_idx[j];
            new_parent = ((index & index_msb_mask) == 0);
        }

        const uint32_t ballot = __ballot_sync(FULL_MASK, new_parent);
        const int warp_count = __popc(ballot);
        if (lane_id == 0) {
            pick_counts[warp_id] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int total = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                if (w < num_warps) total += pick_counts[w];
            }
            *pick_round_base = *pick_num;
            *pick_num += total;
        }
        __syncthreads();

        int warp_prefix = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            if (w < warp_id) warp_prefix += pick_counts[w];
        }
        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        const int lane_prefix = __popc(ballot & lane_mask);
        const int pos = *pick_round_base + warp_prefix + lane_prefix;
        if (new_parent && pos < expand_width) {
            expand_pos[pos] = j;
            frontier_idx[j] |= index_msb_mask;
        }
        __syncthreads();
    }

    return min(*pick_num, expand_width);
}

static __device__ __forceinline__ void block_seed_initial_frontier_from_start_points(
    INDEX_T* hash,
    INDEX_T* frontier_idx,
    DISTANCE_T* frontier_dist,
    int* frontier_size,
    int K,
    int beam_size,
    size_t npoints,
    const vid_t* starting_points) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int num_warps = blockDim.x / WARP_SIZE;
    const int num_starts = min(K, beam_size);

    for (int s = warp_id; s < num_starts; s += num_warps) {
        const vid_t start_vid = starting_points[s % beam_size];
        if (start_vid >= npoints) continue;

        uint32_t is_new = 0;
        if (lane_id == 0) {
            is_new = hashtable_insert(hash, BITLEN, start_vid);
        }
        is_new = __shfl_sync(FULL_MASK, is_new, 0);
        if (!is_new) continue;

        if (lane_id == 0) {
            const int pos = atomicAdd(frontier_size, 1);
            if (pos < beam_size) {
                frontier_idx[pos] = start_vid;
                frontier_dist[pos] = FLT_MAX;
            }
        }
    }
}

static __device__ __forceinline__ void block_sort_candidates_by_dist(
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    int valid_count) {
    const int tid = threadIdx.x;
    if (valid_count <= 1) {
        return;
    }
    int sort_size = 1;
    while (sort_size < valid_count) sort_size <<= 1;
    if (sort_size < 2) {
        sort_size = 2;
    }

    for (int i = tid; i < sort_size; i += blockDim.x) {
        if (i >= valid_count) {
            candidate_idx[i] = MAX_INDEX;
            candidate_dist[i] = FLT_MAX;
        }
    }
    __syncthreads();

    for (int k = 2; k <= sort_size; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            for (int i = tid; i < sort_size; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < sort_size) {
                    const bool up = ((i & k) == 0);
                    const DISTANCE_T di = candidate_dist[i];
                    const DISTANCE_T dj = candidate_dist[ixj];
                    if ((di > dj) == up) {
                        const DISTANCE_T tmp_dist = candidate_dist[i];
                        candidate_dist[i] = candidate_dist[ixj];
                        candidate_dist[ixj] = tmp_dist;

                        const INDEX_T tmp_idx = candidate_idx[i];
                        candidate_idx[i] = candidate_idx[ixj];
                        candidate_idx[ixj] = tmp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
}

static __device__ __forceinline__ void block_sort_candidates_by_dist_nav(
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    DISTANCE_T* candidate_nav,
    int valid_count) {
    const int tid = threadIdx.x;
    if (valid_count <= 1) {
        return;
    }
    int sort_size = 1;
    while (sort_size < valid_count) sort_size <<= 1;
    if (sort_size < 2) {
        sort_size = 2;
    }

    for (int i = tid; i < sort_size; i += blockDim.x) {
        if (i >= valid_count) {
            candidate_idx[i] = MAX_INDEX;
            candidate_dist[i] = FLT_MAX;
            candidate_nav[i] = FLT_MAX;
        }
    }
    __syncthreads();

    for (int k = 2; k <= sort_size; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            for (int i = tid; i < sort_size; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < sort_size) {
                    const bool up = ((i & k) == 0);
                    const DISTANCE_T di = candidate_dist[i];
                    const DISTANCE_T dj = candidate_dist[ixj];
                    if ((di > dj) == up) {
                        const DISTANCE_T tmp_dist = candidate_dist[i];
                        candidate_dist[i] = candidate_dist[ixj];
                        candidate_dist[ixj] = tmp_dist;

                        const DISTANCE_T tmp_nav = candidate_nav[i];
                        candidate_nav[i] = candidate_nav[ixj];
                        candidate_nav[ixj] = tmp_nav;

                        const INDEX_T tmp_idx = candidate_idx[i];
                        candidate_idx[i] = candidate_idx[ixj];
                        candidate_idx[ixj] = tmp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
}

static __device__ __forceinline__ void block_sort_candidates_by_nav(
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    DISTANCE_T* candidate_nav,
    int valid_count) {
    const int tid = threadIdx.x;
    if (valid_count <= 1) {
        return;
    }
    int sort_size = 1;
    while (sort_size < valid_count) sort_size <<= 1;
    if (sort_size < 2) {
        sort_size = 2;
    }

    for (int i = tid; i < sort_size; i += blockDim.x) {
        if (i >= valid_count) {
            candidate_idx[i] = MAX_INDEX;
            candidate_dist[i] = FLT_MAX;
            candidate_nav[i] = FLT_MAX;
        }
    }
    __syncthreads();

    for (int k = 2; k <= sort_size; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            for (int i = tid; i < sort_size; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < sort_size) {
                    const bool up = ((i & k) == 0);
                    const DISTANCE_T ni = candidate_nav[i];
                    const DISTANCE_T nj = candidate_nav[ixj];
                    const DISTANCE_T di = candidate_dist[i];
                    const DISTANCE_T dj = candidate_dist[ixj];
                    const INDEX_T ii = candidate_idx[i];
                    const INDEX_T ij = candidate_idx[ixj];
                    const bool lhs_less =
                        (ni < nj) ||
                        (ni == nj && di < dj) ||
                        (ni == nj && di == dj && ((ii & 0x7fffffffu) < (ij & 0x7fffffffu)));
                    const bool should_swap = up ? !lhs_less : lhs_less;
                    if (should_swap) {
                        const DISTANCE_T tmp_dist = candidate_dist[i];
                        candidate_dist[i] = candidate_dist[ixj];
                        candidate_dist[ixj] = tmp_dist;

                        const DISTANCE_T tmp_nav = candidate_nav[i];
                        candidate_nav[i] = candidate_nav[ixj];
                        candidate_nav[ixj] = tmp_nav;

                        const INDEX_T tmp_idx = candidate_idx[i];
                        candidate_idx[i] = candidate_idx[ixj];
                        candidate_idx[ixj] = tmp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
}

static __device__ __forceinline__ int block_select_candidates_from_neighbors(
    INDEX_T* neighbor_idx,
    DISTANCE_T* neighbor_dist,
    int num_neighbors,
    DISTANCE_T candidate_cutoff,
    int frontier_size,
    int beam_size,
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    INDEX_T* candidate_tmp_idx,
    DISTANCE_T* candidate_tmp_dist,
    int* pick_num,
    int* candidate_count,
    int* candidate_tmp_count) {
    const int tid = threadIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);

    for (int i = tid; i < kCandidateBufferSize; i += blockDim.x) {
        candidate_idx[i] = MAX_INDEX;
        candidate_dist[i] = FLT_MAX;
    }
    if (tid == 0) {
        *candidate_count = 0;
    }
    __syncthreads();

    if (num_neighbors == 1) {
        if (tid == 0) {
            const DISTANCE_T dist = neighbor_dist[0];
            const bool keep_one = (dist < candidate_cutoff || frontier_size < beam_size);
            if (keep_one) {
                candidate_idx[0] = neighbor_idx[0];
                candidate_dist[0] = dist;
                *candidate_count = 1;
            } else {
                candidate_idx[0] = MAX_INDEX;
                candidate_dist[0] = FLT_MAX;
                *candidate_count = 0;
            }
        }
        __syncthreads();
        return *candidate_count;
    }

    constexpr int kNeighborCap = kNeighborBufferSize;
    if (tid == 0) {
        *pick_num = 0;
    }
    __syncthreads();

    int padded_neighbors = num_neighbors;
    if (padded_neighbors % blockDim.x) {
        padded_neighbors += blockDim.x - (padded_neighbors % blockDim.x);
    }

    for (int base = 0; base < padded_neighbors; base += blockDim.x) {
        const int n = base + tid;
        const bool valid = (n < num_neighbors);
        const DISTANCE_T dist = valid ? neighbor_dist[n] : FLT_MAX;
        const bool keep =
            valid && (dist < candidate_cutoff || frontier_size < beam_size);

        const uint32_t ballot = __ballot_sync(FULL_MASK, keep);
        const int warp_count = __popc(ballot);
        int round_base = 0;
        if (lane_id == 0 && warp_count > 0) {
            round_base = atomicAdd(pick_num, warp_count);
        }
        round_base = __shfl_sync(FULL_MASK, round_base, 0);
        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        const int lane_prefix = __popc(ballot & lane_mask);
        const int slot = round_base + lane_prefix;

        if (keep && slot < kNeighborCap) {
            candidate_tmp_idx[slot] = neighbor_idx[n];
            candidate_tmp_dist[slot] = dist;
        }
    }

    __syncthreads();
    if (tid == 0) {
        *candidate_tmp_count = min(*pick_num, kNeighborCap);
    }
    __syncthreads();

    const int valid_count = *candidate_tmp_count;
    int sort_size = 1;
    while (sort_size < valid_count) sort_size <<= 1;
    if (sort_size < 2) sort_size = 2;

    for (int i = tid; i < sort_size; i += blockDim.x) {
        if (i >= valid_count) {
            candidate_tmp_idx[i] = MAX_INDEX;
            candidate_tmp_dist[i] = FLT_MAX;
        }
    }
    __syncthreads();

    for (int k = 2; k <= sort_size; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            for (int i = tid; i < sort_size; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < sort_size) {
                    const bool up = ((i & k) == 0);
                    const DISTANCE_T di = candidate_tmp_dist[i];
                    const DISTANCE_T dj = candidate_tmp_dist[ixj];
                    bool do_swap = false;
                    if (up) {
                        do_swap = (di > dj);
                    } else {
                        do_swap = (di < dj);
                    }
                    if (do_swap) {
                        candidate_tmp_dist[i] = dj;
                        candidate_tmp_dist[ixj] = di;
                        const INDEX_T vi = candidate_tmp_idx[i];
                        candidate_tmp_idx[i] = candidate_tmp_idx[ixj];
                        candidate_tmp_idx[ixj] = vi;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        int nc = 0;
        for (int i = 0; i < valid_count && nc < kCandidateBufferSize; i++) {
            if (candidate_tmp_idx[i] != MAX_INDEX) {
                candidate_idx[nc] = candidate_tmp_idx[i];
                candidate_dist[nc] = candidate_tmp_dist[i];
                nc++;
            }
        }
        *candidate_count = nc;
        for (int i = nc; i < kCandidateBufferSize; i++) {
            candidate_idx[i] = MAX_INDEX;
            candidate_dist[i] = FLT_MAX;
        }
    }
    __syncthreads();

    return *candidate_count;
}

static __device__ __forceinline__ void block_sort_neighbor_prefix_by_dist(
    INDEX_T* neighbor_idx,
    DISTANCE_T* neighbor_approx,
    DISTANCE_T* neighbor_dist,
    int valid_count) {
    const int tid = threadIdx.x;
    int sort_size = 1;
    while (sort_size < valid_count) sort_size <<= 1;
    if (sort_size < 2) {
        sort_size = 2;
    }

    for (int i = tid; i < sort_size; i += blockDim.x) {
        if (i >= valid_count) {
            neighbor_idx[i] = MAX_INDEX;
            neighbor_approx[i] = FLT_MAX;
            neighbor_dist[i] = FLT_MAX;
        } else {
            DISTANCE_T approx = neighbor_approx[i];
            if (!(approx < FLT_MAX) || (approx != approx)) {
                approx = FLT_MAX;
            }
            neighbor_dist[i] = approx;
        }
    }
    __syncthreads();

    for (int k = 2; k <= sort_size; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            for (int i = tid; i < sort_size; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < sort_size) {
                    const bool up = ((i & k) == 0);
                    const DISTANCE_T di = neighbor_dist[i];
                    const DISTANCE_T dj = neighbor_dist[ixj];
                    if ((di > dj) == up) {
                        const DISTANCE_T tmp_dist = neighbor_dist[i];
                        neighbor_dist[i] = neighbor_dist[ixj];
                        neighbor_dist[ixj] = tmp_dist;

                        const DISTANCE_T tmp_approx = neighbor_approx[i];
                        neighbor_approx[i] = neighbor_approx[ixj];
                        neighbor_approx[ixj] = tmp_approx;

                        const INDEX_T tmp_idx = neighbor_idx[i];
                        neighbor_idx[i] = neighbor_idx[ixj];
                        neighbor_idx[ixj] = tmp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
}

static __device__ __forceinline__ void block_merge_candidates_into_frontier(
    INDEX_T* frontier_idx,
    DISTANCE_T* frontier_dist,
    int* frontier_size,
    int beam_size,
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    int candidate_count) {
    const int tid = threadIdx.x;
    const int nf = min(*frontier_size, beam_size);
    const int nc = min(candidate_count, kCandidateBufferSize);
    const int total = nf + nc;
    const int output_size = min(total, beam_size);

    DISTANCE_T local_dist[4];
    INDEX_T local_idx[4];
    int local_count = 0;

    for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
        const int k = out_pos;
        int lo = max(0, k - nf);
        int hi = min(k, nc);

        while (lo < hi) {
            const int cand_take = (lo + hi + 1) >> 1;
            const int front_take = k - cand_take;
            const DISTANCE_T cand_left =
                (cand_take > 0) ? candidate_dist[cand_take - 1] : -FLT_MAX;
            const DISTANCE_T front_right =
                (front_take < nf) ? frontier_dist[front_take] : FLT_MAX;
            if (cand_left <= front_right) {
                lo = cand_take;
            } else {
                hi = cand_take - 1;
            }
        }

        const int cand_take = lo;
        const int front_take = k - cand_take;
        const bool cand_valid = (cand_take < nc);
        const bool front_valid = (front_take < nf);
        const DISTANCE_T cand_out = cand_valid ? candidate_dist[cand_take] : FLT_MAX;
        const DISTANCE_T front_out = front_valid ? frontier_dist[front_take] : FLT_MAX;

        if (cand_valid && cand_out <= front_out) {
            local_dist[local_count] = cand_out;
            local_idx[local_count] = candidate_idx[cand_take];
        } else if (front_valid) {
            local_dist[local_count] = front_out;
            local_idx[local_count] = frontier_idx[front_take];
        } else {
            local_dist[local_count] = FLT_MAX;
            local_idx[local_count] = MAX_INDEX;
        }
        local_count++;
    }
    __syncthreads();

    local_count = 0;
    for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
        frontier_idx[out_pos] = local_idx[local_count];
        frontier_dist[out_pos] = local_dist[local_count];
        local_count++;
    }
    if (tid == 0) {
        *frontier_size = output_size;
    }
    __syncthreads();
}

static __device__ __forceinline__ void block_merge_candidates_into_frontier_nav(
    INDEX_T* frontier_idx,
    DISTANCE_T* frontier_dist,
    DISTANCE_T* frontier_nav,
    int* frontier_size,
    int beam_size,
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    DISTANCE_T* candidate_nav,
    int candidate_count) {
    const int tid = threadIdx.x;
    const int nf = min(*frontier_size, beam_size);
    const int nc = min(candidate_count, kCandidateBufferSize);
    const int total = nf + nc;
    const int output_size = min(total, beam_size);

    DISTANCE_T local_dist[4];
    DISTANCE_T local_nav[4];
    INDEX_T local_idx[4];
    int local_count = 0;

    for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
        const int k = out_pos;
        int lo = max(0, k - nf);
        int hi = min(k, nc);

        while (lo < hi) {
            const int cand_take = (lo + hi + 1) >> 1;
            const int front_take = k - cand_take;
            const DISTANCE_T cand_left =
                (cand_take > 0) ? candidate_dist[cand_take - 1] : -FLT_MAX;
            const DISTANCE_T front_right =
                (front_take < nf) ? frontier_dist[front_take] : FLT_MAX;
            if (cand_left <= front_right) {
                lo = cand_take;
            } else {
                hi = cand_take - 1;
            }
        }

        const int cand_take = lo;
        const int front_take = k - cand_take;
        const bool cand_valid = (cand_take < nc);
        const bool front_valid = (front_take < nf);
        const DISTANCE_T cand_out = cand_valid ? candidate_dist[cand_take] : FLT_MAX;
        const DISTANCE_T front_out = front_valid ? frontier_dist[front_take] : FLT_MAX;

        if (cand_valid && cand_out <= front_out) {
            local_dist[local_count] = cand_out;
            local_nav[local_count] = candidate_nav[cand_take];
            local_idx[local_count] = candidate_idx[cand_take];
        } else if (front_valid) {
            local_dist[local_count] = front_out;
            local_nav[local_count] = frontier_nav[front_take];
            local_idx[local_count] = frontier_idx[front_take];
        } else {
            local_dist[local_count] = FLT_MAX;
            local_nav[local_count] = FLT_MAX;
            local_idx[local_count] = MAX_INDEX;
        }
        local_count++;
    }
    __syncthreads();

    local_count = 0;
    for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
        frontier_idx[out_pos] = local_idx[local_count];
        frontier_dist[out_pos] = local_dist[local_count];
        frontier_nav[out_pos] = local_nav[local_count];
        local_count++;
    }
    if (tid == 0) {
        *frontier_size = output_size;
    }
    __syncthreads();
}

static __device__ __forceinline__ void block_merge_candidates_into_frontier_by_nav(
    INDEX_T* frontier_idx,
    DISTANCE_T* frontier_dist,
    DISTANCE_T* frontier_nav,
    int* frontier_size,
    int beam_size,
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    DISTANCE_T* candidate_nav,
    int candidate_count) {
    const int tid = threadIdx.x;
    const int nf = min(*frontier_size, beam_size);
    const int nc = min(candidate_count, kCandidateBufferSize);
    const int total = nf + nc;
    const int output_size = min(total, beam_size);

    DISTANCE_T local_dist[4];
    DISTANCE_T local_nav[4];
    INDEX_T local_idx[4];
    int local_count = 0;

    for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
        const int k = out_pos;
        int lo = max(0, k - nf);
        int hi = min(k, nc);

        while (lo < hi) {
            const int cand_take = (lo + hi + 1) >> 1;
            const int front_take = k - cand_take;
            const DISTANCE_T cand_left =
                (cand_take > 0) ? candidate_nav[cand_take - 1] : -FLT_MAX;
            const DISTANCE_T front_right =
                (front_take < nf) ? frontier_nav[front_take] : FLT_MAX;
            if (cand_left <= front_right) {
                lo = cand_take;
            } else {
                hi = cand_take - 1;
            }
        }

        const int cand_take = lo;
        const int front_take = k - cand_take;
        const bool cand_valid = (cand_take < nc);
        const bool front_valid = (front_take < nf);
        const DISTANCE_T cand_out = cand_valid ? candidate_nav[cand_take] : FLT_MAX;
        const DISTANCE_T front_out = front_valid ? frontier_nav[front_take] : FLT_MAX;

        if (cand_valid && cand_out <= front_out) {
            local_dist[local_count] = candidate_dist[cand_take];
            local_nav[local_count] = cand_out;
            local_idx[local_count] = candidate_idx[cand_take];
        } else if (front_valid) {
            local_dist[local_count] = frontier_dist[front_take];
            local_nav[local_count] = front_out;
            local_idx[local_count] = frontier_idx[front_take];
        } else {
            local_dist[local_count] = FLT_MAX;
            local_nav[local_count] = FLT_MAX;
            local_idx[local_count] = MAX_INDEX;
        }
        local_count++;
    }
    __syncthreads();

    local_count = 0;
    for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
        frontier_idx[out_pos] = local_idx[local_count];
        frontier_dist[out_pos] = local_dist[local_count];
        frontier_nav[out_pos] = local_nav[local_count];
        local_count++;
    }
    if (tid == 0) {
        *frontier_size = output_size;
    }
    __syncthreads();
}

template <typename T>
static __device__ __forceinline__ DISTANCE_T l2_distance_scalar_device(
    int dim, const T* __restrict__ a, const T* __restrict__ b) {
    DISTANCE_T sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const DISTANCE_T d = (DISTANCE_T)a[i] - (DISTANCE_T)b[i];
        sum = fmaf(d, d, sum);
    }
    return sum;
}

#define PARTIAL_L2_128_LANES 24

template <int VALID_LANES>
static __device__ __forceinline__ float warp_l2_distance_128_prefix(
    const float* __restrict__ a,
    const float* __restrict__ b) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane < VALID_LANES) {
        const float4 av = reinterpret_cast<const float4*>(a)[lane];
        const float4 bv = reinterpret_cast<const float4*>(b)[lane];
        const float dx0 = av.x - bv.x;
        const float dx1 = av.y - bv.y;
        const float dx2 = av.z - bv.z;
        const float dx3 = av.w - bv.w;
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

template <int START_LANE>
static __device__ __forceinline__ float warp_l2_distance_128_suffix(
    const float* __restrict__ a,
    const float* __restrict__ b) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane >= START_LANE) {
        const float4 av = reinterpret_cast<const float4*>(a)[lane];
        const float4 bv = reinterpret_cast<const float4*>(b)[lane];
        const float dx0 = av.x - bv.x;
        const float dx1 = av.y - bv.y;
        const float dx2 = av.z - bv.z;
        const float dx3 = av.w - bv.w;
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

template <bool FRONTIER_FULL>
static __device__ __forceinline__ DISTANCE_T rabitq_fixed_exact_distance_128(
    const float* __restrict__ query,
    const float* __restrict__ nvec,
    DISTANCE_T pass_cutoff,
    bool* __restrict__ computed_full) {
    if constexpr (FRONTIER_FULL) {
        const float partial =
            warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(query, nvec);
        if (partial < pass_cutoff) {
            const float tail =
                warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(query, nvec);
            return partial + tail;
        }
        *computed_full = false;
        return FLT_MAX;
    }
    (void)pass_cutoff;
    return warp_l2_distance_128(query, nvec);
}

template <int VALID_LANES>
static __device__ __forceinline__ float warp_l2_distance_128_cached_prefix(
    const float* __restrict__ query,
    const float* __restrict__ nvec) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane < VALID_LANES) {
        const int base = lane << 2;
        const float dx0 = query[base + 0] - nvec[base + 0];
        const float dx1 = query[base + 1] - nvec[base + 1];
        const float dx2 = query[base + 2] - nvec[base + 2];
        const float dx3 = query[base + 3] - nvec[base + 3];
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

template <int START_LANE>
static __device__ __forceinline__ float warp_l2_distance_128_cached_suffix(
    const float* __restrict__ query,
    const float* __restrict__ nvec) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane >= START_LANE) {
        const int base = lane << 2;
        const float dx0 = query[base + 0] - nvec[base + 0];
        const float dx1 = query[base + 1] - nvec[base + 1];
        const float dx2 = query[base + 2] - nvec[base + 2];
        const float dx3 = query[base + 3] - nvec[base + 3];
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

static __device__ __forceinline__ float warp_l2_distance_128_cached(
    const float* __restrict__ query,
    const float* __restrict__ nvec) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int base = lane << 2;
    const float dx0 = query[base + 0] - nvec[base + 0];
    const float dx1 = query[base + 1] - nvec[base + 1];
    const float dx2 = query[base + 2] - nvec[base + 2];
    const float dx3 = query[base + 3] - nvec[base + 3];
    float sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

template <bool FRONTIER_FULL>
static __device__ __forceinline__ DISTANCE_T rabitq_fixed_exact_distance_128_cached(
    const float* __restrict__ query,
    const float* __restrict__ nvec,
    DISTANCE_T pass_cutoff) {
    if constexpr (FRONTIER_FULL) {
        const float partial =
            warp_l2_distance_128_cached_prefix<PARTIAL_L2_128_LANES>(query, nvec);
        if (partial < pass_cutoff) {
            const float tail =
                warp_l2_distance_128_cached_suffix<PARTIAL_L2_128_LANES>(query, nvec);
            return partial + tail;
        }
        return FLT_MAX;
    }
    (void)pass_cutoff;
    return warp_l2_distance_128_cached(query, nvec);
}

static __device__ __forceinline__ bool bitonic_pair_less(
    DISTANCE_T lhs_dist,
    INDEX_T lhs_idx,
    DISTANCE_T rhs_dist,
    INDEX_T rhs_idx) {
    if (lhs_dist < rhs_dist) return true;
    if (lhs_dist > rhs_dist) return false;
    return (lhs_idx & 0x7fffffffu) < (rhs_idx & 0x7fffffffu);
}

template <int N>
static __device__ __forceinline__ void block_bitonic_sort_pairs(
    INDEX_T* idx,
    DISTANCE_T* dist) {
    const int tid = threadIdx.x;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (tid < N && ixj < N && ixj > tid) {
                const bool ascending = ((tid & k) == 0);
                const DISTANCE_T di = dist[tid];
                const DISTANCE_T dj = dist[ixj];
                const INDEX_T ii = idx[tid];
                const INDEX_T ij = idx[ixj];
                const bool lhs_less = bitonic_pair_less(di, ii, dj, ij);
                const bool should_swap = ascending ? !lhs_less : lhs_less;
                if (should_swap) {
                    dist[tid] = dj;
                    dist[ixj] = di;
                    idx[tid] = ij;
                    idx[ixj] = ii;
                }
            }
            __syncthreads();
        }
    }
}

template <int N>
static __device__ __forceinline__ void block_bitonic_sort_triplets(
    INDEX_T* idx,
    DISTANCE_T* dist,
    DISTANCE_T* nav) {
    const int tid = threadIdx.x;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (tid < N && ixj < N && ixj > tid) {
                const bool ascending = ((tid & k) == 0);
                const DISTANCE_T di = dist[tid];
                const DISTANCE_T dj = dist[ixj];
                const DISTANCE_T ni = nav[tid];
                const DISTANCE_T nj = nav[ixj];
                const INDEX_T ii = idx[tid];
                const INDEX_T ij = idx[ixj];
                const bool lhs_less = bitonic_pair_less(di, ii, dj, ij);
                const bool should_swap = ascending ? !lhs_less : lhs_less;
                if (should_swap) {
                    dist[tid] = dj;
                    dist[ixj] = di;
                    nav[tid] = nj;
                    nav[ixj] = ni;
                    idx[tid] = ij;
                    idx[ixj] = ii;
                }
            }
            __syncthreads();
        }
    }
}

template <int N>
static __device__ __forceinline__ void block_bitonic_sort_triplets_by_nav(
    INDEX_T* idx,
    DISTANCE_T* dist,
    DISTANCE_T* nav) {
    const int tid = threadIdx.x;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (tid < N && ixj < N && ixj > tid) {
                const bool ascending = ((tid & k) == 0);
                const DISTANCE_T di = dist[tid];
                const DISTANCE_T dj = dist[ixj];
                const DISTANCE_T ni = nav[tid];
                const DISTANCE_T nj = nav[ixj];
                const INDEX_T ii = idx[tid];
                const INDEX_T ij = idx[ixj];
                const bool lhs_less =
                    (ni < nj) ||
                    (ni == nj && di < dj) ||
                    (ni == nj && di == dj && ((ii & 0x7fffffffu) < (ij & 0x7fffffffu)));
                const bool should_swap = ascending ? !lhs_less : lhs_less;
                if (should_swap) {
                    dist[tid] = dj;
                    dist[ixj] = di;
                    nav[tid] = nj;
                    nav[ixj] = ni;
                    idx[tid] = ij;
                    idx[ixj] = ii;
                }
            }
            __syncthreads();
        }
    }
}

static __device__ __forceinline__ uint32_t hashtable_insert_noatomic(
    INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size = 1u << bitlen;
    const uint32_t bit_mask = size - 1;
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    uint32_t index = h & bit_mask;
    uint32_t stride = ((h >> bitlen) | 1u);
    #pragma unroll 1
    for (unsigned i = 0; i < size; i++) {
        const INDEX_T cur = table[index];
        if (cur == key) return 0;
        if (cur == MAX_INDEX) {
            table[index] = key;
            return 1;
        }
        index = (index + stride) & bit_mask;
    }
    return 2;
}

static __device__ __forceinline__ uint32_t hashtable_insert_adaptive(
    INDEX_T* const table, const unsigned bitlen, const INDEX_T key, const int staged_neighbors) {
    (void)staged_neighbors;
    // Always use limited probing to prevent infinite loops when hash table
    // is nearly full (critical with BITLEN=10 and 7-point expansion).
    return hashtable_insert_limited(table, bitlen, key, (unsigned)HASH_INSERT_MAX_PROBES);
}

// Sign-bit file IO helpers (`load_sign_bit`, `load_sign_bit_read_bin`,
// `save_sign_bit`) were moved to qleaf_sidecars.hpp so that header is
// self-contained when included from main.cu (which does not pull the
// kernel header). The definitions still have internal linkage in every TU
// that includes qleaf_sidecars.hpp.

#include "qleaf_device_primitives.cuh"
#include "qleaf_rabitq.cuh"
#include "qleaf_gate_policies.cuh"
#include "qleaf_controller.cuh"
#include "qleaf_profile.cuh"

template <typename T>
static __global__ __launch_bounds__(BLOCK_SIZE, kMinBlocksPerSM) void BeamSearch(
    int K, int beam_size, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_count_dc,
    gpu_long_t* d_total_hops,
    gpu_long_t* d_routea_profile_counters,
    vid_t* d_starting_points,
    uint32_t* d_sign_bit,
    int sign_words_per_edge,
    const float* __restrict__ d_rabitq_triple,
    const float* __restrict__ d_rabitq_factor_dq,
    const float* __restrict__ d_rabitq_factor_vq,
    const uint32_t* __restrict__ d_routea_query_bp,
    int routea_query_score_words,
    int routea_query_score_dims,
    const float* __restrict__ d_routea_query_lower,
    const float* __restrict__ d_routea_query_width,
    const int* __restrict__ d_routea_query_sumq,
    GraphGPU<vid_t> gg
) {
#if !ROUTEA_PROFILE_COUNTERS
    (void)d_routea_profile_counters;
#endif

    const int qid = blockIdx.x;
    if (qid >= nq) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int num_warps = blockDim.x / WARP_SIZE;
    const int max_degree = gg.get_max_degree();
    const bool use_prepared_query_bp =
        (d_routea_query_bp != nullptr) &&
        (d_routea_query_lower != nullptr) &&
        (d_routea_query_width != nullptr) &&
        (d_routea_query_sumq != nullptr) &&
        (routea_query_score_words > 0) &&
        (routea_query_score_words <= kStableRabitqScoreWords) &&
        (routea_query_score_dims > 0) &&
        (routea_query_score_dims <= (routea_query_score_words << 5));
    const int rabitq_score_words = use_prepared_query_bp
        ? routea_query_score_words
        : min(kStableRabitqScoreWords, (dim + 31) >> 5);
    const int rabitq_score_dims_for_ready = use_prepared_query_bp
        ? routea_query_score_dims
        : min(dim, rabitq_score_words << 5);
    const int sign_word_stride = max(0, sign_words_per_edge);
    if (beam_size <= 0 || beam_size > BEAM_SIZE) return;

    constexpr INDEX_T kIndexMsbMask = 0x80000000u;
    __shared__ INDEX_T s_all_idx[kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE];
    __shared__ DISTANCE_T s_all_dist[kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE];
    __shared__ DISTANCE_T s_all_nav[kRouteAUseNavArray ? (kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE) : 1];
    // Merge path: separate candidate + merge scratch buffers
    __shared__ INDEX_T s_cand_idx[kUseBitonicMerge ? 1 : kCandidateBufferSize];
    __shared__ DISTANCE_T s_cand_dist[kUseBitonicMerge ? 1 : kCandidateBufferSize];
    __shared__ DISTANCE_T s_cand_nav[kRouteAUseNavArray ? (kUseBitonicMerge ? 1 : kCandidateBufferSize) : 1];
    __shared__ INDEX_T s_hash[kHashSize];
    __shared__ float s_query_cache[kStableRabitqMaxScoreDims];
    __shared__ uint32_t s_query_bp[6][kStableRabitqScoreWords];
    __shared__ INDEX_T s_expand_pos[SEARCH_WIDTH];
    __shared__ int s_frontier_size;
    __shared__ int s_candidate_count;
    __shared__ int s_num_expanders;
    __shared__ int s_dc_count;
    __shared__ int s_iter_count;
    __shared__ int s_no_progress_iters;
    __shared__ int s_break_now;
    __shared__ DISTANCE_T s_rabitq_best_dist;
    __shared__ int s_rabitq_stall_iters;
    __shared__ int s_rabitq_break_now;
    __shared__ int s_adaptive_active_width;
    __shared__ float s_routea_query_lower;
    __shared__ float s_routea_query_width;
    __shared__ int s_routea_query_sumq;
    __shared__ int s_routea_query_ready;
    __shared__ int s_pick_counts[WARPS_PER_BLOCK];
    __shared__ int s_useful_counts[WARPS_PER_BLOCK];
#if ROUTEA_PROFILE_COUNTERS
    __shared__ int s_profile_parents_expanded;
    __shared__ int s_profile_valid_neighbors_scored;
    __shared__ int s_profile_gate_kept_pre_hash;
    __shared__ int s_profile_hash_accepted;
    __shared__ int s_profile_hash_duplicate;
    __shared__ int s_profile_hash_probe_exhausted;
    __shared__ int s_profile_candidate_buffer_overflow;
    __shared__ int s_profile_initial_exact_l2;
    __shared__ int s_profile_candidate_exact_l2;
    __shared__ int s_profile_early_stop_triggers;
    __shared__ int s_profile_no_progress_breaks;
    __shared__ int s_profile_adaptive_parents;
    __shared__ int s_profile_adaptive_keep_sum;
    __shared__ int s_profile_adaptive_keep_le4;
    __shared__ int s_profile_adaptive_keep_le8;
    __shared__ int s_profile_adaptive_keep_le16;
    __shared__ int s_profile_adaptive_keep_gt16;
    __shared__ int s_profile_adaptive_width_rounds;
    __shared__ int s_profile_adaptive_width_sum;
    __shared__ int s_profile_adaptive_threshold_under;
    __shared__ int s_profile_adaptive_exact_useful;
    __shared__ int s_profile_adaptive_feedback_narrow;
    __shared__ int s_profile_adaptive_feedback_normal;
    __shared__ int s_profile_adaptive_feedback_broad;
#if ROUTEA_AUDIT_COUNTERS
    __shared__ int s_profile_gate_exact_topn;
    __shared__ int s_profile_gate_hit_exact_topn;
    __shared__ int s_profile_gate_false_negative_topn;
    __shared__ int s_profile_gate_false_negative_cutoff;
    __shared__ gpu_long_t s_profile_gate_false_negative_cutoff_margin_x1000;
    __shared__ int s_profile_parent_pick_count;
    __shared__ int s_profile_parent_exact_rank_sum;
    __shared__ int s_profile_parent_exact_rank_max;
    __shared__ int s_profile_parent_matches_exact_best;
    __shared__ int s_profile_parent_delays_exact_best;
#endif
#endif

    const T* query = d_queries + static_cast<size_t>(qid) * static_cast<size_t>(dim);
    const bool routea_static_ready =
        (d_sign_bit != nullptr) &&
        (d_rabitq_triple != nullptr) &&
        (d_rabitq_factor_dq != nullptr) &&
        (d_rabitq_factor_vq != nullptr) &&
        (sign_word_stride >= rabitq_score_words) &&
        (rabitq_score_words > 0) &&
        (rabitq_score_dims_for_ready <= (kStableRabitqScoreWords << 5)) &&
        (max_degree <= WARP_SIZE);
    const bool routea_query_static_ready = routea_static_ready;

    hashtable_init(s_hash, BITLEN, 0);
    if constexpr (kUseBitonicMerge) {
        for (int i = tid; i < kResultBufferSize; i += blockDim.x) {
            s_all_idx[i] = MAX_INDEX;
            s_all_dist[i] = FLT_MAX;
            if constexpr (kRouteAUseNavArray) {
                s_all_nav[i] = FLT_MAX;
            }
        }
    } else {
        for (int i = tid; i < BEAM_SIZE; i += blockDim.x) {
            s_all_idx[i] = MAX_INDEX;
            s_all_dist[i] = FLT_MAX;
            if constexpr (kRouteAUseNavArray) {
                s_all_nav[i] = FLT_MAX;
            }
        }
        for (int i = tid; i < kCandidateBufferSize; i += blockDim.x) {
            s_cand_idx[i] = MAX_INDEX;
            s_cand_dist[i] = FLT_MAX;
            if constexpr (kRouteAUseNavArray) {
                s_cand_nav[i] = FLT_MAX;
            }
        }
    }
    for (int i = tid; i < kStableRabitqMaxScoreDims; i += blockDim.x) {
        s_query_cache[i] = 0.0f;
    }
    if (tid < 6 * kStableRabitqScoreWords) {
        const int k = tid / kStableRabitqScoreWords;
        const int w = tid % kStableRabitqScoreWords;
        s_query_bp[k][w] = 0u;
    }
    if (tid == 0) {
        s_frontier_size = 0;
        s_candidate_count = 0;
        s_num_expanders = 0;
        s_dc_count = 0;
        s_iter_count = 0;
        s_no_progress_iters = 0;
        s_break_now = 0;
        s_rabitq_best_dist = FLT_MAX;
        s_rabitq_stall_iters = 0;
        s_rabitq_break_now = 0;
        s_adaptive_active_width = SEARCH_WIDTH;
        s_routea_query_lower = 0.0f;
        s_routea_query_width = 0.0f;
        s_routea_query_sumq = 0;
        s_routea_query_ready = 0;
#if ROUTEA_PROFILE_COUNTERS
        s_profile_parents_expanded = 0;
        s_profile_valid_neighbors_scored = 0;
        s_profile_gate_kept_pre_hash = 0;
        s_profile_hash_accepted = 0;
        s_profile_hash_duplicate = 0;
        s_profile_hash_probe_exhausted = 0;
        s_profile_candidate_buffer_overflow = 0;
        s_profile_initial_exact_l2 = 0;
        s_profile_candidate_exact_l2 = 0;
        s_profile_early_stop_triggers = 0;
        s_profile_no_progress_breaks = 0;
        s_profile_adaptive_parents = 0;
        s_profile_adaptive_keep_sum = 0;
        s_profile_adaptive_keep_le4 = 0;
        s_profile_adaptive_keep_le8 = 0;
        s_profile_adaptive_keep_le16 = 0;
        s_profile_adaptive_keep_gt16 = 0;
        s_profile_adaptive_width_rounds = 0;
        s_profile_adaptive_width_sum = 0;
        s_profile_adaptive_threshold_under = 0;
        s_profile_adaptive_exact_useful = 0;
        s_profile_adaptive_feedback_narrow = 0;
        s_profile_adaptive_feedback_normal = 0;
        s_profile_adaptive_feedback_broad = 0;
#if ROUTEA_AUDIT_COUNTERS
        s_profile_gate_exact_topn = 0;
        s_profile_gate_hit_exact_topn = 0;
        s_profile_gate_false_negative_topn = 0;
        s_profile_gate_false_negative_cutoff = 0;
        s_profile_gate_false_negative_cutoff_margin_x1000 = 0;
        s_profile_parent_pick_count = 0;
        s_profile_parent_exact_rank_sum = 0;
        s_profile_parent_exact_rank_max = 0;
        s_profile_parent_matches_exact_best = 0;
        s_profile_parent_delays_exact_best = 0;
#endif
#endif
    }
    __syncthreads();

    for (int i = tid; i < dim && i < kStableRabitqMaxScoreDims; i += blockDim.x) {
        s_query_cache[i] = static_cast<float>(query[i]);
    }
    __syncthreads();

    if (tid == 0 && routea_query_static_ready) {
        for (int k = 0; k < 6; ++k) {
            #pragma unroll
            for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                s_query_bp[k][w] = 0u;
            }
        }
        if (use_prepared_query_bp) {
            const uint32_t* qbp =
                d_routea_query_bp +
                static_cast<size_t>(qid) * 6u * static_cast<size_t>(routea_query_score_words);
            for (int k = 0; k < 6; ++k) {
                #pragma unroll
                for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                    if (w < routea_query_score_words) {
                        s_query_bp[k][w] =
                            qbp[static_cast<size_t>(k) * static_cast<size_t>(routea_query_score_words) +
                                static_cast<size_t>(w)];
                    }
                }
            }
            s_routea_query_lower = d_routea_query_lower[qid];
            s_routea_query_width = d_routea_query_width[qid];
            s_routea_query_sumq = d_routea_query_sumq[qid];
        } else {
            float lo = FLT_MAX;
            float hi = -FLT_MAX;
            for (int i = 0; i < dim; ++i) {
                const float v = s_query_cache[i];
                lo = fminf(lo, v);
                hi = fmaxf(hi, v);
            }
            const float width = (hi > lo) ? ((hi - lo) / 63.0f) : 0.0f;
            int sumq = 0;
            for (int i = 0; i < dim; ++i) {
                int qv = 0;
                if (width > 0.0f) {
                    const float normalized = ((s_query_cache[i] - lo) / width) + 0.5f;
                    qv = __float2int_rn(normalized);
                    qv = max(0, min(63, qv));
                }
                sumq += qv;
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    if (((qv >> k) & 1) != 0) {
                        s_query_bp[k][i >> 5] |= (1u << (i & 31));
                    }
                }
            }
            s_routea_query_lower = lo;
            s_routea_query_width = width;
            s_routea_query_sumq = sumq;
        }
        s_routea_query_ready = 1;
    }
    __syncthreads();

    block_seed_initial_frontier_from_start_points(
        s_hash, s_all_idx, s_all_dist, &s_frontier_size,
        K, beam_size, npoints, d_starting_points);
    __syncthreads();

    int dc_local = 0;
    if (dim == 128) {
        for (int n = warp_id; n < s_frontier_size; n += num_warps) {
            const vid_t neighbor = s_all_idx[n] & ~kIndexMsbMask;
            const float* nvec128 =
                reinterpret_cast<const float*>(d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim));
            const DISTANCE_T dist = warp_l2_distance_128_cached(s_query_cache, nvec128);
            if (lane_id == 0) {
                s_all_dist[n] = dist;
                if constexpr (kRouteAUseNavArray) {
                    s_all_nav[n] = dist;
                }
                dc_local++;
            }
        }
    } else {
        for (int n = warp_id; n < s_frontier_size; n += num_warps) {
            const vid_t neighbor = s_all_idx[n] & ~kIndexMsbMask;
            const T* nvec = d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim);
            const DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
            if (lane_id == 0) {
                s_all_dist[n] = dist;
                if constexpr (kRouteAUseNavArray) {
                    s_all_nav[n] = dist;
                }
                dc_local++;
            }
        }
    }
    if (lane_id == 0) {
        s_pick_counts[warp_id] = dc_local;
    }
    __syncthreads();
    if (tid == 0) {
        int dc_sum = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            dc_sum += s_pick_counts[w];
        }
        s_dc_count += dc_sum;
#if ROUTEA_PROFILE_COUNTERS
        s_profile_initial_exact_l2 += dc_sum;
#endif
    }
    // Parallel padding: all 256 threads cooperate instead of tid==0 serial loop
    for (int i = s_frontier_size + tid; i < BEAM_SIZE; i += blockDim.x) {
        s_all_idx[i] = MAX_INDEX;
        s_all_dist[i] = FLT_MAX;
        if constexpr (kRouteAUseNavArray) {
            s_all_nav[i] = FLT_MAX;
        }
    }
    __syncthreads();
    if constexpr (kUseBitonicMerge) {
        if constexpr (kRouteAUseNavArray) {
            if constexpr (kRouteASortByNav) {
                block_bitonic_sort_triplets_by_nav<kResultBufferSize>(s_all_idx, s_all_dist, s_all_nav);
            } else {
                block_bitonic_sort_triplets<kResultBufferSize>(s_all_idx, s_all_dist, s_all_nav);
            }
        } else {
            block_bitonic_sort_pairs<kResultBufferSize>(s_all_idx, s_all_dist);
        }
    } else {
        if constexpr (kRouteAUseNavArray) {
            if constexpr (kRouteASortByNav) {
                block_sort_candidates_by_nav(s_all_idx, s_all_dist, s_all_nav, s_frontier_size);
            } else {
                block_sort_candidates_by_dist_nav(s_all_idx, s_all_dist, s_all_nav, s_frontier_size);
            }
        } else {
            block_sort_candidates_by_dist(s_all_idx, s_all_dist, s_frontier_size);
        }
    }
    if (tid == 0) {
        s_frontier_size = min(s_frontier_size, beam_size);
    }
    __syncthreads();

    for (int iter = 0; iter < (int)RABITQ_MAX_ITER_CAP; ++iter) {
        if (iter > 0 && (iter % SMALL_HASH_RESET_INTERVAL) == 0) {
            hashtable_init(s_hash, BITLEN, 0);
            __syncthreads();
            hashtable_restore(s_hash, BITLEN, s_all_idx, min(s_frontier_size, beam_size), 0);
            __syncthreads();
        }

        if (tid == 0 && iter > 0) {
            if constexpr (kRouteASortByNav) {
                s_rabitq_break_now = 0;
            } else {
            const int nf = min(s_frontier_size, beam_size);
            DISTANCE_T curr_best = FLT_MAX;
            if (nf > 0) {
                const int kpos = min(max(K, 1), nf) - 1;
                curr_best = s_all_dist[kpos];
            }
            DISTANCE_T improve_need = (DISTANCE_T)RABITQ_STALL_IMPROVE_EPS_ABS;
            const DISTANCE_T rel_need =
                (DISTANCE_T)(fabsf((float)s_rabitq_best_dist) * (float)RABITQ_STALL_IMPROVE_EPS_REL);
            if (rel_need > improve_need) {
                improve_need = rel_need;
            }
            const DISTANCE_T improve = s_rabitq_best_dist - curr_best;
            if (improve > improve_need) {
                s_rabitq_best_dist = curr_best;
                s_rabitq_stall_iters = 0;
            } else {
                s_rabitq_stall_iters++;
            }
            const bool stop_ok =
                (iter >= (int)RABITQ_EARLY_STOP_WARMUP_ITERS) &&
                (s_frontier_size >= beam_size);
            s_rabitq_break_now =
                stop_ok && (s_rabitq_stall_iters >= (int)RABITQ_EARLY_STOP_STALL_ITERS);
            }
        }
        __syncthreads();
        if (s_rabitq_break_now) {
#if ROUTEA_PROFILE_COUNTERS
            if (tid == 0) {
                s_profile_early_stop_triggers = 1;
            }
            __syncthreads();
#endif
            break;
        }

        if (tid == 0) {
            s_break_now = 0;
            s_candidate_count = 0;
            s_num_expanders = 0;
            const int nf = min(s_frontier_size, beam_size);
            int active_width = SEARCH_WIDTH;
            if constexpr (ROUTEA_ADAPTIVE_GATE) {
                const int width_min = max(1, min((int)ROUTEA_ADAPTIVE_WIDTH_MIN, SEARCH_WIDTH));
                const int width_base = max(width_min, min((int)ROUTEA_ADAPTIVE_WIDTH_BASE, SEARCH_WIDTH));
                const int width_max = max(width_base, min((int)ROUTEA_ADAPTIVE_WIDTH_MAX, SEARCH_WIDTH));

                RouteAAdaptiveWidthInput width_input;
                width_input.width_min = width_min;
                width_input.width_base = width_base;
                width_input.width_max = width_max;
                width_input.frontier_full = (nf >= beam_size);
                width_input.stall_iters = s_rabitq_stall_iters;
                width_input.stall_widen_1 = (int)ROUTEA_ADAPTIVE_STALL_WIDEN_1;
                width_input.stall_widen_2 = (int)ROUTEA_ADAPTIVE_STALL_WIDEN_2;
                active_width = routea_adaptive_width_policy0_decide(width_input).active_width;
            }
            s_adaptive_active_width = active_width;
#if ROUTEA_PROFILE_COUNTERS
            if constexpr (ROUTEA_ADAPTIVE_GATE) {
                s_profile_adaptive_width_rounds++;
                s_profile_adaptive_width_sum += active_width;
            }
#endif
            if constexpr (kRouteAUseNavArray) {
                for (int pick = 0; pick < active_width; ++pick) {
                    int best_pos = -1;
                    DISTANCE_T best_nav = FLT_MAX;
                    DISTANCE_T best_exact = FLT_MAX;
                    INDEX_T best_idx = MAX_INDEX;
#if ROUTEA_PROFILE_COUNTERS && ROUTEA_AUDIT_COUNTERS
                    int exact_best_pos = -1;
#endif
                    for (int j = 0; j < nf; ++j) {
                        const INDEX_T index = s_all_idx[j];
                        const INDEX_T clean = index & ~kIndexMsbMask;
                        if (clean >= npoints || (index & kIndexMsbMask) != 0) {
                            continue;
                        }
#if ROUTEA_PROFILE_COUNTERS && ROUTEA_AUDIT_COUNTERS
                        if (exact_best_pos < 0) {
                            exact_best_pos = j;
                        }
#endif
                        DISTANCE_T nav = s_all_nav[j];
                        if (!(nav < FLT_MAX) || nav != nav) {
                            nav = s_all_dist[j];
                        }
                        const DISTANCE_T exact = s_all_dist[j];
                        const bool better =
                            (nav < best_nav) ||
                            (nav == best_nav && exact < best_exact) ||
                            (nav == best_nav && exact == best_exact &&
                             (clean < (best_idx & ~kIndexMsbMask)));
                        if (better) {
                            best_pos = j;
                            best_nav = nav;
                            best_exact = exact;
                            best_idx = index;
                        }
                    }
                    if (best_pos < 0) {
                        break;
                    }
#if ROUTEA_PROFILE_COUNTERS && ROUTEA_AUDIT_COUNTERS
                    const int exact_rank = best_pos + 1;
                    s_profile_parent_pick_count++;
                    s_profile_parent_exact_rank_sum += exact_rank;
                    s_profile_parent_exact_rank_max =
                        max(s_profile_parent_exact_rank_max, exact_rank);
                    if (best_pos == exact_best_pos) {
                        s_profile_parent_matches_exact_best++;
                    } else {
                        s_profile_parent_delays_exact_best++;
                    }
#endif
                    s_expand_pos[s_num_expanders++] = best_pos;
                    s_all_idx[best_pos] |= kIndexMsbMask;
                }
            } else {
                for (int j = 0; j < nf && s_num_expanders < active_width; ++j) {
                    const INDEX_T index = s_all_idx[j];
                    const INDEX_T clean = index & ~kIndexMsbMask;
                    if (clean >= npoints) {
                        continue;
                    }
                    if ((index & kIndexMsbMask) == 0) {
#if ROUTEA_PROFILE_COUNTERS && ROUTEA_AUDIT_COUNTERS
                        const int exact_rank = j + 1;
                        s_profile_parent_pick_count++;
                        s_profile_parent_exact_rank_sum += exact_rank;
                        s_profile_parent_exact_rank_max =
                            max(s_profile_parent_exact_rank_max, exact_rank);
                        s_profile_parent_matches_exact_best++;
#endif
                        s_expand_pos[s_num_expanders++] = j;
                        s_all_idx[j] = index | kIndexMsbMask;
                    }
                }
            }
#if ROUTEA_PROFILE_COUNTERS
            s_profile_parents_expanded += s_num_expanders;
#endif
        }
        __syncthreads();
        if (s_break_now) {
            break;
        }
        if (s_num_expanders == 0) {
            break;
        }

        if (tid == 0) {
            s_iter_count++;
        }

        const bool routea_filter_gate =
            (ROUTEA_GATE_MODE == 1) &&
            routea_static_ready && (s_routea_query_ready == 1);
        const int rabitq_score_dims = use_prepared_query_bp
            ? routea_query_score_dims
            : min(dim, rabitq_score_words << 5);
        const bool routea_frontier_full = (!kRouteASortByNav) && (s_frontier_size >= beam_size);
        const DISTANCE_T routea_cutoff =
            routea_frontier_full ? s_all_dist[beam_size - 1] : FLT_MAX;
#if ROUTEA_PROFILE_COUNTERS && ROUTEA_AUDIT_COUNTERS
        const bool audit_frontier_full = routea_frontier_full;
        const DISTANCE_T audit_cutoff = routea_cutoff;
#endif

        for (int e = warp_id; e < s_num_expanders; e += num_warps) {
            const int expand_pos = s_expand_pos[e];
            const vid_t current = s_all_idx[expand_pos] & ~kIndexMsbMask;
            if (current >= npoints) {
                continue;
            }
            // ────────────────────────────────────────────────────────────
            // Section 1: Load parent and lane-local neighbor.
            // Each warp expands one frontier slot `e`. Lane `lane_id`
            // owns the lane-th neighbor; lanes >= deg are inactive.
            // ────────────────────────────────────────────────────────────
            const DISTANCE_T parent_dist = s_all_dist[expand_pos];
            const DISTANCE_T parent_nav = kRouteAUseNavArray ? s_all_nav[expand_pos] : parent_dist;
            const vid_t* neighbors = gg.N(current);
            const int deg = min(max_degree, WARP_SIZE);
            const vid_t neighbor = (lane_id < deg) ? neighbors[lane_id] : MAX_INDEX;
            const bool valid = (lane_id < deg) && (neighbor < npoints);

            // ────────────────────────────────────────────────────────────
            // Sections 2-3: Compute approximate score for placements that
            // use RaBitQ AND decide whether this neighbor survives the
            // local gate. The two steps are interleaved inside the
            // if-elseif chain below to avoid recomputing the score for
            // gate paths that don't need it:
            //   - NAV=8 edge-key:    compute edge score, keep all.
            //   - GATE 0:            no score, keep valid neighbors.
            //   - GATE 1:            compute edge bitplane score, then route to
            //                        either the non-adaptive local top-k helper or
            //                        policy-0 adaptive keep.
            // ────────────────────────────────────────────────────────────
            float approx = FLT_MAX;
            bool keep = valid;
            if constexpr (kRouteAUseEdgeRabitqNav) {
                // Sophia-style edge-residual RaBitQ-as-key (NAV_MODE=8).
                //
                // Correctness requirement: approx_base MUST be exact_L2(query, parent),
                // not s_all_dist[expand_pos]. When kRouteASkipSurvivorExact=true, newly
                // admitted candidates have s_all_dist=FLT_MAX until the final rerank
                // loop. If such a candidate is later expanded as a parent, parent_dist
                // would be FLT_MAX, making child_nav = FLT_MAX + residual = garbage.
                //
                // Fix: compute parent exact L2 here (warp-cooperative), write it back
                // to s_all_dist[expand_pos] so the final rerank loop also benefits.
                DISTANCE_T parent_exact_l2 = parent_dist;
                if (!(parent_exact_l2 > 0.0f) || parent_exact_l2 >= FLT_MAX * 0.5f) {
                    // parent_dist is stale (FLT_MAX) — compute it now.
                    const T* pvec = d_data + static_cast<size_t>(current) * static_cast<size_t>(dim);
                    if (dim == 128) {
                        parent_exact_l2 = warp_l2_distance_128_cached(s_query_cache, pvec);
                    } else {
                        parent_exact_l2 = warp_l2_distance(dim, query, pvec);
                    }
                    if (lane_id == 0) {
                        s_all_dist[expand_pos] = parent_exact_l2;
                        // Count this exact L2 so total # distance computations is correct.
                        s_dc_count++;
#if ROUTEA_PROFILE_COUNTERS
                        s_profile_candidate_exact_l2++;
#endif
                    }
                }
                if (valid && routea_static_ready) {
                    const size_t edge_pos =
                        static_cast<size_t>(current) * static_cast<size_t>(max_degree) +
                        static_cast<size_t>(lane_id);
                    const RouteAEdgeRabitqScore edge_score =
                        routea_edge_rabitq_score_bitplane(
                            valid, routea_static_ready, edge_pos,
                            d_sign_bit, sign_word_stride,
                            d_rabitq_triple, d_rabitq_factor_dq, d_rabitq_factor_vq,
                            &s_query_bp[0][0],
                            rabitq_score_words, rabitq_score_dims,
                            s_routea_query_sumq,
                            s_routea_query_width, s_routea_query_lower,
                            parent_exact_l2);
                    if (edge_score.ready) {
                        approx = edge_score.approx;
                    }
                }
                keep = valid;  // no local pruning
            } else if constexpr (ROUTEA_GATE_MODE == 0) {
                keep = routea_gate_keep_all(valid);
            } else if (routea_filter_gate) {
                if (valid) {
                    const size_t edge_pos =
                        static_cast<size_t>(current) * static_cast<size_t>(max_degree) +
                        static_cast<size_t>(lane_id);
                    const float approx_base = kRouteAUseEdgeRabitqNav ? parent_dist :
                                              (kRouteASortByNav ? parent_nav : parent_dist);
                    const RouteAEdgeRabitqScore edge_score =
                        routea_edge_rabitq_score_bitplane(
                            valid, routea_static_ready, edge_pos,
                            d_sign_bit, sign_word_stride,
                            d_rabitq_triple, d_rabitq_factor_dq, d_rabitq_factor_vq,
                            &s_query_bp[0][0],
                            rabitq_score_words, rabitq_score_dims,
                            s_routea_query_sumq,
                            s_routea_query_width, s_routea_query_lower,
                            approx_base);
                    if (edge_score.ready) {
                        approx = edge_score.approx;
                    }
                }
                int keep_k = min((int)REFLOW_TOPN, deg);
                int adaptive_threshold_count = keep_k;
                if constexpr (!ROUTEA_ADAPTIVE_GATE) {
                    // Non-adaptive scored gate (GATE_MODE 1 = local top-k).
                    RouteALocalGateInput gate_in;
                    gate_in.valid = valid;
                    gate_in.approx = approx;
                    gate_in.deg = deg;
                    gate_in.lane_id = lane_id;
                    gate_in.approx_rank = 32;
                    gate_in.keep_k = keep_k;
                    const RouteALocalGateDecision gate =
                        routea_scored_nonadaptive_gate_decide(gate_in);
                    keep = gate.keep;
                    keep_k = gate.keep_k;
                } else if constexpr (ROUTEA_ADAPTIVE_GATE) {
                    // Q-LEAF formal policy=0 keep decision.
                    //
                    // 1. Cutoff-density count: how many of this warp's 32
                    //    neighbours score better than the per-query frontier
                    //    cutoff plus a relative margin. This becomes the
                    //    desired_keep handed to the helper.
                    // 2. budget_keep = ceil(target_candidates / active_width)
                    //    so every iteration produces ~target_candidates
                    //    new candidates regardless of active width.
                    // 3. Stall-aware widening: stall_widen_1 broadens
                    //    adaptive_min/max and the cutoff margin; stall_widen_2
                    //    fully unlocks adaptive_max_keep and doubles
                    //    target_candidates.
                    // 4. routea_adaptive_keep_policy0_decide() is the single
                    //    place where the final keep_k clamp lands.
                    const int approx_rank =
                        rabitq_lane_rank_smallest_f32(approx, valid);
                    const int min_keep =
                        max(1, min((int)ROUTEA_ADAPTIVE_MIN_KEEP, deg));
                    const int base_keep =
                        max(min_keep, min((int)ROUTEA_ADAPTIVE_BASE_KEEP, deg));
                    const int max_keep =
                        max(base_keep, min((int)ROUTEA_ADAPTIVE_MAX_KEEP, deg));
                    int target_candidates =
                        max(1, (int)ROUTEA_ADAPTIVE_TARGET_CANDIDATES);
                    int adaptive_max_keep = max(base_keep, min(max_keep, 16));
                    int adaptive_min_keep = min_keep;
                    int margin_rel_x1000 =
                        max(0, (int)ROUTEA_ADAPTIVE_MARGIN_REL_X1000);

                    if (s_rabitq_stall_iters >= (int)ROUTEA_ADAPTIVE_STALL_WIDEN_2) {
                        target_candidates = target_candidates << 1;
                        adaptive_min_keep = base_keep;
                        adaptive_max_keep = max_keep;
                        margin_rel_x1000 = margin_rel_x1000 << 2;
                    } else if (s_rabitq_stall_iters >= (int)ROUTEA_ADAPTIVE_STALL_WIDEN_1) {
                        target_candidates += (target_candidates >> 1);
                        adaptive_min_keep = max(min_keep, min(base_keep, 8));
                        adaptive_max_keep = max(base_keep, min(max_keep, 16));
                        margin_rel_x1000 = margin_rel_x1000 << 1;
                    }

                    const int active_width = max(1, s_adaptive_active_width);
                    const int budget_keep =
                        max(adaptive_min_keep,
                            min(adaptive_max_keep,
                                (target_candidates + active_width - 1) / active_width));

                    adaptive_threshold_count = adaptive_max_keep;
                    if (routea_frontier_full && routea_cutoff < FLT_MAX) {
                        const float cutoff_f = (float)routea_cutoff;
                        const float margin =
                            fmaxf(1.0e-6f,
                                  fabsf(cutoff_f) *
                                      ((float)margin_rel_x1000 * 0.001f));
                        const float threshold = cutoff_f + margin;
                        const uint32_t under_ballot =
                            __ballot_sync(FULL_MASK,
                                          valid && (approx <= threshold));
                        adaptive_threshold_count = __popc(under_ballot);
                    }

                    const int cutoff_keep =
                        max(adaptive_min_keep,
                            min(adaptive_threshold_count, adaptive_max_keep));

                    RouteAAdaptiveKeepInput input;
                    input.desired_keep = cutoff_keep;
                    input.adaptive_min_keep = adaptive_min_keep;
                    input.adaptive_max_keep = adaptive_max_keep;
                    input.budget_keep = budget_keep;
                    input.max_keep = max_keep;
                    keep_k = routea_adaptive_keep_policy0_decide(input).keep_k;

#if ROUTEA_PROFILE_COUNTERS
                    if (lane_id == 0) {
                        atomicAdd(&s_profile_adaptive_parents, 1);
                        atomicAdd(&s_profile_adaptive_keep_sum, keep_k);
                        atomicAdd(&s_profile_adaptive_threshold_under,
                                  adaptive_threshold_count);
                        if (keep_k <= 4) {
                            atomicAdd(&s_profile_adaptive_keep_le4, 1);
                        } else if (keep_k <= 8) {
                            atomicAdd(&s_profile_adaptive_keep_le8, 1);
                        } else if (keep_k <= 16) {
                            atomicAdd(&s_profile_adaptive_keep_le16, 1);
                        } else {
                            atomicAdd(&s_profile_adaptive_keep_gt16, 1);
                        }
                    }
#endif

                    keep = valid && (approx_rank < keep_k);
                } else {
                    keep = rabitq_lane_keep_topk_smallest_f32(approx, valid, keep_k);
                }
            } else {
                keep = valid;
            }

#if ROUTEA_PROFILE_COUNTERS && ROUTEA_AUDIT_COUNTERS
            DISTANCE_T audit_exact = FLT_MAX;
            if (valid) {
                audit_exact = routea_lane_l2_distance_cached(
                    dim, s_query_cache, d_data, neighbor);
            }
            const int audit_keep_k = min((int)REFLOW_TOPN, deg);
            const bool audit_exact_topn =
                rabitq_lane_keep_topk_smallest_f32(audit_exact, valid, audit_keep_k);
            const bool audit_gate_hit_exact_topn = keep && audit_exact_topn;
            const bool audit_gate_false_negative_topn = (!keep) && audit_exact_topn;
            const bool audit_gate_false_negative_cutoff =
                audit_frontier_full && valid && (!keep) && (audit_exact < audit_cutoff);
            gpu_long_t margin_x1000 = 0;
            if (audit_gate_false_negative_cutoff) {
                const DISTANCE_T margin = audit_cutoff - audit_exact;
                const DISTANCE_T scaled = fminf(fmaxf(margin * 1000.0f, 0.0f), 9000000000000000.0f);
                margin_x1000 = static_cast<gpu_long_t>(scaled);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                margin_x1000 += __shfl_down_sync(FULL_MASK, margin_x1000, offset);
            }
            const uint32_t exact_topn_ballot = __ballot_sync(FULL_MASK, audit_exact_topn);
            const uint32_t hit_exact_topn_ballot =
                __ballot_sync(FULL_MASK, audit_gate_hit_exact_topn);
            const uint32_t false_negative_topn_ballot =
                __ballot_sync(FULL_MASK, audit_gate_false_negative_topn);
            const uint32_t false_negative_cutoff_ballot =
                __ballot_sync(FULL_MASK, audit_gate_false_negative_cutoff);
            if (lane_id == 0) {
                atomicAdd(&s_profile_gate_exact_topn, __popc(exact_topn_ballot));
                atomicAdd(&s_profile_gate_hit_exact_topn, __popc(hit_exact_topn_ballot));
                atomicAdd(&s_profile_gate_false_negative_topn, __popc(false_negative_topn_ballot));
                atomicAdd(&s_profile_gate_false_negative_cutoff, __popc(false_negative_cutoff_ballot));
                atomicAdd(&s_profile_gate_false_negative_cutoff_margin_x1000, margin_x1000);
            }
#endif

            // ────────────────────────────────────────────────────────────
            // Section 4: Admit kept candidates through the visited hash.
            //
            // `keep` was computed by Sections 2-3 above. For each lane that
            // wants to keep its neighbour, run the warp-cooperative hash
            // insert; lanes that successfully insert (is_new == 1) become
            // candidates and reserve a slot in the candidate buffer below.
            //
            // Hash admission stays in this file. It is coupled to `s_hash`,
            // the visited-hash shared array, and to the profile counters
            // `s_profile_hash_*`. The candidate buffer write at the end of
            // this block also stays here for the same reason.
            // ────────────────────────────────────────────────────────────
            uint32_t is_new = 0u;
            if (keep) {
                // Note: do NOT call warp_read_volatile_shared_i32 here —
                // it contains __shfl_sync(FULL_MASK) which deadlocks when
                // called inside a divergent branch (keep is false for
                // ~12/32 lanes after RouteA topk filtering).
                is_new = hashtable_insert_limited(
                    s_hash, BITLEN, neighbor, (unsigned)HASH_INSERT_MAX_PROBES);
            }
#if ROUTEA_PROFILE_COUNTERS
            const uint32_t valid_ballot = __ballot_sync(FULL_MASK, valid);
            const uint32_t keep_ballot = __ballot_sync(FULL_MASK, keep);
            const uint32_t hash_inserted_ballot = __ballot_sync(FULL_MASK, is_new == 1u);
            const uint32_t hash_duplicate_ballot = __ballot_sync(FULL_MASK, keep && is_new == 0u);
            const uint32_t hash_probe_exhausted_ballot = __ballot_sync(FULL_MASK, is_new == 2u);
            if (lane_id == 0) {
                atomicAdd(&s_profile_valid_neighbors_scored, __popc(valid_ballot));
                atomicAdd(&s_profile_gate_kept_pre_hash, __popc(keep_ballot));
                atomicAdd(&s_profile_hash_accepted, __popc(hash_inserted_ballot));
                atomicAdd(&s_profile_hash_duplicate, __popc(hash_duplicate_ballot));
                atomicAdd(&s_profile_hash_probe_exhausted, __popc(hash_probe_exhausted_ballot));
            }
#endif
            const uint32_t ballot = __ballot_sync(FULL_MASK, is_new == 1u);
            const int warp_count = __popc(ballot);
            int base_offset = 0;
            if (lane_id == 0 && warp_count > 0) {
                base_offset = atomicAdd(&s_candidate_count, warp_count);
            }
            base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
            if (is_new == 1u) {
                const int prefix = __popc(ballot & ((1u << lane_id) - 1u));
                const int slot = base_offset + prefix;
                if (slot < kCandidateBufferSize) {
                    if constexpr (kUseBitonicMerge) {
                        s_all_idx[BEAM_SIZE + slot] = neighbor;
                        s_all_dist[BEAM_SIZE + slot] = FLT_MAX;
                        if constexpr (kRouteAUseNavArray) {
                            DISTANCE_T candidate_nav =
                                (kRouteAUseEdgeRabitqNav && approx < FLT_MAX) ? approx :
                                (routea_filter_gate && approx < FLT_MAX) ? approx : parent_nav;
                            s_all_nav[BEAM_SIZE + slot] = candidate_nav;
                        }
                    } else {
                        s_cand_idx[slot] = neighbor;
                        s_cand_dist[slot] = FLT_MAX;
                        if constexpr (kRouteAUseNavArray) {
                            DISTANCE_T candidate_nav =
                                (kRouteAUseEdgeRabitqNav && approx < FLT_MAX) ? approx :
                                (routea_filter_gate && approx < FLT_MAX) ? approx : parent_nav;
                            s_cand_nav[slot] = candidate_nav;
                        }
                    }
                }
            }
        }
        __syncthreads();

        const int cand_count = min(s_candidate_count, kCandidateBufferSize);
        if (tid == 0) {
#if ROUTEA_PROFILE_COUNTERS
            if (s_candidate_count > kCandidateBufferSize) {
                s_profile_candidate_buffer_overflow +=
                    (s_candidate_count - kCandidateBufferSize);
            }
#endif
            if (cand_count == 0) {
                s_no_progress_iters++;
                if (s_no_progress_iters >= kNoProgressBreakIters) {
                    s_break_now = 1;
#if ROUTEA_PROFILE_COUNTERS
                    s_profile_no_progress_breaks = 1;
#endif
                }
            } else {
                s_no_progress_iters = 0;
            }
        }
        __syncthreads();
        if (s_break_now) {
            break;
        }
        if (cand_count == 0) {
            continue;
        }

        const bool frontier_full = (s_frontier_size >= beam_size);
        const DISTANCE_T cutoff = frontier_full ? s_all_dist[beam_size - 1] : FLT_MAX;
        int dc_round = 0;
        int useful_round = 0;
        if constexpr (kRouteASkipSurvivorExact) {
            // NAV=8 edge-RaBitQ-as-key orders candidates by navigation score
            // during search and defers survivor exact L2 until final rerank.
        } else if (dim == 128) {
            for (int n = warp_id; n < cand_count; n += num_warps) {
                const vid_t neighbor = kUseBitonicMerge ? s_all_idx[BEAM_SIZE + n] : s_cand_idx[n];
                const float* nvec128 =
                    reinterpret_cast<const float*>(d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim));
                DISTANCE_T dist = FLT_MAX;
                if (frontier_full) {
                    dist = rabitq_fixed_exact_distance_128_cached<true>(
                        s_query_cache, nvec128, cutoff);
                } else {
                    dist = rabitq_fixed_exact_distance_128_cached<false>(
                        s_query_cache, nvec128, cutoff);
                }
                if (lane_id == 0) {
                    if constexpr (kUseBitonicMerge) {
                        s_all_dist[BEAM_SIZE + n] = dist;
                    } else {
                        s_cand_dist[n] = dist;
                    }
                    dc_round++;
                    if (!frontier_full || dist < cutoff) {
                        useful_round++;
                    }
                }
            }
        } else {
            for (int n = warp_id; n < cand_count; n += num_warps) {
                const vid_t neighbor = kUseBitonicMerge ? s_all_idx[BEAM_SIZE + n] : s_cand_idx[n];
                const T* nvec = d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim);
                const DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
                if (lane_id == 0) {
                    if constexpr (kUseBitonicMerge) {
                        s_all_dist[BEAM_SIZE + n] = dist;
                    } else {
                        s_cand_dist[n] = dist;
                    }
                    dc_round++;
                    if (!frontier_full || dist < cutoff) {
                        useful_round++;
                    }
                }
            }
        }
        if (lane_id == 0) {
            s_pick_counts[warp_id] = dc_round;
            s_useful_counts[warp_id] = useful_round;
        }
        __syncthreads();
        if (tid == 0) {
            int dc_sum = 0;
            int useful_sum = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                dc_sum += s_pick_counts[w];
                useful_sum += s_useful_counts[w];
            }
            s_dc_count += dc_sum;
#if ROUTEA_PROFILE_COUNTERS
            s_profile_candidate_exact_l2 += dc_sum;
            s_profile_adaptive_exact_useful += useful_sum;
#endif
        }
        __syncthreads();

        // ────────────────────────────────────────────────────────────────
        // Section 5: Merge candidate buffer back into frontier.
        //
        // Two paths depending on kUseBitonicMerge:
        //   - bitonic path (small result buffer fits in one block):
        //     pad the candidate region with sentinels, then run the
        //     in-place block bitonic sort across BEAM_SIZE + cand_count.
        //   - merge path: sort the candidate buffer alone, then binary-
        //     merge it into the existing sorted frontier with the
        //     appropriate distance / nav comparator.
        //
        // The merge step is not extracted into a separate file by Phase 5
        // — only the named section comments are added here. The actual
        // merge helpers live above (block_merge_candidates_into_frontier*
        // and block_bitonic_sort_*).
        // ────────────────────────────────────────────────────────────────
        if constexpr (kUseBitonicMerge) {
            for (int i = BEAM_SIZE + cand_count + tid; i < kResultBufferSize; i += blockDim.x) {
                s_all_idx[i] = MAX_INDEX;
                s_all_dist[i] = FLT_MAX;
                if constexpr (kRouteAUseNavArray) {
                    s_all_nav[i] = FLT_MAX;
                }
            }
            __syncthreads();
            if constexpr (kRouteAUseNavArray) {
                if constexpr (kRouteASortByNav) {
                    block_bitonic_sort_triplets_by_nav<kResultBufferSize>(s_all_idx, s_all_dist, s_all_nav);
                } else {
                    block_bitonic_sort_triplets<kResultBufferSize>(s_all_idx, s_all_dist, s_all_nav);
                }
            } else {
                block_bitonic_sort_pairs<kResultBufferSize>(s_all_idx, s_all_dist);
            }
            if (tid == 0) {
                s_frontier_size = min(beam_size, s_frontier_size + cand_count);
            }
            __syncthreads();
        } else {
            // Merge path: sort candidates, then binary merge into frontier
            if constexpr (kRouteASortByNav) {
                block_sort_candidates_by_nav(
                    s_cand_idx, s_cand_dist, s_cand_nav, cand_count);
                block_merge_candidates_into_frontier_by_nav(
                    s_all_idx, s_all_dist, s_all_nav, &s_frontier_size, beam_size,
                    s_cand_idx, s_cand_dist, s_cand_nav, cand_count);
            } else {
                block_sort_candidates_by_dist(
                    s_cand_idx, s_cand_dist, cand_count);
                block_merge_candidates_into_frontier(
                    s_all_idx, s_all_dist, &s_frontier_size, beam_size,
                    s_cand_idx, s_cand_dist, cand_count);
            }
            __syncthreads();
        }
    }

    if constexpr (kRouteASortByNav) {
        const int final_count = min(s_frontier_size, beam_size);
        int dc_final = 0;
        if (dim == 128) {
            for (int n = warp_id; n < final_count; n += num_warps) {
                const vid_t neighbor = s_all_idx[n] & ~kIndexMsbMask;
                if (neighbor >= npoints) {
                    continue;
                }
                const float* nvec128 =
                    reinterpret_cast<const float*>(d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim));
                const DISTANCE_T dist = warp_l2_distance_128_cached(s_query_cache, nvec128);
                if (lane_id == 0) {
                    s_all_dist[n] = dist;
                    dc_final++;
                }
            }
        } else {
            for (int n = warp_id; n < final_count; n += num_warps) {
                const vid_t neighbor = s_all_idx[n] & ~kIndexMsbMask;
                if (neighbor >= npoints) {
                    continue;
                }
                const T* nvec = d_data + static_cast<size_t>(neighbor) * static_cast<size_t>(dim);
                const DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
                if (lane_id == 0) {
                    s_all_dist[n] = dist;
                    dc_final++;
                }
            }
        }
        if (lane_id == 0) {
            s_pick_counts[warp_id] = dc_final;
        }
        __syncthreads();
        if (tid == 0) {
            int dc_sum = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                dc_sum += s_pick_counts[w];
            }
            s_dc_count += dc_sum;
#if ROUTEA_PROFILE_COUNTERS
            s_profile_candidate_exact_l2 += dc_sum;
#endif
        }
        __syncthreads();
        if constexpr (kUseBitonicMerge) {
            for (int i = final_count + tid; i < kResultBufferSize; i += blockDim.x) {
                s_all_idx[i] = MAX_INDEX;
                s_all_dist[i] = FLT_MAX;
                s_all_nav[i] = FLT_MAX;
            }
            __syncthreads();
            block_bitonic_sort_triplets<kResultBufferSize>(s_all_idx, s_all_dist, s_all_nav);
        } else {
            block_sort_candidates_by_dist_nav(s_all_idx, s_all_dist, s_all_nav, final_count);
        }
        __syncthreads();
    }

    for (int i = tid; i < K; i += blockDim.x) {
        if (i < s_frontier_size) {
            d_results[qid * K + i] = s_all_idx[i] & ~kIndexMsbMask;
        } else {
            d_results[qid * K + i] = 0;
        }
    }
    if (tid == 0) {
        atomicAdd(d_total_count_dc, (gpu_long_t)s_dc_count);
        if (d_total_hops != nullptr) {
            atomicAdd(d_total_hops, (gpu_long_t)s_iter_count);
        }
#if ROUTEA_PROFILE_COUNTERS
        if (d_routea_profile_counters != nullptr) {
            routea_profile_add(d_routea_profile_counters, kRouteAProfileParentsExpanded,
                      (gpu_long_t)s_profile_parents_expanded);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileValidNeighborsScored,
                      (gpu_long_t)s_profile_valid_neighbors_scored);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileGateKeptPreHash,
                      (gpu_long_t)s_profile_gate_kept_pre_hash);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileHashAccepted,
                      (gpu_long_t)s_profile_hash_accepted);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileHashDuplicate,
                      (gpu_long_t)s_profile_hash_duplicate);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileHashProbeExhausted,
                      (gpu_long_t)s_profile_hash_probe_exhausted);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileCandidateBufferOverflow,
                      (gpu_long_t)s_profile_candidate_buffer_overflow);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileInitialExactL2,
                      (gpu_long_t)s_profile_initial_exact_l2);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileCandidateExactL2,
                      (gpu_long_t)s_profile_candidate_exact_l2);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileEarlyStopTriggers,
                      (gpu_long_t)s_profile_early_stop_triggers);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileNoProgressBreaks,
                      (gpu_long_t)s_profile_no_progress_breaks);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveParents,
                      (gpu_long_t)s_profile_adaptive_parents);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveKeepSum,
                      (gpu_long_t)s_profile_adaptive_keep_sum);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveKeepLe4,
                      (gpu_long_t)s_profile_adaptive_keep_le4);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveKeepLe8,
                      (gpu_long_t)s_profile_adaptive_keep_le8);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveKeepLe16,
                      (gpu_long_t)s_profile_adaptive_keep_le16);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveKeepGt16,
                      (gpu_long_t)s_profile_adaptive_keep_gt16);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveWidthRounds,
                      (gpu_long_t)s_profile_adaptive_width_rounds);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveWidthSum,
                      (gpu_long_t)s_profile_adaptive_width_sum);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveThresholdUnder,
                      (gpu_long_t)s_profile_adaptive_threshold_under);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveExactUseful,
                      (gpu_long_t)s_profile_adaptive_exact_useful);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveFeedbackNarrow,
                      (gpu_long_t)s_profile_adaptive_feedback_narrow);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveFeedbackNormal,
                      (gpu_long_t)s_profile_adaptive_feedback_normal);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileAdaptiveFeedbackBroad,
                      (gpu_long_t)s_profile_adaptive_feedback_broad);
#if ROUTEA_AUDIT_COUNTERS
            routea_profile_add(d_routea_profile_counters, kRouteAProfileGateExactTopN,
                      (gpu_long_t)s_profile_gate_exact_topn);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileGateHitExactTopN,
                      (gpu_long_t)s_profile_gate_hit_exact_topn);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileGateFalseNegativeTopN,
                      (gpu_long_t)s_profile_gate_false_negative_topn);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileGateFalseNegativeCutoff,
                      (gpu_long_t)s_profile_gate_false_negative_cutoff);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileGateFalseNegativeCutoffMarginX1000,
                      s_profile_gate_false_negative_cutoff_margin_x1000);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileParentPickCount,
                      (gpu_long_t)s_profile_parent_pick_count);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileParentExactRankSum,
                      (gpu_long_t)s_profile_parent_exact_rank_sum);
            routea_profile_max(d_routea_profile_counters, kRouteAProfileParentExactRankMax,
                      (gpu_long_t)s_profile_parent_exact_rank_max);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileParentMatchesExactBest,
                      (gpu_long_t)s_profile_parent_matches_exact_best);
            routea_profile_add(d_routea_profile_counters, kRouteAProfileParentDelaysExactBest,
                      (gpu_long_t)s_profile_parent_delays_exact_best);
#endif
        }
#endif
    }
}
