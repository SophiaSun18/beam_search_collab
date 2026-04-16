// Stable mainline kernel main file.
// Keep the stable path readable from top to bottom in this one file:
// helper routines first, then the BeamSearch kernel body.

#pragma once

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

static __device__ __forceinline__ bool rabitq_lane_keep_topk_smallest_i32(
    int value,
    bool valid,
    int keep_k) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    // Invalid lanes carry INT_MAX — detected via value comparison,
    // eliminating the need for a separate valid-flag shuffle.
    const int my_value = valid ? value : INT_MAX;
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const int other_value = __shfl_sync(FULL_MASK, my_value, src);
        if (other_value >= INT_MAX) {
            continue;
        }
        if ((other_value < my_value) || ((other_value == my_value) && (src < lane_id))) {
            ++better;
        }
    }
    if (!valid || keep_k <= 0) {
        return false;
    }
    return better < keep_k;
}

static __device__ __forceinline__ bool rabitq_lane_keep_topk_smallest_f32(
    float value,
    bool valid,
    int keep_k) {
    // NOTE: do NOT early-return here — all 32 lanes must reach the
    // __shfl_sync(FULL_MASK, ...) calls below.  Invalid lanes participate
    // with FLT_MAX so they never win the topk selection.
    // The valid flag is implicit: invalid lanes carry FLT_MAX, so
    // 'other_value >= FLT_MAX' reliably signals invalidity.
    // This halves the shuffle count (64 → 32 per call).
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const float my_value = valid ? value : FLT_MAX;
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other_value = __shfl_sync(FULL_MASK, my_value, src);
        if (other_value >= FLT_MAX) {
            continue;
        }
        if ((other_value < my_value) || ((other_value == my_value) && (src < lane_id))) {
            ++better;
        }
    }
    if (!valid || keep_k <= 0) {
        return false;
    }
    return better < keep_k;
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

static inline std::vector<uint32_t> load_sign_bit(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamoff bytes = f.tellg();
    if (bytes <= 0) return {};
    if ((bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(static_cast<size_t>(bytes / sizeof(uint32_t)));
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    if (!f) return {};
    return buf;
}

static inline std::vector<uint32_t> load_sign_bit_read_bin(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};

    int32_t n = 0;
    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!f || n <= 0 || dim <= 0) return {};

    const size_t words = static_cast<size_t>(n) * static_cast<size_t>(dim);
    std::vector<uint32_t> buf(words);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(words * sizeof(uint32_t)));
    if (!f) return {};
    return buf;
}

static inline bool save_sign_bit(const char* filename, const uint32_t* data, size_t words) {
    std::ofstream f(filename, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(words * sizeof(uint32_t)));
    return static_cast<bool>(f);
}

template <typename T>
static __global__ __launch_bounds__(BLOCK_SIZE, kMinBlocksPerSM) void BeamSearch(
    int K, int beam_size, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_count_dc,
    gpu_long_t* d_total_hops,
    vid_t* d_starting_points,
    uint32_t* d_sign_bit,
    int sign_words_per_edge,
    const float* __restrict__ d_rabitq_triple,
    const float* __restrict__ d_rabitq_factor_dq,
    const uint16_t* __restrict__ d_routea_query_lut,
    int routea_lut_groups,
    const float* __restrict__ d_routea_query_lower,
    const float* __restrict__ d_routea_query_width,
    const int* __restrict__ d_routea_query_sumq,
    GraphGPU<vid_t> gg
) {
    (void)d_routea_query_lut;
    (void)routea_lut_groups;
    (void)d_routea_query_lower;
    (void)d_routea_query_width;
    (void)d_routea_query_sumq;

    const int qid = blockIdx.x;
    if (qid >= nq) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int num_warps = blockDim.x / WARP_SIZE;
    const int max_degree = gg.get_max_degree();
    const int rabitq_score_words = min(kStableRabitqScoreWords, (dim + 31) >> 5);
    const int sign_word_stride = max(0, sign_words_per_edge);
    if (beam_size <= 0 || beam_size > BEAM_SIZE) return;

    constexpr INDEX_T kIndexMsbMask = 0x80000000u;
    __shared__ INDEX_T s_all_idx[kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE];
    __shared__ DISTANCE_T s_all_dist[kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE];
    // Merge path: separate candidate + merge scratch buffers
    __shared__ INDEX_T s_cand_idx[kUseBitonicMerge ? 1 : kCandidateBufferSize];
    __shared__ DISTANCE_T s_cand_dist[kUseBitonicMerge ? 1 : kCandidateBufferSize];
    __shared__ INDEX_T s_hash[kHashSize];
    __shared__ float s_query_cache[128];
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
    __shared__ float s_routea_query_lower;
    __shared__ float s_routea_query_width;
    __shared__ int s_routea_query_sumq;
    __shared__ int s_routea_query_ready;
    __shared__ int s_pick_counts[WARPS_PER_BLOCK];

    const T* query = d_queries + static_cast<size_t>(qid) * static_cast<size_t>(dim);
    const bool routea_static_ready =
        (d_sign_bit != nullptr) &&
        (d_rabitq_triple != nullptr) &&
        (d_rabitq_factor_dq != nullptr) &&
        (sign_word_stride >= rabitq_score_words) &&
        (rabitq_score_words > 0) &&
        (dim <= (kStableRabitqScoreWords << 5)) &&
        (REFLOW_TOPN > 0) &&
        (REFLOW_TOPN < 32) &&
        (max_degree <= WARP_SIZE);

    hashtable_init(s_hash, BITLEN, 0);
    if constexpr (kUseBitonicMerge) {
        for (int i = tid; i < kResultBufferSize; i += blockDim.x) {
            s_all_idx[i] = MAX_INDEX;
            s_all_dist[i] = FLT_MAX;
        }
    } else {
        for (int i = tid; i < BEAM_SIZE; i += blockDim.x) {
            s_all_idx[i] = MAX_INDEX;
            s_all_dist[i] = FLT_MAX;
        }
        for (int i = tid; i < kCandidateBufferSize; i += blockDim.x) {
            s_cand_idx[i] = MAX_INDEX;
            s_cand_dist[i] = FLT_MAX;
        }
    }
    for (int i = tid; i < 128; i += blockDim.x) {
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
        s_routea_query_lower = 0.0f;
        s_routea_query_width = 0.0f;
        s_routea_query_sumq = 0;
        s_routea_query_ready = 0;
    }
    __syncthreads();

    for (int i = tid; i < dim && i < 128; i += blockDim.x) {
        s_query_cache[i] = static_cast<float>(query[i]);
    }
    __syncthreads();

    if (tid == 0 && routea_static_ready) {
        float lo = FLT_MAX;
        float hi = -FLT_MAX;
        for (int i = 0; i < dim; ++i) {
            const float v = s_query_cache[i];
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
        const float width = (hi > lo) ? ((hi - lo) / 63.0f) : 0.0f;
        int sumq = 0;
        for (int k = 0; k < 6; ++k) {
            #pragma unroll
            for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                s_query_bp[k][w] = 0u;
            }
        }
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
    }
    // Parallel padding: all 256 threads cooperate instead of tid==0 serial loop
    for (int i = s_frontier_size + tid; i < BEAM_SIZE; i += blockDim.x) {
        s_all_idx[i] = MAX_INDEX;
        s_all_dist[i] = FLT_MAX;
    }
    __syncthreads();
    if constexpr (kUseBitonicMerge) {
        block_bitonic_sort_pairs<kResultBufferSize>(s_all_idx, s_all_dist);
    } else {
        block_sort_candidates_by_dist(s_all_idx, s_all_dist, s_frontier_size);
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
        __syncthreads();
        if (s_rabitq_break_now) {
            break;
        }

        if (tid == 0) {
            s_break_now = 0;
            s_candidate_count = 0;
            s_num_expanders = 0;
            const int nf = min(s_frontier_size, beam_size);
            for (int j = 0; j < nf && s_num_expanders < SEARCH_WIDTH; ++j) {
                const INDEX_T index = s_all_idx[j];
                const INDEX_T clean = index & ~kIndexMsbMask;
                if (clean >= npoints) {
                    continue;
                }
                if ((index & kIndexMsbMask) == 0) {
                    s_expand_pos[s_num_expanders++] = j;
                    s_all_idx[j] = index | kIndexMsbMask;
                }
            }
        }
        __syncthreads();
        if (s_num_expanders == 0) {
            break;
        }

        if (tid == 0) {
            s_iter_count++;
        }

        const bool routea_filter_gate =
            routea_static_ready && (s_routea_query_ready == 1);
        const int rabitq_score_dims = min(dim, rabitq_score_words << 5);
        const int tail_bits = rabitq_score_dims & 31;
        const uint32_t tail_mask =
            (tail_bits == 0) ? 0xFFFFFFFFu : ((1u << tail_bits) - 1u);

        for (int e = warp_id; e < s_num_expanders; e += num_warps) {
            const int expand_pos = s_expand_pos[e];
            const vid_t current = s_all_idx[expand_pos] & ~kIndexMsbMask;
            if (current >= npoints) {
                continue;
            }
            const DISTANCE_T parent_dist = s_all_dist[expand_pos];
            const vid_t* neighbors = gg.N(current);
            const int deg = min(max_degree, WARP_SIZE);
            const vid_t neighbor = (lane_id < deg) ? neighbors[lane_id] : MAX_INDEX;
            const bool valid = (lane_id < deg) && (neighbor < npoints);

            float approx = FLT_MAX;
            bool keep = valid;
            if (routea_filter_gate) {
                if (valid) {
                    const size_t edge_pos =
                        static_cast<size_t>(current) * static_cast<size_t>(max_degree) +
                        static_cast<size_t>(lane_id);
                    const uint32_t* sign_words =
                        d_sign_bit + edge_pos * static_cast<size_t>(sign_word_stride);
                    uint32_t sw[kStableRabitqScoreWords] = {0u, 0u, 0u, 0u};
                    #pragma unroll
                    for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                        if (w < rabitq_score_words) {
                            sw[w] = sign_words[w];
                        }
                    }
                    if (tail_bits != 0 && rabitq_score_words > 0) {
                        sw[rabitq_score_words - 1] &= tail_mask;
                    }
                    int set_sum = 0;
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) {
                        int plane_sum = 0;
                        #pragma unroll
                        for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                            plane_sum += __popc(s_query_bp[k][w] & sw[w]);
                        }
                        set_sum += (plane_sum << k);
                    }
                    int pop_sum = 0;
                    #pragma unroll
                    for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                        pop_sum += __popc(sw[w]);
                    }
                    const int signed_sum = (set_sum << 1) - s_routea_query_sumq;
                    const float triple = d_rabitq_triple[edge_pos];
                    const float fdq = d_rabitq_factor_dq[edge_pos];
                    const int sign_sum = (pop_sum << 1) - rabitq_score_dims;
                    const float fvq = fdq * (float)sign_sum;
                    approx = parent_dist + triple +
                             (fdq * (s_routea_query_width * (float)signed_sum)) +
                             (fvq * s_routea_query_lower);
                    if (!(approx > 0.0f) || (approx != approx) || (approx >= FLT_MAX)) {
                        approx = FLT_MAX * 0.5f;
                    }
                }
                const int keep_k = min((int)REFLOW_TOPN, deg);
                keep = rabitq_lane_keep_topk_smallest_f32(approx, valid, keep_k);
            }

            uint32_t is_new = 0u;
            if (keep) {
                // Note: do NOT call warp_read_volatile_shared_i32 here —
                // it contains __shfl_sync(FULL_MASK) which deadlocks when
                // called inside a divergent branch (keep is false for
                // ~12/32 lanes after RouteA topk filtering).
                is_new = hashtable_insert_limited(
                    s_hash, BITLEN, neighbor, (unsigned)HASH_INSERT_MAX_PROBES);
            }
            const uint32_t ballot = __ballot_sync(FULL_MASK, is_new != 0u);
            const int warp_count = __popc(ballot);
            int base_offset = 0;
            if (lane_id == 0 && warp_count > 0) {
                base_offset = atomicAdd(&s_candidate_count, warp_count);
            }
            base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
            if (is_new) {
                const int prefix = __popc(ballot & ((1u << lane_id) - 1u));
                const int slot = base_offset + prefix;
                if (slot < kCandidateBufferSize) {
                    if constexpr (kUseBitonicMerge) {
                        s_all_idx[BEAM_SIZE + slot] = neighbor;
                        s_all_dist[BEAM_SIZE + slot] = FLT_MAX;
                    } else {
                        s_cand_idx[slot] = neighbor;
                        s_cand_dist[slot] = FLT_MAX;
                    }
                }
            }
        }
        __syncthreads();

        const int cand_count = min(s_candidate_count, kCandidateBufferSize);
        if (tid == 0) {
            if (cand_count == 0) {
                s_no_progress_iters++;
                if (s_no_progress_iters >= kNoProgressBreakIters) {
                    s_break_now = 1;
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
        if (dim == 128) {
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
                }
            }
        }
        if (lane_id == 0) {
            s_pick_counts[warp_id] = dc_round;
        }
        __syncthreads();
        if (tid == 0) {
            int dc_sum = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                dc_sum += s_pick_counts[w];
            }
            s_dc_count += dc_sum;
        }
        __syncthreads();

        if constexpr (kUseBitonicMerge) {
            for (int i = BEAM_SIZE + cand_count + tid; i < kResultBufferSize; i += blockDim.x) {
                s_all_idx[i] = MAX_INDEX;
                s_all_dist[i] = FLT_MAX;
            }
            __syncthreads();
            block_bitonic_sort_pairs<kResultBufferSize>(s_all_idx, s_all_dist);
            if (tid == 0) {
                s_frontier_size = min(beam_size, s_frontier_size + cand_count);
            }
            __syncthreads();
        } else {
            // Merge path: sort candidates, then binary merge into frontier
            block_sort_candidates_by_dist(
                s_cand_idx, s_cand_dist, cand_count);
            block_merge_candidates_into_frontier(
                s_all_idx, s_all_dist, &s_frontier_size, beam_size,
                s_cand_idx, s_cand_dist, cand_count);
            __syncthreads();
        }
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
    }
}
