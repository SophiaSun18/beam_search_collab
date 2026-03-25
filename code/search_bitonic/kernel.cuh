// Stable mainline kernel main file.
// Keep the stable path readable from top to bottom in this one file:
// helper routines first, then the BeamSearch kernel body.

#pragma once

static __device__ __forceinline__ int warp_read_volatile_shared_i32(
    volatile int* ptr) {
    int v = 0;
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        v = *ptr;
    }
    return __shfl_sync(FULL_MASK, v, 0);
}

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
    if (!valid || keep_k <= 0) {
        return false;
    }
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const int other_value = __shfl_sync(FULL_MASK, value, src);
        const int other_valid = __shfl_sync(FULL_MASK, valid ? 1 : 0, src);
        if (!other_valid) {
            continue;
        }
        if ((other_value < value) || ((other_value == value) && (src < lane_id))) {
            ++better;
        }
    }
    return better < keep_k;
}

static __device__ __forceinline__ bool rabitq_lane_keep_topk_smallest_f32(
    float value,
    bool valid,
    int keep_k) {
    if (!valid || keep_k <= 0) {
        return false;
    }
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other_value = __shfl_sync(FULL_MASK, value, src);
        const int other_valid = __shfl_sync(FULL_MASK, valid ? 1 : 0, src);
        if (!other_valid) {
            continue;
        }
        if ((other_value < value) || ((other_value == value) && (src < lane_id))) {
            ++better;
        }
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
    int candidate_count,
    INDEX_T* merge_idx,
    DISTANCE_T* merge_dist) {
    const int tid = threadIdx.x;
    const int nf = min(*frontier_size, beam_size);
    const int nc = min(candidate_count, kCandidateBufferSize);
    const int total = nf + nc;
    const int output_size = min(total, beam_size);

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
            merge_dist[out_pos] = cand_out;
            merge_idx[out_pos] = candidate_idx[cand_take];
        } else if (front_valid) {
            merge_dist[out_pos] = front_out;
            merge_idx[out_pos] = frontier_idx[front_take];
        } else {
            merge_dist[out_pos] = FLT_MAX;
            merge_idx[out_pos] = MAX_INDEX;
        }
    }
    __syncthreads();

    for (int i = tid; i < output_size; i += blockDim.x) {
        frontier_idx[i] = merge_idx[i];
        frontier_dist[i] = merge_dist[i];
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
    if (staged_neighbors >= (int)HASH_INSERT_LIMIT_START_FILL) {
        return hashtable_insert_limited(table, bitlen, key, (unsigned)HASH_INSERT_MAX_PROBES);
    }
    return hashtable_insert(table, bitlen, key);
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
static __global__ void BeamSearch(
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
    const int qid = blockIdx.x;
    if (qid >= nq) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    const int max_degree = gg.get_max_degree();
    const int rabitq_score_words = min(kStableRabitqScoreWords, (dim + 31) >> 5);
    const int sign_word_stride = max(0, sign_words_per_edge);
    if (beam_size <= 0 || beam_size > BEAM_SIZE) return;

    constexpr INDEX_T kIndexMsbMask = 0x80000000;
    __shared__ INDEX_T s_frontier_idx[BEAM_SIZE];
    __shared__ DISTANCE_T s_frontier_dist[BEAM_SIZE];
    __shared__ int s_frontier_size;
    __shared__ int s_dc_count;
    __shared__ int s_iter_count;
    __shared__ int s_no_progress_iters;
    __shared__ int s_break_now;
    __shared__ INDEX_T s_hash[kHashSize];
    __shared__ INDEX_T s_neighbor_idx[kNeighborBufferSize];
    __shared__ DISTANCE_T s_neighbor_dist[kNeighborBufferSize];
    __shared__ int s_rabitq_dualpool_anchor_budget;
    __shared__ int s_num_neighbors;
    __shared__ INDEX_T s_candidate_idx[kCandidateBufferSize];
    __shared__ DISTANCE_T s_candidate_dist[kCandidateBufferSize];
    __shared__ int s_candidate_count;
    __shared__ INDEX_T s_candidate_tmp_idx[kNeighborBufferSize];
    __shared__ DISTANCE_T s_candidate_tmp_dist[kNeighborBufferSize];
    __shared__ int s_candidate_tmp_count;
    #define s_neighbor_approx s_candidate_tmp_dist
    __shared__ INDEX_T s_expand_pos[SEARCH_WIDTH];
    __shared__ int s_num_expanders;
    __shared__ DISTANCE_T s_rabitq_best_dist;
    __shared__ int s_rabitq_stall_iters;
    __shared__ int s_rabitq_break_now;
    __shared__ uint16_t s_routea_q6_lut[kStableRabitqScoreWords * 8][16];
    __shared__ float s_routea_query_lower;
    __shared__ float s_routea_query_width;
    __shared__ int s_routea_query_sumq;
    __shared__ int s_routea_query_ready;
    __shared__ int s_pick_counts[WARPS_PER_BLOCK];
    __shared__ int s_pick_num;
    __shared__ int s_pick_round_base;

    const T* query = d_queries + qid * dim;
    hashtable_init(s_hash, BITLEN, 0);

    for (int i = tid; i < BEAM_SIZE; i += blockDim.x) {
        s_frontier_idx[i] = MAX_INDEX;
        s_frontier_dist[i] = FLT_MAX;
    }
    for (int i = tid; i < kCandidateBufferSize; i += blockDim.x) {
        s_candidate_idx[i] = MAX_INDEX;
        s_candidate_dist[i] = FLT_MAX;
    }

    if (tid == 0) {
        s_frontier_size = 0;
        s_dc_count = 0;
        s_iter_count = 0;
        s_no_progress_iters = 0;
        s_break_now = 0;
        s_num_neighbors = 0;
        s_candidate_count = 0;
        s_rabitq_best_dist = FLT_MAX;
        s_rabitq_stall_iters = 0;
        s_rabitq_break_now = 0;
        s_routea_query_lower = 0.0f;
        s_routea_query_width = 0.0f;
        s_routea_query_sumq = 0;
        s_routea_query_ready = 0;
    }
    __syncthreads();

    const int routea_expected_groups = min(routea_lut_groups, kStableRabitqScoreWords * 8);
    const bool routea_query_available =
        (d_routea_query_lut != nullptr) &&
        (d_routea_query_lower != nullptr) &&
        (d_routea_query_width != nullptr) &&
        (d_routea_query_sumq != nullptr) &&
        (routea_expected_groups > 0);
    if (routea_query_available) {
        const size_t q_lut_base =
            static_cast<size_t>(qid) * static_cast<size_t>(routea_lut_groups) * 16u;
        for (int i = tid; i < routea_expected_groups * 16; i += blockDim.x) {
            const int g = i >> 4;
            const int code = i & 15;
            s_routea_q6_lut[g][code] = d_routea_query_lut[q_lut_base + static_cast<size_t>(i)];
        }
        if (tid == 0) {
            s_routea_query_lower = d_routea_query_lower[qid];
            s_routea_query_width = d_routea_query_width[qid];
            s_routea_query_sumq = d_routea_query_sumq[qid];
            s_routea_query_ready = 1;
        }
    }
    __syncthreads();

    block_seed_initial_frontier_from_start_points(
        s_hash, s_frontier_idx, s_frontier_dist, &s_frontier_size,
        K, beam_size, npoints, d_starting_points);
    __syncthreads();

    for (int iter = 0; iter < (int)RABITQ_MAX_ITER_CAP; iter++) {
        if (tid == 0) {
            s_break_now = 0;
        }
        __syncthreads();

        if (iter > 0 && (iter % SMALL_HASH_RESET_INTERVAL) == 0) {
            hashtable_init(s_hash, BITLEN, 0);
            __syncthreads();
            hashtable_restore(s_hash, BITLEN, s_frontier_idx, min(s_frontier_size, beam_size), 0);
            __syncthreads();
        }

        int expand_width = max(1, (int)REFLOW_POP_BATCH);
        const int M = min(s_frontier_size, beam_size);
        const int picked_expanders = block_pick_new_parents_to_expand(
            s_frontier_idx, M, expand_width, s_expand_pos,
            s_pick_counts, &s_pick_num, &s_pick_round_base, kIndexMsbMask);
        if (tid == 0) s_num_expanders = picked_expanders;
        __syncthreads();
        if (s_num_expanders == 0) break;

        if (tid == 0) {
            s_iter_count++;
        }
        if (tid == 0) s_num_neighbors = 0;
        __syncthreads();

        if (tid == 0 && iter > 0) {
            const int nf = min(s_frontier_size, beam_size);
            DISTANCE_T curr_best = FLT_MAX;
            if (nf > 0) {
                const int kpos = min(max(K, 1), nf) - 1;
                curr_best = s_frontier_dist[kpos];
            }
            const DISTANCE_T prev_best = s_rabitq_best_dist;
            DISTANCE_T improve_need = (DISTANCE_T)RABITQ_STALL_IMPROVE_EPS_ABS;
            const DISTANCE_T rel_need =
                (DISTANCE_T)(fabsf((float)prev_best) * (float)RABITQ_STALL_IMPROVE_EPS_REL);
            if (rel_need > improve_need) improve_need = rel_need;
            const DISTANCE_T improve = prev_best - curr_best;
            if (improve > improve_need) {
                s_rabitq_best_dist = curr_best;
                s_rabitq_stall_iters = 0;
            } else {
                s_rabitq_stall_iters++;
            }
        }
        __syncthreads();

        if (tid == 0) {
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

        const int rabitq_cool_start = (int)RABITQ_MAX_ITER_CAP;
        const bool routea_static_ready =
            (s_routea_query_ready == 1) &&
            (d_sign_bit != nullptr) &&
            (d_rabitq_triple != nullptr) &&
            (d_rabitq_factor_dq != nullptr) &&
            (sign_word_stride >= rabitq_score_words) &&
            (routea_expected_groups >= (rabitq_score_words << 3)) &&
            (dim <= (kStableRabitqScoreWords << 5)) &&
            (REFLOW_TOPN > 0) &&
            (REFLOW_TOPN < 32) &&
            (rabitq_score_words > 0);
        const bool routea_filter_gate =
            routea_static_ready &&
            (iter < rabitq_cool_start);

        for (int e = warp_id; e < s_num_expanders; e += num_warps) {
            INDEX_T expand_pos = s_expand_pos[e];
            if (expand_pos >= beam_size) continue;
            vid_t current = s_frontier_idx[expand_pos] & ~kIndexMsbMask;
            if (current >= npoints) continue;
            const int rabitq_score_dims = min(dim, rabitq_score_words << 5);
            const vid_t* neighbors = gg.N(current);
            for (int base = 0; base < max_degree; base += 32) {
                const int deg = min(32, max_degree - base);
                const bool routea_active_base =
                    routea_filter_gate && (deg > 1);
                const int n = base + lane_id;
                vid_t neighbor = (lane_id < deg) ? neighbors[n] : UINT_MAX;
                const bool valid = (lane_id < deg) && (neighbor < npoints);
                bool keep = valid;
                DISTANCE_T neighbor_approx = FLT_MAX;

                if (routea_filter_gate && routea_active_base) {
                    float routea_neighbor_approx = FLT_MAX;
                    if (valid) {
                        const size_t edge_pos =
                            (size_t)current * (size_t)max_degree + (size_t)n;
                        const uint32_t* sign_words =
                            d_sign_bit + edge_pos * (size_t)sign_word_stride;
                        int set_sum = 0;
                        int pop_sum = 0;
                        const int tail_bits = rabitq_score_dims & 31;
                        const uint32_t tail_mask =
                            (tail_bits == 0) ? 0xFFFFFFFFu : ((1u << tail_bits) - 1u);
                        #pragma unroll
                        for (int w = 0; w < kStableRabitqScoreWords; ++w) {
                            if (w >= rabitq_score_words) {
                                break;
                            }
                            uint32_t sw = sign_words[w];
                            if ((w == rabitq_score_words - 1) && (tail_bits != 0)) {
                                sw &= tail_mask;
                            }
                            pop_sum += __popc(sw);
                            const int lut_base = w << 3;
                            set_sum += (int)s_routea_q6_lut[lut_base + 0][(sw >> 0) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 1][(sw >> 4) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 2][(sw >> 8) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 3][(sw >> 12) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 4][(sw >> 16) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 5][(sw >> 20) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 6][(sw >> 24) & 0xFu];
                            set_sum += (int)s_routea_q6_lut[lut_base + 7][(sw >> 28) & 0xFu];
                        }
                        const int signed_sum = (set_sum << 1) - s_routea_query_sumq;
                        const float triple = d_rabitq_triple[edge_pos];
                        const float fdq = d_rabitq_factor_dq[edge_pos];
                        const int sign_sum = (pop_sum << 1) - rabitq_score_dims;
                        const float fvq = fdq * (float)sign_sum;
                        routea_neighbor_approx =
                            s_frontier_dist[expand_pos] + triple +
                            (fdq * (s_routea_query_width * (float)signed_sum)) +
                            (fvq * s_routea_query_lower);
                        if (!(routea_neighbor_approx > 0.0f) ||
                            (routea_neighbor_approx != routea_neighbor_approx) ||
                            (routea_neighbor_approx >= FLT_MAX)) {
                            routea_neighbor_approx = FLT_MAX * 0.5f;
                        }
                    }
                    neighbor_approx = (DISTANCE_T)routea_neighbor_approx;
                    const int keep_k = min((int)REFLOW_TOPN, deg);
                    keep = rabitq_lane_keep_topk_smallest_f32(
                        routea_neighbor_approx, valid, keep_k);
                }

                uint32_t is_new = 0;
                const uint32_t active = __ballot_sync(FULL_MASK, keep);
                if (active != 0u) {
                    const int staged_neighbors_base =
                        warp_read_volatile_shared_i32((volatile int*)&s_num_neighbors);
                    if (keep) {
                        is_new = hashtable_insert_adaptive(
                            s_hash, BITLEN, neighbor, staged_neighbors_base);
                    }
                }
                const uint32_t ballot = __ballot_sync(FULL_MASK, is_new);
                const int warp_count = __popc(ballot);
                int base_offset = 0;
                if (lane_id == 0 && warp_count > 0) {
                    base_offset = atomicAdd(&s_num_neighbors, warp_count);
                }
                base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                if (is_new) {
                    const int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                    const int slot = base_offset + my_offset;
                    if (slot < kNeighborBufferSize) {
                        s_neighbor_idx[slot] = neighbor;
                        s_neighbor_approx[slot] = neighbor_approx;
                    }
                }
            }
        }
        __syncthreads();

        int num_neighbors = min(s_num_neighbors, kNeighborBufferSize);
        if (tid == 0) {
            if (num_neighbors == 0) {
                s_no_progress_iters++;
                if (s_no_progress_iters >= kNoProgressBreakIters) {
                    s_break_now = 1;
                }
            } else {
                s_no_progress_iters = 0;
            }
        }
        __syncthreads();
        if (s_break_now) break;
        if (num_neighbors == 0) continue;

        const bool frontier_full = (s_frontier_size >= beam_size);
        const DISTANCE_T cutoff = frontier_full ? s_frontier_dist[beam_size - 1] : FLT_MAX;
        int dc_warp = 0;
        int dc_local = 0;
        if (tid == 0) {
            s_rabitq_dualpool_anchor_budget = 0;
        }
        __syncthreads();

        if (tid == 0 &&
            routea_filter_gate &&
            (num_neighbors > 0)) {
            const int tiles_per_expander = max(1, (max_degree + 31) / 32);
            const int shortlist_cap = (int)REFLOW_TOPN;
            const int per_pop = max(shortlist_cap, 0) * tiles_per_expander;
            int budget = per_pop * max(s_num_expanders, 1);
            if (budget < 1) budget = 1;
            if (budget > num_neighbors) budget = num_neighbors;
            s_rabitq_dualpool_anchor_budget = budget;
        }
        __syncthreads();

        int exact_limit = num_neighbors;
        if (routea_filter_gate) {
            exact_limit =
                min(num_neighbors, max(s_rabitq_dualpool_anchor_budget, 0));
        }
        const DISTANCE_T pass_cutoff = cutoff;
        const bool need_routea_approx_order =
            routea_filter_gate &&
            (num_neighbors > 1) &&
            (exact_limit < num_neighbors);

        if (need_routea_approx_order) {
            block_sort_neighbor_prefix_by_dist(
                s_neighbor_idx, s_neighbor_approx, s_neighbor_dist, num_neighbors);
        }
        __syncthreads();

        if (dim == 128) {
            if (frontier_full) {
                for (int n = warp_id; n < exact_limit; n += num_warps) {
                    vid_t neighbor = s_neighbor_idx[n];
                    const float* nvec128 = (const float*)(d_data + (size_t)neighbor * (size_t)dim);
                    bool computed_full = true;
                    const DISTANCE_T dist =
                        rabitq_fixed_exact_distance_128<true>(
                            (const float*)query, nvec128, pass_cutoff, &computed_full);
                    if (lane_id == 0) {
                        s_neighbor_dist[n] = dist;
                        dc_local++;
                    }
                }
            } else {
                for (int n = warp_id; n < exact_limit; n += num_warps) {
                    vid_t neighbor = s_neighbor_idx[n];
                    const float* nvec128 = (const float*)(d_data + (size_t)neighbor * (size_t)dim);
                    bool computed_full = true;
                    const DISTANCE_T dist =
                        rabitq_fixed_exact_distance_128<false>(
                            (const float*)query, nvec128, pass_cutoff, &computed_full);
                    if (lane_id == 0) {
                        s_neighbor_dist[n] = dist;
                        dc_local++;
                    }
                }
            }
        } else {
            for (int n = warp_id; n < exact_limit; n += num_warps) {
                vid_t neighbor = s_neighbor_idx[n];
                const T* nvec = d_data + neighbor * dim;
                const DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
                if (lane_id == 0) {
                    s_neighbor_dist[n] = dist;
                    dc_local++;
                }
            }
        }
        __syncthreads();

        if (lane_id == 0) {
            dc_warp = dc_local;
        }
        if (lane_id == 0) {
            s_pick_counts[warp_id] = dc_warp;
        }

        __syncthreads();
        if (tid == 0) {
            int dc_sum = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                if (w < num_warps) dc_sum += s_pick_counts[w];
            }
            s_dc_count += dc_sum;
        }
        __syncthreads();

        const int candidate_neighbor_count =
            routea_filter_gate ? exact_limit : num_neighbors;
        const DISTANCE_T candidate_cutoff = cutoff;
        s_candidate_count = block_select_candidates_from_neighbors(
            s_neighbor_idx, s_neighbor_dist, candidate_neighbor_count, candidate_cutoff,
            s_frontier_size, beam_size,
            s_candidate_idx, s_candidate_dist,
            s_candidate_tmp_idx, s_candidate_tmp_dist,
            &s_pick_num, &s_candidate_count, &s_candidate_tmp_count);

        if (tid == 0) {
            if (s_candidate_count == 0) {
                s_no_progress_iters++;
                if (s_no_progress_iters >= kNoProgressBreakIters) {
                    s_break_now = 1;
                }
            } else {
                s_no_progress_iters = 0;
            }
        }
        __syncthreads();

        if (s_break_now) break;
        if (s_candidate_count == 0) continue;

        block_merge_candidates_into_frontier(
            s_frontier_idx, s_frontier_dist, &s_frontier_size, beam_size,
            s_candidate_idx, s_candidate_dist, s_candidate_count,
            s_neighbor_idx, s_neighbor_dist);
        __syncthreads();
    }

    for (int i = tid; i < K; i += blockDim.x) {
        if (i < s_frontier_size) {
            d_results[qid * K + i] = s_frontier_idx[i] & ~kIndexMsbMask;
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
    #undef s_neighbor_approx
}
