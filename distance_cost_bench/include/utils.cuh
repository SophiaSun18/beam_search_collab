#pragma once
#include <cfloat>
#include <cstdint>
#include <cassert>
#include <climits>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "access_trace.h"

#define FULL_MASK     0xffffffff
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define GPU_HASH_BASE_BITLEN 11
#define GPU_HASH_BITLEN_FOR_TOPK(topk)                                            \
  (((topk) * 4 > 3 * (1 << GPU_HASH_BASE_BITLEN)) ? (GPU_HASH_BASE_BITLEN + 1) : GPU_HASH_BASE_BITLEN)

//#define FLT_MAX 3.40282347e+38
#ifndef GPU_SEARCH_STATS
#define GPU_SEARCH_STATS 0
#endif

#ifndef GPU_EXPANDER_TRACE
#define GPU_EXPANDER_TRACE 0
#endif

#ifndef GPU_EXPANDER_TRACE_MAX_STEPS
#define GPU_EXPANDER_TRACE_MAX_STEPS 2048
#endif

#define GPU_EXPANDER_TRACE_FIELDS 4

#ifndef GPU_ACCESS_TRACE
#define GPU_ACCESS_TRACE 0
#endif

#ifndef GPU_ACCESS_TRACE_MAX_RECORDS_PER_QUERY
#define GPU_ACCESS_TRACE_MAX_RECORDS_PER_QUERY 4096
#endif

#ifndef GPU_FRONTIER_AUDIT
#define GPU_FRONTIER_AUDIT 0
#endif

#ifndef GPU_FRONTIER_AUDIT_MAX_RECORDS_PER_QUERY
#define GPU_FRONTIER_AUDIT_MAX_RECORDS_PER_QUERY 2048
#endif

#ifndef GPU_QUALITY_HIT_TRACE
#define GPU_QUALITY_HIT_TRACE 0
#endif

#ifndef GPU_QUALITY_HIT_TRACE_MAX_RECORDS_PER_QUERY
#define GPU_QUALITY_HIT_TRACE_MAX_RECORDS_PER_QUERY 2048
#endif

#define GPU_QUALITY_HIT_STAGE_EXPAND 0
#define GPU_QUALITY_HIT_STAGE_FINAL 1

#ifndef GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES
#define GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES 32
#endif

#ifndef GPU_RABITQ_USE_LUT
#define GPU_RABITQ_USE_LUT 1
#endif

#define MAX_INDEX UINT_MAX

#define SHFL_DOWN(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define SHFL(val, lane) __shfl_sync(FULL_MASK, val, lane)
#define SHFL_XOR(val, mask) __shfl_xor_sync(FULL_MASK, val, mask)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 900)
#define GPU_MAX_WARPS_PER_SM 48
#else
#define GPU_MAX_WARPS_PER_SM 64
#endif

#define GPU_MIN_BLOCKS_PER_SM_FOR(block_size) (GPU_MAX_WARPS_PER_SM / WARPS_PER_BLOCK)
#define GPU_LAUNCH_BOUNDS(block_size) __launch_bounds__((block_size), GPU_MIN_BLOCKS_PER_SM_FOR(block_size))

typedef unsigned long long gpu_long_t; // for counters
typedef uint32_t vidType;
using vid_t = vidType;
typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

enum GpuSearchStatIndex : int {
  GPU_SEARCH_STAT_DIST_CMPS = 0,
  GPU_SEARCH_STAT_DIST_EST = 1,
  GPU_SEARCH_STAT_ITERATIONS = 2,
  GPU_SEARCH_STAT_PRUNED = 3,
  GPU_SEARCH_STAT_COUNT = 4
};

enum GpuExpanderTraceField : int {
  GPU_TRACE_EXPANDER_DIST = 0,
  GPU_TRACE_SIGNED_DELTA = 1,
  GPU_TRACE_POSITIVE_DELTA = 2,
  GPU_TRACE_BEST_EXPANDER_DIST = 3
};

static __device__ inline void record_gpu_access_trace(
    AccessTraceRecord* records,
    uint32_t* counts,
    int max_records_per_query,
    uint32_t query_id,
    uint32_t step,
    uint32_t op,
    uint32_t node_id,
    uint32_t active_mask) {
#if GPU_ACCESS_TRACE
  if (records == nullptr || counts == nullptr || max_records_per_query <= 0) {
    return;
  }
  uint32_t pos = counts[query_id];
  if (pos < static_cast<uint32_t>(max_records_per_query)) {
    AccessTraceRecord& r =
        records[static_cast<size_t>(query_id) * max_records_per_query + pos];
    r.query_id = query_id;
    r.step = step;
    r.op = op;
    r.node_id = node_id;
    r.active_mask = active_mask;
  }
  counts[query_id] = pos + 1;
#else
  (void)records;
  (void)counts;
  (void)max_records_per_query;
  (void)query_id;
  (void)step;
  (void)op;
  (void)node_id;
  (void)active_mask;
#endif
}

static __device__ inline void record_gpu_expander_trace(
    double* trace_sums,
    gpu_long_t* trace_counts,
    int trace_max_steps,
    int step,
    double expander_dist,
    double& previous_expander_dist,
    bool& has_previous_expander_dist,
    double& first_expander_dist,
    double& best_expander_dist) {
#if GPU_EXPANDER_TRACE
  const double signed_delta =
      has_previous_expander_dist ? previous_expander_dist - expander_dist : 0.0;
  const double positive_delta = signed_delta > 0.0 ? signed_delta : 0.0;
  best_expander_dist = has_previous_expander_dist
                           ? (expander_dist < best_expander_dist ? expander_dist : best_expander_dist)
                           : expander_dist;
  if (!has_previous_expander_dist) {
    first_expander_dist = expander_dist;
  }

  if (trace_sums != nullptr && trace_counts != nullptr && step < trace_max_steps) {
    const int base = step * GPU_EXPANDER_TRACE_FIELDS;
    atomicAdd(trace_sums + base + GPU_TRACE_EXPANDER_DIST, expander_dist);
    atomicAdd(trace_sums + base + GPU_TRACE_SIGNED_DELTA, signed_delta);
    atomicAdd(trace_sums + base + GPU_TRACE_POSITIVE_DELTA, positive_delta);
    atomicAdd(trace_sums + base + GPU_TRACE_BEST_EXPANDER_DIST, best_expander_dist);
    atomicAdd(trace_counts + step, static_cast<gpu_long_t>(1));
  }

  previous_expander_dist = expander_dist;
  has_previous_expander_dist = true;
#else
  (void)trace_sums;
  (void)trace_counts;
  (void)trace_max_steps;
  (void)step;
  (void)expander_dist;
  (void)previous_expander_dist;
  (void)has_previous_expander_dist;
  (void)first_expander_dist;
  (void)best_expander_dist;
#endif
}

template <typename T>
static __device__ inline T gpu_scalar_distance(int dim, const T* a, const T* b, bool use_ip) {
  T acc = 0;
  if (use_ip) {
    for (int i = 0; i < dim; ++i) {
      acc += a[i] * b[i];
    }
    return -acc;
  }
  for (int i = 0; i < dim; ++i) {
    const T diff = a[i] - b[i];
    acc += diff * diff;
  }
  return acc;
}

template <typename T>
static __device__ inline void record_gpu_frontier_audit(
    GpuFrontierAuditRecord* records,
    uint32_t* counts,
    int max_records_per_query,
    uint32_t query_id,
    uint32_t step,
    INDEX_T* topk_index,
    int beam_sz,
    const T* query,
    const T* data,
    size_t row_stride,
    int dim,
    bool use_ip,
    vidType chosen_node,
    DISTANCE_T* exact_scratch) {
#if GPU_FRONTIER_AUDIT
  if (records == nullptr || counts == nullptr || max_records_per_query <= 0 ||
      topk_index == nullptr || query == nullptr || data == nullptr ||
      row_stride == 0 || dim <= 0 || exact_scratch == nullptr) {
    return;
  }

  constexpr INDEX_T index_msb_1_mask = 0x80000000u;
  for (int i = threadIdx.x; i < beam_sz; i += blockDim.x) {
    const INDEX_T raw = topk_index[i];
    const INDEX_T node = raw & ~index_msb_1_mask;
    const bool valid = raw != MAX_INDEX &&
        (((raw & index_msb_1_mask) == 0) || node == chosen_node);
    exact_scratch[i] = valid
        ? gpu_scalar_distance(dim, query, data + static_cast<size_t>(node) * row_stride, use_ip)
        : FLT_MAX;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float best_dist = FLT_MAX;
    vidType best_node = MAX_INDEX;
    float chosen_dist = FLT_MAX;
    bool found_chosen = false;

    for (int i = 0; i < beam_sz; ++i) {
      const INDEX_T raw = topk_index[i];
      const INDEX_T node = raw & ~index_msb_1_mask;
      const bool valid = raw != MAX_INDEX &&
          (((raw & index_msb_1_mask) == 0) || node == chosen_node);
      if (!valid) continue;

      const float dist = exact_scratch[i];
      if (dist < best_dist || (dist == best_dist && node < best_node)) {
        best_dist = dist;
        best_node = node;
      }
      if (node == chosen_node) {
        chosen_dist = dist;
        found_chosen = true;
      }
    }

    if (found_chosen && best_node != MAX_INDEX && isfinite(chosen_dist) && isfinite(best_dist)) {
      uint32_t rank = 1;
      for (int i = 0; i < beam_sz; ++i) {
        const INDEX_T raw = topk_index[i];
        const INDEX_T node = raw & ~index_msb_1_mask;
        const bool valid = raw != MAX_INDEX &&
            (((raw & index_msb_1_mask) == 0) || node == chosen_node);
        if (!valid || node == chosen_node) continue;

        const float dist = exact_scratch[i];
        if (dist < chosen_dist || (dist == chosen_dist && node < chosen_node)) {
          ++rank;
        }
      }

      const uint32_t pos = counts[query_id];
      if (pos < static_cast<uint32_t>(max_records_per_query)) {
        GpuFrontierAuditRecord& rec =
            records[static_cast<size_t>(query_id) * max_records_per_query + pos];
        rec.query_id = query_id;
        rec.step = step;
        rec.chosen_node = chosen_node;
        rec.chosen_exact_rank = rank;
        rec.frontier_exact_best_l2 = best_dist;
        rec.frontier_chosen_exact_l2 = chosen_dist;
        const float denom = fmaxf(fabsf(best_dist), 1e-12f);
        rec.frontier_exact_gap = (chosen_dist - best_dist) / denom;
      }
      counts[query_id] = pos + 1;
    }
  }
  __syncthreads();
#else
  (void)records;
  (void)counts;
  (void)max_records_per_query;
  (void)query_id;
  (void)step;
  (void)topk_index;
  (void)beam_sz;
  (void)query;
  (void)data;
  (void)row_stride;
  (void)dim;
  (void)use_ip;
  (void)chosen_node;
  (void)exact_scratch;
#endif
}

static __device__ inline int gpu_popcount_masks(const uint32_t* masks, int mask_words) {
  int total = 0;
  for (int i = 0; i < mask_words; ++i) {
    total += __popc(masks[i]);
  }
  return total;
}

static __device__ inline void gpu_mark_quality_hits_from_list(
    const INDEX_T* topk_index,
    int list_size,
    int topk,
    const int* query_gt,
    int gt_n,
    uint32_t* cur,
    bool scan_full_list) {
#if GPU_QUALITY_HIT_TRACE
  if (topk_index == nullptr || query_gt == nullptr || cur == nullptr ||
      list_size <= 0 || topk <= 0 || gt_n <= 0) {
    return;
  }

  const int scan_n = scan_full_list ? list_size : (list_size < topk ? list_size : topk);
  for (int i = threadIdx.x; i < scan_n; i += blockDim.x) {
    const INDEX_T raw = topk_index[i];
    if (raw == MAX_INDEX) {
      continue;
    }
    const INDEX_T node = raw & ~0x80000000u;
    for (int rank = 0; rank < gt_n; ++rank) {
      if (query_gt[rank] == static_cast<int>(node)) {
        const int word = rank >> 5;
        const uint32_t bit = 1u << (rank & 31);
        atomicOr(cur + word, bit);
        break;
      }
    }
  }
#else
  (void)topk_index;
  (void)list_size;
  (void)topk;
  (void)query_gt;
  (void)gt_n;
  (void)cur;
  (void)scan_full_list;
#endif
}

static __device__ inline void record_gpu_quality_hit_trace_impl(
    GpuQualityHitTraceRecord* records,
    uint32_t* counts,
    int max_records_per_query,
    uint32_t query_id,
    uint32_t step,
    uint8_t stage,
    const INDEX_T* topk_index,
    int list_size,
    int topk,
    const int* groundtruth,
    int gt_k,
    uint32_t* previous_gt_masks,
    uint32_t* seen_gt_masks,
    uint32_t* current_gt_masks,
    int mask_words,
    bool scan_full_list) {
#if GPU_QUALITY_HIT_TRACE
  if (records == nullptr || counts == nullptr || max_records_per_query <= 0 ||
      topk_index == nullptr || groundtruth == nullptr || previous_gt_masks == nullptr ||
      seen_gt_masks == nullptr || current_gt_masks == nullptr || topk <= 0 ||
      gt_k <= 0 || mask_words <= 0) {
    return;
  }

  const int gt_n = topk < gt_k ? topk : gt_k;
  uint32_t* prev = previous_gt_masks + static_cast<size_t>(query_id) * mask_words;
  uint32_t* seen = seen_gt_masks + static_cast<size_t>(query_id) * mask_words;
  uint32_t* cur = current_gt_masks + static_cast<size_t>(query_id) * mask_words;
  const int* query_gt = groundtruth + static_cast<size_t>(query_id) * gt_k;

  for (int word = threadIdx.x; word < mask_words; word += blockDim.x) {
    cur[word] = 0;
  }
  __syncthreads();

  gpu_mark_quality_hits_from_list(topk_index, list_size, topk, query_gt, gt_n,
                                  cur, scan_full_list);
  __syncthreads();

  if (threadIdx.x == 0) {
    int new_gt_count = 0;
    int first_seen_gt_count = 0;
    int dropped_gt_count = 0;
    for (int word = 0; word < mask_words; ++word) {
      const uint32_t now = cur[word];
      const uint32_t was = prev[word];
      const uint32_t ever = seen[word];
      new_gt_count += __popc(now & ~was);
      first_seen_gt_count += __popc(now & ~ever);
      dropped_gt_count += __popc(was & ~now);
      seen[word] = ever | now;
      prev[word] = now;
    }

    const int current_hit_count = gpu_popcount_masks(cur, mask_words);
    const int cumulative_seen_gt_count = gpu_popcount_masks(seen, mask_words);
    const uint32_t pos = counts[query_id];
    if (pos < static_cast<uint32_t>(max_records_per_query)) {
      GpuQualityHitTraceRecord& rec =
          records[static_cast<size_t>(query_id) * max_records_per_query + pos];
      rec.query_id = query_id;
      rec.step = step;
      rec.stage = stage;
      rec.new_gt_count = static_cast<uint16_t>(new_gt_count);
      rec.first_seen_gt_count = static_cast<uint16_t>(first_seen_gt_count);
      rec.dropped_gt_count = static_cast<uint16_t>(dropped_gt_count);
      rec.current_hit_count = static_cast<uint16_t>(current_hit_count);
      rec.cumulative_seen_gt_count = static_cast<uint16_t>(cumulative_seen_gt_count);
      rec.topk_recall = static_cast<float>(current_hit_count) / static_cast<float>(topk);
    }
    counts[query_id] = pos + 1;
  }
  __syncthreads();
#else
  (void)records;
  (void)counts;
  (void)max_records_per_query;
  (void)query_id;
  (void)step;
  (void)stage;
  (void)topk_index;
  (void)list_size;
  (void)topk;
  (void)groundtruth;
  (void)gt_k;
  (void)previous_gt_masks;
  (void)seen_gt_masks;
  (void)current_gt_masks;
  (void)mask_words;
  (void)scan_full_list;
#endif
}

static __device__ inline void record_gpu_quality_hit_trace(
    GpuQualityHitTraceRecord* records,
    uint32_t* counts,
    int max_records_per_query,
    uint32_t query_id,
    uint32_t step,
    uint8_t stage,
    const INDEX_T* topk_index,
    int list_size,
    int topk,
    const int* groundtruth,
    int gt_k,
    uint32_t* previous_gt_masks,
    uint32_t* seen_gt_masks,
    uint32_t* current_gt_masks,
    int mask_words) {
  record_gpu_quality_hit_trace_impl(
      records, counts, max_records_per_query, query_id, step, stage,
      topk_index, list_size, topk, groundtruth, gt_k,
      previous_gt_masks, seen_gt_masks, current_gt_masks, mask_words, false);
}

static __device__ inline void record_gpu_quality_beam_hit_trace(
    GpuQualityHitTraceRecord* records,
    uint32_t* counts,
    int max_records_per_query,
    uint32_t query_id,
    uint32_t step,
    uint8_t stage,
    const INDEX_T* topk_index,
    int list_size,
    int topk,
    const int* groundtruth,
    int gt_k,
    uint32_t* previous_gt_masks,
    uint32_t* seen_gt_masks,
    uint32_t* current_gt_masks,
    int mask_words) {
  record_gpu_quality_hit_trace_impl(
      records, counts, max_records_per_query, query_id, step, stage,
      topk_index, list_size, topk, groundtruth, gt_k,
      previous_gt_masks, seen_gt_masks, current_gt_masks, mask_words, true);
}

static __device__ inline void record_gpu_quality_search_state_hit_trace(
    GpuQualityHitTraceRecord* records,
    uint32_t* counts,
    int max_records_per_query,
    uint32_t query_id,
    uint32_t step,
    uint8_t stage,
    const INDEX_T* result_index,
    int result_size,
    const INDEX_T* beam_index,
    int beam_size,
    int topk,
    const int* groundtruth,
    int gt_k,
    uint32_t* previous_gt_masks,
    uint32_t* seen_gt_masks,
    uint32_t* current_gt_masks,
    int mask_words) {
#if GPU_QUALITY_HIT_TRACE
  if (records == nullptr || counts == nullptr || max_records_per_query <= 0 ||
      groundtruth == nullptr || previous_gt_masks == nullptr ||
      seen_gt_masks == nullptr || current_gt_masks == nullptr || topk <= 0 ||
      gt_k <= 0 || mask_words <= 0) {
    return;
  }

  const int gt_n = topk < gt_k ? topk : gt_k;
  uint32_t* prev = previous_gt_masks + static_cast<size_t>(query_id) * mask_words;
  uint32_t* seen = seen_gt_masks + static_cast<size_t>(query_id) * mask_words;
  uint32_t* cur = current_gt_masks + static_cast<size_t>(query_id) * mask_words;
  const int* query_gt = groundtruth + static_cast<size_t>(query_id) * gt_k;

  for (int word = threadIdx.x; word < mask_words; word += blockDim.x) {
    cur[word] = 0;
  }
  __syncthreads();

  gpu_mark_quality_hits_from_list(result_index, result_size, topk, query_gt, gt_n,
                                  cur, false);
  gpu_mark_quality_hits_from_list(beam_index, beam_size, topk, query_gt, gt_n,
                                  cur, true);
  __syncthreads();

  if (threadIdx.x == 0) {
    int new_gt_count = 0;
    int first_seen_gt_count = 0;
    int dropped_gt_count = 0;
    for (int word = 0; word < mask_words; ++word) {
      const uint32_t now = cur[word];
      const uint32_t was = prev[word];
      const uint32_t ever = seen[word];
      new_gt_count += __popc(now & ~was);
      first_seen_gt_count += __popc(now & ~ever);
      dropped_gt_count += __popc(was & ~now);
      seen[word] = ever | now;
      prev[word] = now;
    }

    const int current_hit_count = gpu_popcount_masks(cur, mask_words);
    const int cumulative_seen_gt_count = gpu_popcount_masks(seen, mask_words);
    const uint32_t pos = counts[query_id];
    if (pos < static_cast<uint32_t>(max_records_per_query)) {
      GpuQualityHitTraceRecord& rec =
          records[static_cast<size_t>(query_id) * max_records_per_query + pos];
      rec.query_id = query_id;
      rec.step = step;
      rec.stage = stage;
      rec.new_gt_count = static_cast<uint16_t>(new_gt_count);
      rec.first_seen_gt_count = static_cast<uint16_t>(first_seen_gt_count);
      rec.dropped_gt_count = static_cast<uint16_t>(dropped_gt_count);
      rec.current_hit_count = static_cast<uint16_t>(current_hit_count);
      rec.cumulative_seen_gt_count = static_cast<uint16_t>(cumulative_seen_gt_count);
      rec.topk_recall = static_cast<float>(current_hit_count) / static_cast<float>(topk);
    }
    counts[query_id] = pos + 1;
  }
  __syncthreads();
#else
  (void)records;
  (void)counts;
  (void)max_records_per_query;
  (void)query_id;
  (void)step;
  (void)stage;
  (void)result_index;
  (void)result_size;
  (void)beam_index;
  (void)beam_size;
  (void)topk;
  (void)groundtruth;
  (void)gt_k;
  (void)previous_gt_masks;
  (void)seen_gt_masks;
  (void)current_gt_masks;
  (void)mask_words;
#endif
}

static __device__ inline void record_gpu_query_expander_efficiency(
    double* query_l2_progress_sum,
    double* query_l2_progress_per_expander_sum,
    gpu_long_t* query_expander_steps_sum,
    gpu_long_t* query_expander_count,
    double* query_first_expander_l2_by_query,
    gpu_long_t* query_expander_steps_by_query,
    int query_id,
    int expander_steps,
    double first_expander_dist,
    double best_expander_dist) {
#if GPU_EXPANDER_TRACE
  if (query_l2_progress_sum == nullptr ||
      query_l2_progress_per_expander_sum == nullptr ||
      query_expander_steps_sum == nullptr || query_expander_count == nullptr ||
      expander_steps <= 0) {
    return;
  }
  const double first_l2 = sqrt(fmax(0.0, first_expander_dist));
  const double best_l2 = sqrt(fmax(0.0, best_expander_dist));
  const double progress = fmax(0.0, first_l2 - best_l2);
  if (query_first_expander_l2_by_query != nullptr) {
    query_first_expander_l2_by_query[query_id] = first_l2;
  }
  if (query_expander_steps_by_query != nullptr) {
    query_expander_steps_by_query[query_id] = static_cast<gpu_long_t>(expander_steps);
  }
  atomicAdd(query_l2_progress_sum, progress);
  atomicAdd(query_l2_progress_per_expander_sum,
            progress / static_cast<double>(expander_steps));
  atomicAdd(query_expander_steps_sum, static_cast<gpu_long_t>(expander_steps));
  atomicAdd(query_expander_count, static_cast<gpu_long_t>(1));
#else
  (void)query_l2_progress_sum;
  (void)query_l2_progress_per_expander_sum;
  (void)query_expander_steps_sum;
  (void)query_expander_count;
  (void)query_first_expander_l2_by_query;
  (void)query_expander_steps_by_query;
  (void)query_id;
  (void)expander_steps;
  (void)first_expander_dist;
  (void)best_expander_dist;
#endif
}

struct QueryFactors {
  float* rotated_query = nullptr;
  uint8_t* quantized_query = nullptr;
  uint8_t* lut = nullptr;
  float low_val = 0.0f;
  float high_val = 0.0f;
  float width = 0.0f;
  int32_t sum_q = 0;
};

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);              \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                           \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if ( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
    } }

template <typename T>
class GraphGPU {
protected:
  vidType nv;           // number of vertices
  vidType maxDeg;       // maximun degree
  vidType *d_edges;
public:
  GraphGPU(int n, int d, vidType *h_edges) : nv(n), maxDeg(d) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges, n * d * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, n * d * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  inline __device__ __host__ vidType V() { return nv; }
  inline __device__ __host__ vidType get_max_degree() { return maxDeg; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_edges + vid * maxDeg; }
};

template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T* a, const T* b) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T val = 0.;
  for(int i = thread_lane; i < dim; i += WARP_SIZE)
    val += (a[i] - b[i]) * (a[i] - b[i]);
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum  = SHFL(sum, 0);
  return sum;
}

template <typename T = float>
__device__ __forceinline__ T warp_ip_distance(int dim, const T* a, const T* b) {
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  T val = 0.;
  for (int i = thread_lane; i < dim; i += WARP_SIZE)
    val += a[i] * b[i];
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum = SHFL(sum, 0);
  return -sum;
}

template <typename T = float>
__device__ __forceinline__ T warp_distance(int dim, const T* a, const T* b, bool use_ip) {
  return use_ip ? warp_ip_distance(dim, a, b) : warp_l2_distance(dim, a, b);
}

/*-------------------------------------------- sort --------------------------------------------*/
template <typename T>
__device__ inline void print_buffer(T* buffer, uint32_t buffer_size, uint32_t index_mask=0x80000000) {
    if constexpr (std::is_same<T, INDEX_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%d, ", buffer[i] & ~index_mask);
        }
    } else if constexpr (std::is_same<T, DISTANCE_T>::value || std::is_same<T, DATA_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%f, ", buffer[i]);
        }
    } else {
        printf("Unsupported type\n");
        assert(false);
    }
    printf("\n");
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc) {
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc) {
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0;
    k0               = k1;
    k1               = tmp_k;
    const auto tmp_v = v0;
    v0               = v1;
    v1               = tmp_v;
  }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc) {
  if (_N == 2) {
    constexpr unsigned N = 2;
    const auto lane_id   = threadIdx.x % warp_size;

    if (range == 1) {
      const auto p = ((lane_id & 1) == 0);
      swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
      return;
    }

    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
      }
    }
    const auto p = ((lane_id & b) == 0);
    swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
  } else if (_N == 4) {
    constexpr unsigned N = 4;
    const auto lane_id = threadIdx.x % warp_size;
    if (range == 1) {
      for (std::uint32_t b = 2; b <= N; b <<= 1) {
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
          #pragma unroll
          for (std::uint32_t i = 0; i < N; i++) {
            std::uint32_t j = i ^ c;
            if (i >= j) continue;
            const auto line_id = i + (N * lane_id);
            const auto p       = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
            swap_if_needed(k[i], v[i], k[j], v[j], p);
          }
        }
      }
      return;
    }

    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed(k[i], v[i], c, p);
      }
    }
    const auto p = ((lane_id & b) == 0);
    for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
    #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        std::uint32_t j = i ^ c;
        if (i >= j) continue;
        swap_if_needed(k[i], v[i], k[j], v[j], p);
      }
    }
  }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
  warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // sort 1
  if (warp_id > 0) return;
  if (CANDIDATE_BUFFER_SIZE > 64) {
    printf("CANDIDATE_BUFFER_SIZE must be <= 64\n");
    assert(false);
  }
  // constexpr unsigned N_1 = 2;
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];

  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort_inverse(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // sort 1
  if (warp_id > 0) { return; }
  if (CANDIDATE_BUFFER_SIZE > 64) {
    printf("CANDIDATE_BUFFER_SIZE must be <= 64\n");
    assert(false);
  }
  // constexpr unsigned N_1 = 2;
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];

  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
    if (j < CANDIDATE_BUFFER_SIZE) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_128_by_bitonic_sort(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // Sort part 1
  if (warp_id > 0) return;
  assert (CANDIDATE_BUFFER_SIZE <= 64);

  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];
  auto candidate_distances = result_distances_ptr + internal_topk;
  auto candidate_indices = result_indices_ptr + internal_topk;
  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE && j < internal_topk) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }

  // Sort part 2
  // constexpr unsigned N_2 = 4;
  DISTANCE_T key_2[N_2];
  INDEX_T val_2[N_2];
  if (first) {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < internal_topk) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = MAX_INDEX;
      }
    }
    /* Warp Sort */
    warp_sort<float, uint32_t, N_2>(key_2, val_2);
  } else {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      if (j < internal_topk) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = MAX_INDEX;
      }
    }
  }

  /* Merge candidates */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;  // [0:MAX_ITOPK-1]
    unsigned k = internal_topk - 1 - j;
    if (k >= internal_topk || k >= CANDIDATE_BUFFER_SIZE) continue;
    auto candidate_key = candidate_distances[k];
    if (key_2[i] > candidate_key) {
      key_2[i] = candidate_key;
      val_2[i] = candidate_indices[k];
    }
  }
  /* Warp Merge */
  warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

  /* Store new itopk results */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;
    if (j < internal_topk) {
      result_distances_ptr[j] = key_2[i];
      result_indices_ptr[j]   = val_2[i];
    }
  }

  (void)RESULT_BUFFER_SIZE;
  (void)first;
}

__device__ inline int merge_path_partition(
    const DISTANCE_T* a,
    int a_count,
    const DISTANCE_T* b,
    int b_count,
    int diag) {
  int low = max(0, diag - b_count);
  int high = min(diag, a_count);

  while (low <= high) {
    const int a_idx = (low + high) >> 1;
    const int b_idx = diag - a_idx;

    const DISTANCE_T a_left = (a_idx > 0) ? a[a_idx - 1] : -FLT_MAX;
    const DISTANCE_T a_right = (a_idx < a_count) ? a[a_idx] : FLT_MAX;
    const DISTANCE_T b_left = (b_idx > 0) ? b[b_idx - 1] : -FLT_MAX;
    const DISTANCE_T b_right = (b_idx < b_count) ? b[b_idx] : FLT_MAX;

    if (a_left > b_right) {
      high = a_idx - 1;
    } else if (b_left > a_right) {
      low = a_idx + 1;
    } else {
      return a_idx;
    }
  }

  return low;
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_candidate_sort_and_merge(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    INDEX_T* merged_topk_index_shared,
    DISTANCE_T* merged_topk_dist_shared,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  auto candidate_indices = result_indices_ptr + internal_topk;
  auto candidate_distances = result_distances_ptr + internal_topk;

  candidate_by_bitonic_sort<N_1, 0>(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE);
  __syncthreads();

  constexpr int kItemsPerThread = 4;
  int active_threads = (internal_topk + kItemsPerThread - 1) / kItemsPerThread;
  active_threads = min(static_cast<int>(blockDim.x),
                       ((active_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);

  if (threadIdx.x < active_threads) {
    const int out_begin =
        min(internal_topk, (static_cast<int>(threadIdx.x) * internal_topk) / active_threads);
    const int out_end =
        min(internal_topk, (static_cast<int>(threadIdx.x + 1) * internal_topk) / active_threads);

    const int top_begin = merge_path_partition(
        result_distances_ptr, static_cast<int>(internal_topk),
        candidate_distances, static_cast<int>(CANDIDATE_BUFFER_SIZE),
        out_begin);
    const int top_end = merge_path_partition(
        result_distances_ptr, static_cast<int>(internal_topk),
        candidate_distances, static_cast<int>(CANDIDATE_BUFFER_SIZE),
        out_end);

    int top_i = top_begin;
    int cand_i = out_begin - top_begin;
    const int cand_end = out_end - top_end;

    for (int out_i = out_begin; out_i < out_end; ++out_i) {
      const bool take_topk =
          (cand_i >= cand_end) ||
          (top_i < top_end &&
           result_distances_ptr[top_i] <= candidate_distances[cand_i]);

      if (take_topk) {
        merged_topk_dist_shared[out_i] = result_distances_ptr[top_i];
        merged_topk_index_shared[out_i] = result_indices_ptr[top_i];
        ++top_i;
      } else {
        merged_topk_dist_shared[out_i] = candidate_distances[cand_i];
        merged_topk_index_shared[out_i] = candidate_indices[cand_i];
        ++cand_i;
      }
    }
  }
  __syncthreads();

  for (unsigned i = threadIdx.x; i < internal_topk; i += blockDim.x) {
    result_distances_ptr[i] = merged_topk_dist_shared[i];
    result_indices_ptr[i] = merged_topk_index_shared[i];
  }
  __syncthreads();

  (void)RESULT_BUFFER_SIZE;
  (void)first;
}

template <unsigned kPerThread>
__device__ __forceinline__ void merge_tail_candidates(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    unsigned lane_id) {
  DISTANCE_T merge_keys[kPerThread];
  INDEX_T merge_vals[kPerThread];
  const uint32_t tail_begin = internal_topk - CANDIDATE_BUFFER_SIZE;

  #pragma unroll
  for (unsigned i = 0; i < kPerThread; ++i) {
    const uint32_t j = lane_id + (WARP_SIZE * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      merge_keys[i] = result_distances_ptr[tail_begin + j];
      merge_vals[i] = result_indices_ptr[tail_begin + j];
    } else {
      const uint32_t cj = j - CANDIDATE_BUFFER_SIZE;
      merge_keys[i] = candidate_distances[cj];
      merge_vals[i] = candidate_indices[cj];
    }
  }

  warp_sort<float, uint32_t, kPerThread>(merge_keys, merge_vals);

  #pragma unroll
  for (unsigned i = 0; i < kPerThread; ++i) {
    const uint32_t j = (kPerThread * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE) {
      result_distances_ptr[tail_begin + j] = merge_keys[i];
      result_indices_ptr[tail_begin + j] = merge_vals[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_tail_compare_radix_sort(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk) {
  // step 1: sort the candidate buffer by bitonic sort
  auto candidate_indices = result_indices_ptr + internal_topk;
  auto candidate_distances = result_distances_ptr + internal_topk;

  candidate_by_bitonic_sort<N_1, 0>(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE);
  __syncthreads();

  // step 2: load the tail of current TOPK, size = CANDIDATE_BUFFER_SIZE, to registers
  // then do a bitonic sort on the tail of TOPK and current candidates
  // final result: unsorted TOPK with the worst part replaced by candidates
  const unsigned lane_id = threadIdx.x % WARP_SIZE;
  const unsigned warp_id = threadIdx.x / WARP_SIZE;
  if (warp_id == 0) {
    assert(internal_topk >= CANDIDATE_BUFFER_SIZE);
    if (CANDIDATE_BUFFER_SIZE == 32) {
      merge_tail_candidates<2>(
          result_indices_ptr, result_distances_ptr,
          candidate_indices, candidate_distances,
          CANDIDATE_BUFFER_SIZE, internal_topk, lane_id);
    } else {
      assert(CANDIDATE_BUFFER_SIZE == 64);
      merge_tail_candidates<4>(
          result_indices_ptr, result_distances_ptr,
          candidate_indices, candidate_distances,
          CANDIDATE_BUFFER_SIZE, internal_topk, lane_id);
    }
  }
  __syncthreads();

  // step 3: call external library cub to do radix sort on the TOPK
  constexpr unsigned TopKPerThread = (N_2 * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using TopkRadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, TopKPerThread, INDEX_T>;
  __shared__ typename TopkRadixSort::TempStorage radix_sort_storage;

  DISTANCE_T topk_dist[TopKPerThread];
  INDEX_T topk_idx[TopKPerThread];
  for (unsigned i = 0; i < TopKPerThread; ++i) {
    const uint32_t j = (threadIdx.x * TopKPerThread) + i;
    if (j < internal_topk) {
      topk_dist[i] = result_distances_ptr[j];
      topk_idx[i] = result_indices_ptr[j];
    } else {
      topk_dist[i] = FLT_MAX;
      topk_idx[i] = MAX_INDEX;
    }
  }
  TopkRadixSort(radix_sort_storage).Sort(topk_dist, topk_idx);
  for (unsigned i = 0; i < TopKPerThread; ++i) {
    const uint32_t j = (threadIdx.x * TopKPerThread) + i;
    if (j < internal_topk) {
      result_distances_ptr[j] = topk_dist[i];
      result_indices_ptr[j] = topk_idx[i];
    }
  }
  __syncthreads();

  (void)RESULT_BUFFER_SIZE;
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_sort_candidates_and_merge(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    INDEX_T* merged_topk_index_shared,
    DISTANCE_T* merged_topk_dist_shared,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  if constexpr (N_2 == 2 || N_2 == 4) {
    topk_128_by_bitonic_sort<N_1, N_2>(
        result_indices_ptr, result_distances_ptr, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, internal_topk, first);
  } else {
    topk_by_candidate_sort_and_merge<N_1, N_2>(
        result_indices_ptr, result_distances_ptr,
        merged_topk_index_shared, merged_topk_dist_shared, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, internal_topk, first);
    // tok_by_tail_compare_radix_sort<N_1, N_2>(
    //     result_indices_ptr, result_distances_ptr, RESULT_BUFFER_SIZE,
    //     CANDIDATE_BUFFER_SIZE, internal_topk);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_sort_candidates_and_merge(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  if constexpr (N_2 == 2 || N_2 == 4) {
    topk_by_sort_candidates_and_merge<N_1, N_2>(
        result_indices_ptr, result_distances_ptr, nullptr, nullptr,
        RESULT_BUFFER_SIZE, CANDIDATE_BUFFER_SIZE, internal_topk, first);
  } else {
    __shared__ DISTANCE_T merged_topk_dist_shared[N_2 * WARP_SIZE];
    __shared__ INDEX_T merged_topk_index_shared[N_2 * WARP_SIZE];
    topk_by_sort_candidates_and_merge<N_1, N_2>(
        result_indices_ptr, result_distances_ptr,
        merged_topk_index_shared, merged_topk_dist_shared,
        RESULT_BUFFER_SIZE, CANDIDATE_BUFFER_SIZE, internal_topk, first);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ __noinline__ void topk_by_sort_candidates_and_merge_case(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    INDEX_T* merged_topk_index_shared,
    DISTANCE_T* merged_topk_dist_shared,
    uint32_t result_buffer_size,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
  topk_by_sort_candidates_and_merge<N_1, N_2>(
      result_indices_ptr, result_distances_ptr,
      merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
      candidate_buffer_size, internal_topk, first);
}

template <unsigned N_1>
__device__ void dispatch_topk_by_sort_candidates_and_merge(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    INDEX_T* merged_topk_index_shared,
    DISTANCE_T* merged_topk_dist_shared,
    uint32_t result_buffer_size,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
  switch ((internal_topk + WARP_SIZE - 1) / WARP_SIZE) {
    case 1:
      topk_by_sort_candidates_and_merge_case<N_1, 1>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 2:
      topk_by_sort_candidates_and_merge_case<N_1, 2>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 3:
      topk_by_sort_candidates_and_merge_case<N_1, 3>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 4:
      topk_by_sort_candidates_and_merge_case<N_1, 4>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 5:
      topk_by_sort_candidates_and_merge_case<N_1, 5>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 6:
      topk_by_sort_candidates_and_merge_case<N_1, 6>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 7:
      topk_by_sort_candidates_and_merge_case<N_1, 7>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 8:
      topk_by_sort_candidates_and_merge_case<N_1, 8>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 9:
      topk_by_sort_candidates_and_merge_case<N_1, 9>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 10:
      topk_by_sort_candidates_and_merge_case<N_1, 10>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 11:
      topk_by_sort_candidates_and_merge_case<N_1, 11>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 12:
      topk_by_sort_candidates_and_merge_case<N_1, 12>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 13:
      topk_by_sort_candidates_and_merge_case<N_1, 13>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 14:
      topk_by_sort_candidates_and_merge_case<N_1, 14>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 15:
      topk_by_sort_candidates_and_merge_case<N_1, 15>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 16:
      topk_by_sort_candidates_and_merge_case<N_1, 16>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 17:
      topk_by_sort_candidates_and_merge_case<N_1, 17>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 18:
      topk_by_sort_candidates_and_merge_case<N_1, 18>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 19:
      topk_by_sort_candidates_and_merge_case<N_1, 19>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 20:
      topk_by_sort_candidates_and_merge_case<N_1, 20>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 21:
      topk_by_sort_candidates_and_merge_case<N_1, 21>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 22:
      topk_by_sort_candidates_and_merge_case<N_1, 22>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 23:
      topk_by_sort_candidates_and_merge_case<N_1, 23>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 24:
      topk_by_sort_candidates_and_merge_case<N_1, 24>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 25:
      topk_by_sort_candidates_and_merge_case<N_1, 25>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 26:
      topk_by_sort_candidates_and_merge_case<N_1, 26>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 27:
      topk_by_sort_candidates_and_merge_case<N_1, 27>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 28:
      topk_by_sort_candidates_and_merge_case<N_1, 28>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 29:
      topk_by_sort_candidates_and_merge_case<N_1, 29>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 30:
      topk_by_sort_candidates_and_merge_case<N_1, 30>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 31:
      topk_by_sort_candidates_and_merge_case<N_1, 31>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    case 32:
      topk_by_sort_candidates_and_merge_case<N_1, 32>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
    default:
      topk_by_sort_candidates_and_merge_case<N_1, 4>(
          result_indices_ptr, result_distances_ptr,
          merged_topk_index_shared, merged_topk_dist_shared, result_buffer_size,
          candidate_buffer_size, internal_topk, first);
      break;
  }
}

#define SMALL_HASH_RESET_INTERVAL 16

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
  return 1 << bitlen;
}

__device__ inline void hashtable_init(INDEX_T* const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
  if (threadIdx.x < FIRST_TID) return;
  for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = MAX_INDEX;
  }
}

__device__ inline uint32_t hashtable_insert(INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    // Open addressing is used for collision resolution
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    // Linear probing
    INDEX_T index             = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = atomicCAS(&table[index], ~static_cast<INDEX_T>(0), key);
        if (old == ~static_cast<INDEX_T>(0)) {
            return 1;
        } else if (old == key) {
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
  return 0;
}

__device__ inline uint32_t hashtable_contains(INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    INDEX_T index             = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    // only check if the slot is filled, no actual insertion
    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = table[index];
        if (old == key) return 1;
        if (old == ~static_cast<INDEX_T>(0)) return 0;
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ inline void hashtable_restore(
    INDEX_T* const table,
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashtable_insert(table, BITLEN, key);
  }
}

__device__ inline void hashtable_restore_expand_only(
    INDEX_T* const table,
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    const INDEX_T raw = itopk_indices[i];
    // only restore nodes that are expanded already (MSB is 0)
    if ((raw & index_msb_1_mask) == 0 || raw == MAX_INDEX) continue;
    const INDEX_T key = raw & ~index_msb_1_mask;
    hashtable_insert(table, BITLEN, key);
  }
}

// select top-N un-expanded nodes in the priority queue of size M
static __device__ uint32_t pick_expanders(int N, INDEX_T* output_list, int M, INDEX_T* pri_queue) {
  uint32_t num_exp = 0; // number of expanders actually seleceted; must be <= N
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  uint32_t itopk_max = M;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }   // round up to be multiple of 32
  for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    INDEX_T index;
    int new_parent = 0;
    if (j < M) {    // within the priority queue bound
      index = pri_queue[j];
      if ((index & index_msb_1_mask) == 0) {   // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_exp;
      if (i < N) {
        output_list[i] = j;
        //if (blockIdx.x == 0) printf("pick %d-th node %d as the %d-th parent\n", j, pri_queue[j], i);
        // set most significant bit as used node
        pri_queue[j] |= index_msb_1_mask;
      }
    }
    num_exp += __popc(ballot_mask);
    if (num_exp >= N) { break; }
  }
  return num_exp;
}

static __device__ inline void result_pool_recompute_worst(
    const DISTANCE_T* pool_distance,
    uint32_t pool_size,
    uint32_t* worst_idx,
    DISTANCE_T* worst_dist) {
  if (pool_size == 0) {
    *worst_idx = 0;
    *worst_dist = -FLT_MAX;
    return;
  }
  uint32_t local_worst_idx = 0;
  DISTANCE_T local_worst_dist = pool_distance[0];
  for (uint32_t i = 1; i < pool_size; ++i) {
    if (pool_distance[i] > local_worst_dist) {
      local_worst_dist = pool_distance[i];
      local_worst_idx = i;
    }
  }
  *worst_idx = local_worst_idx;
  *worst_dist = local_worst_dist;
}

static __device__ inline void result_pool_push_unique_unsorted(
    INDEX_T* pool_index,
    DISTANCE_T* pool_distance,
    uint32_t* pool_size,
    uint32_t* worst_idx,
    DISTANCE_T* worst_dist,
    int capacity,
    INDEX_T node,
    DISTANCE_T distance) {
  uint32_t size = *pool_size;

  if (size < static_cast<uint32_t>(capacity)) {
    pool_index[size] = node;
    pool_distance[size] = distance;
    *pool_size = size + 1;
    if (size == 0 || distance > *worst_dist) {
      *worst_idx = size;
      *worst_dist = distance;
    }
    return;
  }

  if (distance >= *worst_dist) return;

  pool_index[*worst_idx] = node;
  pool_distance[*worst_idx] = distance;
  result_pool_recompute_worst(pool_distance, size, worst_idx, worst_dist);
}

static __device__ inline void result_pool_sort_small(
    INDEX_T* pool_index,
    DISTANCE_T* pool_distance,
    uint32_t pool_size) {
  for (uint32_t i = 1; i < pool_size; ++i) {
    INDEX_T key_index = pool_index[i];
    DISTANCE_T key_dist = pool_distance[i];
    int j = static_cast<int>(i) - 1;
    while (j >= 0 && pool_distance[j] > key_dist) {
      pool_distance[j + 1] = pool_distance[j];
      pool_index[j + 1] = pool_index[j];
      --j;
    }
    pool_distance[j + 1] = key_dist;
    pool_index[j + 1] = key_index;
  }
}

/*-------------------------------------------- rabitq --------------------------------------------*/
static __device__ inline void rotate_vector_gpu(const float* src, float* dst,
   const float* signs, int dim, int padded_dim) {

  int tid = threadIdx.x;

  for (size_t i = tid; i < dim; i += blockDim.x) {
    dst[i] = src[i] * signs[i];
  }
  for (int i = dim + tid; i < padded_dim; i += blockDim.x) {
    dst[i] = 0.0f;
  }
  __syncthreads();

  // Fast Walsh-Hadamard Transform (in-place, power-of-2 B, B >= 8)
  for (size_t len = 1; len < padded_dim; len <<= 1) {
    int pairs = padded_dim >> 1;
    for (int pair_idx = tid; pair_idx < pairs; pair_idx += blockDim.x) {
      int block = pair_idx / len;
      int j = pair_idx % len;
      int base = block * (len << 1);

      int idx1 = base + j;
      int idx2 = idx1 + len;

      float u = dst[idx1];
      float v = dst[idx2];
      dst[idx1] = u + v;
      dst[idx2] = u - v;
    }
    __syncthreads();
  }
}

static __device__ void query_prepare_gpu(const float* query_raw, QueryFactors& scratch,
    const float* signs_ptr, int dim, int padded_dim) {
  constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  // rotate the query
  rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
  __syncthreads();

  __shared__ float warp_min[WARPS_PER_BLOCK];
  __shared__ float warp_max[WARPS_PER_BLOCK];
  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  // data range: find the lowest and highest dimension in the rotated query
  for (size_t i = tid; i < padded_dim; i += blockDim.x) {
      float tmp = scratch.rotated_query[i];
      local_min = fminf(local_min, tmp);
      local_max = fmaxf(local_max, tmp);
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
    local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
  }

  if (lane_id == 0) {
    warp_min[warp_id] = local_min;
    warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
    local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
      local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
      scratch.low_val = local_min;
      scratch.high_val = local_max;
      const float query_span = scratch.high_val - scratch.low_val;
      scratch.width = query_span * kQueryLevelsInv;
    }
  }
  __syncthreads();

  // quantize the query
  const float inv_width = 1.0f / scratch.width;

  __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
  int32_t local_sum = 0;
  for (size_t i = tid; i < padded_dim; i += blockDim.x) {
      const float scaled = ((scratch.rotated_query[i] - scratch.low_val) * inv_width) + 0.5f;
      uint8_t cur = static_cast<uint8_t>(lroundf(scaled));
      scratch.quantized_query[i] = cur;
      local_sum += cur;
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
  }

  if (lane_id == 0) {
    warp_sum[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
      scratch.sum_q = local_sum;
    }
  }
  __syncthreads();
}

static __device__ inline DISTANCE_T estimate_one_neighbor_warp_no_lut_gpu(const QueryFactors& qf,
  const uint64_t* packed_code, float triple_x, float factor_dq, float factor_vq,
  float exact_dist, int padded_dim_) {

  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;

  if (triple_x == FLT_MAX) return 0.0f;

  int32_t local_sum = 0;
  for (int word = 0; word < padded_dim_ / 64; ++word) {
      uint64_t bits = packed_code[word];
      int base = word * 64;

      int d0 = base + lane_id;
      if ((bits >> lane_id) & 1ULL)
          local_sum += static_cast<int32_t>(qf.quantized_query[d0]);

      int d1 = d0 + 32;
      if (d1 < padded_dim_ && ((bits >> (lane_id + 32)) & 1ULL))
          local_sum += static_cast<int32_t>(qf.quantized_query[d1]);
  }

  // reduce the sum
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  float result_float = static_cast<float>((local_sum << 1) - qf.sum_q);
  DISTANCE_T est_dist = triple_x + exact_dist;
  est_dist += factor_dq * qf.width * result_float;
  est_dist += factor_vq * qf.low_val;
  return est_dist;
}


static __device__ __forceinline__ int fastscan_lane_pack_index(int lane_mod16) {
  return ((lane_mod16 & 7) << 1) | (lane_mod16 >> 3);
}

static __device__ __forceinline__ uint8_t fastscan_decode_code_gpu(
    const uint8_t* packed_codes_block, int lane_id, int codebook_idx) {
  const int pair_idx = codebook_idx >> 1;
  const int nibble_group = codebook_idx & 1;
  const int lane_mod16 = lane_id & 15;
  const int byte_idx = fastscan_lane_pack_index(lane_mod16);
  const uint8_t packed =
      packed_codes_block[pair_idx * 32 + nibble_group * 16 + byte_idx];
  return (lane_id < 16) ? static_cast<uint8_t>(packed & 0x0f)
                        : static_cast<uint8_t>(packed >> 4);
}

static __device__ __forceinline__ uint8_t fastscan_decode_code_for_neighbor_gpu(
    const uint8_t* packed_codes_block, int neighbor_idx, int codebook_idx) {
  const int pair_idx = codebook_idx >> 1;
  const int nibble_group = codebook_idx & 1;
  const int neighbor_mod16 = neighbor_idx & 15;
  const int byte_idx = fastscan_lane_pack_index(neighbor_mod16);
  const uint8_t packed =
      packed_codes_block[pair_idx * 32 + nibble_group * 16 + byte_idx];
  return (neighbor_idx < 16) ? static_cast<uint8_t>(packed & 0x0f)
                             : static_cast<uint8_t>(packed >> 4);
}

template <int LUT_LAYOUT_LANES = GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES>
static __device__ __forceinline__ uint8_t fastscan_decode_code_for_neighbor_seq_lut_gpu(
    const uint8_t* code_base, int codebook_idx, int neighbor_in_tile = 0) {
  static_assert(LUT_LAYOUT_LANES == 2 || LUT_LAYOUT_LANES == 4 || LUT_LAYOUT_LANES == 8 || LUT_LAYOUT_LANES == 16 || LUT_LAYOUT_LANES == 32,
                "GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES must be 2, 4, 8, 16, or 32");
  const int pair_idx = codebook_idx >> 1;
  uint8_t packed;
  if constexpr (LUT_LAYOUT_LANES == 32) {
    packed = code_base[pair_idx];
  } else {
    constexpr int neighbors_per_tile = WARP_SIZE / LUT_LAYOUT_LANES;
    packed = code_base[pair_idx * neighbors_per_tile + neighbor_in_tile];
  }
  return (codebook_idx & 1) ? static_cast<uint8_t>(packed >> 4)
                            : static_cast<uint8_t>(packed & 0x0f);
}

static __device__ inline void pack_lut_gpu(
    int dim, const uint8_t* __restrict__ quantized_query, uint8_t* __restrict__ lut) {
  const int tid = threadIdx.x;
  const int num_codebook = dim >> 2;

  for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
    const uint8_t* query_chunk = quantized_query + (cb << 2);
    uint8_t* lut_chunk = lut + (cb << 4);

    lut_chunk[0] = 0;
    for (int mask = 1; mask < 16; ++mask) {
      const int lowbit = mask & -mask;
      const int pos = (lowbit == 8) ? 0 : ((lowbit == 4) ? 1 : ((lowbit == 2) ? 2 : 3));
      lut_chunk[mask] = static_cast<uint8_t>(lut_chunk[mask - lowbit] + query_chunk[pos]);
    }
  }
}

static __device__ inline void scan_all_neighbors_32_gpu(
    const QueryFactors& qf, const uint8_t* packed_codes_block,
    const float* triple_x, const float* factor_dq, const float* factor_vq,
    float exact_dist, int padded_dim, float* appro_dist, int active_neighbors = WARP_SIZE) {
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  if (lane_id >= active_neighbors) return;
  if (triple_x[lane_id] == FLT_MAX) {
    appro_dist[lane_id] = 0.0f;
    return;
  }

  const int num_codebook = padded_dim >> 2;
  uint32_t raw_sum = 0;

  for (int cb = 0; cb < num_codebook; ++cb) {
    const uint8_t code = fastscan_decode_code_gpu(packed_codes_block, lane_id, cb);
    raw_sum += static_cast<uint32_t>(qf.lut[(cb << 4) + code]);
  }

  const float result_float = static_cast<float>((static_cast<int32_t>(raw_sum) << 1) - qf.sum_q);
  float est_dist = triple_x[lane_id] + exact_dist;
  est_dist += factor_dq[lane_id] * qf.width * result_float;
  est_dist += factor_vq[lane_id] * qf.low_val;
  appro_dist[lane_id] = est_dist;
}

template <int LANES_PER_NEIGHBOR>
static __device__ inline DISTANCE_T scan_one_neighbor_lanes_gpu(
    const QueryFactors &qf, const uint8_t *packed_codes_block, int neighbor_idx,
    const float *triple_x, const float *factor_dq, const float *factor_vq,
    float exact_dist, int padded_dim, int bytes_per_neighbor) {
  static_assert(LANES_PER_NEIGHBOR == 2 || LANES_PER_NEIGHBOR == 4 ||
                    LANES_PER_NEIGHBOR == 8 || LANES_PER_NEIGHBOR == 16 ||
                    LANES_PER_NEIGHBOR == 32,
                "LANES_PER_NEIGHBOR must be one of 2, 4, 8, 16, or 32");
  const float triple_x_value = triple_x[0];
  if (triple_x_value == FLT_MAX)
    return 0.0f;

  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  const int group_lane = lane_id & (LANES_PER_NEIGHBOR - 1);
  unsigned group_mask = FULL_MASK;
  if constexpr (LANES_PER_NEIGHBOR != WARP_SIZE) {
    const int group_start = lane_id - group_lane;
    constexpr unsigned group_bits = (1u << LANES_PER_NEIGHBOR) - 1u;
    group_mask = group_bits << group_start;
  }
  const int num_codebook = padded_dim >> 2;
  uint32_t local_raw_sum = 0;

  constexpr int lut_layout_neighbors = WARP_SIZE / GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES;
  const uint8_t *seq_lut_tile = packed_codes_block;
  int seq_lut_neighbor_in_tile = 0;
  if constexpr (GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES == 32) {
    seq_lut_tile = packed_codes_block + neighbor_idx * bytes_per_neighbor;
  } else {
    const int tile_idx = neighbor_idx / lut_layout_neighbors;
    seq_lut_neighbor_in_tile = neighbor_idx & (lut_layout_neighbors - 1);
    seq_lut_tile = packed_codes_block + tile_idx * lut_layout_neighbors * bytes_per_neighbor;
  }

  for (int cb = group_lane; cb < num_codebook; cb += LANES_PER_NEIGHBOR) {
    const uint8_t code = fastscan_decode_code_for_neighbor_seq_lut_gpu(
        seq_lut_tile, cb, seq_lut_neighbor_in_tile);
    local_raw_sum += static_cast<uint32_t>(qf.lut[(cb << 4) + code]);
  }

  for (int offset = LANES_PER_NEIGHBOR / 2; offset > 0; offset >>= 1) {
    local_raw_sum += __shfl_down_sync(group_mask, local_raw_sum, offset, LANES_PER_NEIGHBOR);
  }

  if (group_lane != 0) return 0.0f;

  const float result_float = static_cast<float>((static_cast<int32_t>(local_raw_sum) << 1) - qf.sum_q);
  DISTANCE_T est_dist = triple_x_value + exact_dist;
  est_dist += factor_dq[0] * qf.width * result_float;
  est_dist += factor_vq[0] * qf.low_val;
  return est_dist;
}
