//#include <cuda_profiler_api.h>
//#define USE_RADIX // Use bitonic if not defined

#include "include/index.hpp"
#ifdef USE_RADIX
#include "include/search_radix.cuh"
#else
#include "include/search_bitonic.cuh"
#endif

#include <chrono>
#include <array>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace {

bool trace_file_requested(const char* path) {
  return path != nullptr && path[0] != '\0';
}

void dump_gpu_expander_trace(const char* path,
                             int trace_max_steps,
                             double* d_trace_sums,
                             gpu_long_t* d_trace_counts,
                             double* d_trace_net_descent,
                             double* d_query_l2_progress_sum,
                             double* d_query_l2_progress_per_expander_sum,
                             gpu_long_t* d_query_expander_steps_sum,
                             gpu_long_t* d_query_expander_count,
                             double* d_query_first_expander_l2_by_query,
                             gpu_long_t* d_query_expander_steps_by_query,
                             int nq,
                             GpuRunStats* gpu_stats) {
  if (!trace_file_requested(path)) return;

  std::vector<double> h_sums(static_cast<size_t>(trace_max_steps) * GPU_EXPANDER_TRACE_FIELDS);
  std::vector<gpu_long_t> h_counts(trace_max_steps);
  double h_net_descent = 0.0;
  double h_query_l2_progress_sum = 0.0;
  double h_query_l2_progress_per_expander_sum = 0.0;
  gpu_long_t h_query_expander_steps_sum = 0;
  gpu_long_t h_query_expander_count = 0;
  std::vector<double> h_query_first_expander_l2(static_cast<size_t>(nq), 0.0);
  std::vector<gpu_long_t> h_query_expander_steps(static_cast<size_t>(nq), 0);
  CUDA_SAFE_CALL(cudaMemcpy(h_sums.data(), d_trace_sums,
                            h_sums.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_counts.data(), d_trace_counts,
                            h_counts.size() * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_net_descent, d_trace_net_descent,
                            sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_query_l2_progress_sum, d_query_l2_progress_sum,
                            sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_query_l2_progress_per_expander_sum,
                            d_query_l2_progress_per_expander_sum,
                            sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_query_expander_steps_sum, d_query_expander_steps_sum,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_query_expander_count, d_query_expander_count,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_query_first_expander_l2.data(),
                            d_query_first_expander_l2_by_query,
                            h_query_first_expander_l2.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_query_expander_steps.data(),
                            d_query_expander_steps_by_query,
                            h_query_expander_steps.size() * sizeof(gpu_long_t),
                            cudaMemcpyDeviceToHost));

  RunStats trace_stats;
  trace_stats.total_expander_net_descent = h_net_descent;
  trace_stats.total_query_l2_progress = h_query_l2_progress_sum;
  trace_stats.total_query_l2_progress_per_expander = h_query_l2_progress_per_expander_sum;
  trace_stats.total_query_expander_steps = static_cast<int64_t>(h_query_expander_steps_sum);
  trace_stats.total_query_expander_count = static_cast<int64_t>(h_query_expander_count);
  trace_stats.query_first_expander_l2_values = h_query_first_expander_l2;
  trace_stats.query_expander_step_values.assign(h_query_expander_steps.begin(),
                                                h_query_expander_steps.end());
  trace_stats.expander_trace_by_step.resize(trace_max_steps);
  for (int i = 0; i < trace_max_steps; ++i) {
    const auto count = h_counts[i];
    if (count != 0) {
      const size_t base = static_cast<size_t>(i) * GPU_EXPANDER_TRACE_FIELDS;
      auto& step = trace_stats.expander_trace_by_step[i];
      step.expander_dist_sum = h_sums[base + GPU_TRACE_EXPANDER_DIST];
      step.signed_delta_sum = h_sums[base + GPU_TRACE_SIGNED_DELTA];
      step.positive_delta_sum = h_sums[base + GPU_TRACE_POSITIVE_DELTA];
      step.best_expander_dist_sum = h_sums[base + GPU_TRACE_BEST_EXPANDER_DIST];
      step.count = static_cast<size_t>(count);
      trace_stats.total_expander_trace_events += static_cast<int64_t>(count);
      trace_stats.total_expander_positive_delta += step.positive_delta_sum;
    }
  }
  while (!trace_stats.expander_trace_by_step.empty() &&
         trace_stats.expander_trace_by_step.back().count == 0) {
    trace_stats.expander_trace_by_step.pop_back();
  }
  if (gpu_stats != nullptr) {
    gpu_stats->common.total_expander_trace_events = trace_stats.total_expander_trace_events;
    gpu_stats->common.total_expander_positive_delta = trace_stats.total_expander_positive_delta;
    gpu_stats->common.total_expander_net_descent = trace_stats.total_expander_net_descent;
    gpu_stats->common.total_query_expander_steps = trace_stats.total_query_expander_steps;
    gpu_stats->common.total_query_expander_count = trace_stats.total_query_expander_count;
    gpu_stats->common.total_query_l2_progress = trace_stats.total_query_l2_progress;
    gpu_stats->common.total_query_l2_progress_per_expander =
        trace_stats.total_query_l2_progress_per_expander;
    gpu_stats->common.query_first_expander_l2_values =
        trace_stats.query_first_expander_l2_values;
    gpu_stats->common.query_expander_step_values =
        trace_stats.query_expander_step_values;
    gpu_stats->common.expander_trace_by_step = trace_stats.expander_trace_by_step;
  }
  dump_expander_trace_stats(path, trace_stats);
}

void dump_gpu_query_steps(const char* path,
                          const std::vector<gpu_long_t>& h_block_stats,
                          size_t num_blocks) {
  if (!trace_file_requested(path)) return;

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error(std::string("cannot open GPU query steps file: ") + path);
  }
  out << "query_id,expander_steps\n";
  for (size_t q = 0; q < num_blocks; ++q) {
    const size_t base = q * GPU_SEARCH_STAT_COUNT;
    out << q << "," << h_block_stats[base + GPU_SEARCH_STAT_ITERATIONS] << "\n";
  }
}

void dump_gpu_access_trace(const char* path,
                           int nq,
                           int max_records_per_query,
                           AccessTraceRecord* d_records,
                           uint32_t* d_counts) {
  if (!trace_file_requested(path)) return;

  std::vector<uint32_t> h_counts(static_cast<size_t>(nq), 0);
  std::vector<AccessTraceRecord> h_records(
      static_cast<size_t>(nq) * max_records_per_query);
  CUDA_SAFE_CALL(cudaMemcpy(h_counts.data(), d_counts,
                            h_counts.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_records.data(), d_records,
                            h_records.size() * sizeof(AccessTraceRecord),
                            cudaMemcpyDeviceToHost));

  std::vector<AccessTraceRecord> flat;
  size_t dropped = 0;
  for (int q = 0; q < nq; ++q) {
    const uint32_t count = h_counts[static_cast<size_t>(q)];
    const uint32_t kept = std::min<uint32_t>(
        count, static_cast<uint32_t>(max_records_per_query));
    if (count > kept) {
      dropped += static_cast<size_t>(count - kept);
    }
    const size_t base = static_cast<size_t>(q) * max_records_per_query;
    for (uint32_t i = 0; i < kept; ++i) {
      flat.push_back(h_records[base + i]);
    }
  }
  write_access_trace_csv(path, flat);
  std::cout << "GPU access trace records: " << flat.size();
  if (dropped != 0) {
    std::cout << " (dropped " << dropped
              << "; increase GPU_ACCESS_TRACE_MAX_RECORDS_PER_QUERY)";
  }
  std::cout << "\n";
}

void dump_gpu_frontier_audit(const char* path,
                             const char* method,
                             int nq,
                             int max_records_per_query,
                             GpuFrontierAuditRecord* d_records,
                             uint32_t* d_counts) {
  if (!trace_file_requested(path)) return;

  std::vector<uint32_t> h_counts(static_cast<size_t>(nq), 0);
  std::vector<GpuFrontierAuditRecord> h_records(
      static_cast<size_t>(nq) * max_records_per_query);
  CUDA_SAFE_CALL(cudaMemcpy(h_counts.data(), d_counts,
                            h_counts.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_records.data(), d_records,
                            h_records.size() * sizeof(GpuFrontierAuditRecord),
                            cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error(std::string("cannot open GPU frontier audit file: ") + path);
  }
  out << "method,query_id,step,chosen_node,chosen_exact_rank,"
      << "frontier_exact_best_l2,frontier_chosen_exact_l2,frontier_exact_gap\n";
  out << std::setprecision(10);

  size_t kept_total = 0;
  size_t dropped = 0;
  for (int q = 0; q < nq; ++q) {
    const uint32_t count = h_counts[static_cast<size_t>(q)];
    const uint32_t kept = std::min<uint32_t>(
        count, static_cast<uint32_t>(max_records_per_query));
    if (count > kept) {
      dropped += static_cast<size_t>(count - kept);
    }
    const size_t base = static_cast<size_t>(q) * max_records_per_query;
    for (uint32_t i = 0; i < kept; ++i) {
      const auto& r = h_records[base + i];
      out << method << ","
          << r.query_id << ","
          << r.step << ","
          << r.chosen_node << ","
          << r.chosen_exact_rank << ","
          << r.frontier_exact_best_l2 << ","
          << r.frontier_chosen_exact_l2 << ","
          << r.frontier_exact_gap << "\n";
      ++kept_total;
    }
  }

  std::cout << "GPU frontier audit records: " << kept_total;
  if (dropped != 0) {
    std::cout << " (dropped " << dropped
              << "; increase GPU_FRONTIER_AUDIT_MAX_RECORDS_PER_QUERY)";
  }
  std::cout << "\n";
}

void dump_gpu_quality_hit_trace(const char* path,
                                const char* method,
                                int nq,
                                int max_records_per_query,
                                GpuQualityHitTraceRecord* d_records,
                                uint32_t* d_counts) {
  if (!trace_file_requested(path)) return;

  std::vector<uint32_t> h_counts(static_cast<size_t>(nq), 0);
  std::vector<GpuQualityHitTraceRecord> h_records(
      static_cast<size_t>(nq) * max_records_per_query);
  CUDA_SAFE_CALL(cudaMemcpy(h_counts.data(), d_counts,
                            h_counts.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_records.data(), d_records,
                            h_records.size() * sizeof(GpuQualityHitTraceRecord),
                            cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error(std::string("cannot open GPU quality hit trace file: ") + path);
  }
  out << "method,query_id,step,stage,new_gt_count,first_seen_gt_count,"
      << "dropped_gt_count,current_hit_count,cumulative_seen_gt_count,"
      << "topk_recall\n";
  out << std::setprecision(10);

  size_t kept_total = 0;
  size_t dropped = 0;
  for (int q = 0; q < nq; ++q) {
    const uint32_t count = h_counts[static_cast<size_t>(q)];
    const uint32_t kept = std::min<uint32_t>(
        count, static_cast<uint32_t>(max_records_per_query));
    if (count > kept) {
      dropped += static_cast<size_t>(count - kept);
    }
    const size_t base = static_cast<size_t>(q) * max_records_per_query;
    for (uint32_t i = 0; i < kept; ++i) {
      const auto& r = h_records[base + i];
      const char* stage = r.stage == GPU_QUALITY_HIT_STAGE_FINAL ? "final" : "expand";
      out << method << ","
          << r.query_id << ","
          << r.step << ","
          << stage << ","
          << r.new_gt_count << ","
          << r.first_seen_gt_count << ","
          << r.dropped_gt_count << ","
          << r.current_hit_count << ","
          << r.cumulative_seen_gt_count << ","
          << r.topk_recall << "\n";
      ++kept_total;
    }
  }

  std::cout << "GPU quality hit trace records: " << kept_total;
  if (dropped != 0) {
    std::cout << " (dropped " << dropped
              << "; increase GPU_QUALITY_HIT_TRACE_MAX_RECORDS_PER_QUERY)";
  }
  std::cout << "\n";
}

}  // namespace

static vid_t compute_gpu_base_entry_point(const float* data, size_t npoints, size_t dim) {
  std::vector<float> centroid(dim, 0.0f);
  const float inv_npoints = 1.0f / static_cast<float>(npoints);
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * dim;
    for (size_t d = 0; d < dim; ++d) {
      centroid[d] += row[d] * inv_npoints;
    }
  }

  vid_t best = 0;
  float best_dist = FLT_MAX;
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * dim;
    float dist = compute_distance(static_cast<int>(dim), row, centroid.data());
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<vid_t>(i);
    }
  }
  return best;
}

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T *result_vec, vid_t *result_idx, int beam_sz, double& elapsed,
                           GpuRunStats* gpu_stats, bool debug_mode, const char* expander_trace_file,
                           const char* query_steps_file,
                           const char* access_trace_file,
                           const char* frontier_audit_file,
                           const int* groundtruth,
                           int gt_k,
                           const char* quality_hit_trace_file,
                           const char* quality_beam_hit_trace_file) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  auto bitlen = GPU_HASH_BITLEN_FOR_TOPK(beam_size);
  const bool use_ip = (this->metric == METRIC_IP);
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d, internal_topk=%d\n", K, nq, dim, npoints, beam_size, INTERNAL_TOPK);
  printf("Metric: %s\n", metric_name(this->metric));

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = nq;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
  int numBlocksPerSM = 0;
  int blockSize = 256;   // threads per block

  // allocate device memory
  T* d_queries, *d_data, *h_data = this->get_data_ptr();
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_queries, nq * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, nq * dim * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, npoints * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, npoints * dim * sizeof(T), cudaMemcpyHostToDevice));

  vid_t *h_results = &result_idx[0];
  vid_t *d_results;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, nq * K * sizeof(vid_t)));

  // counters
  gpu_long_t *d_total_stats = nullptr;
  double *d_expander_trace_sums = nullptr;
  gpu_long_t *d_expander_trace_counts = nullptr;
  double *d_expander_trace_net_descent = nullptr;
  double *d_query_l2_progress_sum = nullptr;
  double *d_query_l2_progress_per_expander_sum = nullptr;
  gpu_long_t *d_query_expander_steps_sum = nullptr;
  gpu_long_t *d_query_expander_count = nullptr;
  double *d_query_first_expander_l2_by_query = nullptr;
  gpu_long_t *d_query_expander_steps_by_query = nullptr;
  AccessTraceRecord* d_access_trace_records = nullptr;
  uint32_t* d_access_trace_counts = nullptr;
  GpuFrontierAuditRecord* d_frontier_audit_records = nullptr;
  uint32_t* d_frontier_audit_counts = nullptr;
  GpuQualityHitTraceRecord* d_quality_hit_records = nullptr;
  uint32_t* d_quality_hit_counts = nullptr;
  GpuQualityHitTraceRecord* d_quality_beam_hit_records = nullptr;
  uint32_t* d_quality_beam_hit_counts = nullptr;
  int* d_groundtruth = nullptr;
  uint32_t* d_quality_hit_previous_masks = nullptr;
  uint32_t* d_quality_hit_seen_masks = nullptr;
  uint32_t* d_quality_hit_current_masks = nullptr;
  uint32_t* d_quality_beam_hit_previous_masks = nullptr;
  uint32_t* d_quality_beam_hit_seen_masks = nullptr;
  uint32_t* d_quality_beam_hit_current_masks = nullptr;
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  const bool collect_expander_trace = trace_file_requested(expander_trace_file);
  const bool collect_query_steps = trace_file_requested(query_steps_file);
  const bool collect_access_trace = trace_file_requested(access_trace_file);
  const bool collect_frontier_audit = trace_file_requested(frontier_audit_file);
  const bool collect_quality_hit_trace = trace_file_requested(quality_hit_trace_file);
  const bool collect_quality_beam_hit_trace = trace_file_requested(quality_beam_hit_trace_file);
  const bool collect_any_quality_hit_trace = collect_quality_hit_trace || collect_quality_beam_hit_trace;
  const int expander_trace_max_steps = GPU_EXPANDER_TRACE_MAX_STEPS;
  const int access_trace_max_records_per_query = GPU_ACCESS_TRACE_MAX_RECORDS_PER_QUERY;
  const int frontier_audit_max_records_per_query = GPU_FRONTIER_AUDIT_MAX_RECORDS_PER_QUERY;
  const int quality_hit_max_records_per_query = GPU_QUALITY_HIT_TRACE_MAX_RECORDS_PER_QUERY;
  const int quality_hit_gt_n = std::min(K, gt_k);
  const int quality_hit_mask_words = (quality_hit_gt_n + 31) / 32;
  if (collect_expander_trace && !GPU_EXPANDER_TRACE) {
    throw std::runtime_error("rebuild GPU base with -DGPU_EXPANDER_TRACE=1 to collect expander traces");
  }
  if (collect_query_steps && !GPU_SEARCH_STATS) {
    throw std::runtime_error("rebuild GPU base with -DGPU_SEARCH_STATS=1 to collect query steps");
  }
  if (collect_access_trace && !GPU_ACCESS_TRACE) {
    throw std::runtime_error("rebuild GPU base with -DGPU_ACCESS_TRACE=1 to collect access traces");
  }
  if (collect_frontier_audit && !GPU_FRONTIER_AUDIT) {
    throw std::runtime_error("rebuild GPU base with -DGPU_FRONTIER_AUDIT=1 to collect frontier audit traces");
  }
  if (collect_any_quality_hit_trace && !GPU_QUALITY_HIT_TRACE) {
    throw std::runtime_error("rebuild GPU base with -DGPU_QUALITY_HIT_TRACE=1 to collect quality hit traces");
  }
  if (collect_any_quality_hit_trace && (groundtruth == nullptr || gt_k <= 0 || quality_hit_mask_words <= 0)) {
    throw std::runtime_error("GPU quality hit trace requires groundtruth and gt_k > 0");
  }
  const size_t stats_rows = num_blocks;
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_stats, stats_rows * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_total_stats, 0, stats_rows * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  }
  if (collect_expander_trace) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_expander_trace_sums,
                              static_cast<size_t>(expander_trace_max_steps) *
                                  GPU_EXPANDER_TRACE_FIELDS * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_expander_trace_counts,
                              static_cast<size_t>(expander_trace_max_steps) * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_expander_trace_net_descent, sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_query_l2_progress_sum, sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_query_l2_progress_per_expander_sum, sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_query_expander_steps_sum, sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_query_expander_count, sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_query_first_expander_l2_by_query,
                              static_cast<size_t>(nq) * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_query_expander_steps_by_query,
                              static_cast<size_t>(nq) * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_expander_trace_sums, 0,
                              static_cast<size_t>(expander_trace_max_steps) *
                                  GPU_EXPANDER_TRACE_FIELDS * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_expander_trace_counts, 0,
                              static_cast<size_t>(expander_trace_max_steps) * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_expander_trace_net_descent, 0, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_query_l2_progress_sum, 0, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_query_l2_progress_per_expander_sum, 0, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_query_expander_steps_sum, 0, sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_query_expander_count, 0, sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_query_first_expander_l2_by_query, 0,
                              static_cast<size_t>(nq) * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(d_query_expander_steps_by_query, 0,
                              static_cast<size_t>(nq) * sizeof(gpu_long_t)));
  }
  if (collect_access_trace) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_access_trace_records,
                              static_cast<size_t>(nq) *
                                  access_trace_max_records_per_query *
                                  sizeof(AccessTraceRecord)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_access_trace_counts,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_access_trace_counts, 0,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
  }
  if (collect_frontier_audit) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_audit_records,
                              static_cast<size_t>(nq) *
                                  frontier_audit_max_records_per_query *
                                  sizeof(GpuFrontierAuditRecord)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_audit_counts,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_frontier_audit_counts, 0,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
  }
  if (collect_any_quality_hit_trace) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_groundtruth,
                              static_cast<size_t>(nq) * gt_k * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_groundtruth, groundtruth,
                              static_cast<size_t>(nq) * gt_k * sizeof(int),
                              cudaMemcpyHostToDevice));
  }
  if (collect_quality_hit_trace) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_hit_records,
                              static_cast<size_t>(nq) *
                                  quality_hit_max_records_per_query *
                                  sizeof(GpuQualityHitTraceRecord)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_hit_counts,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_hit_previous_masks,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_hit_seen_masks,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_hit_current_masks,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_hit_counts, 0,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_hit_previous_masks, 0,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_hit_seen_masks, 0,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_hit_current_masks, 0,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
  }
  if (collect_quality_beam_hit_trace) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_beam_hit_records,
                              static_cast<size_t>(nq) *
                                  quality_hit_max_records_per_query *
                                  sizeof(GpuQualityHitTraceRecord)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_beam_hit_counts,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_beam_hit_previous_masks,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_beam_hit_seen_masks,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_quality_beam_hit_current_masks,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_beam_hit_counts, 0,
                              static_cast<size_t>(nq) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_beam_hit_previous_masks, 0,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_beam_hit_seen_masks, 0,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(d_quality_beam_hit_current_masks, 0,
                              static_cast<size_t>(nq) * quality_hit_mask_words * sizeof(uint32_t)));
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(npoints, max_degree, this->edges.data()); // GPU graph

  const int starting_point_count = std::max(beam_size, CANDIDATE_BUFFER_SIZE);
  vid_t *starting_points = new vid_t[starting_point_count];
  vid_t *d_starting_points;
  const vid_t entry_point = compute_gpu_base_entry_point(h_data, npoints, dim);
  printf("GPU base entry point: %u\n", entry_point);
  for (int i = 0; i < starting_point_count; i++) {
    starting_points[i] = (i == 0) ? entry_point : MAX_INDEX;
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_starting_points, starting_point_count * sizeof(vid_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, starting_point_count * sizeof(vid_t), cudaMemcpyHostToDevice));

#ifdef USE_RADIX
  //vid_t *d_filters = nullptr;
  //printf("filter size = %d bytes\n", num_blocks * HASH_FILTER_SIZE * sizeof(vid_t));
  //CUDA_SAFE_CALL(cudaMalloc((void **)&d_filters, num_blocks * HASH_FILTER_SIZE * sizeof(vid_t)));
  BeamSearch<T><<<num_blocks, num_threads>>>(K, nq, dim, npoints, d_queries, d_data,
                                             d_results, d_total_stats,
                                             d_expander_trace_sums, d_expander_trace_counts,
                                             d_expander_trace_net_descent,
                                             d_query_l2_progress_sum,
                                             d_query_l2_progress_per_expander_sum,
                                             d_query_expander_steps_sum,
                                             d_query_expander_count,
                                             d_query_first_expander_l2_by_query,
                                             d_query_expander_steps_by_query,
                                             expander_trace_max_steps,
                                             d_access_trace_records, d_access_trace_counts,
                                             access_trace_max_records_per_query,
	                                             d_frontier_audit_records, d_frontier_audit_counts,
	                                             frontier_audit_max_records_per_query,
	                                             d_quality_hit_records, d_quality_hit_counts,
	                                             quality_hit_max_records_per_query,
	                                             d_groundtruth, gt_k,
	                                             d_quality_hit_previous_masks,
	                                             d_quality_hit_seen_masks,
	                                             d_quality_hit_current_masks,
	                                             quality_hit_mask_words,
	                                             d_starting_points, gg);
#else
  uint32_t shm_size = calculate_shared_mem_size(dim, beam_size, max_degree, SEARCH_WIDTH, bitlen);
  vid_t *d_seed_map = nullptr;
  if (USE_SEED) {
    size_t ghost_scale_factor = 100;
    size_t ghost_num = npoints / ghost_scale_factor;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_seed_map, ghost_num * sizeof(vid_t)));
    init_identity_map<<<(ghost_num-1)/num_threads+1, num_threads>>>(ghost_num, d_seed_map);
  }
  uint32_t *d_sign_bit = nullptr;
  if (PRUNE_CONFIG) {
    auto sign_bit = load_sign_bit("sign_bit.bin");
    size_t sign_bit_size = sign_bit.size();
    uint32_t *h_sign_bit = sign_bit.data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sign_bit, sign_bit_size * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit, sign_bit_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, BeamSearch<T>, blockSize, shm_size);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  // Execute the search
  printf("\nStarting GPU beam search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  BeamSearch<T><<<num_blocks, num_threads, shm_size>>>(K, nq, dim, beam_size, bitlen, npoints, d_queries, d_data, d_results,
                                                         d_total_stats,
                                                         d_expander_trace_sums, d_expander_trace_counts,
                                                         d_expander_trace_net_descent,
                                                         d_query_l2_progress_sum,
                                                         d_query_l2_progress_per_expander_sum,
                                                         d_query_expander_steps_sum,
                                                         d_query_expander_count,
                                                         d_query_first_expander_l2_by_query,
                                                         d_query_expander_steps_by_query,
                                                         expander_trace_max_steps,
                                                         d_access_trace_records, d_access_trace_counts,
                                                         access_trace_max_records_per_query,
                                                         d_frontier_audit_records, d_frontier_audit_counts,
                                                         frontier_audit_max_records_per_query,
                                                         d_quality_hit_records, d_quality_hit_counts,
                                                         quality_hit_max_records_per_query,
                                                         d_groundtruth, gt_k,
                                                         d_quality_hit_previous_masks,
                                                         d_quality_hit_seen_masks,
                                                         d_quality_hit_current_masks,
                                                         quality_hit_mask_words,
                                                         d_quality_beam_hit_records,
                                                         d_quality_beam_hit_counts,
                                                         d_quality_beam_hit_previous_masks,
                                                         d_quality_beam_hit_seen_masks,
                                                         d_quality_beam_hit_current_masks,
                                                         d_starting_points, d_seed_map, d_sign_bit, gg, use_ip);
  CUDA_SAFE_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, nq * K * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<gpu_long_t> h_block_stats;
  if (GPU_SEARCH_STATS) {
    h_block_stats.resize(stats_rows * GPU_SEARCH_STAT_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_stats.data(), d_total_stats,
                              h_block_stats.size() * sizeof(gpu_long_t),
                              cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < stats_rows; ++b) {
      for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
        total_stats[i] += h_block_stats[b * GPU_SEARCH_STAT_COUNT + i];
      }
    }
    if (collect_query_steps) {
      dump_gpu_query_steps(query_steps_file, h_block_stats, stats_rows);
    }
  }
  if (collect_expander_trace) {
    dump_gpu_expander_trace(expander_trace_file, expander_trace_max_steps,
                            d_expander_trace_sums, d_expander_trace_counts,
                            d_expander_trace_net_descent,
                            d_query_l2_progress_sum,
                            d_query_l2_progress_per_expander_sum,
                            d_query_expander_steps_sum,
                            d_query_expander_count,
                            d_query_first_expander_l2_by_query,
                            d_query_expander_steps_by_query,
                            nq,
                            gpu_stats);
  }
  if (collect_access_trace) {
    dump_gpu_access_trace(access_trace_file, nq, access_trace_max_records_per_query,
                          d_access_trace_records, d_access_trace_counts);
  }
  if (collect_frontier_audit) {
    dump_gpu_frontier_audit(frontier_audit_file, "gpu_base", nq,
                            frontier_audit_max_records_per_query,
                            d_frontier_audit_records, d_frontier_audit_counts);
  }
  if (collect_quality_hit_trace) {
    dump_gpu_quality_hit_trace(quality_hit_trace_file, "gpu_base", nq,
                               quality_hit_max_records_per_query,
                               d_quality_hit_records, d_quality_hit_counts);
  }
  if (collect_quality_beam_hit_trace) {
    dump_gpu_quality_hit_trace(quality_beam_hit_trace_file, "gpu_base", nq,
                               quality_hit_max_records_per_query,
                               d_quality_beam_hit_records, d_quality_beam_hit_counts);
  }
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_starting_points));
  delete[] starting_points;
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaFree(d_total_stats));
  }
  if (collect_expander_trace) {
    CUDA_SAFE_CALL(cudaFree(d_expander_trace_sums));
    CUDA_SAFE_CALL(cudaFree(d_expander_trace_counts));
    CUDA_SAFE_CALL(cudaFree(d_expander_trace_net_descent));
    CUDA_SAFE_CALL(cudaFree(d_query_l2_progress_sum));
    CUDA_SAFE_CALL(cudaFree(d_query_l2_progress_per_expander_sum));
    CUDA_SAFE_CALL(cudaFree(d_query_expander_steps_sum));
    CUDA_SAFE_CALL(cudaFree(d_query_expander_count));
    CUDA_SAFE_CALL(cudaFree(d_query_first_expander_l2_by_query));
    CUDA_SAFE_CALL(cudaFree(d_query_expander_steps_by_query));
  }
  if (collect_access_trace) {
    CUDA_SAFE_CALL(cudaFree(d_access_trace_records));
    CUDA_SAFE_CALL(cudaFree(d_access_trace_counts));
  }
  if (collect_frontier_audit) {
    CUDA_SAFE_CALL(cudaFree(d_frontier_audit_records));
    CUDA_SAFE_CALL(cudaFree(d_frontier_audit_counts));
  }
  if (collect_quality_hit_trace) {
    CUDA_SAFE_CALL(cudaFree(d_quality_hit_records));
    CUDA_SAFE_CALL(cudaFree(d_quality_hit_counts));
    CUDA_SAFE_CALL(cudaFree(d_quality_hit_previous_masks));
    CUDA_SAFE_CALL(cudaFree(d_quality_hit_seen_masks));
    CUDA_SAFE_CALL(cudaFree(d_quality_hit_current_masks));
  }
  if (collect_quality_beam_hit_trace) {
    CUDA_SAFE_CALL(cudaFree(d_quality_beam_hit_records));
    CUDA_SAFE_CALL(cudaFree(d_quality_beam_hit_counts));
    CUDA_SAFE_CALL(cudaFree(d_quality_beam_hit_previous_masks));
    CUDA_SAFE_CALL(cudaFree(d_quality_beam_hit_seen_masks));
    CUDA_SAFE_CALL(cudaFree(d_quality_beam_hit_current_masks));
  }
  if (collect_any_quality_hit_trace) {
    CUDA_SAFE_CALL(cudaFree(d_groundtruth));
  }
  if (GPU_SEARCH_STATS) {
    std::cout << "total # exact distance computations: " << total_stats[GPU_SEARCH_STAT_DIST_CMPS] << "\n";
    std::cout << "average # exact distance computations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]) / nq << "\n";
    std::cout << "total # distance estimations: " << total_stats[GPU_SEARCH_STAT_DIST_EST] << "\n";
    std::cout << "average # distance estimations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_EST]) / nq << "\n";
    std::cout << "total # expansion steps: " << total_stats[GPU_SEARCH_STAT_ITERATIONS] << "\n";
    std::cout << "average # expansion steps: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_ITERATIONS]) / nq << "\n";
  }
  if (gpu_stats != nullptr) {
    gpu_stats->common.total_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_ds = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_EST]);
    gpu_stats->common.total_extra_dc = 0;
    gpu_stats->common.total_full_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_iter = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_ITERATIONS]);
    gpu_stats->common.total_pruned = 0;
    gpu_stats->common.time_search_total = elapsed;
  }
  (void)debug_mode;
}

template class IndexGraph<float>;
