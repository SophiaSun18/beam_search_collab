//#include <cuda_profiler_api.h>

#include "include/qg.hpp"

#ifndef GPU_RABITQ_PRUNED_SPEC
#define GPU_RABITQ_PRUNED_SPEC 0
#endif

#ifndef GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
#define GPU_RABITQ_PRUNED_SPEC_ADAPTIVE 0
#endif

#ifndef SEARCH_WIDTH
#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
#define SEARCH_WIDTH 16
#else
#define SEARCH_WIDTH 1
#endif
#endif

#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE && (SEARCH_WIDTH < 1)
#error "GPU RaBitQ-pruned adaptive mode requires SEARCH_WIDTH >= 1"
#endif

#if !GPU_RABITQ_PRUNED_SPEC_ADAPTIVE && !GPU_RABITQ_PRUNED_SPEC && (SEARCH_WIDTH != 1)
#error "GPU RaBitQ-pruned clean mode requires SEARCH_WIDTH=1; use GPU_RABITQ_PRUNED_SPEC=1 for wider fixed-width search"
#endif

#if !GPU_RABITQ_PRUNED_SPEC_ADAPTIVE && GPU_RABITQ_PRUNED_SPEC && (SEARCH_WIDTH < 1 || SEARCH_WIDTH > 16)
#error "GPU RaBitQ-pruned spec mode requires SEARCH_WIDTH in [1, 16]"
#endif

#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
#include "include/search_rabitq_pruned_spec_adaptive.cuh"
#elif GPU_RABITQ_PRUNED_SPEC && (SEARCH_WIDTH > 1)
#include "include/search_rabitq_pruned_spec.cuh"
#else
#include "include/search_rabitq_pruned.cuh"
#endif

#include <chrono>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <vector>

static vidType compute_gpu_rabitq_entry_point(const float* data, size_t npoints,
                                              size_t dim, size_t row_offset) {
  std::vector<float> centroid(dim, 0.0f);
  const float inv_npoints = 1.0f / static_cast<float>(npoints);
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * row_offset;
    for (size_t d = 0; d < dim; ++d) {
      centroid[d] += row[d] * inv_npoints;
    }
  }

  vidType best = 0;
  float best_dist = FLT_MAX;
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * row_offset;
    float dist = compute_distance(static_cast<int>(dim), row, centroid.data());
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<vidType>(i);
    }
  }
  return best;
}

void QuantizationGraph::gpu_search_pruned(int nq, const float* queries, int K,
                                          float* result_vec, vidType* result_idx,
                                          int beam_sz, float prune_ratio,
                                          float approx_ratio, double& elapsed,
                                          GpuRunStats* gpu_stats, bool debug_mode) {
  if (this->is_bins_mode()) {
    throw std::runtime_error("gpu_search_pruned currently supports only codebook-mode QuantizationGraph");
  }
  if (beam_sz <= 0) {
    throw std::runtime_error("GPU RaBitQ-pruned requires beam_size > 0");
  }
  int padded_beam_size = static_cast<int>(gpu_effective_sort_beam_size(static_cast<uint32_t>(beam_sz)));
  const size_t candidate_work_count = static_cast<size_t>(SEARCH_WIDTH) * this->degree_;
#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
  const size_t candidate_buffer_size = round_up_power2_u32(
      rabitq_pruned_adaptive_candidate_buffer_capacity(static_cast<uint32_t>(this->degree_)));
#else
  const size_t candidate_buffer_size = rabitq_pruned_candidate_buffer_size(static_cast<uint32_t>(this->degree_), prune_ratio, approx_ratio);
#endif
#if !(GPU_RABITQ_PRUNED_SPEC_ADAPTIVE || (GPU_RABITQ_PRUNED_SPEC && (SEARCH_WIDTH > 1)))
  if (candidate_work_count > candidate_buffer_size) {
    throw std::runtime_error("GPU RaBitQ-pruned clean kernel requires degree <= candidate buffer size");
  }
#endif
  auto dim = this->dim_;
  auto npoints = this->num_nodes_;
  auto max_degree = this->degree_;
  auto beam_size = beam_sz;
  auto bitlen = gpu_hash_bitlen_for_pruned_workload(
      static_cast<uint32_t>(padded_beam_size),
      static_cast<uint32_t>(candidate_buffer_size),
      static_cast<uint32_t>(SMALL_HASH_RESET_INTERVAL));
  const bool use_ip = (this->get_metric() == METRIC_IP);
  const int active_nq = nq;

  printf("\nBeam search RaBitQ-pruned on GPU: K=%d, nq=%d, dim=%ld, npoints=%ld, beam_size=%d, padded_beam_size=%d, internal_topk=%d\n",
         K, active_nq, dim, npoints, beam_size, padded_beam_size, INTERNAL_TOPK);
  printf("Metric: %s\n", metric_name(this->get_metric()));
  printf("RaBitQ-pruned params: prune_ratio=%.4f, keep_fraction=%.4f, prune_iteration_ratio=%.4f\n",
         prune_ratio, 1.0f - prune_ratio, approx_ratio);
#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
  constexpr const char* mode_name = "spec_adaptive";
#elif GPU_RABITQ_PRUNED_SPEC && (SEARCH_WIDTH > 1)
  constexpr const char* mode_name = "spec";
#elif GPU_RABITQ_PRUNED_SPEC
  constexpr const char* mode_name = "clean_spec_width1_fallback";
#else
  constexpr const char* mode_name = "clean";
#endif
  printf("GPU rabitq-pruned mode: %s, search_width=%d, candidate_work_count=%zu, candidate_buffer_size=%zu, hash_bitlen=%d, hash_size=%u\n",
         mode_name, SEARCH_WIDTH, candidate_work_count, candidate_buffer_size, bitlen, hashtable_getsize(bitlen));
#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
  uint32_t adaptive_collect_capacity = static_cast<uint32_t>(RABITQ_ADAPTIVE_BUFFER_BOUND);
  if (adaptive_collect_capacity > static_cast<uint32_t>(candidate_buffer_size)) {
    adaptive_collect_capacity = static_cast<uint32_t>(candidate_buffer_size);
  }
  const uint32_t phase1_theta = static_cast<uint32_t>(RABITQ_ADAPTIVE_PHASE1_THETA);
  const uint32_t phase2_theta = static_cast<uint32_t>(RABITQ_ADAPTIVE_PHASE2_THETA);
  const uint32_t phase1_raw_keep =
      keep_count_per_parent(static_cast<uint32_t>(this->degree_), static_cast<float>(RABITQ_ADAPTIVE_PHASE1_RHO));
  const uint32_t phase2_raw_keep =
      keep_count_per_parent(static_cast<uint32_t>(this->degree_), static_cast<float>(RABITQ_ADAPTIVE_PHASE2_RHO));
  uint32_t phase1_budget_keep = adaptive_collect_capacity / phase1_theta;
  uint32_t phase2_budget_keep = adaptive_collect_capacity / phase2_theta;
  if (phase1_budget_keep < 1) phase1_budget_keep = 1;
  if (phase2_budget_keep < 1) phase2_budget_keep = 1;
  if (phase1_budget_keep > static_cast<uint32_t>(this->degree_)) phase1_budget_keep = static_cast<uint32_t>(this->degree_);
  if (phase2_budget_keep > static_cast<uint32_t>(this->degree_)) phase2_budget_keep = static_cast<uint32_t>(this->degree_);
  const uint32_t phase1_raw_expected = phase1_theta * phase1_raw_keep;
  const uint32_t phase2_raw_expected = phase2_theta * phase2_raw_keep;
  const uint32_t phase1_expected_admit = phase1_theta *
      ((phase1_raw_keep < phase1_budget_keep) ? phase1_raw_keep : phase1_budget_keep);
  const uint32_t phase2_expected_admit = phase2_theta *
      ((phase2_raw_keep < phase2_budget_keep) ? phase2_raw_keep : phase2_budget_keep);
  printf("RaBitQ-pruned adaptive params: theta=[%d, %d], phase1=(w%d, rho=%.4f, raw=%u, budgeted=%u), phase2=(w%d, rho=%.4f, raw=%u, budgeted=%u), buffer_bound=%d, collect_capacity=%u, budget_ok=(%d,%d), rho=[%.4f, %.4f], head_stall_iters=%d, kth_cutoff=%d\n",
         static_cast<int>(RABITQ_ADAPTIVE_THETA_MIN),
         static_cast<int>(RABITQ_ADAPTIVE_THETA_MAX),
         static_cast<int>(RABITQ_ADAPTIVE_PHASE1_THETA),
         static_cast<double>(RABITQ_ADAPTIVE_PHASE1_RHO),
         phase1_raw_expected,
         phase1_expected_admit,
         static_cast<int>(RABITQ_ADAPTIVE_PHASE2_THETA),
         static_cast<double>(RABITQ_ADAPTIVE_PHASE2_RHO),
         phase2_raw_expected,
         phase2_expected_admit,
         static_cast<int>(RABITQ_ADAPTIVE_BUFFER_BOUND),
         adaptive_collect_capacity,
         static_cast<int>(phase1_expected_admit <= adaptive_collect_capacity),
         static_cast<int>(phase2_expected_admit <= adaptive_collect_capacity),
         static_cast<double>(RABITQ_ADAPTIVE_RHO_MIN),
         static_cast<double>(RABITQ_ADAPTIVE_RHO_MAX),
         static_cast<int>(RABITQ_ADAPTIVE_TOP1_WARMUP_STALL_ITERS),
         static_cast<int>(RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF));
#endif

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = active_nq;
  std::cout << "\nnum_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  CUDA_SAFE_CALL(cudaGetDevice(&dev));
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  int numBlocksPerSM = 0;
  int blockSize = 256;

  float* d_queries = nullptr;
  float* d_qg_data = nullptr;
  float* d_qg_signs = nullptr;
  size_t free_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));

  const size_t query_bytes = static_cast<size_t>(active_nq) * dim * sizeof(float);
  const size_t results_bytes = static_cast<size_t>(active_nq) * K * sizeof(vidType);
  const size_t qg_data_bytes = this->get_data_bytes();
  const size_t qg_signs_bytes = this->get_signs_bytes();
  const size_t required_bytes = query_bytes + results_bytes + qg_data_bytes + qg_signs_bytes;

  constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
  printf("GPU memory: free=%.2f GiB, total=%.2f GiB, required=%.2f GiB (qg_data=%.2f GiB, qg_signs=%.4f GiB)\n",
         static_cast<double>(free_mem_bytes) / kBytesPerGiB,
         static_cast<double>(total_mem_bytes) / kBytesPerGiB,
         static_cast<double>(required_bytes) / kBytesPerGiB,
         static_cast<double>(qg_data_bytes) / kBytesPerGiB,
         static_cast<double>(qg_signs_bytes) / kBytesPerGiB);
  if (required_bytes > free_mem_bytes) {
    throw std::runtime_error("QG upload exceeds currently available GPU memory");
  }

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_queries, query_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, query_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_data, qg_data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_data, this->get_data_ptr(), qg_data_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_signs, qg_signs_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_signs, this->get_signs_ptr(), qg_signs_bytes, cudaMemcpyHostToDevice));

  vidType* d_results = nullptr;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, results_bytes));

  gpu_long_t* d_total_stats = nullptr;
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_stats, num_blocks * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_total_stats, 0, num_blocks * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  vidType entry_point = this->get_entry_point();
  if (entry_point == UINT32_MAX) {
    entry_point =
        compute_gpu_rabitq_entry_point(this->get_data_ptr(), npoints, dim, this->get_row_offset());
  }
  printf("GPU rabitq-pruned entry point: %u\n", entry_point);

  uint32_t shm_size = calculate_pruned_shared_mem_size(
      static_cast<int>(dim), padded_beam_size, K, static_cast<int>(max_degree), bitlen,
      prune_ratio, approx_ratio);
  printf("Dynamic shared memory size = %u bytes\n", shm_size);
  CUDA_SAFE_CALL(cudaFuncSetAttribute(
      QuantizedPrunedBeamSearch,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shm_size)));

  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, QuantizedPrunedBeamSearch, blockSize, shm_size);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  printf("\nStarting GPU rabitq-pruned search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  QuantizedPrunedBeamSearch<<<num_blocks, num_threads, shm_size>>>(
      K, active_nq, static_cast<int>(dim), beam_size, bitlen, static_cast<int>(max_degree),
      npoints, d_queries, d_qg_data, d_qg_signs, d_results, d_total_stats,
      entry_point, this->get_row_offset(), this->get_neighbor_offset(),
      this->get_code_offset(), this->get_factor_offset(), use_ip, prune_ratio, approx_ratio);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, results_bytes, cudaMemcpyDeviceToHost));
  if (GPU_SEARCH_STATS) {
    std::vector<gpu_long_t> h_block_stats(num_blocks * GPU_SEARCH_STAT_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_stats.data(), d_total_stats,
                              h_block_stats.size() * sizeof(gpu_long_t),
                              cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < num_blocks; ++b) {
      for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
        total_stats[i] += h_block_stats[b * GPU_SEARCH_STAT_COUNT + i];
      }
    }
  }
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_qg_data));
  CUDA_SAFE_CALL(cudaFree(d_qg_signs));
  CUDA_SAFE_CALL(cudaFree(d_results));
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaFree(d_total_stats));
  }

  if (GPU_SEARCH_STATS) {
    std::cout << "total # exact distance computations: " << total_stats[GPU_SEARCH_STAT_DIST_CMPS] << "\n";
    std::cout << "average # exact distance computations: "
              << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]) / active_nq << "\n";
    std::cout << "total # distance estimations: " << total_stats[GPU_SEARCH_STAT_DIST_EST] << "\n";
    std::cout << "average # distance estimations: "
              << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_EST]) / active_nq << "\n";
    std::cout << "total # expansion steps: " << total_stats[GPU_SEARCH_STAT_ITERATIONS] << "\n";
    std::cout << "average # expansion steps: "
              << static_cast<double>(total_stats[GPU_SEARCH_STAT_ITERATIONS]) / active_nq << "\n";
    std::cout << "total # pruned neighbors: " << total_stats[GPU_SEARCH_STAT_PRUNED] << "\n";
    std::cout << "average # pruned neighbors: "
              << static_cast<double>(total_stats[GPU_SEARCH_STAT_PRUNED]) / active_nq << "\n";
#if GPU_RABITQ_PRUNED_SPEC_ADAPTIVE
    std::cout << "AP total # search expansion steps: "
              << total_stats[GPU_SEARCH_STAT_OUTER_EXPANSION_STEPS] << "\n";
    std::cout << "AP average # search expansion steps: "
              << static_cast<double>(total_stats[GPU_SEARCH_STAT_OUTER_EXPANSION_STEPS]) / active_nq << "\n";
#endif
  }
  if (gpu_stats != nullptr) {
    gpu_stats->common.total_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_ds = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_EST]);
    gpu_stats->common.total_extra_dc = 0;
    gpu_stats->common.total_full_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_iter = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_ITERATIONS]);
    gpu_stats->common.total_pruned = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_PRUNED]);
    gpu_stats->common.time_search_total = elapsed;
  }

  (void)result_vec;
  (void)debug_mode;
}
