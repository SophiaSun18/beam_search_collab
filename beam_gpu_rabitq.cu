//#include <cuda_profiler_api.h>

#include "include/qg.hpp"
#include "include/search_rabitq.cuh"

#include <chrono>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cfloat>
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

void QuantizationGraph::gpu_search(int nq, const float* queries, int K, float* result_vec,
                                   vidType* result_idx, int beam_sz, double& elapsed,
                                   GpuRunStats* gpu_stats, bool debug_mode) {
  if (this->is_bins_mode()) {
    throw std::runtime_error("gpu_search currently supports only codebook-mode QuantizationGraph");
  }

  auto dim = this->dim_;
  auto npoints = this->num_nodes_;
  auto max_degree = this->degree_;
  auto beam_size = beam_sz;
  auto bitlen = GPU_HASH_BITLEN_FOR_TOPK(beam_size);
  const bool use_ip = (this->get_metric() == METRIC_IP);
  const int active_nq = GPU_DISTANCE_TIMING ? 1 : nq;
  printf("\nBeam search on GPU: K=%d, nq=%d, dim=%ld, npoints=%ld, beam_size=%d, internal_topk=%d\n", K, active_nq, dim, npoints, beam_size, INTERNAL_TOPK);
  printf("Metric: %s\n", metric_name(this->get_metric()));
  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = active_nq;
  std::cout << "\nnum_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
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

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_queries, static_cast<size_t>(active_nq) * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, static_cast<size_t>(active_nq) * dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_data, qg_data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_data, this->get_data_ptr(), qg_data_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_signs, qg_signs_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_signs, this->get_signs_ptr(), qg_signs_bytes, cudaMemcpyHostToDevice));

  vidType* d_results = nullptr;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, static_cast<size_t>(active_nq) * K * sizeof(vidType)));

  gpu_long_t* d_total_stats = nullptr;
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_stats, num_blocks * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  }
  gpu_long_t* d_total_timing = nullptr;
  std::array<gpu_long_t, GPU_TIMING_COUNT> total_timing{};
  if (GPU_DISTANCE_TIMING) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_timing, num_blocks * GPU_TIMING_COUNT * sizeof(gpu_long_t)));
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  vidType entry_point = this->get_entry_point();
  if (entry_point == UINT32_MAX) {
    entry_point =
        compute_gpu_rabitq_entry_point(this->get_data_ptr(), npoints, dim, this->get_row_offset());
  }
  printf("GPU rabitq entry point: %u\n", entry_point);

  uint32_t shm_size = calculate_shared_mem_size(static_cast<int>(dim), beam_size, K,
                                                static_cast<int>(max_degree), bitlen);
  CUDA_SAFE_CALL(cudaFuncSetAttribute(
      QuantizedBeamSearch,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shm_size)));

  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, QuantizedBeamSearch, blockSize, shm_size);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  printf("\nStarting GPU rabitq search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  QuantizedBeamSearch<<<num_blocks, num_threads, shm_size>>>(
      K, active_nq, static_cast<int>(dim), beam_size, bitlen, static_cast<int>(max_degree),
      d_queries, d_qg_data, d_qg_signs, d_results, d_total_stats, d_total_timing,
      entry_point, this->get_row_offset(), this->get_neighbor_offset(),
      this->get_code_offset(), this->get_factor_offset(), use_ip, debug_mode);
  CUDA_SAFE_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, static_cast<size_t>(active_nq) * K * sizeof(int), cudaMemcpyDeviceToHost));
  if (GPU_SEARCH_STATS) {
    std::vector<gpu_long_t> h_block_stats(num_blocks * GPU_SEARCH_STAT_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_stats.data(), d_total_stats, h_block_stats.size() * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < num_blocks; ++b) {
      for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
        total_stats[i] += h_block_stats[b * GPU_SEARCH_STAT_COUNT + i];
      }
    }
  }
  if (GPU_DISTANCE_TIMING) {
    std::vector<gpu_long_t> h_block_timing(num_blocks * GPU_TIMING_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_timing.data(), d_total_timing, h_block_timing.size() * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < num_blocks; ++b) {
      for (int i = 0; i < GPU_TIMING_COUNT; ++i) {
        total_timing[i] += h_block_timing[b * GPU_TIMING_COUNT + i];
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
  if (GPU_DISTANCE_TIMING) {
    CUDA_SAFE_CALL(cudaFree(d_total_timing));
  }

  if (GPU_SEARCH_STATS) {
    std::cout << "total # exact distance computations: " << total_stats[GPU_SEARCH_STAT_DIST_CMPS] << "\n";
    std::cout << "average # exact distance computations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]) / active_nq << "\n";
    std::cout << "total # distance estimations: " << total_stats[GPU_SEARCH_STAT_DIST_EST] << "\n";
    std::cout << "average # distance estimations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_EST]) / active_nq << "\n";
    std::cout << "total # expansion steps: " << total_stats[GPU_SEARCH_STAT_ITERATIONS] << "\n";
    std::cout << "average # expansion steps: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_ITERATIONS]) / active_nq << "\n";
  }
  if (GPU_DISTANCE_TIMING) {
    const double avg_l2_cycles =
        total_timing[GPU_TIMING_L2_CALLS] ? static_cast<double>(total_timing[GPU_TIMING_L2_CYCLES]) / total_timing[GPU_TIMING_L2_CALLS] : 0.0;
    const double avg_est_cycles =
        total_timing[GPU_TIMING_EST_CALLS] ? static_cast<double>(total_timing[GPU_TIMING_EST_CYCLES]) / total_timing[GPU_TIMING_EST_CALLS] : 0.0;
    const double avg_est_region_cycles =
        total_timing[GPU_TIMING_EST_REGION_CANDIDATES]
            ? static_cast<double>(total_timing[GPU_TIMING_EST_REGION_CYCLES]) / total_timing[GPU_TIMING_EST_REGION_CANDIDATES]
            : 0.0;
    std::cout << "total # l2 timing calls: " << total_timing[GPU_TIMING_L2_CALLS] << "\n";
    std::cout << "average cycles per warp_l2_distance: " << avg_l2_cycles << "\n";
    std::cout << "total # est timing calls: " << total_timing[GPU_TIMING_EST_CALLS] << "\n";
    std::cout << "average cycles per estimated_distance_gpu: " << avg_est_cycles << "\n";
    std::cout << "total # est region candidates: " << total_timing[GPU_TIMING_EST_REGION_CANDIDATES] << "\n";
    std::cout << "average cycles per estimation region candidate: " << avg_est_region_cycles << "\n";
    const struct {
      int idx;
      const char* label;
    } timing_sections[] = {
        {GPU_TIMING_QUERY_PREP_CYCLES, "query_prepare cycles"},
        {GPU_TIMING_SORT_CYCLES, "topk sort cycles"},
        {GPU_TIMING_PICK_CYCLES, "pick expander cycles"},
        {GPU_TIMING_VISITED_CYCLES, "visited check cycles"},
        {GPU_TIMING_STAGE_CYCLES, "stage code/factor cycles"},
        {GPU_TIMING_FINAL_SORT_CYCLES, "final sort cycles"},
        {GPU_TIMING_RESULT_POOL_CYCLES, "result pool update cycles"},
        {GPU_TIMING_TENSOR_QUERY_TILE_CYCLES, "tensor query tile cycles"},
        {GPU_TIMING_TENSOR_UNPACK_CYCLES, "tensor unpack cycles"},
        {GPU_TIMING_TENSOR_MMA_CYCLES, "tensor mma cycles"},
        {GPU_TIMING_TENSOR_FINAL_FACTOR_CYCLES, "tensor final factor cycles"},
    };
    for (const auto& section : timing_sections) {
      if (total_timing[section.idx] != 0) {
        std::cout << section.label << ": " << total_timing[section.idx] << "\n";
      }
    }
  }

  if (gpu_stats != nullptr) {
    gpu_stats->common.total_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_ds = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_EST]);
    gpu_stats->common.total_extra_dc = 0;
    gpu_stats->common.total_full_dc = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]);
    gpu_stats->common.total_iter = static_cast<int64_t>(total_stats[GPU_SEARCH_STAT_ITERATIONS]);
    gpu_stats->common.total_pruned = 0;
    gpu_stats->common.time_search_total = elapsed;
    gpu_stats->gpu_l2_calls = static_cast<int64_t>(total_timing[GPU_TIMING_L2_CALLS]);
    gpu_stats->gpu_l2_cycles = static_cast<int64_t>(total_timing[GPU_TIMING_L2_CYCLES]);
    gpu_stats->gpu_est_calls = static_cast<int64_t>(total_timing[GPU_TIMING_EST_CALLS]);
    gpu_stats->gpu_est_cycles = static_cast<int64_t>(total_timing[GPU_TIMING_EST_CYCLES]);
  }

  (void)result_vec;
}
