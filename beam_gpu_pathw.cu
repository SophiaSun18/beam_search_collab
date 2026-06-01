//#include <cuda_profiler_api.h>

#include "include/index.hpp"
#include "include/search_pathw.cuh"

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

static vid_t compute_gpu_pathw_entry_point(const float* data, size_t npoints, size_t dim) {
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

static bool get_gpu_pathw_seed_override(size_t npoints, vid_t& seed_node) {
  const char* seed_env = std::getenv("PATHW_SEED_NODE");
  if (seed_env == nullptr || seed_env[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  const unsigned long long value = std::strtoull(seed_env, &end, 10);
  if (end == seed_env || *end != '\0' || value >= npoints) {
    throw std::runtime_error("Invalid PATHW_SEED_NODE: " + std::string(seed_env));
  }
  seed_node = static_cast<vid_t>(value);
  return true;
}

template <typename T>
void IndexGraph<T>::search_pathw(int nq, const T* queries, int K, T* result_vec,
                                 vid_t* result_idx, int beam_sz,
                                 const char* signbit_file,
                                 float neighbor_keep_ratio,
                                 float iteration_prune_ratio,
                                 double& elapsed, GpuRunStats* gpu_stats,
                                 bool debug_mode) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  auto padded_beam_size = static_cast<int>(gpu_effective_sort_beam_size(static_cast<uint32_t>(beam_size), static_cast<uint32_t>(K)));
  auto bitlen = pathw_hash_bitlen_for_topk(padded_beam_size);
  const bool use_ip = (this->metric == METRIC_IP);
  const int active_nq = nq;
  if (use_ip) {
    throw std::runtime_error("GPU PathW currently supports L2 only");
  }
  if (K > beam_size) {
    throw std::runtime_error("GPU PathW requires K <= beam_size");
  }
  if (beam_size <= 0) {
    throw std::runtime_error("GPU PathW requires beam_size > 0");
  }
  const size_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(SEARCH_WIDTH) * static_cast<uint32_t>(max_degree));

  printf("Beam search PathW on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d, padded_beam_size=%d, compiled_internal_topk=%d\n",
         K, active_nq, dim, npoints, beam_size, padded_beam_size, INTERNAL_TOPK);
  printf("Metric: %s\n", metric_name(this->metric));
  printf("PathW params: neighbor_keep_ratio=%.4f, iteration_prune_ratio=%.4f, iteration_limit=beam_size\n",
         neighbor_keep_ratio, iteration_prune_ratio);
  printf("GPU PathW speculation: search_width=%d, candidate_buffer_size=%zu\n",
         SEARCH_WIDTH, candidate_buffer_size);

  size_t sign_n = 0;
  size_t sign_width = 0;
  uint32_t* h_sign_bit_raw = read_bin<uint32_t>(signbit_file, sign_n, sign_width);
  auto signbit_deleter = [sign_width](uint32_t* ptr) {
    release_bin_buffer(ptr, sign_width);
  };
  std::unique_ptr<uint32_t, decltype(signbit_deleter)> h_sign_bit(
      h_sign_bit_raw, signbit_deleter);
  const size_t packed_dim = (static_cast<size_t>(dim) + 31) / 32;
  const size_t expected_width = static_cast<size_t>(max_degree) * packed_dim;
  if (sign_n != npoints || sign_width != expected_width) {
    throw std::runtime_error(
        "PathW signbit shape mismatch: got n=" + std::to_string(sign_n) +
        ", width=" + std::to_string(sign_width) +
        "; expected n=" + std::to_string(npoints) +
        ", width=" + std::to_string(expected_width));
  }

  size_t num_threads = PATHW_BLOCK_SIZE;
  size_t num_blocks = active_nq;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  CUDA_SAFE_CALL(cudaGetDevice(&dev));
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  T* d_queries = nullptr;
  T* d_data = nullptr;
  T* h_data = this->get_data_ptr();
  vid_t* d_results = nullptr;
  uint32_t* d_sign_bit = nullptr;

  size_t free_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));
  const size_t query_bytes = static_cast<size_t>(active_nq) * dim * sizeof(T);
  const size_t data_bytes = static_cast<size_t>(npoints) * dim * sizeof(T);
  const size_t result_bytes = static_cast<size_t>(active_nq) * K * sizeof(vid_t);
  const size_t sign_bytes = static_cast<size_t>(sign_n) * sign_width * sizeof(uint32_t);
  const size_t required_bytes = query_bytes + data_bytes + result_bytes + sign_bytes;
  constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
  printf("GPU memory: free=%.2f GiB, total=%.2f GiB, required=%.2f GiB (data=%.2f GiB, signbit=%.2f GiB)\n",
         static_cast<double>(free_mem_bytes) / kBytesPerGiB,
         static_cast<double>(total_mem_bytes) / kBytesPerGiB,
         static_cast<double>(required_bytes) / kBytesPerGiB,
         static_cast<double>(data_bytes) / kBytesPerGiB,
         static_cast<double>(sign_bytes) / kBytesPerGiB);
  if (required_bytes > free_mem_bytes) {
    throw std::runtime_error("PathW GPU upload exceeds currently available GPU memory");
  }

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_queries, query_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, query_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, result_bytes));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_sign_bit, sign_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit.get(), sign_bytes, cudaMemcpyHostToDevice));

  gpu_long_t* d_total_stats = nullptr;
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  const size_t stats_rows = num_blocks;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_stats,
                            stats_rows * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemset(d_total_stats, 0,
                            stats_rows * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(npoints, max_degree, this->edges.data());

  const vid_t entry_point = compute_gpu_pathw_entry_point(h_data, npoints, dim);
  printf("GPU pathw entry point: %u\n", entry_point);
  vid_t search_entry_point = entry_point;
  if (get_gpu_pathw_seed_override(npoints, search_entry_point)) {
    printf("GPU pathw seed node: %u\n", search_entry_point);
  }

  uint32_t shm_size = calculate_pathw_shared_mem_size(dim, padded_beam_size, bitlen, max_degree);
  int numBlocksPerSM = 0;
  CUDA_SAFE_CALL(cudaFuncSetAttribute(PathWBeamSearch<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shm_size)));
  CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, PathWBeamSearch<T>, num_threads, shm_size));
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  printf("\nStarting GPU PathW search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  PathWBeamSearch<T><<<num_blocks, num_threads, shm_size>>>(
      K, active_nq, dim, beam_size, bitlen, npoints, d_queries, d_data,
      d_sign_bit, d_results, d_total_stats, search_entry_point, gg, neighbor_keep_ratio,
      iteration_prune_ratio);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, result_bytes, cudaMemcpyDeviceToHost));
  std::vector<gpu_long_t> h_block_stats(stats_rows * GPU_SEARCH_STAT_COUNT);
  CUDA_SAFE_CALL(cudaMemcpy(h_block_stats.data(), d_total_stats,
                            h_block_stats.size() * sizeof(gpu_long_t),
                            cudaMemcpyDeviceToHost));
  for (size_t b = 0; b < stats_rows; ++b) {
    for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
      total_stats[i] += h_block_stats[b * GPU_SEARCH_STAT_COUNT + i];
    }
  }
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_sign_bit));
  CUDA_SAFE_CALL(cudaFree(d_total_stats));
  gg.release();

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

template class IndexGraph<float>;
