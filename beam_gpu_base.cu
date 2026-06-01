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
#include <vector>

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
                           GpuRunStats* gpu_stats, bool debug_mode) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  auto padded_beam_size = static_cast<int>(gpu_effective_sort_beam_size(static_cast<uint32_t>(beam_size), static_cast<uint32_t>(K)));
  auto bitlen = GPU_HASH_BITLEN_FOR_TOPK(padded_beam_size);
  const bool use_ip = (this->metric == METRIC_IP);
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d, padded_beam_size=%d, internal_topk=%d\n",
         K, nq, dim, npoints, beam_size, padded_beam_size, INTERNAL_TOPK);
  printf("Metric: %s\n", metric_name(this->metric));
  const size_t candidate_work_count = static_cast<size_t>(SEARCH_WIDTH) * max_degree;
  const size_t candidate_buffer_size = round_up_power2_u32(candidate_work_count);
  printf("GPU base speculation: search_width=%d, candidate_work_count=%zu, candidate_buffer_size=%zu\n",
         SEARCH_WIDTH, candidate_work_count, candidate_buffer_size);

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
  std::array<gpu_long_t, GPU_SEARCH_STAT_COUNT> total_stats{};
  const size_t stats_rows = num_blocks;
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_stats, stats_rows * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemset(d_total_stats, 0, stats_rows * GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(npoints, max_degree, this->edges.data()); // GPU graph 

  const int starting_point_count = std::max(beam_size, static_cast<int>(candidate_buffer_size));
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
                                             d_results, d_total_stats, d_starting_points, gg);
#else
  uint32_t shm_size = calculate_shared_mem_size(dim, padded_beam_size, max_degree, SEARCH_WIDTH, bitlen);
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
                                                         d_total_stats, d_starting_points, d_seed_map, d_sign_bit, gg, use_ip);
  CUDA_SAFE_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, nq * K * sizeof(int), cudaMemcpyDeviceToHost));
  if (GPU_SEARCH_STATS) {
    std::vector<gpu_long_t> h_block_stats(stats_rows * GPU_SEARCH_STAT_COUNT);
    CUDA_SAFE_CALL(cudaMemcpy(h_block_stats.data(), d_total_stats,
                              h_block_stats.size() * sizeof(gpu_long_t),
                              cudaMemcpyDeviceToHost));
    for (size_t b = 0; b < stats_rows; ++b) {
      for (int i = 0; i < GPU_SEARCH_STAT_COUNT; ++i) {
        total_stats[i] += h_block_stats[b * GPU_SEARCH_STAT_COUNT + i];
      }
    }
  }
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_starting_points));
  gg.release();
  delete[] starting_points;
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaFree(d_total_stats));
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
