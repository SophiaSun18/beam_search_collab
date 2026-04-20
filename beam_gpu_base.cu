//#include <cuda_profiler_api.h>
//#define USE_RADIX // Use bitonic if not defined

#include "include/index.hpp"
#ifdef USE_RADIX
#include "include/search_radix.cuh"
#else
#include "include/search_bitonic.cuh"
#endif

#include <algorithm>
#include <chrono>

namespace {

struct BeamBucketConfig {
  int internal_topk;
  int bitlen;
};

inline BeamBucketConfig select_beam_bucket(int beam_size) {
  if (beam_size <= 128) return {128, 10};
  if (beam_size <= 256) return {256, 10};
  if (beam_size <= 512) return {512, 10};
  return {1024, 12};
}

template <typename T, int kInternalTopk, int kBitlen>
double launch_gpu_base_bucket(int K, int nq, int dim, size_t npoints, int beam_size,
                              size_t num_blocks, size_t num_threads, const T* d_queries, const T* d_data,
                              vid_t* d_results, gpu_long_t* d_total_stats, gpu_long_t* d_total_timing,
                              vid_t* d_seed_map, uint32_t* d_sign_bit, GraphGPU<vid_t> gg) {
  const int active_beam = std::max(1, std::min(beam_size, kInternalTopk));
  if (beam_size <= 0) {
    printf("WARNING: beam_size=%d is invalid, clamping to 1 (bucket_topk=%d, bucket_bitlen=%d)\n",
           beam_size, kInternalTopk, kBitlen);
  } else if (beam_size > kInternalTopk) {
    printf("WARNING: beam_size=%d exceeds bucket_topk=%d, clamping to %d (bucket_bitlen=%d)\n",
           beam_size, kInternalTopk, kInternalTopk, kBitlen);
  }
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d, active_beam=%d, bucket_topk=%d, bucket_bitlen=%d\n",
         K, nq, dim, npoints, beam_size, active_beam, kInternalTopk, kBitlen);

  int num_starting = std::max(active_beam, kCandidateBufferSize);
  vid_t *starting_points = new vid_t[num_starting];
  vid_t *d_starting_points = nullptr;
  for (int i = 0; i < num_starting; i++) starting_points[i] = rand() % npoints;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_starting_points, num_starting * sizeof(vid_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, num_starting * sizeof(vid_t), cudaMemcpyHostToDevice));

  int numBlocksPerSM = 0;
  int blockSize = 256;
  uint32_t shm_size = calculate_shared_mem_size<kInternalTopk, kBitlen>(dim);
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, BeamSearch<T, kInternalTopk, kBitlen>, blockSize, shm_size);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    delete[] starting_points;
    CUDA_SAFE_CALL(cudaFree(d_starting_points));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  printf("\nStarting GPU beam search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  BeamSearch<T, kInternalTopk, kBitlen><<<num_blocks, num_threads, shm_size>>>(
      K, nq, dim, npoints, d_queries, d_data, d_results, d_total_stats, d_total_timing,
      d_starting_points, d_seed_map, d_sign_bit, gg, active_beam);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  CUDA_SAFE_CALL(cudaFree(d_starting_points));
  delete[] starting_points;
  return std::chrono::duration<double>(end - start).count();
}

}  // namespace

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T *result_vec, vid_t *result_idx, int beam_sz, double& elapsed,
                           GpuRunStats* gpu_stats) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  const BeamBucketConfig bucket = select_beam_bucket(beam_size);

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = GPU_DISTANCE_TIMING ? 1 : nq;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

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
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_stats, GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_total_stats, total_stats.data(), GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  }
  gpu_long_t *d_total_timing;
  std::array<gpu_long_t, GPU_TIMING_COUNT> total_timing{};
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_timing, GPU_TIMING_COUNT * sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_timing, total_timing.data(), GPU_TIMING_COUNT * sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(npoints, max_degree, this->edges.data()); // GPU graph 
  vid_t *d_seed_map = nullptr;
  uint32_t *d_sign_bit = nullptr;

#ifdef USE_RADIX
  //vid_t *d_filters = nullptr;
  //printf("filter size = %d bytes\n", num_blocks * HASH_FILTER_SIZE * sizeof(vid_t));
  //CUDA_SAFE_CALL(cudaMalloc((void **)&d_filters, num_blocks * HASH_FILTER_SIZE * sizeof(vid_t)));
  BeamSearch<T><<<num_blocks, num_threads>>>(K, nq, dim, npoints, d_queries, d_data,
                                             d_results, d_total_stats, d_total_timing, nullptr, d_total_hops, gg);
#else
  if (kUseSeed) {
    size_t ghost_scale_factor = 100;
    size_t ghost_num = npoints / ghost_scale_factor;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_seed_map, ghost_num * sizeof(vid_t)));
    init_identity_map<<<(ghost_num-1)/num_threads+1, num_threads>>>(ghost_num, d_seed_map);
  }
  if (kPruneConfig) {
    auto sign_bit = load_sign_bit("sign_bit.bin");
    size_t sign_bit_size = sign_bit.size();
    uint32_t *h_sign_bit = sign_bit.data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sign_bit, sign_bit_size * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit, sign_bit_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  switch (bucket.internal_topk) {
    case 128:
      elapsed = launch_gpu_base_bucket<T, 128, 10>(
          K, nq, dim, npoints, beam_size, num_blocks, num_threads, d_queries, d_data,
          d_results, d_total_stats, d_total_timing, d_seed_map, d_sign_bit, gg);
      break;
    case 256:
      elapsed = launch_gpu_base_bucket<T, 256, 10>(
          K, nq, dim, npoints, beam_size, num_blocks, num_threads, d_queries, d_data,
          d_results, d_total_stats, d_total_timing, d_seed_map, d_sign_bit, gg);
      break;
    case 512:
      elapsed = launch_gpu_base_bucket<T, 512, 10>(
          K, nq, dim, npoints, beam_size, num_blocks, num_threads, d_queries, d_data,
          d_results, d_total_stats, d_total_timing, d_seed_map, d_sign_bit, gg);
      break;
    case 1024:
      elapsed = launch_gpu_base_bucket<T, 1024, 12>(
          K, nq, dim, npoints, beam_size, num_blocks, num_threads, d_queries, d_data,
          d_results, d_total_stats, d_total_timing, d_seed_map, d_sign_bit, gg);
      break;
    default:
      printf("Unsupported beam bucket INTERNAL_TOPK=%d\n", bucket.internal_topk);
      exit(-1);
  }

#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, nq * K * sizeof(int), cudaMemcpyDeviceToHost));
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaMemcpy(total_stats.data(), d_total_stats, GPU_SEARCH_STAT_COUNT * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  }
  CUDA_SAFE_CALL(cudaMemcpy(total_timing.data(), d_total_timing, GPU_TIMING_COUNT * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  if (GPU_SEARCH_STATS) {
    CUDA_SAFE_CALL(cudaFree(d_total_stats));
  }
  CUDA_SAFE_CALL(cudaFree(d_total_timing));
  if (d_seed_map != nullptr) {
    CUDA_SAFE_CALL(cudaFree(d_seed_map));
  }
  if (d_sign_bit != nullptr) {
    CUDA_SAFE_CALL(cudaFree(d_sign_bit));
  }
  if (GPU_SEARCH_STATS) {
    std::cout << "total # exact distance computations: " << total_stats[GPU_SEARCH_STAT_DIST_CMPS] << "\n";
    std::cout << "average # exact distance computations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_CMPS]) / nq << "\n";
    std::cout << "total # distance estimations: " << total_stats[GPU_SEARCH_STAT_DIST_EST] << "\n";
    std::cout << "average # distance estimations: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_DIST_EST]) / nq << "\n";
    std::cout << "total # expansion steps: " << total_stats[GPU_SEARCH_STAT_ITERATIONS] << "\n";
    std::cout << "average # expansion steps: " << static_cast<double>(total_stats[GPU_SEARCH_STAT_ITERATIONS]) / nq << "\n";
  }
  if (GPU_DISTANCE_TIMING) {
    const double avg_l2_cycles =
        total_timing[GPU_TIMING_L2_CALLS] ? static_cast<double>(total_timing[GPU_TIMING_L2_CYCLES]) / total_timing[GPU_TIMING_L2_CALLS] : 0.0;
    std::cout << "total # l2 timing calls: " << total_timing[GPU_TIMING_L2_CALLS] << "\n";
    std::cout << "average cycles per warp_l2_distance: " << avg_l2_cycles << "\n";
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
}

template class IndexGraph<float>;
