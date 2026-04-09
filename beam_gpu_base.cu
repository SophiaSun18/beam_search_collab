//#include <cuda_profiler_api.h>
//#define USE_RADIX // Use bitonic if not defined

#include "include/index.hpp"
#ifdef USE_RADIX
#include "include/search_radix.cuh"
#else
#include "include/search_bitonic.cuh"
#endif

#include <chrono>

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T *result_vec, vid_t *result_idx, int beam_sz, double& elapsed) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d\n", K, nq, dim, npoints, beam_size);

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = GPU_DISTANCE_TIMING ? 1 : nq;
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

  vid_t *starting_points = new vid_t[beam_size];
  vid_t *d_starting_points;
  for (int i = 0; i < beam_size; i++) starting_points[i] = rand() % npoints;
  // printf("random nodes generated: ");
  // for (size_t i = 0; i < beam_size; i++) {
  //   printf("%d, ", starting_points[i]);
  // }
  // printf("\n");
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_starting_points, beam_size * sizeof(vid_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, beam_size * sizeof(vid_t), cudaMemcpyHostToDevice));

#ifdef USE_RADIX
  //vid_t *d_filters = nullptr;
  //printf("filter size = %d bytes\n", num_blocks * HASH_FILTER_SIZE * sizeof(vid_t));
  //CUDA_SAFE_CALL(cudaMalloc((void **)&d_filters, num_blocks * HASH_FILTER_SIZE * sizeof(vid_t)));
  BeamSearch<T><<<num_blocks, num_threads>>>(K, nq, dim, npoints, d_queries, d_data,
                                             d_results, d_total_stats, d_total_timing, d_starting_points, d_total_hops, gg);
#else
  uint32_t shm_size = calculate_shared_mem_size(dim, INTERNAL_TOPK, max_degree, SEARCH_WIDTH, BITLEN);
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

  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
                      BeamSearch<T>, blockSize, shm_size);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  // Execute the search
  printf("\nStarting GPU beam search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  BeamSearch<T><<<num_blocks, num_threads, shm_size>>>(K, nq, dim, npoints, d_queries, d_data, d_results, 
                                                       d_total_stats, d_total_timing, d_starting_points, d_seed_map, d_sign_bit, gg);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

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
}

template class IndexGraph<float>;
