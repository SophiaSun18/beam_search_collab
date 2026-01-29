//#include <cuda_profiler_api.h>
//#define USE_RADIX // Use bitonic if not defined

#include "include/index.hpp"
#ifdef USE_RADIX
#include "include/search_radix.cuh"
#else
#include "include/search_bitonic.cuh"
#endif

#define NUM_START BEAM_SIZE // number of starting nodes

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T *result_vec, vid_t *result_idx, int beam_sz) {
  // bool use_ip = (this->metric == METRIC_IP);
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  assert(beam_sz <= BEAM_SIZE); // todo: set BEAM_SIZE as user specified
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d\n", K, nq, dim, npoints, BEAM_SIZE);

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
  size_t dynamicSMemSize = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
                      BeamSearch<T>, blockSize, dynamicSMemSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

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
  gpu_long_t *d_total_count_dc, *d_total_hops;
  gpu_long_t total_count_dc = 0, total_hops = 0; 
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_hops,     sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_count_dc, &total_count_dc, sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_hops,     &total_hops,     sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(npoints, max_degree, this->edges.data()); // GPU graph 

  vid_t *starting_points = new vid_t[NUM_START];
  vid_t *d_starting_points;
  for (int i = 0; i < NUM_START; i++) starting_points[i] = rand() % npoints;
  // printf("random nodes generated: ");
  // for (size_t i = 0; i < NUM_START; i++) {
  //   printf("%d, ", starting_points[i]);
  // }
  // printf("\n");
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_starting_points, NUM_START * sizeof(vid_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, NUM_START * sizeof(vid_t), cudaMemcpyHostToDevice));

#ifdef USE_RADIX
  //vid_t *d_filters = nullptr;
  //printf("filter size = %d bytes\n", num_blocks * HASH_FILTER_SIZE * sizeof(vid_t));
  //CUDA_SAFE_CALL(cudaMalloc((void **)&d_filters, num_blocks * HASH_FILTER_SIZE * sizeof(vid_t)));
  BeamSearch<T><<<num_blocks, num_threads>>>(K, nq, dim, npoints, d_queries, d_data,
                                             d_results, d_starting_points, d_total_count_dc, d_total_hops, gg);
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
  BeamSearch<T><<<num_blocks, num_threads, shm_size>>>(K, nq, dim, npoints, d_queries, d_data, d_results, 
                                                       d_total_count_dc, d_starting_points, d_seed_map, d_sign_bit, gg);
#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, nq * K * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, d_total_count_dc, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_hops, d_total_hops, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_total_hops));
  CUDA_SAFE_CALL(cudaFree(d_total_count_dc));
  std::cout << "total # distance computation: " << total_count_dc << "\n";
  std::cout << "average # distance computation: " << total_count_dc / nq << "\n";
  std::cout << "total # hops: " << total_hops << "\n";
  std::cout << "average # hops: " << total_hops / nq << "\n";
}

template class IndexGraph<float>;