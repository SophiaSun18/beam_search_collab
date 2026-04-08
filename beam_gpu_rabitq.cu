//#include <cuda_profiler_api.h>

#include "include/qg.hpp"
#include "include/search_rabitq.cuh"

#include <chrono>
#include <cstddef>
#include <array>

void QuantizationGraph::gpu_search(int nq, const float* queries, int K, float* result_vec,
                                   vidType* result_idx, int beam_sz, double& elapsed) {
  if (this->is_bins_mode()) {
    throw std::runtime_error("gpu_search currently supports only codebook-mode QuantizationGraph");
  }

  auto dim = this->dim_;
  auto npoints = this->num_nodes_;
  auto max_degree = this->degree_;
  auto beam_size = beam_sz;
  printf("\nBeam search on GPU: K=%d, nq=%d, dim=%ld, npoints=%ld, beam_size=%d\n", K, nq, dim, npoints, beam_size);
  printf("QG debug: entry_point=%u degree=%d padded_dim=%d row_offset=%zu codes_row_words=%zu\n",
         this->get_entry_point(), this->get_degree(), this->get_padded_dim(), this->get_row_offset(), this->get_codes_row_words());
  if (npoints > 0 && max_degree > 1) {
    const float* row0 = this->get_row(this->get_entry_point());
    const uint32_t* nbr0 = this->get_neighbors(this->get_entry_point());
    const float* fac0 = this->get_factor(this->get_entry_point());
    printf("QG debug node0: row[0]=%.6f row[1]=%.6f row[2]=%.6f nbr[0]=%u nbr[1]=%u factor[0]=%.6f factor[1]=%.6f\n",
           row0[0], row0[1], row0[2], nbr0[0], nbr0[1], fac0[0], fac0[1]);
  }

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = nq;
  std::cout << "\nnum_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  int numBlocksPerSM = 0;
  int blockSize = 256;
  size_t dynamicSMemSize = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
                      QuantizedBeamSearch<float>, blockSize, dynamicSMemSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  float* d_queries = nullptr;
  float* d_qg_data = nullptr;
  float* d_qg_signs = nullptr;
  size_t free_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));

  const size_t query_bytes = static_cast<size_t>(nq) * dim * sizeof(float);
  const size_t results_bytes = static_cast<size_t>(nq) * K * sizeof(vidType);
  const size_t counters_bytes = GPU_RABITQ_STAT_COUNT * sizeof(gpu_long_t);
  const size_t qg_data_bytes = this->get_data_bytes();
  const size_t qg_signs_bytes = this->get_signs_bytes();
  const size_t required_bytes = query_bytes + results_bytes + counters_bytes + qg_data_bytes + qg_signs_bytes;

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

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_queries, static_cast<size_t>(nq) * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, static_cast<size_t>(nq) * dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_data, qg_data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_data, this->get_data_ptr(), qg_data_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_qg_signs, qg_signs_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_qg_signs, this->get_signs_ptr(), qg_signs_bytes, cudaMemcpyHostToDevice));

  vidType* d_results = nullptr;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, static_cast<size_t>(nq) * K * sizeof(vidType)));

  gpu_long_t* d_total_stats = nullptr;
  std::array<gpu_long_t, GPU_RABITQ_STAT_COUNT> total_stats{};
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_total_stats, GPU_RABITQ_STAT_COUNT * sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_stats, total_stats.data(), GPU_RABITQ_STAT_COUNT * sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  vidType entry_point = this->get_entry_point();

  uint32_t shm_size = calculate_shared_mem_size(static_cast<int>(dim), INTERNAL_TOPK, max_degree, SEARCH_WIDTH, BITLEN);

  printf("\nStarting GPU rabitq search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  QuantizedBeamSearch<<<num_blocks, num_threads, shm_size>>>(
      K, nq, static_cast<int>(dim), max_degree, npoints, d_queries, d_qg_data, d_qg_signs, d_results,
      d_total_stats, entry_point, this->get_row_offset(), this->get_neighbor_offset(),
      this->get_code_offset(), this->get_factor_offset());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, static_cast<size_t>(nq) * K * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_stats.data(), d_total_stats, GPU_RABITQ_STAT_COUNT * sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_qg_data));
  CUDA_SAFE_CALL(cudaFree(d_qg_signs));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_total_stats));

  std::cout << "total # expanded nodes: " << total_stats[GPU_RABITQ_STAT_EXPANDED] << "\n";
  std::cout << "average # expanded nodes: " << total_stats[GPU_RABITQ_STAT_EXPANDED] / nq << "\n";
  std::cout << "total # neighbors scanned: " << total_stats[GPU_RABITQ_STAT_NEIGHBORS_SCANNED] << "\n";
  std::cout << "average # neighbors scanned: " << total_stats[GPU_RABITQ_STAT_NEIGHBORS_SCANNED] / nq << "\n";
  std::cout << "total # candidates inserted: " << total_stats[GPU_RABITQ_STAT_CANDIDATES_INSERTED] << "\n";
  std::cout << "average # candidates inserted: " << total_stats[GPU_RABITQ_STAT_CANDIDATES_INSERTED] / nq << "\n";
  std::cout << "total # duplicate/visited prunes: " << total_stats[GPU_RABITQ_STAT_DUPLICATE_VISITED_PRUNES] << "\n";
  std::cout << "average # duplicate/visited prunes: " << total_stats[GPU_RABITQ_STAT_DUPLICATE_VISITED_PRUNES] / nq << "\n";
  std::cout << "total # distance computation: " << total_stats[GPU_RABITQ_STAT_DIST_CMPS] << "\n";
  std::cout << "average # distance computation: " << total_stats[GPU_RABITQ_STAT_DIST_CMPS] / nq << "\n";
  std::cout << "total # distance estimation: " << total_stats[GPU_RABITQ_STAT_DIST_EST] << "\n";
  std::cout << "average # distance estimation: " << total_stats[GPU_RABITQ_STAT_DIST_EST] / nq << "\n";
  std::cout << "total # iterations: " << total_stats[GPU_RABITQ_STAT_ITERATIONS] << "\n";
  std::cout << "average # iterations: " << total_stats[GPU_RABITQ_STAT_ITERATIONS] / nq << "\n";

  (void)result_vec;
}
