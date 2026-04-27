#include "index.hpp"
#include "gpu_base_search.cuh"
#include "cpu_graph.hpp"
#include "gpu_graph.cuh"

#include <chrono>
#include <array>
#include <cfloat>
#include <vector>

static constexpr unsigned BASE_BLOCK_SIZE = BLOCK_SIZE;  // from utils.cuh (256)

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
    float dist = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      const float diff = row[d] - centroid[d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<vid_t>(i);
    }
  }
  return best;
}

// ── kernel ──────────────────────────────────────────────────────────────────

__global__ void gpu_base_search_kernel(
    GPUGraphView graph,
    GPUBaseSearchCfg cfg,
    int nq, int dim,
    const float* __restrict__ d_queries,
    vid_t* __restrict__ d_results,
    int K)
{
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;

    extern __shared__ char smem[];
    char* smem_ptr = smem;

    GPUBaseSearch<BASE_BLOCK_SIZE> searcher(smem_ptr, cfg, graph);
    __syncthreads();

    searcher.search(d_queries + query_id * dim, d_results + query_id * K, K);
}

// ── host entry point ────────────────────────────────────────────────────────

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T* result_vec,
                           vid_t* result_idx, int beam_sz, double& elapsed,
                           GpuRunStats* gpu_stats, bool debug_mode) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  printf("GPUBaseSearch: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d\n",
         K, nq, dim, npoints, beam_sz);

  // build CPUGraph → GPUGraph
  CPUGraph cpu_graph(npoints, dim, max_degree,
                     std::vector<float>(this->data),
                     std::vector<vid_t>(this->edges));
  GPUGraph gpu_graph(cpu_graph);

  // entry point
  const vid_t entry_point = compute_gpu_base_entry_point(
      this->get_data_ptr(), npoints, dim);
  printf("GPU base entry point: %u\n", entry_point);

  // search config
  GPUBaseSearchCfg cfg;
  cfg.topk_size      = BASE_BLOCK_SIZE;  // must be >= BX * REG
  cfg.bitlen         = 10;
  cfg.entry_point    = entry_point;
  cfg.max_iterations = 1024;
  cfg.reset_interval = 16;

  // shared memory size
  size_t shm_size = GPUBaseSearch<BASE_BLOCK_SIZE>::shared_mem_bytes(cfg, dim);
  printf("shared memory size: %ld\n", shm_size);

  // device memory for queries and results
  float* d_queries;
  vid_t* d_results;
  CUDA_SAFE_CALL(cudaMalloc(&d_queries, nq * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, nq * dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc(&d_results, nq * K * sizeof(vid_t)));

  // occupancy
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
  int numBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, gpu_base_search_kernel, BASE_BLOCK_SIZE, shm_size);
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  // launch
  size_t num_blocks = nq;
  printf("\nStarting GPU base beam search...\n");
  auto start = std::chrono::high_resolution_clock::now();
  gpu_base_search_kernel<<<num_blocks, BASE_BLOCK_SIZE, shm_size>>>(
      gpu_graph.view(), cfg, nq, dim, d_queries, d_results, K);
  CUDA_SAFE_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(end - start).count();

  // copy results back
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, nq * K * sizeof(vid_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_results));

  if (gpu_stats != nullptr) {
    gpu_stats->common.time_search_total = elapsed;
    gpu_stats->common.total_dc = 0;
    gpu_stats->common.total_ds = 0;
    gpu_stats->common.total_extra_dc = 0;
    gpu_stats->common.total_full_dc = 0;
    gpu_stats->common.total_iter = 0;
    gpu_stats->common.total_pruned = 0;
    gpu_stats->gpu_l2_calls = 0;
    gpu_stats->gpu_l2_cycles = 0;
    gpu_stats->gpu_est_calls = 0;
    gpu_stats->gpu_est_cycles = 0;
  }
  (void)result_vec;
  (void)beam_sz;
  (void)debug_mode;
}

template class IndexGraph<float>;
