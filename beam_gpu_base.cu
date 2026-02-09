#include "index.hpp"
#ifdef USE_RADIX
#include "search_radix.cuh"
#else
#include "search_bitonic.cuh"
#endif
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <cstdint>
#include <cmath>

#if ENABLE_SQ8
static bool build_sq8_data(
    const float* data, size_t npoints, int dim,
    std::vector<float>& scale,
    std::vector<float>& scale2,
    std::vector<int8_t>& data_sq8) {
  if (dim != 128) {
    return false;
  }
  scale.assign(dim, 1.0f);
  scale2.assign(dim, 1.0f);
  std::vector<float> max_abs(dim, 0.0f);

  for (size_t i = 0; i < npoints; i++) {
    const float* v = data + i * (size_t)dim;
    for (int d = 0; d < dim; d++) {
      const float a = std::fabs(v[d]);
      if (a > max_abs[d]) max_abs[d] = a;
    }
  }
  for (int d = 0; d < dim; d++) {
    const float m = max_abs[d];
    const float s = (m > 0.0f) ? (m / 127.0f) : 1.0f;
    scale[d] = s;
    scale2[d] = s * s;
  }

  data_sq8.resize(npoints * (size_t)dim);
  for (size_t i = 0; i < npoints; i++) {
    const float* v = data + i * (size_t)dim;
    int8_t* dst = data_sq8.data() + i * (size_t)dim;
    for (int d = 0; d < dim; d++) {
      const float inv = 1.0f / scale[d];
      int q = (int)lrintf(v[d] * inv);
      if (q < -127) q = -127;
      if (q > 127) q = 127;
      dst[d] = (int8_t)q;
    }
  }
  return true;
}
#endif

#if ENABLE_DGS
static std::vector<uint32_t> build_dir_sign_bits_host(
    const float* data, size_t npoints, int dim,
    const std::vector<vid_t>& edges, int max_degree) {
  const int W = (dim + 31) >> 5;
  const size_t total_edges = (size_t)npoints * (size_t)max_degree;
  std::vector<uint32_t> out(total_edges * (size_t)W);

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int64_t uu = 0; uu < (int64_t)npoints; uu++) {
    const size_t u = (size_t)uu;
    const float* uvec = data + u * (size_t)dim;
    const size_t edge_base = u * (size_t)max_degree;
    for (int j = 0; j < max_degree; j++) {
      const vid_t v = edges[edge_base + (size_t)j];
      uint32_t* dst = out.data() + (edge_base + (size_t)j) * (size_t)W;
      if ((size_t)v >= npoints) {
        std::memset(dst, 0, (size_t)W * sizeof(uint32_t));
        continue;
      }
      const float* vvec = data + (size_t)v * (size_t)dim;
      for (int w = 0; w < W; w++) {
        uint32_t word = 0;
        const int base_dim = w << 5;
        for (int b = 0; b < 32; b++) {
          const int i = base_dim + b;
          if (i >= dim) break;
          const float diff = vvec[i] - uvec[i];
          const uint32_t bit = (diff >= 0.0f) ? 1u : 0u;
          word |= (bit << b);
        }
        dst[w] = word;
      }
    }
  }
  return out;
}
#endif

namespace {
template <typename T>
struct DeviceCache {
  T* d_data = nullptr;
  vid_t* d_edges = nullptr;
#if ENABLE_SQ8
  int8_t* d_data_sq8 = nullptr;
  size_t data_sq8_bytes = 0;
#endif
#if 1
  T* d_queries = nullptr;
  size_t queries_bytes = 0;
  vid_t* d_results = nullptr;
  size_t results_bytes = 0;
  gpu_long_t* d_total_count_dc = nullptr;
  gpu_long_t* d_total_hops = nullptr;
  vid_t* d_starting_points = nullptr;
#endif
#if ENABLE_DGS
  uint32_t* d_sign_bit = nullptr;
  size_t sign_words = 0;
#endif
  size_t npoints = 0;
  int dim = 0;
  int max_degree = 0;
};

template <typename T>
static DeviceCache<T>& cache_for(const IndexGraph<T>* idx) {
  static std::unordered_map<const void*, DeviceCache<T>> m;
  return m[(const void*)idx];
}

template <typename PtrT>
static inline void cuda_free_if(PtrT*& p) {
  if (p) {
    CUDA_SAFE_CALL(cudaFree(p));
    p = nullptr;
  }
}
} // namespace

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, T *result_vec, vid_t *result_idx, int beam_sz) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  assert(beam_sz > 0 && beam_sz <= BEAM_SIZE);
  assert(K > 0 && K <= beam_sz);
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d (compile-time max=%d)\n", K, nq, dim, npoints, beam_sz, BEAM_SIZE);
#if ENABLE_DGS
  printf("DGS compile-time: ENABLE_DGS=1, DGS_TOPN=%d, DGS_COOL_RATIO=%f, DGS_EST_ITERS=%d, cool_start=%d, DGS_MAX_WORDS=%d\n",
         (int)DGS_TOPN, (double)DGS_COOL_RATIO, (int)DGS_EST_ITERS,
         (int)((1.0f - (float)DGS_COOL_RATIO) * (float)DGS_EST_ITERS), (int)DGS_MAX_WORDS);
#else
  printf("DGS compile-time: ENABLE_DGS=0\n");
#endif
#ifndef SEARCH_NUM_THREADS
#define SEARCH_NUM_THREADS BLOCK_SIZE
#endif
  size_t num_threads = (size_t)SEARCH_NUM_THREADS;
  size_t num_blocks = nq;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
  int numBlocksPerSM = 0;
  int blockSize = (int)num_threads;   // threads per block
  size_t dynamicSMemSize = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
                      BeamSearch<T>, blockSize, dynamicSMemSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);
  auto& cache = cache_for(this);
  const bool cache_mismatch =
      (cache.d_data == nullptr) ||
      (cache.d_edges == nullptr) ||
#if ENABLE_DGS
      false ||
#endif
      (cache.npoints != npoints) || (cache.dim != dim) || (cache.max_degree != max_degree);
  if (cache_mismatch) {
    // Reset any stale cache if index shape changed.
    cuda_free_if(cache.d_data);
    cuda_free_if(cache.d_edges);
#if ENABLE_SQ8
    cuda_free_if(cache.d_data_sq8);
    cache.data_sq8_bytes = 0;
#endif
#if 1
    cuda_free_if(cache.d_queries);
    cache.queries_bytes = 0;
    cuda_free_if(cache.d_results);
    cache.results_bytes = 0;
    cuda_free_if(cache.d_total_count_dc);
    cuda_free_if(cache.d_total_hops);
    cuda_free_if(cache.d_starting_points);
#endif
#if ENABLE_DGS
    cuda_free_if(cache.d_sign_bit);
    cache.sign_words = 0;
#endif
    cache.npoints = npoints;
    cache.dim = dim;
    cache.max_degree = max_degree;
  }
  T* h_data = this->get_data_ptr();
  if (cache.d_data == nullptr) {
    const size_t bytes = (size_t)npoints * (size_t)dim * sizeof(T);
    printf("GPU cache: uploading data vectors (%zu bytes)\n", bytes);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_data, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(cache.d_data, h_data, bytes, cudaMemcpyHostToDevice));
  }
#if ENABLE_SQ8
  if (dim == 128 && cache.d_data_sq8 == nullptr) {
    std::vector<float> scale, scale2;
    std::vector<int8_t> data_sq8;
    if (build_sq8_data((const float*)h_data, npoints, dim, scale, scale2, data_sq8)) {
      const size_t bytes = (size_t)npoints * (size_t)dim * sizeof(int8_t);
      printf("GPU cache: uploading SQ8 data vectors (%zu bytes)\n", bytes);
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_data_sq8, bytes));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_data_sq8, data_sq8.data(), bytes, cudaMemcpyHostToDevice));
      cache.data_sq8_bytes = bytes;
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_sq8_scale, scale.data(), dim * sizeof(float)));
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_sq8_scale2, scale2.data(), dim * sizeof(float)));
    } else {
      printf("SQ8: disabled (dim=%d not supported)\n", dim);
    }
  }
#endif
  if (cache.d_edges == nullptr) {
    const size_t bytes = (size_t)npoints * (size_t)max_degree * sizeof(vid_t);
    printf("GPU cache: uploading graph edges (%zu bytes)\n", bytes);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_edges, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(cache.d_edges, this->edges.data(), bytes, cudaMemcpyHostToDevice));
  }
  const size_t q_bytes = (size_t)nq * (size_t)dim * sizeof(T);
  if (cache.d_queries == nullptr || cache.queries_bytes < q_bytes) {
    cuda_free_if(cache.d_queries);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_queries, q_bytes));
    cache.queries_bytes = q_bytes;
  }
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_queries, queries, q_bytes, cudaMemcpyHostToDevice));

  vid_t *h_results = &result_idx[0];
  const size_t r_bytes = (size_t)nq * (size_t)K * sizeof(vid_t);
  if (cache.d_results == nullptr || cache.results_bytes < r_bytes) {
    cuda_free_if(cache.d_results);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_results, r_bytes));
    cache.results_bytes = r_bytes;
  }

  gpu_long_t total_count_dc = 0, total_hops = 0; 
  if (cache.d_total_count_dc == nullptr) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_total_count_dc, sizeof(gpu_long_t)));
  }
  if (cache.d_total_hops == nullptr) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_total_hops, sizeof(gpu_long_t)));
  }
  CUDA_SAFE_CALL(cudaMemset(cache.d_total_count_dc, 0, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemset(cache.d_total_hops, 0, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg((vid_t)npoints, (vid_t)max_degree, cache.d_edges); // GPU graph (device-resident)

  std::vector<vid_t> starting_points((size_t)beam_sz);
  for (int i = 0; i < beam_sz; i++) starting_points[i] = rand() % (vid_t)npoints;
  if (cache.d_starting_points == nullptr) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_starting_points, BEAM_SIZE * sizeof(vid_t)));
  }
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_starting_points, starting_points.data(),
                            (size_t)beam_sz * sizeof(vid_t), cudaMemcpyHostToDevice));

#ifdef USE_RADIX
  BeamSearch<T><<<num_blocks, num_threads>>>(K, beam_sz, nq, dim, npoints, d_queries, d_data,
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
  uint32_t* d_sign_bit = nullptr;
#if ENABLE_DGS
  // Upload (once) the sign-bit table to device.
  if (cache.d_sign_bit == nullptr) {
    const std::string primary_path = this->graph_prefix.empty()
                                       ? std::string("sign_bit.bin")
                                       : (this->graph_prefix + ".signbit.bin");
    const std::string fallback_path = "sign_bit.bin";
    const int W = (dim + 31) >> 5;
    const size_t expected_words = (size_t)npoints * (size_t)max_degree * (size_t)W;

    const char* loaded_from = primary_path.c_str();
    std::vector<uint32_t> sign_bit = load_sign_bit(loaded_from);
    if (sign_bit.size() != expected_words && primary_path != fallback_path) {
      auto alt = load_sign_bit(fallback_path.c_str());
      if (alt.size() == expected_words) {
        sign_bit.swap(alt);
        loaded_from = fallback_path.c_str();
        printf("DGS: loaded sign-bit table from fallback %s (%zu bytes)\n",
               fallback_path.c_str(), sign_bit.size() * sizeof(uint32_t));
      }
    }
    if (sign_bit.size() != expected_words) {
      printf("DGS: building sign-bit table on host (expected_words=%zu, got=%zu)\n",
             expected_words, sign_bit.size());
      printf("DGS: primary sign-bit path = %s\n", primary_path.c_str());
      sign_bit = build_dir_sign_bits_host((const float*)h_data, (size_t)npoints, dim, this->edges, max_degree);
      if (sign_bit.size() == expected_words) {
        bool saved_primary = save_sign_bit(primary_path.c_str(), sign_bit.data(), sign_bit.size());
        printf("DGS: save %s %s (%zu bytes)\n", primary_path.c_str(), saved_primary ? "OK" : "FAILED",
               sign_bit.size() * sizeof(uint32_t));
        if (!saved_primary && primary_path != fallback_path) {
          bool saved_fallback = save_sign_bit(fallback_path.c_str(), sign_bit.data(), sign_bit.size());
          printf("DGS: save fallback %s %s\n", fallback_path.c_str(), saved_fallback ? "OK" : "FAILED");
        }
      }
    } else {
      printf("DGS: loaded sign-bit table from %s (%zu bytes)\n",
             loaded_from, sign_bit.size() * sizeof(uint32_t));
    }
    if (sign_bit.size() == expected_words) {
      printf("GPU cache: uploading DGS sign-bit table (%zu bytes)\n", expected_words * sizeof(uint32_t));
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_sign_bit, expected_words * sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_sign_bit, sign_bit.data(),
                                expected_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
      cache.sign_words = expected_words;
    } else {
      printf("DGS: sign-bit table unavailable; running without DGS.\n");
      cache.d_sign_bit = nullptr;
      cache.sign_words = 0;
    }
  }
  d_sign_bit = cache.d_sign_bit;
#else
  if (PRUNE_CONFIG) {
    auto sign_bit = load_sign_bit("sign_bit.bin");
    size_t sign_bit_size = sign_bit.size();
    uint32_t *h_sign_bit = sign_bit.data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sign_bit, sign_bit_size * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit, sign_bit_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }
#endif
  // Measure kernel+sync time only (exclude sign-bit build/load).
  cudaEvent_t ev0, ev1;
  CUDA_SAFE_CALL(cudaEventCreate(&ev0));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1));
  CUDA_SAFE_CALL(cudaEventRecord(ev0));
  BeamSearch<T><<<num_blocks, num_threads, shm_size>>>(K, beam_sz, nq, dim, npoints, cache.d_queries, cache.d_data,
#if ENABLE_SQ8
                                                       cache.d_data_sq8,
#else
                                                       (const int8_t*)nullptr,
#endif
                                                       cache.d_results,
                                                       cache.d_total_count_dc, cache.d_starting_points, d_seed_map, d_sign_bit, gg);
  CUDA_SAFE_CALL(cudaEventRecord(ev1));
  CUDA_SAFE_CALL(cudaEventSynchronize(ev1));
  float kernel_ms = 0.0f;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
  CUDA_SAFE_CALL(cudaEventDestroy(ev0));
  CUDA_SAFE_CALL(cudaEventDestroy(ev1));
  printf("GPU kernel time (BeamSearch + sync): %.3f ms\n", kernel_ms);
#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, cache.d_results, (size_t)nq * (size_t)K * sizeof(vid_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, cache.d_total_count_dc, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_hops, cache.d_total_hops, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  if (d_seed_map) CUDA_SAFE_CALL(cudaFree(d_seed_map));
#if ENABLE_DGS
  // d_sign_bit is cached on device; do not free here.
#endif
  std::cout << "total # distance computation: " << total_count_dc << "\n";
  std::cout << "average # distance computation: " << total_count_dc / nq << "\n";
  std::cout << "total # hops: " << total_hops << "\n";
  std::cout << "average # hops: " << total_hops / nq << "\n";
}

template class IndexGraph<float>;
