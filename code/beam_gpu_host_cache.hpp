#pragma once

#include "index.hpp"
#include "utils.cuh"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace beam_gpu_host {

template <typename T>
struct DeviceCache {
  T* d_data = nullptr;
  vid_t* d_edges = nullptr;

  T* d_queries = nullptr;
  size_t queries_bytes = 0;
  vid_t* d_results = nullptr;
  size_t results_bytes = 0;
  gpu_long_t* d_total_count_dc = nullptr;
  gpu_long_t* d_total_hops = nullptr;
  vid_t* d_starting_points = nullptr;
  size_t starting_points_bytes = 0;
  uint16_t* d_routea_query_lut = nullptr;
  size_t routea_query_lut_words = 0;
  float* d_routea_query_lower = nullptr;
  float* d_routea_query_width = nullptr;
  int* d_routea_query_sumq = nullptr;
  size_t routea_query_count = 0;
  uint32_t* d_sign_bit = nullptr;
  size_t sign_words = 0;
  int sign_words_per_edge = 0;
  float* d_rabitq_triple = nullptr;
  float* d_rabitq_factor_dq = nullptr;
  size_t rabitq_factor_count = 0;
  size_t npoints = 0;
  int dim = 0;
  int max_degree = 0;
};

template <typename T>
inline DeviceCache<T>& cache_for(const IndexGraph<T>* idx) {
  static std::unordered_map<const void*, DeviceCache<T>> cache_map;
  return cache_map[static_cast<const void*>(idx)];
}

template <typename PtrT>
inline void cuda_free_if(PtrT*& p) {
  if (p) {
    CUDA_SAFE_CALL(cudaFree(p));
    p = nullptr;
  }
}

}  // namespace beam_gpu_host
