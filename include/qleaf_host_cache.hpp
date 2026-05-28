#pragma once

#include "index.hpp"
#include "utils.cuh"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace beam_gpu_host {

// ─── Logical substructures within DeviceCache ─────────────────────────────
// DeviceCache groups its device-owned RouteA buffers under two
// substructures: edge_sidecar (per-edge RaBitQ used by GATE=1 and NAV=8)
// and query_prep (per-query q6 bitplanes + factors). Per-node sidecar
// fields were removed when NAV 1-7 paths were dropped from the
// formal-only Q-LEAF surface.

template <typename T>
struct BaseGpuBuffers {
  T* d_data = nullptr;
  vid_t* d_edges = nullptr;
  T* d_queries = nullptr;
  size_t queries_bytes = 0;
  vid_t* d_results = nullptr;
  size_t results_bytes = 0;
};

struct RouteAEdgeSidecarDevice {
  uint32_t* d_sign_bit = nullptr;
  size_t sign_words = 0;
  int sign_words_per_edge = 0;
  float* d_rabitq_triple = nullptr;
  float* d_rabitq_factor_dq = nullptr;
  float* d_rabitq_factor_vq = nullptr;
  size_t rabitq_factor_count = 0;
};

struct RouteAQueryPrepDevice {
  uint32_t* d_routea_query_bp = nullptr;
  size_t routea_query_bp_words = 0;
  float* d_routea_query_lower = nullptr;
  float* d_routea_query_width = nullptr;
  int* d_routea_query_sumq = nullptr;
  size_t routea_query_count = 0;
  int routea_query_score_words = 0;
  int routea_query_score_dims = 0;
};

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
  gpu_long_t* d_routea_profile_counters = nullptr;
  vid_t* d_starting_points = nullptr;
  size_t starting_points_bytes = 0;

  // Grouped device-owned RouteA buffers.
  RouteAEdgeSidecarDevice edge_sidecar;
  RouteAQueryPrepDevice query_prep;

  // QG metadata. Used by query-prep rotation / score_dims.
  std::vector<float> qg_signs;
  float* d_qg_signs = nullptr;
  size_t qg_signs_count = 0;
  int qg_padded_dim = 0;
  bool routea_bitplane_qg_loaded = false;

  // Cached graph shape; set when the graph is uploaded so subsequent
  // calls can detect a shape change and rebuild the cache.
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

// Free helpers per substructure. Idempotent — safe to call twice.

inline void release_edge_sidecar(RouteAEdgeSidecarDevice& s) {
  cuda_free_if(s.d_sign_bit);
  s.sign_words = 0;
  s.sign_words_per_edge = 0;
  cuda_free_if(s.d_rabitq_triple);
  cuda_free_if(s.d_rabitq_factor_dq);
  cuda_free_if(s.d_rabitq_factor_vq);
  s.rabitq_factor_count = 0;
}

inline void release_query_prep(RouteAQueryPrepDevice& q) {
  cuda_free_if(q.d_routea_query_bp);
  q.routea_query_bp_words = 0;
  cuda_free_if(q.d_routea_query_lower);
  cuda_free_if(q.d_routea_query_width);
  cuda_free_if(q.d_routea_query_sumq);
  q.routea_query_count = 0;
  q.routea_query_score_words = 0;
  q.routea_query_score_dims = 0;
}

// Explicit cache release helper. Frees every device-owned buffer in
// `DeviceCache<T>`. NOT called automatically — the current benchmark
// process owns the cache until process exit. Static cache map lifetime
// around CUDA shutdown is fragile, so we keep this explicit and do not
// add a destructor.
template <typename T>
inline void release_device_cache(DeviceCache<T>& cache) {
  cuda_free_if(cache.d_data);
  cuda_free_if(cache.d_edges);
  cuda_free_if(cache.d_queries);
  cache.queries_bytes = 0;
  cuda_free_if(cache.d_results);
  cache.results_bytes = 0;
  cuda_free_if(cache.d_total_count_dc);
  cuda_free_if(cache.d_total_hops);
  cuda_free_if(cache.d_routea_profile_counters);
  cuda_free_if(cache.d_starting_points);
  cache.starting_points_bytes = 0;
  release_edge_sidecar(cache.edge_sidecar);
  release_query_prep(cache.query_prep);
  cache.qg_signs.clear();
  cuda_free_if(cache.d_qg_signs);
  cache.qg_signs_count = 0;
  cache.qg_padded_dim = 0;
  cache.routea_bitplane_qg_loaded = false;
  cache.npoints = 0;
  cache.dim = 0;
  cache.max_degree = 0;
}

}  // namespace beam_gpu_host
