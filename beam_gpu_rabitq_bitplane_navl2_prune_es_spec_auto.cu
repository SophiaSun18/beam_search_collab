#include "index.hpp"
#include "search_rabitq_bitplane_navl2_prune_es_spec_auto.cuh"
#include "qleaf_host_cache.hpp"
#include "qleaf_sidecars.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cfloat>

namespace {
// Phase 3.1 of the Q-LEAF architecture hardening plan: shared host types
// (`StableRabitqSidecars`, `RouteAQuerySidecars`, `RouteAQueryPrepTiming`,
// `HostTimePoint` plus `host_now` / `host_elapsed_ms` /
// `routea_runtime_enabled`) moved into `beam_gpu_routea_types.hpp`. The
// types are visible at file scope, so unqualified names inside this
// anonymous namespace continue to resolve correctly via ordinary lookup.
//
// Phase 3.2: `beam_gpu_routea_sidecars.hpp` is now included at file scope
// (below, after this anonymous namespace closes) instead of from inside
// it. Helpers in the sidecar header no longer depend on file-scope
// context that only this TU provided.

inline bool routea_timing_breakdown_enabled() {
  const char* raw = std::getenv("ROUTEA_TIMING_BREAKDOWN");
  return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

struct RouteASearchTiming {
  double reset_cache_ms = 0.0;
  double graph_cache_ms = 0.0;
  double query_buffer_ms = 0.0;
  double counter_reset_ms = 0.0;
  double starting_points_ms = 0.0;
  double rabitq_sidecars_ms = 0.0;
  double query_sidecar_total_ms = 0.0;
  double query_sidecar_host_ms = 0.0;
  double query_sidecar_alloc_ms = 0.0;
  double query_sidecar_upload_ms = 0.0;
  double query_sidecar_gpu_ms = 0.0;
  double kernel_host_ms = 0.0;
  double kernel_event_ms = 0.0;
  double result_copy_ms = 0.0;
  double total_ms = 0.0;
};

inline void print_routea_timing_breakdown(
    int call_id,
    int nq,
    int dim,
    int K,
    int beam_sz,
    const RouteASearchTiming& t) {
  printf("ROUTEA_TIMING_BREAKDOWN call=%d nq=%d dim=%d K=%d beam=%d "
         "total_ms=%.6f reset_cache_ms=%.6f graph_cache_ms=%.6f "
         "query_buffer_ms=%.6f counter_reset_ms=%.6f starting_points_ms=%.6f "
         "rabitq_sidecars_ms=%.6f query_sidecar_total_ms=%.6f "
         "query_sidecar_host_ms=%.6f query_sidecar_alloc_ms=%.6f "
         "query_sidecar_upload_ms=%.6f query_sidecar_gpu_ms=%.6f "
         "kernel_host_ms=%.6f kernel_event_ms=%.6f "
         "result_copy_ms=%.6f\n",
         call_id, nq, dim, K, beam_sz,
         t.total_ms, t.reset_cache_ms, t.graph_cache_ms,
         t.query_buffer_ms, t.counter_reset_ms, t.starting_points_ms,
         t.rabitq_sidecars_ms, t.query_sidecar_total_ms,
         t.query_sidecar_host_ms, t.query_sidecar_alloc_ms,
         t.query_sidecar_upload_ms, t.query_sidecar_gpu_ms,
         t.kernel_host_ms, t.kernel_event_ms,
         t.result_copy_ms);
}

inline bool consume_runtime_log_once() {
  static bool s_runtime_logged_once = false;
  if (s_runtime_logged_once) {
    return false;
  }
  s_runtime_logged_once = true;
  return true;
}

}  // namespace (close anonymous namespace so the routea query-prep
   //            kernel can live at file scope. This is required by Phase
   //            3.2 of the architecture hardening plan, which makes
   //            beam_gpu_routea_sidecars.hpp callable without depending
   //            on this TU's anonymous namespace.)

template <typename T>
__global__ void routea_query_prep_gpu_kernel(
    const T* __restrict__ d_queries,
    int nq,
    int dim,
    int score_dims,
    const float* __restrict__ d_qg_signs,
    int qg_mode,
    uint32_t* __restrict__ d_query_bp,
    float* __restrict__ d_query_lower,
    float* __restrict__ d_query_width,
    int* __restrict__ d_query_sumq) {
  const int qid = blockIdx.x;
  if (qid >= nq || score_dims <= 0 || score_dims > kStableRabitqMaxScoreDims) {
    return;
  }

  __shared__ float s_values[kStableRabitqMaxScoreDims];
  const T* query = d_queries + static_cast<size_t>(qid) * static_cast<size_t>(dim);

  for (int i = threadIdx.x; i < score_dims; i += blockDim.x) {
    float v = 0.0f;
    if (i < dim) {
      v = static_cast<float>(query[i]);
    }
    if (qg_mode && d_qg_signs != nullptr) {
      v *= d_qg_signs[i];
    }
    s_values[i] = v;
  }
  __syncthreads();

  if (qg_mode) {
    for (int h = 1; h < score_dims; h <<= 1) {
      for (int i = threadIdx.x; i < score_dims; i += blockDim.x) {
        if ((i & h) == 0) {
          const float a = s_values[i];
          const float b = s_values[i + h];
          s_values[i] = a + b;
          s_values[i + h] = a - b;
        }
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    float lo = FLT_MAX;
    float hi = -FLT_MAX;
    for (int i = 0; i < score_dims; ++i) {
      const float v = s_values[i];
      lo = fminf(lo, v);
      hi = fmaxf(hi, v);
    }

    const float width = (hi > lo) ? ((hi - lo) / 63.0f) : 0.0f;
    int sumq = 0;
    uint32_t local_bp[6][kStableRabitqScoreWords];
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
      #pragma unroll
      for (int w = 0; w < kStableRabitqScoreWords; ++w) {
        local_bp[k][w] = 0u;
      }
    }

    for (int i = 0; i < score_dims; ++i) {
      int qv = 0;
      if (width > 0.0f) {
        const float normalized = ((s_values[i] - lo) / width) + 0.5f;
        qv = __float2int_rn(normalized);
        qv = max(0, min(63, qv));
      }
      sumq += qv;
      #pragma unroll
      for (int k = 0; k < 6; ++k) {
        if (((qv >> k) & 1) != 0) {
          local_bp[k][i >> 5] |= (1u << (i & 31));
        }
      }
    }

    const int score_words = (score_dims + 31) >> 5;
    uint32_t* bp =
        d_query_bp + static_cast<size_t>(qid) * 6u * static_cast<size_t>(score_words);
    for (int k = 0; k < 6; ++k) {
      for (int w = 0; w < score_words; ++w) {
        bp[static_cast<size_t>(k) * static_cast<size_t>(score_words) +
           static_cast<size_t>(w)] = local_bp[k][w];
      }
    }
    d_query_lower[qid] = lo;
    d_query_width[qid] = width;
    d_query_sumq[qid] = sumq;
  }
}

namespace {  // reopen anonymous namespace for the rest of the host helpers

inline cudaDeviceProp load_device_properties_with_fallback() {
  cudaDeviceProp prop{};
  int dev = 0;
  cudaError_t dev_err = cudaGetDevice(&dev);
  if (dev_err != cudaSuccess) {
    printf("WARN: cudaGetDevice failed (%s), fallback dev=0.\n",
           cudaGetErrorString(dev_err));
    (void)cudaGetLastError();
    dev = 0;
  }

  cudaError_t prop_err = cudaGetDeviceProperties(&prop, dev);
  if (prop_err != cudaSuccess) {
    printf("WARN: cudaGetDeviceProperties failed (%s), continue with unknown device info.\n",
           cudaGetErrorString(prop_err));
    (void)cudaGetLastError();
    std::strncpy(prop.name, "unknown", sizeof(prop.name) - 1);
    prop.name[sizeof(prop.name) - 1] = '\0';
    prop.multiProcessorCount = 0;
  }
  return prop;
}

template <typename T>
int query_max_active_blocks_per_sm(size_t num_threads) {
  int numBlocksPerSM = 0;
  const int blockSize = static_cast<int>(num_threads);
  const size_t dynamicSMemSize = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, BeamSearch<T>, blockSize, dynamicSMemSize);
  if (err != cudaSuccess) {
    printf("WARN: occupancy query failed (%s), continue without occupancy hint.\n",
           cudaGetErrorString(err));
    (void)cudaGetLastError();
    return -1;
  }
  return numBlocksPerSM;
}

template <typename T>
void print_search_runtime_banner_once(
    bool print_runtime_info, int K, int nq, int dim, size_t npoints,
    int beam_sz, size_t num_blocks, size_t num_threads) {
  if (!print_runtime_info) {
    return;
  }
  printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%zu, beam_size=%d (compile-time max=%d)\n",
         K, nq, dim, npoints, beam_sz, BEAM_SIZE);
  printf("Stable compile-time: REFLOW_TOPN=%d, REFLOW_POP_BATCH=%d, SCORE_WORDS=%d, RABITQ_EARLY_STOP_STALL_ITERS=%d, RABITQ_EARLY_STOP_WARMUP_ITERS=%d, RABITQ_MAX_ITER_CAP=%d\n",
         (int)REFLOW_TOPN, (int)REFLOW_POP_BATCH,
         kStableRabitqScoreWords,
         (int)RABITQ_EARLY_STOP_STALL_ITERS,
         (int)RABITQ_EARLY_STOP_WARMUP_ITERS,
         (int)RABITQ_MAX_ITER_CAP);
  printf("Search compile-time: BITLEN=%d, HASH_SIZE=%u, SMALL_HASH_RESET_INTERVAL=%d, CANDIDATE_BUFFER_SIZE=%d, MAX_PREHASH_CANDIDATES=%d, RESULT_BUFFER_SIZE=%d, USE_BITONIC_MERGE=%d, SMEM_PER_BLOCK=%d, MIN_BLOCKS_PER_SM=%d, ROUTEA_GATE_MODE=%d, NAV_MODE=%d, ROUTEA_HASH_MAX_PROBES=%d, ROUTEA_GPU_QUERY_PREP=%d, ROUTEA_PROFILE_COUNTERS=%d, ROUTEA_AUDIT_COUNTERS=%d, ROUTEA_ADAPTIVE_GATE=%d, ADAPTIVE_KEEP=[%d,%d,%d], ADAPTIVE_WIDTH=[%d,%d,%d], ADAPTIVE_TARGET=%d, ADAPTIVE_MARGIN_REL_X1000=%d, ADAPTIVE_POLICY=%d\n",
         (int)BITLEN, (unsigned)kHashSize, (int)SMALL_HASH_RESET_INTERVAL,
         kCandidateBufferSize, kRouteAMaxPreHashCandidates,
         kResultBufferSize,
         (int)kUseBitonicMerge, (int)kSmemPerBlock, (int)kMinBlocksPerSM,
         (int)ROUTEA_GATE_MODE, (int)NAV_MODE, (int)ROUTEA_HASH_MAX_PROBES,
         (int)ROUTEA_GPU_QUERY_PREP,
         (int)ROUTEA_PROFILE_COUNTERS, (int)ROUTEA_AUDIT_COUNTERS,
         (int)ROUTEA_ADAPTIVE_GATE,
         (int)ROUTEA_ADAPTIVE_MIN_KEEP, (int)ROUTEA_ADAPTIVE_BASE_KEEP,
         (int)ROUTEA_ADAPTIVE_MAX_KEEP,
         (int)ROUTEA_ADAPTIVE_WIDTH_MIN, (int)ROUTEA_ADAPTIVE_WIDTH_BASE,
         (int)ROUTEA_ADAPTIVE_WIDTH_MAX,
         (int)ROUTEA_ADAPTIVE_TARGET_CANDIDATES,
         (int)ROUTEA_ADAPTIVE_MARGIN_REL_X1000,
         (int)ROUTEA_ADAPTIVE_POLICY);
  printf("Q-LEAF adaptive policy: stall_widen=[%d,%d]\n",
         (int)ROUTEA_ADAPTIVE_STALL_WIDEN_1,
         (int)ROUTEA_ADAPTIVE_STALL_WIDEN_2);
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";

  const cudaDeviceProp prop = load_device_properties_with_fallback();
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  const int numBlocksPerSM = query_max_active_blocks_per_sm<T>(num_threads);
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  // First-layer ablation manifest. One canonical JSON line per binary
  // launch so smoke logs can be parsed unambiguously. Variant identity
  // (`variant`, `description`) is supplied via env by the variant config
  // emitter — defaults are used for ad-hoc/manual runs.
  auto env_or = [](const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && v[0] != '\0') ? v : fallback;
  };
  const char* qleaf_variant = env_or("QLEAF_VARIANT", "manual");
  const char* qleaf_placement = env_or("QLEAF_PLACEMENT", "manual");
  const char* qleaf_description = env_or("QLEAF_DESCRIPTION", "manual_run");
  const char* qleaf_category = env_or("QLEAF_CATEGORY", "unknown");
  const char* qleaf_global_key = env_or("QLEAF_GLOBAL_KEY", "unknown");
  const char* qleaf_local_gate = env_or("QLEAF_LOCAL_GATE", "unknown");
  const char* qleaf_scorer_domain = env_or("QLEAF_SCORER_DOMAIN", "unknown");
  const char* qleaf_scorer_layout_env = env_or("QLEAF_SCORER_LAYOUT", "unknown");
  const char* qleaf_width_policy = env_or("QLEAF_WIDTH_POLICY", "unknown");
  const char* qleaf_keep_policy = env_or("QLEAF_KEEP_POLICY", "unknown");
  const char* qleaf_result_policy = env_or("QLEAF_RESULT_POLICY", "unknown");
  const char* qleaf_formal_eligible = env_or("QLEAF_FORMAL_ELIGIBLE", "0");
  const char* qleaf_deprecated_alias_used = env_or("QLEAF_DEPRECATED_ALIAS_USED", "0");
  const bool gate_local_pruning =
      (ROUTEA_GATE_MODE == ROUTEA_GATE_LOCAL_TOPK);
  const bool routea_global_ordering = kRouteASortByNav;
  const char* nav_mode_symbol = "unknown";
  switch ((int)NAV_MODE) {
    case 0: nav_mode_symbol = "EXACT"; break;
    case 8: nav_mode_symbol = "EDGE_RABITQ_KEY_FINAL_RERANK"; break;
    default: break;
  }
  const char* gate_mode_symbol = "unknown";
  switch ((int)ROUTEA_GATE_MODE) {
    case 0: gate_mode_symbol = "EXACT"; break;
    case 1: gate_mode_symbol = "LOCAL_TOPK"; break;
    default: break;
  }
  const char* scorer_layout_symbol;
  if (kRouteAUseEdgeRabitqNav) {
    scorer_layout_symbol = "edge_bitplane_global"; // NAV=8: edge-residual bitplane/popcount
  } else if (gate_local_pruning) {
    scorer_layout_symbol = "edge_bitplane_local";
  } else {
    scorer_layout_symbol = "exact";
  }
  printf("__QLEAF_FIRST_LAYER_MANIFEST__ "
         "{\"variant\":\"%s\","
         "\"placement\":\"%s\","
         "\"description\":\"%s\","
         "\"category\":\"%s\","
         "\"global_key\":\"%s\","
         "\"local_gate\":\"%s\","
         "\"scorer_domain\":\"%s\","
         "\"scorer_layout_semantic\":\"%s\","
         "\"width_policy\":\"%s\","
         "\"keep_policy\":\"%s\","
         "\"result_policy\":\"%s\","
         "\"formal_eligible\":%s,"
         "\"deprecated_alias_used\":%s,"
         "\"nav_mode\":%d,"
         "\"nav_mode_symbol\":\"%s\","
         "\"gate_mode\":%d,"
         "\"gate_mode_symbol\":\"%s\","
         "\"scorer_layout\":\"%s\","
         "\"use_rabitq_for_global_ordering\":%s,"
         "\"local_pruning_enabled\":%s,"
         "\"query_prep_runs\":%s,"
         "\"sidecar_edge_loaded\":%s,"
         "\"search_width\":%d,"
         "\"reflow_topn\":%d,"
         "\"candidate_buffer_size\":%d,"
         "\"required_prehash_candidates\":%d,"
         "\"candidate_buffer_safe\":%s,"
         "\"bitlen\":%d,"
         "\"score_words\":%d,"
         "\"timing_scope\":\"kernel_event_and_end_to_end\"}\n",
         qleaf_variant, qleaf_placement, qleaf_description,
         qleaf_category, qleaf_global_key, qleaf_local_gate,
         qleaf_scorer_domain, qleaf_scorer_layout_env,
         qleaf_width_policy, qleaf_keep_policy, qleaf_result_policy,
         qleaf_formal_eligible, qleaf_deprecated_alias_used,
         (int)NAV_MODE, nav_mode_symbol,
         (int)ROUTEA_GATE_MODE, gate_mode_symbol,
         scorer_layout_symbol,
         routea_global_ordering ? "true" : "false",
         gate_local_pruning ? "true" : "false",
         kRouteANeedsQueryPrep ? "true" : "false",
         kRouteANeedsEdgeSidecar ? "true" : "false",
         (int)SEARCH_WIDTH, (int)REFLOW_TOPN,
         kCandidateBufferSize, kRouteAMaxPreHashCandidates,
         (kCandidateBufferSize >= kRouteAMaxPreHashCandidates) ? "true" : "false",
         (int)BITLEN,
         kStableRabitqScoreWords);
  printf("Sidecar load: edge=%s query_prep=%s\n",
         kRouteANeedsEdgeSidecar ? "YES" : "NO",
         kRouteANeedsQueryPrep ? "YES" : "NO");
}

template <typename T>
bool cache_shape_matches(
    const beam_gpu_host::DeviceCache<T>& cache,
    size_t npoints,
    int dim,
    int max_degree) {
  return (cache.d_data != nullptr) && (cache.d_edges != nullptr) &&
         (cache.npoints == npoints) && (cache.dim == dim) &&
         (cache.max_degree == max_degree);
}

template <typename T>
void clear_search_buffers(beam_gpu_host::DeviceCache<T>& cache) {
  beam_gpu_host::cuda_free_if(cache.d_data);
  beam_gpu_host::cuda_free_if(cache.d_edges);
  beam_gpu_host::cuda_free_if(cache.d_queries);
  cache.queries_bytes = 0;
  beam_gpu_host::cuda_free_if(cache.d_results);
  cache.results_bytes = 0;
  beam_gpu_host::cuda_free_if(cache.d_total_count_dc);
  beam_gpu_host::cuda_free_if(cache.d_total_hops);
  beam_gpu_host::cuda_free_if(cache.d_routea_profile_counters);
  beam_gpu_host::cuda_free_if(cache.d_starting_points);
  cache.starting_points_bytes = 0;
}

template <typename T>
void clear_rabitq_sidecars(beam_gpu_host::DeviceCache<T>& cache) {
  beam_gpu_host::release_query_prep(cache.query_prep);
  beam_gpu_host::release_edge_sidecar(cache.edge_sidecar);
  cache.qg_signs.clear();
  beam_gpu_host::cuda_free_if(cache.d_qg_signs);
  cache.qg_signs_count = 0;
  cache.qg_padded_dim = 0;
  cache.routea_bitplane_qg_loaded = false;
}

template <typename T>
void update_cache_shape(
    beam_gpu_host::DeviceCache<T>& cache, size_t npoints, int dim, int max_degree) {
  cache.npoints = npoints;
  cache.dim = dim;
  cache.max_degree = max_degree;
}

template <typename T>
void reset_cache_if_shape_changed(
    beam_gpu_host::DeviceCache<T>& cache, size_t npoints, int dim, int max_degree) {
  if (cache_shape_matches(cache, npoints, dim, max_degree)) {
    return;
  }

  clear_search_buffers(cache);
  clear_rabitq_sidecars(cache);
  update_cache_shape(cache, npoints, dim, max_degree);
}

template <typename T>
void ensure_graph_cache(
    beam_gpu_host::DeviceCache<T>& cache,
    const T* h_data,
    const std::vector<vid_t>& edges,
    size_t npoints,
    int dim,
    int max_degree) {
  if (cache.d_data == nullptr) {
    const size_t bytes = npoints * static_cast<size_t>(dim) * sizeof(T);
    printf("GPU cache: uploading data vectors (%zu bytes)\n", bytes);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_data, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(cache.d_data, h_data, bytes, cudaMemcpyHostToDevice));
  }
  if (cache.d_edges == nullptr) {
    const size_t bytes = npoints * static_cast<size_t>(max_degree) * sizeof(vid_t);
    printf("GPU cache: uploading graph edges (%zu bytes)\n", bytes);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_edges, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(cache.d_edges, edges.data(), bytes, cudaMemcpyHostToDevice));
  }
}

template <typename T>
void ensure_query_and_result_buffers(
    beam_gpu_host::DeviceCache<T>& cache,
    int nq,
    int dim,
    int K,
    const T* queries) {
  const size_t query_bytes = static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(T);
  if (cache.d_queries == nullptr || cache.queries_bytes < query_bytes) {
    beam_gpu_host::cuda_free_if(cache.d_queries);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_queries, query_bytes));
    cache.queries_bytes = query_bytes;
  }
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_queries, queries, query_bytes, cudaMemcpyHostToDevice));

  const size_t result_bytes = static_cast<size_t>(nq) * static_cast<size_t>(K) * sizeof(vid_t);
  if (cache.d_results == nullptr || cache.results_bytes < result_bytes) {
    beam_gpu_host::cuda_free_if(cache.d_results);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_results, result_bytes));
    cache.results_bytes = result_bytes;
  }
}

template <typename T>
void reset_search_counters(beam_gpu_host::DeviceCache<T>& cache) {
  if (cache.d_total_count_dc == nullptr) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_total_count_dc, sizeof(gpu_long_t)));
  }
  if (cache.d_total_hops == nullptr) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_total_hops, sizeof(gpu_long_t)));
  }
#if ROUTEA_PROFILE_COUNTERS
  if (cache.d_routea_profile_counters == nullptr) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_routea_profile_counters,
                              kRouteAProfileCounterCount * sizeof(gpu_long_t)));
  }
#endif
  CUDA_SAFE_CALL(cudaMemset(cache.d_total_count_dc, 0, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemset(cache.d_total_hops, 0, sizeof(gpu_long_t)));
#if ROUTEA_PROFILE_COUNTERS
  CUDA_SAFE_CALL(cudaMemset(cache.d_routea_profile_counters, 0,
                            kRouteAProfileCounterCount * sizeof(gpu_long_t)));
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

inline std::vector<vid_t> build_starting_points(size_t npoints, int beam_sz, int K) {
  std::vector<vid_t> starting_points(static_cast<size_t>(beam_sz));
  for (int i = 0; i < beam_sz; i++) {
    starting_points[static_cast<size_t>(i)] = rand() % static_cast<vid_t>(npoints);
  }

  const int start_eval = std::min(K, beam_sz);
  std::vector<vid_t> sp(starting_points.begin(), starting_points.begin() + start_eval);
  std::sort(sp.begin(), sp.end());
  const auto it = std::unique(sp.begin(), sp.end());
  const int unique_cnt = static_cast<int>(std::distance(sp.begin(), it));
  printf("Starting points: eval=%d unique=%d duplicate_ratio=%.3f\n",
         start_eval, unique_cnt, 1.0 - (double)unique_cnt / (double)start_eval);
  return starting_points;
}

template <typename T>
void upload_starting_points(
    beam_gpu_host::DeviceCache<T>& cache,
    const std::vector<vid_t>& starting_points,
    int beam_sz) {
  const size_t start_cap_bytes = static_cast<size_t>(BEAM_SIZE) * sizeof(vid_t);
  if (cache.d_starting_points == nullptr || cache.starting_points_bytes < start_cap_bytes) {
    beam_gpu_host::cuda_free_if(cache.d_starting_points);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_starting_points, start_cap_bytes));
    cache.starting_points_bytes = start_cap_bytes;
  }
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_starting_points, starting_points.data(),
                            static_cast<size_t>(beam_sz) * sizeof(vid_t),
                            cudaMemcpyHostToDevice));
}

}  // close anonymous namespace so the sidecar manager is included at
   // file scope (Phase 3.2 of the architecture hardening plan).

// Sidecar orchestration templates are now provided by qleaf_sidecars.hpp
// (included at the top of this TU). The forward declaration of
// routea_query_prep_gpu_kernel<T> in that header binds to the file-scope
// definition above when search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto is instantiated at the bottom
// of this file.

namespace {  // reopen anonymous namespace for the rest of the helpers

template <typename T>
float run_beam_search_kernel(
    int K,
    int beam_sz,
    int nq,
    int dim,
    size_t npoints,
    size_t num_blocks,
    size_t num_threads,
    const beam_gpu_host::DeviceCache<T>& cache,
    const StableRabitqSidecars& sidecars,
    const RouteAQuerySidecars& routea_query_sidecars) {
  const uint32_t shm_size = 0;
  GraphGPU<vid_t> gg(static_cast<vid_t>(npoints), static_cast<vid_t>(cache.max_degree), cache.d_edges);

  cudaEvent_t ev0, ev1;
  CUDA_SAFE_CALL(cudaEventCreate(&ev0));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1));
  CUDA_SAFE_CALL(cudaEventRecord(ev0));
  BeamSearch<T><<<num_blocks, num_threads, shm_size>>>(
      K, beam_sz, nq, dim, npoints, cache.d_queries, cache.d_data,
      cache.d_results, cache.d_total_count_dc, cache.d_total_hops,
      cache.d_routea_profile_counters,
      cache.d_starting_points, sidecars.d_sign_bit, sidecars.sign_words_per_edge,
      sidecars.d_rabitq_triple, sidecars.d_rabitq_factor_dq,
      sidecars.d_rabitq_factor_vq,
      routea_query_sidecars.d_bp, routea_query_sidecars.score_words,
      routea_query_sidecars.score_dims,
      routea_query_sidecars.d_lower, routea_query_sidecars.d_width,
      routea_query_sidecars.d_sumq,
      gg);
  CUDA_SAFE_CALL(cudaEventRecord(ev1));
  CUDA_SAFE_CALL(cudaEventSynchronize(ev1));

  float kernel_ms = 0.0f;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
  CUDA_SAFE_CALL(cudaEventDestroy(ev0));
  CUDA_SAFE_CALL(cudaEventDestroy(ev1));
  return kernel_ms;
}

template <typename T>
void copy_search_results_and_counters(
    const beam_gpu_host::DeviceCache<T>& cache,
    int nq,
    int K,
    vid_t* result_idx,
    gpu_long_t* total_count_dc,
    gpu_long_t* total_hops,
    std::array<gpu_long_t, kRouteAProfileCounterCount>* routea_profile_counters) {
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, cache.d_results,
                            static_cast<size_t>(nq) * static_cast<size_t>(K) * sizeof(vid_t),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_count_dc, cache.d_total_count_dc,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_hops, cache.d_total_hops,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
#if ROUTEA_PROFILE_COUNTERS
  if (routea_profile_counters != nullptr && cache.d_routea_profile_counters != nullptr) {
    CUDA_SAFE_CALL(cudaMemcpy(routea_profile_counters->data(), cache.d_routea_profile_counters,
                              kRouteAProfileCounterCount * sizeof(gpu_long_t),
                              cudaMemcpyDeviceToHost));
  }
#else
  (void)routea_profile_counters;
#endif
}

inline void print_search_runtime_counters(
    bool print_runtime_info,
    gpu_long_t total_count_dc,
    gpu_long_t total_hops,
    int nq) {
  if (!print_runtime_info) {
    return;
  }
  std::cout << "total # distance computation: " << total_count_dc << "\n";
  std::cout << "average # distance computation: " << total_count_dc / nq << "\n";
  std::cout << "total # hops: " << total_hops << "\n";
  std::cout << "average # hops: " << total_hops / nq << "\n";
}

#if ROUTEA_PROFILE_COUNTERS
inline double routea_safe_ratio(gpu_long_t numerator, gpu_long_t denominator) {
  return denominator > 0 ? static_cast<double>(numerator) / static_cast<double>(denominator) : 0.0;
}
#endif

inline void print_routea_profile_counters(
    int call_id,
    int nq,
    const std::array<gpu_long_t, kRouteAProfileCounterCount>& c) {
#if ROUTEA_PROFILE_COUNTERS
  const gpu_long_t parents = c[kRouteAProfileParentsExpanded];
  const gpu_long_t valid = c[kRouteAProfileValidNeighborsScored];
  const gpu_long_t kept = c[kRouteAProfileGateKeptPreHash];
  const gpu_long_t hash_accepted = c[kRouteAProfileHashAccepted];
  const gpu_long_t hash_duplicate = c[kRouteAProfileHashDuplicate];
  const gpu_long_t hash_probe_exhausted = c[kRouteAProfileHashProbeExhausted];
  const gpu_long_t overflow = c[kRouteAProfileCandidateBufferOverflow];
  const gpu_long_t initial_l2 = c[kRouteAProfileInitialExactL2];
  const gpu_long_t candidate_l2 = c[kRouteAProfileCandidateExactL2];
  const gpu_long_t early_stop = c[kRouteAProfileEarlyStopTriggers];
  const gpu_long_t no_progress = c[kRouteAProfileNoProgressBreaks];
  const gpu_long_t total_l2 = initial_l2 + candidate_l2;
  const gpu_long_t adaptive_parents = c[kRouteAProfileAdaptiveParents];
  const gpu_long_t adaptive_keep_sum = c[kRouteAProfileAdaptiveKeepSum];
  const gpu_long_t adaptive_keep_le4 = c[kRouteAProfileAdaptiveKeepLe4];
  const gpu_long_t adaptive_keep_le8 = c[kRouteAProfileAdaptiveKeepLe8];
  const gpu_long_t adaptive_keep_le16 = c[kRouteAProfileAdaptiveKeepLe16];
  const gpu_long_t adaptive_keep_gt16 = c[kRouteAProfileAdaptiveKeepGt16];
  const gpu_long_t adaptive_width_rounds = c[kRouteAProfileAdaptiveWidthRounds];
  const gpu_long_t adaptive_width_sum = c[kRouteAProfileAdaptiveWidthSum];
  const gpu_long_t adaptive_threshold_under = c[kRouteAProfileAdaptiveThresholdUnder];
  const gpu_long_t adaptive_exact_useful = c[kRouteAProfileAdaptiveExactUseful];
  const gpu_long_t adaptive_feedback_narrow = c[kRouteAProfileAdaptiveFeedbackNarrow];
  const gpu_long_t adaptive_feedback_normal = c[kRouteAProfileAdaptiveFeedbackNormal];
  const gpu_long_t adaptive_feedback_broad = c[kRouteAProfileAdaptiveFeedbackBroad];
#if ROUTEA_AUDIT_COUNTERS
  const gpu_long_t gate_exact_topn = c[kRouteAProfileGateExactTopN];
  const gpu_long_t gate_hit_exact_topn = c[kRouteAProfileGateHitExactTopN];
  const gpu_long_t gate_false_negative_topn = c[kRouteAProfileGateFalseNegativeTopN];
  const gpu_long_t gate_false_negative_cutoff = c[kRouteAProfileGateFalseNegativeCutoff];
  const gpu_long_t gate_false_negative_cutoff_margin_x1000 =
      c[kRouteAProfileGateFalseNegativeCutoffMarginX1000];
  const gpu_long_t parent_pick_count = c[kRouteAProfileParentPickCount];
  const gpu_long_t parent_exact_rank_sum = c[kRouteAProfileParentExactRankSum];
  const gpu_long_t parent_exact_rank_max = c[kRouteAProfileParentExactRankMax];
  const gpu_long_t parent_matches_exact_best = c[kRouteAProfileParentMatchesExactBest];
  const gpu_long_t parent_delays_exact_best = c[kRouteAProfileParentDelaysExactBest];
#endif
  printf("ROUTEA_PROFILE_COUNTERS call=%d nq=%d "
         "parents_expanded=%lld avg_parents_expanded=%.6f "
         "valid_neighbors_scored=%lld gate_kept_pre_hash=%lld "
         "hash_accepted=%lld hash_duplicate=%lld hash_probe_exhausted=%lld "
         "hash_table_full=%lld "
         "candidate_buffer_overflow=%lld initial_exact_l2=%lld "
         "candidate_exact_l2=%lld total_exact_l2=%lld "
         "early_stop_triggers=%lld no_progress_breaks=%lld "
         "gate_keep_rate=%.6f hash_accept_rate=%.6f "
         "hash_duplicate_rate=%.6f hash_probe_exhausted_rate=%.6f "
         "overflow_rate=%.6f candidate_exact_eval_rate=%.6f "
         "adaptive_parents=%lld adaptive_keep_sum=%lld adaptive_keep_avg=%.6f "
         "adaptive_keep_le4=%lld adaptive_keep_le8=%lld "
         "adaptive_keep_le16=%lld adaptive_keep_gt16=%lld "
         "adaptive_width_rounds=%lld adaptive_width_sum=%lld "
         "adaptive_width_avg=%.6f adaptive_threshold_under=%lld "
         "adaptive_threshold_under_avg=%.6f adaptive_exact_useful=%lld "
         "adaptive_exact_useful_rate=%.6f "
         "adaptive_feedback_narrow=%lld adaptive_feedback_normal=%lld "
         "adaptive_feedback_broad=%lld"
#if ROUTEA_AUDIT_COUNTERS
         " gate_exact_topn=%lld gate_hit_exact_topn=%lld "
         "gate_false_negative_topn=%lld gate_false_negative_cutoff=%lld "
         "gate_false_negative_cutoff_margin_sum=%.6f "
         "gate_exact_topn_recall=%.6f gate_false_negative_topn_rate=%.6f "
         "gate_false_negative_cutoff_rate=%.6f "
         "gate_false_negative_cutoff_margin_avg=%.6f "
         "parent_pick_count=%lld parent_exact_rank_sum=%lld "
         "parent_exact_rank_avg=%.6f parent_exact_rank_max=%lld "
         "parent_matches_exact_best=%lld parent_delays_exact_best=%lld "
         "parent_exact_best_match_rate=%.6f parent_exact_best_delay_rate=%.6f"
#endif
         "\n",
         call_id, nq,
         static_cast<long long>(parents), routea_safe_ratio(parents, nq),
         static_cast<long long>(valid), static_cast<long long>(kept),
         static_cast<long long>(hash_accepted), static_cast<long long>(hash_duplicate),
         static_cast<long long>(hash_probe_exhausted),
         static_cast<long long>(hash_probe_exhausted),
         static_cast<long long>(overflow), static_cast<long long>(initial_l2),
         static_cast<long long>(candidate_l2), static_cast<long long>(total_l2),
         static_cast<long long>(early_stop), static_cast<long long>(no_progress),
         routea_safe_ratio(kept, valid),
         routea_safe_ratio(hash_accepted, kept),
         routea_safe_ratio(hash_duplicate, kept),
         routea_safe_ratio(hash_probe_exhausted, kept),
         routea_safe_ratio(overflow, kept),
         routea_safe_ratio(candidate_l2, valid),
         static_cast<long long>(adaptive_parents),
         static_cast<long long>(adaptive_keep_sum),
         routea_safe_ratio(adaptive_keep_sum, adaptive_parents),
         static_cast<long long>(adaptive_keep_le4),
         static_cast<long long>(adaptive_keep_le8),
         static_cast<long long>(adaptive_keep_le16),
         static_cast<long long>(adaptive_keep_gt16),
         static_cast<long long>(adaptive_width_rounds),
         static_cast<long long>(adaptive_width_sum),
         routea_safe_ratio(adaptive_width_sum, adaptive_width_rounds),
         static_cast<long long>(adaptive_threshold_under),
         routea_safe_ratio(adaptive_threshold_under, adaptive_parents),
         static_cast<long long>(adaptive_exact_useful),
         routea_safe_ratio(adaptive_exact_useful, candidate_l2),
         static_cast<long long>(adaptive_feedback_narrow),
         static_cast<long long>(adaptive_feedback_normal),
         static_cast<long long>(adaptive_feedback_broad)
#if ROUTEA_AUDIT_COUNTERS
         ,
         static_cast<long long>(gate_exact_topn),
         static_cast<long long>(gate_hit_exact_topn),
         static_cast<long long>(gate_false_negative_topn),
         static_cast<long long>(gate_false_negative_cutoff),
         static_cast<double>(gate_false_negative_cutoff_margin_x1000) / 1000.0,
         routea_safe_ratio(gate_hit_exact_topn, gate_exact_topn),
         routea_safe_ratio(gate_false_negative_topn, gate_exact_topn),
         routea_safe_ratio(gate_false_negative_cutoff, valid),
         routea_safe_ratio(gate_false_negative_cutoff_margin_x1000,
                           gate_false_negative_cutoff) / 1000.0,
         static_cast<long long>(parent_pick_count),
         static_cast<long long>(parent_exact_rank_sum),
         routea_safe_ratio(parent_exact_rank_sum, parent_pick_count),
         static_cast<long long>(parent_exact_rank_max),
         static_cast<long long>(parent_matches_exact_best),
         static_cast<long long>(parent_delays_exact_best),
         routea_safe_ratio(parent_matches_exact_best, parent_pick_count),
         routea_safe_ratio(parent_delays_exact_best, parent_pick_count)
#endif
         );
#else
  (void)call_id;
  (void)nq;
  (void)c;
#endif
}
}

template <typename T>
void IndexGraph<T>::search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto(int nq, const T* queries, int K, vid_t *result_idx, int beam_sz) {
  static int s_search_call_id = 0;
  const int search_call_id = ++s_search_call_id;
  const bool timing_enabled = routea_timing_breakdown_enabled();
  RouteASearchTiming timing;
  const HostTimePoint total_start = host_now();
  const int dim = this->d;
  const size_t npoints = this->ntotal;
  const int max_degree = this->maxDeg;
  assert(beam_sz > 0 && beam_sz <= BEAM_SIZE);
  assert(K > 0 && K <= beam_sz);
  const bool print_runtime_info = consume_runtime_log_once();
  const size_t num_threads = static_cast<size_t>(BLOCK_SIZE);
  const size_t num_blocks = static_cast<size_t>(nq);
  print_search_runtime_banner_once<T>(
      print_runtime_info, K, nq, dim, npoints, beam_sz, num_blocks, num_threads);
  auto& cache = beam_gpu_host::cache_for(this);
  HostTimePoint stage_start = host_now();
  reset_cache_if_shape_changed(cache, npoints, dim, max_degree);
  timing.reset_cache_ms = host_elapsed_ms(stage_start);
  T* h_data = this->get_data_ptr();
  stage_start = host_now();
  ensure_graph_cache(cache, h_data, this->edges, npoints, dim, max_degree);
  timing.graph_cache_ms = host_elapsed_ms(stage_start);
  stage_start = host_now();
  ensure_query_and_result_buffers(cache, nq, dim, K, queries);
  timing.query_buffer_ms = host_elapsed_ms(stage_start);
  gpu_long_t total_count_dc = 0;
  gpu_long_t total_hops = 0;
  std::array<gpu_long_t, kRouteAProfileCounterCount> routea_profile_counters{};
  stage_start = host_now();
  reset_search_counters(cache);
  timing.counter_reset_ms = host_elapsed_ms(stage_start);

  stage_start = host_now();
  const std::vector<vid_t> starting_points = build_starting_points(npoints, beam_sz, K);
  upload_starting_points(cache, starting_points, beam_sz);
  timing.starting_points_ms = host_elapsed_ms(stage_start);

  stage_start = host_now();
  const StableRabitqSidecars sidecars =
      ensure_rabitq_sidecars(this, cache, h_data, npoints, dim, max_degree);
  timing.rabitq_sidecars_ms = host_elapsed_ms(stage_start);
  RouteAQueryPrepTiming query_timing;
  stage_start = host_now();
  build_and_upload_routea_query_sidecars(
      cache, queries, nq, dim, timing_enabled ? &query_timing : nullptr);
  timing.query_sidecar_total_ms = host_elapsed_ms(stage_start);
  timing.query_sidecar_host_ms = query_timing.host_ms;
  timing.query_sidecar_alloc_ms = query_timing.alloc_ms;
  timing.query_sidecar_upload_ms = query_timing.upload_ms;
  timing.query_sidecar_gpu_ms = query_timing.gpu_ms;
  const RouteAQuerySidecars routea_query_sidecars =
      routea_query_sidecars_from_cache(cache);
  stage_start = host_now();
  const float kernel_ms = run_beam_search_kernel(
      K, beam_sz, nq, dim, npoints, num_blocks, num_threads,
      cache, sidecars, routea_query_sidecars);
  timing.kernel_host_ms = host_elapsed_ms(stage_start);
  timing.kernel_event_ms = static_cast<double>(kernel_ms);
  if (print_runtime_info) {
    printf("GPU kernel time (BeamSearch + sync): %.3f ms\n", kernel_ms);
    printf("Approx filter mode: %s\n",
           routea_runtime_enabled() ? "routea" : "off");
    const bool routea_active =
        (sidecars.d_sign_bit != nullptr) &&
        (sidecars.d_rabitq_triple != nullptr) &&
        (sidecars.d_rabitq_factor_dq != nullptr) &&
        (sidecars.d_rabitq_factor_vq != nullptr);
    printf("Q-LEAF edge-residual approx: %s\n", routea_active ? "enabled" : "disabled");
  }

  stage_start = host_now();
  copy_search_results_and_counters(
      cache, nq, K, result_idx, &total_count_dc, &total_hops,
      &routea_profile_counters);
  timing.result_copy_ms = host_elapsed_ms(stage_start);
  timing.total_ms = host_elapsed_ms(total_start);
  if (timing_enabled) {
    print_routea_timing_breakdown(search_call_id, nq, dim, K, beam_sz, timing);
  }
  print_routea_profile_counters(search_call_id, nq, routea_profile_counters);
  print_search_runtime_counters(print_runtime_info, total_count_dc, total_hops, nq);
}

template void IndexGraph<float>::search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto(int, const float*, int, vid_t*, int);
