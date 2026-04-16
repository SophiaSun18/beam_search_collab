#include "index.hpp"
#include "search_bitonic.cuh"
#include "beam_gpu_host_cache.hpp"
#include "beam_gpu_host_sidecars.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace {
enum class ApproxFilterMode {
  kOff,
  kRouteA,
};

inline ApproxFilterMode load_approx_filter_mode() {
  const char* raw = std::getenv("REFLOW_APPROX_MODE");
  if (raw == nullptr || raw[0] == '\0') {
    return ApproxFilterMode::kRouteA;
  }
  std::string mode(raw);
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (mode == "off" || mode == "exact" || mode == "0") {
    return ApproxFilterMode::kOff;
  }
  return ApproxFilterMode::kRouteA;
}

inline bool routea_runtime_enabled() {
  return load_approx_filter_mode() == ApproxFilterMode::kRouteA;
}

struct StableRabitqSidecars {
  uint32_t* d_sign_bit = nullptr;
  int sign_words_per_edge = 0;
  const float* d_rabitq_triple = nullptr;
  const float* d_rabitq_factor_dq = nullptr;
};

struct RouteAQuerySidecars {
  const uint16_t* d_lut = nullptr;
  const float* d_lower = nullptr;
  const float* d_width = nullptr;
  const int* d_sumq = nullptr;
  int lut_groups = 0;
};

inline bool consume_runtime_log_once() {
  static bool s_runtime_logged_once = false;
  if (s_runtime_logged_once) {
    return false;
  }
  s_runtime_logged_once = true;
  return true;
}

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
  printf("Search compile-time: BITLEN=%d, SMALL_HASH_RESET_INTERVAL=%d, CANDIDATE_BUFFER_SIZE=%d, RESULT_BUFFER_SIZE=%d\n",
         (int)BITLEN, (int)SMALL_HASH_RESET_INTERVAL,
         kCandidateBufferSize, kResultBufferSize);
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";

  const cudaDeviceProp prop = load_device_properties_with_fallback();
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  const int numBlocksPerSM = query_max_active_blocks_per_sm<T>(num_threads);
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);
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
  beam_gpu_host::cuda_free_if(cache.d_starting_points);
  cache.starting_points_bytes = 0;
}

template <typename T>
void clear_rabitq_sidecars(beam_gpu_host::DeviceCache<T>& cache) {
  beam_gpu_host::cuda_free_if(cache.d_routea_query_lut);
  cache.routea_query_lut_words = 0;
  beam_gpu_host::cuda_free_if(cache.d_routea_query_lower);
  beam_gpu_host::cuda_free_if(cache.d_routea_query_width);
  beam_gpu_host::cuda_free_if(cache.d_routea_query_sumq);
  cache.routea_query_count = 0;
  beam_gpu_host::cuda_free_if(cache.d_sign_bit);
  cache.sign_words = 0;
  cache.sign_words_per_edge = 0;
  beam_gpu_host::cuda_free_if(cache.d_rabitq_triple);
  beam_gpu_host::cuda_free_if(cache.d_rabitq_factor_dq);
  cache.rabitq_factor_count = 0;
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
  CUDA_SAFE_CALL(cudaMemset(cache.d_total_count_dc, 0, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemset(cache.d_total_hops, 0, sizeof(gpu_long_t)));
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

template <typename T>
void ensure_sign_bit_sidecar(
    const IndexGraph<T>* index,
    beam_gpu_host::DeviceCache<T>& cache,
    const T* host_data,
    size_t npoints,
    int dim,
    int max_degree) {
  if (cache.d_sign_bit == nullptr) {
    const std::string raw_primary_path =
        index->graph_prefix.empty() ? std::string("sign_bit.bin")
                                    : (index->graph_prefix + ".signbit.bin");
    const std::string ref_primary_path =
        index->graph_prefix.empty() ? std::string("graph-d32.ref.signbit.bin")
                                    : (index->graph_prefix + ".ref.signbit.bin");
    const std::string raw_fallback_path = "graph-d32.signbit.bin";
    const std::string ref_fallback_path = "graph-d32.ref.signbit.bin";
    const int raw_words_per_edge = (dim + 31) >> 5;
    const int sign_words_upload =
        std::min(raw_words_per_edge, std::max(1, kStableRabitqScoreWords));
    const int upload_words_per_edge = std::max(1, sign_words_upload);
    const size_t edge_count = npoints * static_cast<size_t>(max_degree);
    const size_t expected_words = edge_count * static_cast<size_t>(raw_words_per_edge);
    const size_t upload_words = edge_count * static_cast<size_t>(upload_words_per_edge);

    std::vector<std::string> candidates;
    candidates.reserve(5);
    const char* signbit_override_env = std::getenv("SIGNBIT_FILE");
    if (signbit_override_env && signbit_override_env[0] != '\0') {
      candidates.push_back(std::string(signbit_override_env));
    }
    candidates.push_back(ref_primary_path);
    candidates.push_back(raw_primary_path);
    if (ref_fallback_path != ref_primary_path) {
      candidates.push_back(ref_fallback_path);
    }
    if (raw_fallback_path != raw_primary_path) {
      candidates.push_back(raw_fallback_path);
    }

    std::string loaded_from;
    std::string loaded_format;
    std::vector<uint32_t> sign_bit =
        beam_gpu_host::try_load_sign_bits_with_fallback(
            candidates, expected_words, loaded_from, loaded_format);
    if (sign_bit.size() != expected_words) {
      printf("Residual-sign: building raw host fallback sign-bit table (expected_words=%zu, got=%zu)\n",
             expected_words, sign_bit.size());
      printf("Residual-sign: raw fallback save target = %s\n", raw_primary_path.c_str());
      sign_bit = beam_gpu_host::build_dir_sign_bits_host(
          reinterpret_cast<const float*>(host_data), npoints, dim, index->edges, max_degree);
      if (sign_bit.size() == expected_words) {
        const bool saved = save_sign_bit(raw_primary_path.c_str(), sign_bit.data(), sign_bit.size());
        printf("Residual-sign: save %s %s (%zu bytes)\n",
               raw_primary_path.c_str(), saved ? "OK" : "FAILED",
               sign_bit.size() * sizeof(uint32_t));
      }
    } else {
      printf("Residual-sign: loaded sign-bit table from %s (format=%s, %zu bytes)\n",
             loaded_from.c_str(), loaded_format.c_str(),
             sign_bit.size() * sizeof(uint32_t));
    }

    if (sign_bit.size() == expected_words) {
      std::vector<uint32_t> sign_bit_upload;
      const uint32_t* sign_src = sign_bit.data();
      if (upload_words_per_edge < raw_words_per_edge) {
        sign_bit_upload.resize(upload_words);
        for (size_t e = 0; e < edge_count; ++e) {
          const uint32_t* src = sign_src + e * static_cast<size_t>(raw_words_per_edge);
          uint32_t* dst = sign_bit_upload.data() + e * static_cast<size_t>(upload_words_per_edge);
          std::memcpy(dst, src, static_cast<size_t>(upload_words_per_edge) * sizeof(uint32_t));
        }
        sign_src = sign_bit_upload.data();
        printf("Residual-sign: compacted sign-bit words/edge from %d to %d (bytes: %zu -> %zu)\n",
               raw_words_per_edge, upload_words_per_edge,
               expected_words * sizeof(uint32_t), upload_words * sizeof(uint32_t));
      }
      printf("GPU cache: uploading residual sign-bit table (%zu bytes)\n", upload_words * sizeof(uint32_t));
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_sign_bit, upload_words * sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_sign_bit, sign_src,
                                upload_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
      cache.sign_words = upload_words;
      cache.sign_words_per_edge = upload_words_per_edge;
    } else {
      printf("Residual-sign: sign-bit table unavailable.\n");
      cache.d_sign_bit = nullptr;
      cache.sign_words = 0;
      cache.sign_words_per_edge = 0;
    }
  }
}

template <typename T>
RouteAQuerySidecars routea_query_sidecars_from_cache(
    const beam_gpu_host::DeviceCache<T>& cache) {
  RouteAQuerySidecars sidecars;
  sidecars.d_lut = cache.d_routea_query_lut;
  sidecars.d_lower = cache.d_routea_query_lower;
  sidecars.d_width = cache.d_routea_query_width;
  sidecars.d_sumq = cache.d_routea_query_sumq;
  sidecars.lut_groups = (cache.dim + 3) >> 2;
  return sidecars;
}

template <typename T>
void build_and_upload_routea_query_sidecars(
    beam_gpu_host::DeviceCache<T>& cache,
    const T* queries,
    int nq,
    int dim) {
  auto clear_query_sidecars = [&]() {
    beam_gpu_host::cuda_free_if(cache.d_routea_query_lut);
    cache.routea_query_lut_words = 0;
    beam_gpu_host::cuda_free_if(cache.d_routea_query_lower);
    beam_gpu_host::cuda_free_if(cache.d_routea_query_width);
    beam_gpu_host::cuda_free_if(cache.d_routea_query_sumq);
    cache.routea_query_count = 0;
  };
  if (!routea_runtime_enabled()) {
    clear_query_sidecars();
    return;
  }
  if (cache.d_sign_bit == nullptr ||
      cache.d_rabitq_triple == nullptr ||
      cache.d_rabitq_factor_dq == nullptr ||
      nq <= 0 || queries == nullptr) {
    clear_query_sidecars();
    return;
  }

  const int lut_groups = (dim + 3) >> 2;
  const size_t lut_words = static_cast<size_t>(nq) * static_cast<size_t>(lut_groups) * 16u;
  std::vector<uint16_t> lut_host(lut_words, 0u);
  std::vector<float> lower_host(static_cast<size_t>(nq), 0.0f);
  std::vector<float> width_host(static_cast<size_t>(nq), 0.0f);
  std::vector<int> sumq_host(static_cast<size_t>(nq), 0);

  for (int q = 0; q < nq; ++q) {
    beam_gpu_host::RouteAQueryQ6Prepared prep;
    const float* query_f = reinterpret_cast<const float*>(queries) +
                           static_cast<size_t>(q) * static_cast<size_t>(dim);
    const bool ok = beam_gpu_host::routea_prepare_raw_query_q6_lut_host(
        query_f, dim, prep);
    if (!ok ||
        prep.lut.size() != static_cast<size_t>(lut_groups) * 16u ||
        prep.qint.size() != static_cast<size_t>(dim)) {
      printf("RouteA query-prep: failed for q=%d (dim=%d) in raw-query path\n", q, dim);
      clear_query_sidecars();
      return;
    }
    std::memcpy(lut_host.data() + static_cast<size_t>(q) * static_cast<size_t>(lut_groups) * 16u,
                prep.lut.data(),
                prep.lut.size() * sizeof(uint16_t));
    lower_host[static_cast<size_t>(q)] = prep.lower;
    width_host[static_cast<size_t>(q)] = prep.width;
    sumq_host[static_cast<size_t>(q)] = prep.sumq;
  }

  if (cache.d_routea_query_lut == nullptr || cache.routea_query_lut_words < lut_words) {
    beam_gpu_host::cuda_free_if(cache.d_routea_query_lut);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_routea_query_lut,
                              lut_words * sizeof(uint16_t)));
    cache.routea_query_lut_words = lut_words;
  }
  if (cache.d_routea_query_lower == nullptr || cache.routea_query_count < static_cast<size_t>(nq)) {
    beam_gpu_host::cuda_free_if(cache.d_routea_query_lower);
    beam_gpu_host::cuda_free_if(cache.d_routea_query_width);
    beam_gpu_host::cuda_free_if(cache.d_routea_query_sumq);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_routea_query_lower,
                              static_cast<size_t>(nq) * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_routea_query_width,
                              static_cast<size_t>(nq) * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_routea_query_sumq,
                              static_cast<size_t>(nq) * sizeof(int)));
    cache.routea_query_count = static_cast<size_t>(nq);
  }

  CUDA_SAFE_CALL(cudaMemcpy(cache.d_routea_query_lut, lut_host.data(),
                            lut_words * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_routea_query_lower, lower_host.data(),
                            static_cast<size_t>(nq) * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_routea_query_width, width_host.data(),
                            static_cast<size_t>(nq) * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_routea_query_sumq, sumq_host.data(),
                            static_cast<size_t>(nq) * sizeof(int), cudaMemcpyHostToDevice));
}

template <typename T>
void ensure_factor_sidecars(
    const IndexGraph<T>* index,
    beam_gpu_host::DeviceCache<T>& cache,
    size_t npoints,
    int max_degree) {
  if (cache.d_rabitq_triple != nullptr &&
      cache.d_rabitq_factor_dq != nullptr) {
    return;
  }

  const size_t factor_count = npoints * static_cast<size_t>(max_degree);
  auto load_factor_with_fallback =
      [&](const char* override_env_name,
          const char* local_fallback_name,
          const char* graph_suffix,
          const char* tag,
          std::string& loaded_from) {
        std::vector<std::string> candidates;
        candidates.reserve(3);
        const char* override_env = std::getenv(override_env_name);
        if (override_env && override_env[0] != '\0') {
          candidates.push_back(std::string(override_env));
        }
        const std::string primary =
            index->graph_prefix.empty() ? std::string(local_fallback_name)
                                        : (index->graph_prefix + graph_suffix);
        candidates.push_back(primary);
        if (std::string(local_fallback_name) != primary) {
          candidates.push_back(std::string(local_fallback_name));
        }

        for (const auto& path : candidates) {
          auto v = beam_gpu_host::load_float_vector_raw(path.c_str());
          if (v.size() == factor_count) {
            loaded_from = path;
            printf("RaBitQ: loaded %s from %s\n", tag, path.c_str());
            return v;
          }
        }
        loaded_from.clear();
        return std::vector<float>{};
      };

  std::string triple_from;
  std::string dq_from;
  std::vector<float> triple_host = load_factor_with_fallback(
      "RABITQ_TRIPLE_FILE", "rabitq.triple.f32.bin", ".rabitq.triple.f32.bin", "triple", triple_from);
  std::vector<float> dq_host = load_factor_with_fallback(
      "RABITQ_FACTOR_DQ_FILE", "rabitq.factor_dq.f32.bin", ".rabitq.factor_dq.f32.bin", "factor_dq", dq_from);

  if (triple_host.size() != factor_count ||
      dq_host.size() != factor_count) {
    printf("RaBitQ: factor tables unavailable; current stable kernel cannot use CPU-style compensated approx.\n");
    return;
  }

  printf("GPU cache: uploading RaBitQ factors (%zu edges)\n", factor_count);
  CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_triple, factor_count * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_factor_dq, factor_count * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_triple, triple_host.data(),
                            factor_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_factor_dq, dq_host.data(),
                            factor_count * sizeof(float), cudaMemcpyHostToDevice));
  cache.rabitq_factor_count = factor_count;
}

template <typename T>
StableRabitqSidecars ensure_rabitq_sidecars(
    const IndexGraph<T>* index,
    beam_gpu_host::DeviceCache<T>& cache,
    const T* host_data,
    size_t npoints,
    int dim,
    int max_degree) {
  StableRabitqSidecars sidecars;
  if (!routea_runtime_enabled()) {
    return sidecars;
  }
  ensure_sign_bit_sidecar(index, cache, host_data, npoints, dim, max_degree);
  ensure_factor_sidecars(index, cache, npoints, max_degree);

  sidecars.d_sign_bit = cache.d_sign_bit;
  sidecars.sign_words_per_edge = cache.sign_words_per_edge;
  sidecars.d_rabitq_triple = cache.d_rabitq_triple;
  sidecars.d_rabitq_factor_dq = cache.d_rabitq_factor_dq;
  return sidecars;
}

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
      cache.d_starting_points, sidecars.d_sign_bit, sidecars.sign_words_per_edge,
      sidecars.d_rabitq_triple, sidecars.d_rabitq_factor_dq,
      routea_query_sidecars.d_lut, routea_query_sidecars.lut_groups,
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
    gpu_long_t* total_hops) {
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, cache.d_results,
                            static_cast<size_t>(nq) * static_cast<size_t>(K) * sizeof(vid_t),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_count_dc, cache.d_total_count_dc,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_hops, cache.d_total_hops,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
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
}

template <typename T>
void IndexGraph<T>::search(int nq, const T* queries, int K, vid_t *result_idx, int beam_sz) {
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
  reset_cache_if_shape_changed(cache, npoints, dim, max_degree);
  T* h_data = this->get_data_ptr();
  ensure_graph_cache(cache, h_data, this->edges, npoints, dim, max_degree);
  ensure_query_and_result_buffers(cache, nq, dim, K, queries);
  gpu_long_t total_count_dc = 0;
  gpu_long_t total_hops = 0;
  reset_search_counters(cache);

  const std::vector<vid_t> starting_points = build_starting_points(npoints, beam_sz, K);
  upload_starting_points(cache, starting_points, beam_sz);

  const StableRabitqSidecars sidecars =
      ensure_rabitq_sidecars(this, cache, h_data, npoints, dim, max_degree);
  const RouteAQuerySidecars routea_query_sidecars{};
  const float kernel_ms = run_beam_search_kernel(
      K, beam_sz, nq, dim, npoints, num_blocks, num_threads,
      cache, sidecars, routea_query_sidecars);
  if (print_runtime_info) {
    printf("GPU kernel time (BeamSearch + sync): %.3f ms\n", kernel_ms);
    printf("Approx filter mode: %s\n",
           routea_runtime_enabled() ? "routea" : "off");
    const bool routea_active =
        (sidecars.d_sign_bit != nullptr) &&
        (sidecars.d_rabitq_triple != nullptr) &&
        (sidecars.d_rabitq_factor_dq != nullptr);
    printf("RouteA edge-residual approx: %s\n", routea_active ? "enabled" : "disabled");
  }

  copy_search_results_and_counters(cache, nq, K, result_idx, &total_count_dc, &total_hops);
  print_search_runtime_counters(print_runtime_info, total_count_dc, total_hops, nq);
}

template class IndexGraph<float>;
