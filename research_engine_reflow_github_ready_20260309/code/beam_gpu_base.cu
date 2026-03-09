#include "index.hpp"
#ifdef USE_RADIX
#include "search_radix.cuh"
#else
#include "search_bitonic_reflow_entry.cuh"
#endif
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

#if ENABLE_DGS || ENABLE_RABITQ
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

#if ENABLE_SQG || ENABLE_RABITQ
static inline void fwht_128_host(float* x) {
  for (int h = 1; h < 128; h <<= 1) {
    for (int i = 0; i < 128; i += (h << 1)) {
      for (int j = 0; j < h; j++) {
        const float a = x[i + j];
        const float b = x[i + j + h];
        x[i + j] = a + b;
        x[i + j + h] = a - b;
      }
    }
  }
}

#if ENABLE_RABITQ && RABITQ_QUERY_PROJ_ENABLE
template <typename T>
static bool build_rabitq_projected_queries_host(
    const T* queries,
    int nq,
    int dim,
    const std::vector<float>& proj_steps,
    std::vector<float>& out_proj) {
  out_proj.clear();
  if (queries == nullptr || nq <= 0 || dim != 128) return false;
  const size_t required = (size_t)RABITQ_QUERY_PROJ_STEPS * 128u;
  if (proj_steps.size() < required) return false;

  out_proj.resize((size_t)nq * 128u);
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int64_t qq = 0; qq < (int64_t)nq; ++qq) {
    const T* qsrc = queries + (size_t)qq * 128u;
    float* qdst = out_proj.data() + (size_t)qq * 128u;
    for (int i = 0; i < 128; ++i) qdst[i] = (float)qsrc[i];

#if RABITQ_QUERY_PROJ_MODE == 1
    const float* proj_row = proj_steps.data();
    for (int i = 0; i < 128; ++i) qdst[i] *= proj_row[i];
#elif RABITQ_QUERY_PROJ_MODE == 2
    const float* proj_row = proj_steps.data();
    for (int i = 0; i < 128; ++i) qdst[i] *= proj_row[i];
    fwht_128_host(qdst);
#else
    for (int st = 0; st < (int)RABITQ_QUERY_PROJ_STEPS; ++st) {
      const float* proj_row = proj_steps.data() + (size_t)st * 128u;
      for (int i = 0; i < 128; ++i) qdst[i] *= proj_row[i];
      fwht_128_host(qdst);
    }
#endif
  }
  return true;
}
#endif

static std::vector<float> build_vector_mean_host(
    const float* data, size_t npoints, int dim) {
  std::vector<float> mean((size_t)dim, 0.0f);
  if (npoints == 0 || dim <= 0) return mean;

#ifdef _OPENMP
  const int nth = omp_get_max_threads();
  std::vector<double> partial((size_t)nth * (size_t)dim, 0.0);
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    double* local = partial.data() + (size_t)tid * (size_t)dim;
    #pragma omp for schedule(static)
    for (int64_t uu = 0; uu < (int64_t)npoints; uu++) {
      const float* vec = data + (size_t)uu * (size_t)dim;
      for (int d = 0; d < dim; d++) {
        local[d] += (double)vec[d];
      }
    }
  }
  for (int d = 0; d < dim; d++) {
    double s = 0.0;
    for (int t = 0; t < nth; t++) s += partial[(size_t)t * (size_t)dim + (size_t)d];
    mean[(size_t)d] = (float)(s / (double)npoints);
  }
#else
  std::vector<double> sum((size_t)dim, 0.0);
  for (size_t u = 0; u < npoints; u++) {
    const float* vec = data + u * (size_t)dim;
    for (int d = 0; d < dim; d++) {
      sum[(size_t)d] += (double)vec[d];
    }
  }
  for (int d = 0; d < dim; d++) {
    mean[(size_t)d] = (float)(sum[(size_t)d] / (double)npoints);
  }
#endif
#if SQG_USE_FHT
  if (dim == 128) {
    // FWHT is linear, so mean(FWHT(x)) == FWHT(mean(x)).
    fwht_128_host(mean.data());
  }
#endif
  return mean;
}

static bool save_float_vector_raw(const char* filename, const float* data, size_t n) {
  std::ofstream f(filename, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
  return static_cast<bool>(f);
}

static std::vector<float> load_float_vector_raw(const char* filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  std::streamoff bytes = f.tellg();
  if (bytes <= 0 || (bytes % (std::streamoff)sizeof(float)) != 0) return {};
  f.seekg(0, std::ios::beg);
  std::vector<float> out((size_t)(bytes / (std::streamoff)sizeof(float)));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!f) return {};
  return out;
}

static std::vector<uint32_t> build_vector_sign_bits_host(
    const float* data, size_t npoints, int dim, const float* center) {
  const int W = (dim + 31) >> 5;
  std::vector<uint32_t> out((size_t)npoints * (size_t)W);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 256)
#endif
  for (int64_t uu = 0; uu < (int64_t)npoints; uu++) {
    const size_t u = (size_t)uu;
    const float* vec = data + u * (size_t)dim;
    uint32_t* dst = out.data() + u * (size_t)W;
    float fht_buf[128];
    const float* sqg_vec = vec;
#if SQG_USE_FHT
    if (dim == 128) {
      for (int i = 0; i < 128; i++) fht_buf[i] = vec[i];
      fwht_128_host(fht_buf);
      sqg_vec = fht_buf;
    }
#endif
    for (int w = 0; w < W; w++) {
      uint32_t word = 0;
      const int base_dim = w << 5;
      for (int b = 0; b < 32; b++) {
        const int i = base_dim + b;
        if (i >= dim) break;
        float thr = 0.0f;
#if SQG_SIGNBIT_MODE == 1
        if (center != nullptr) thr = center[i];
#endif
        word |= ((sqg_vec[i] >= thr) ? 1u : 0u) << b;
      }
      dst[w] = word;
    }
  }
  return out;
}

#if ENABLE_RABITQ && RABITQ_USE_FACTORS
static void build_rabitq_factors_host(
    const float* data,
    size_t npoints,
    int dim,
    const std::vector<vid_t>& edges,
    int max_degree,
    std::vector<float>& triple,
    std::vector<float>& factor_dq,
    std::vector<float>& factor_vq) {
  const size_t total_edges = (size_t)npoints * (size_t)max_degree;
  triple.assign(total_edges, FLT_MAX);
  factor_dq.assign(total_edges, 0.0f);
  factor_vq.assign(total_edges, 0.0f);
  if (npoints == 0 || dim <= 0 || max_degree <= 0) {
    return;
  }

  const float fac_norm = 1.0f / std::sqrt((float)dim);
  const float eps = (float)RABITQ_FACTOR_EPS;

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 32)
#endif
  for (int64_t uu = 0; uu < (int64_t)npoints; uu++) {
    const size_t u = (size_t)uu;
    const float* cvec = data + u * (size_t)dim;
    const size_t edge_base = u * (size_t)max_degree;

    for (int j = 0; j < max_degree; j++) {
      const size_t edge_pos = edge_base + (size_t)j;
      const vid_t v = edges[edge_pos];
      if ((size_t)v >= npoints) {
        continue;
      }
      const float* xvec = data + (size_t)v * (size_t)dim;

      float norm2 = 0.0f;
      float dot_x_obar = 0.0f;
      float dot_c_obar = 0.0f;
      int sign_sum = 0;

      for (int d = 0; d < dim; d++) {
        const float r = xvec[d] - cvec[d];
        norm2 = fmaf(r, r, norm2);
        const float s = (r >= 0.0f) ? 1.0f : -1.0f;
        dot_x_obar += r * s;     // equivalent to |r| sum
        dot_c_obar += cvec[d] * s;
        sign_sum += (r >= 0.0f) ? 1 : -1;
      }

      const float norm = std::sqrt(norm2);
      if (norm <= eps || dot_x_obar <= eps) {
        triple[edge_pos] = norm2;
        continue;
      }

      const float fac_x0 = (dot_x_obar * fac_norm) / norm;
      if (std::fabs(fac_x0) <= eps) {
        triple[edge_pos] = norm2;
        continue;
      }

      const float x_div_x0 = norm / fac_x0;
      const float fac_x1 = dot_c_obar * fac_norm;
      triple[edge_pos] = norm2 + (2.0f * x_div_x0 * fac_x1);
      factor_dq[edge_pos] = -2.0f * x_div_x0 * fac_norm;
      factor_vq[edge_pos] = -2.0f * x_div_x0 * fac_norm * (float)sign_sum;
    }
  }
}
#endif
#endif

#if ENABLE_DGS || ENABLE_SQG || ENABLE_RABITQ
static std::vector<uint32_t> try_load_sign_bits_with_fallback(const std::vector<std::string>& candidates,
                                                              size_t expected_words,
                                                              std::string& loaded_from,
                                                              std::string& loaded_format) {
  for (const auto& path : candidates) {
    auto raw = load_sign_bit(path.c_str());
    if (raw.size() == expected_words) {
      loaded_from = path;
      loaded_format = "raw";
      return raw;
    }

    auto rb = load_sign_bit_read_bin(path.c_str());
    if (rb.size() == expected_words) {
      loaded_from = path;
      loaded_format = "read_bin";
      return rb;
    }
  }
  loaded_from.clear();
  loaded_format.clear();
  return {};
}

static std::vector<uint32_t> load_read_bin_payload_block(
    const char* filename, int payload_index) {
  if (payload_index < 0) return {};
  std::ifstream f(filename, std::ios::binary);
  if (!f) return {};

  int32_t n = 0;
  int32_t dim = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
  f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
  if (!f || n <= 0 || dim <= 0) return {};

  const size_t words = static_cast<size_t>(n) * static_cast<size_t>(dim);
  const size_t payload_bytes = words * sizeof(uint32_t);
  const size_t data_off = sizeof(int32_t) * 2 + static_cast<size_t>(payload_index) * payload_bytes;
  f.seekg(static_cast<std::streamoff>(data_off), std::ios::beg);
  if (!f) return {};

  std::vector<uint32_t> out(words);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(payload_bytes));
  if (!f) return {};
  return out;
}

static std::vector<float> load_proj_with_header(
    const char* filename, int& out_rows, int& out_cols) {
  out_rows = 0;
  out_cols = 0;
  std::ifstream f(filename, std::ios::binary);
  if (!f) return {};
  int32_t rows = 0;
  int32_t cols = 0;
  f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
  f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
  if (!f || rows <= 0 || cols <= 0) return {};
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  std::vector<float> out(n);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) return {};
  out_rows = rows;
  out_cols = cols;
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
  size_t starting_points_bytes = 0;
  const T* starting_points_query_ptr = nullptr;
  int starting_points_nq = 0;
  int starting_points_beam = 0;
  std::vector<vid_t> ghost_ids_host;
  std::vector<vid_t> ghost_adj_host;
  int ghost_deg = 0;
  size_t ghost_num = 0;
#endif
#if ENABLE_DGS || ENABLE_RABITQ
  uint32_t* d_sign_bit = nullptr;
  size_t sign_words = 0;
  int sign_words_per_edge = 0;
#endif
#if ENABLE_RABITQ
  float* d_rabitq_triple = nullptr;
  float* d_rabitq_factor_dq = nullptr;
  float* d_rabitq_factor_vq = nullptr;
  float2* d_rabitq_factor_dqvq = nullptr;
  half2* d_rabitq_factor_dqvq_h2 = nullptr;
  size_t rabitq_factor_count = 0;
  uint8_t* d_rabitq_packed_codes = nullptr;
  size_t rabitq_packed_code_bytes = 0;
  int rabitq_packed_code_bytes_per_edge = 0;
  float* d_rabitq_proj = nullptr;
  int rabitq_proj_rows = 0;
  int rabitq_proj_dim = 0;
  std::vector<float> rabitq_proj_host;
  float* d_rabitq_queries_proj = nullptr;
  size_t rabitq_queries_proj_bytes = 0;
  const T* rabitq_queries_proj_src_ptr = nullptr;
  int rabitq_queries_proj_nq = 0;
  int rabitq_queries_proj_dim = 0;
#endif
#if ENABLE_SQG
  uint32_t* d_sqg_sign_bit = nullptr;
  size_t sqg_sign_words = 0;
  float* d_sqg_center = nullptr;
  int sqg_center_dim = 0;
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

static std::vector<vid_t> sample_ghost_ids(size_t npoints, size_t ghost_num) {
  std::vector<vid_t> ghost_ids(ghost_num);
  if (ghost_num == npoints) {
    for (size_t i = 0; i < ghost_num; i++) ghost_ids[i] = (vid_t)i;
    return ghost_ids;
  }
  std::mt19937 rng(12345);
  std::uniform_int_distribution<vid_t> uni(0, (vid_t)(npoints - 1));
  std::unordered_set<vid_t> picked;
  picked.reserve(ghost_num * 2);
  while (picked.size() < ghost_num) {
    picked.insert(uni(rng));
  }
  size_t pos = 0;
  for (vid_t x : picked) {
    if (pos < ghost_num) ghost_ids[pos++] = x;
  }
  while (pos < ghost_num) {
    ghost_ids[pos] = (vid_t)pos;
    pos++;
  }
  return ghost_ids;
}

template <typename T>
static std::vector<vid_t> build_ghost_adj_topk(
    const T* data,
    int dim,
    const std::vector<vid_t>& ghost_ids,
    int ghost_deg) {
  const size_t ng = ghost_ids.size();
  std::vector<vid_t> adj(ng * (size_t)ghost_deg, UINT_MAX);
  if (ng == 0 || ghost_deg <= 0) return adj;

  for (size_t gi = 0; gi < ng; gi++) {
    const vid_t gid = ghost_ids[gi];
    const T* gvec = data + (size_t)gid * (size_t)dim;

    std::vector<std::pair<float, vid_t>> best;
    best.reserve((size_t)ghost_deg);

    for (size_t gj = 0; gj < ng; gj++) {
      if (gj == gi) continue;
      const vid_t nid = ghost_ids[gj];
      const T* nvec = data + (size_t)nid * (size_t)dim;
      float dist = 0.0f;
      for (int d = 0; d < dim; d++) {
        const float diff = (float)gvec[d] - (float)nvec[d];
        dist += diff * diff;
      }
      if ((int)best.size() < ghost_deg) {
        best.emplace_back(dist, nid);
      } else {
        int worst = 0;
        for (int k = 1; k < ghost_deg; k++) {
          if (best[k].first > best[worst].first) worst = k;
        }
        if (dist < best[worst].first) {
          best[worst] = std::make_pair(dist, nid);
        }
      }
    }

    std::sort(best.begin(), best.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    const int fill = std::min((int)best.size(), ghost_deg);
    for (int k = 0; k < fill; k++) {
      adj[gi * (size_t)ghost_deg + (size_t)k] = best[(size_t)k].second;
    }
    for (int k = fill; k < ghost_deg; k++) {
      adj[gi * (size_t)ghost_deg + (size_t)k] = gid;
    }
  }
  return adj;
}

template <typename T>
static void fill_starting_points_ghost_stage_with_graph(
    const T* data,
    size_t npoints,
    int dim,
    int beam_sz,
    int nq,
    const T* queries,
    const std::vector<vid_t>& ghost_ids,
    const std::vector<vid_t>& ghost_adj,
    int ghost_deg,
    std::vector<vid_t>& out) {
  const size_t ng = ghost_ids.size();
  out.resize((size_t)nq * (size_t)beam_sz);
  if (ng == 0) {
    for (int q = 0; q < nq; q++) {
      for (int s = 0; s < beam_sz; s++) {
        out[(size_t)q * (size_t)beam_sz + (size_t)s] = (vid_t)(rand() % (int)npoints);
      }
    }
    return;
  }

  const int m0 = std::max(1, std::min((int)ng, (int)GHOST_STAGE_M));
  const int r = std::max(1, std::min((int)ng, (int)GHOST_STAGE_R));
  const int iters = std::max(1, (int)GHOST_STAGE_ITERS);
  std::mt19937 rng(12345);

  for (int q = 0; q < nq; q++) {
    const T* query = queries + (size_t)q * (size_t)dim;

    std::vector<vid_t> frontier;
    frontier.reserve((size_t)m0);
    for (int i = 0; i < m0; i++) {
      frontier.push_back(ghost_ids[(size_t)(rng() % (uint32_t)ng)]);
    }

    for (int it = 0; it < iters; it++) {
      std::vector<vid_t> cand = frontier;
      for (vid_t fid : frontier) {
        // fid should be in ghost_ids; find local index by linear scan (ng is small)
        size_t loc = ng;
        for (size_t gi = 0; gi < ng; gi++) {
          if (ghost_ids[gi] == fid) {
            loc = gi;
            break;
          }
        }
        if (loc >= ng) continue;
        const size_t base = loc * (size_t)ghost_deg;
        for (int k = 0; k < ghost_deg; k++) {
          const vid_t nb = ghost_adj[base + (size_t)k];
          if ((size_t)nb < npoints) cand.push_back(nb);
        }
      }

      std::sort(cand.begin(), cand.end());
      cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

      std::vector<std::pair<float, vid_t>> scored;
      scored.reserve(cand.size());
      for (vid_t id : cand) {
        const T* vec = data + (size_t)id * (size_t)dim;
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
          const float diff = (float)query[d] - (float)vec[d];
          dist += diff * diff;
        }
        scored.emplace_back(dist, id);
      }
      std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

      frontier.clear();
      const int keep = std::min(r, (int)scored.size());
      for (int i = 0; i < keep; i++) frontier.push_back(scored[(size_t)i].second);
      if (frontier.empty()) break;
    }

    std::vector<std::pair<float, vid_t>> final_scored;
    final_scored.reserve(frontier.size());
    for (vid_t id : frontier) {
      const T* vec = data + (size_t)id * (size_t)dim;
      float dist = 0.0f;
      for (int d = 0; d < dim; d++) {
        const float diff = (float)query[d] - (float)vec[d];
        dist += diff * diff;
      }
      final_scored.emplace_back(dist, id);
    }
    std::sort(final_scored.begin(), final_scored.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    const int fill = std::max(1, std::min((int)final_scored.size(), beam_sz));
    for (int s = 0; s < beam_sz; s++) {
      out[(size_t)q * (size_t)beam_sz + (size_t)s] = final_scored[(size_t)(s % fill)].second;
    }
  }
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
  static int s_runtime_log_mode = -1;  // 0=quiet, 1=once (default), 2=every call
  static bool s_runtime_logged_once = false;
  if (s_runtime_log_mode < 0) {
    const char* log_env = std::getenv("SEARCH_RUNTIME_LOG");
    if (!log_env || log_env[0] == '\0') {
      s_runtime_log_mode = 1;
    } else {
      const int v = std::atoi(log_env);
      if (v <= 0) {
        s_runtime_log_mode = 0;
      } else if (v == 1) {
        s_runtime_log_mode = 1;
      } else {
        s_runtime_log_mode = 2;
      }
    }
  }
  const bool print_runtime_info =
      (s_runtime_log_mode == 2) || ((s_runtime_log_mode == 1) && !s_runtime_logged_once);
  if (print_runtime_info && s_runtime_log_mode == 1) {
    s_runtime_logged_once = true;
  }
#ifndef SEARCH_NUM_THREADS
#define SEARCH_NUM_THREADS BLOCK_SIZE
#endif
  size_t num_threads = (size_t)SEARCH_NUM_THREADS;
  size_t num_blocks = nq;
  if (print_runtime_info) {
    printf("Beam search on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d (compile-time max=%d)\n",
           K, nq, dim, npoints, beam_sz, BEAM_SIZE);
#if ENABLE_DGS
    printf("DGS compile-time: ENABLE_DGS=1, DGS_TOPN=%d, DGS_COOL_RATIO=%f, DGS_EST_ITERS=%d, cool_start=%d, DGS_MAX_WORDS=%d\n",
           (int)DGS_TOPN, (double)DGS_COOL_RATIO, (int)DGS_EST_ITERS,
           (int)((1.0f - (float)DGS_COOL_RATIO) * (float)DGS_EST_ITERS), (int)DGS_MAX_WORDS);
#else
    printf("DGS compile-time: ENABLE_DGS=0\n");
#endif
#if ENABLE_SQG
    printf("SQG compile-time: ENABLE_SQG=1, SQG_SIGNBIT_MODE=%d, SQG_USE_FHT=%d, SQG_TOPN=%d, SQG_TOPN_EARLY=%d, SQG_TOPN_EARLY_ITERS=%d, SQG_FULLSCAN_STRIDE=%d, SQG_SCORE_MODE=%d, SQG_REFINE_TOPM=%d, SQG_SQ8_STRIDE=%d, SQG_SAMPLE_DIMS=%d, SQG_MAX_WORDS=%d, SQG_RAW_KEEP=%d, SQG_WARMUP_ITERS=%d, SQG_COOL_RATIO=%f, SQG_MIN_BEAM=%d, SQG_EST_ITERS=%d\n",
           (int)SQG_SIGNBIT_MODE,
           (int)SQG_USE_FHT,
           (int)SQG_TOPN, (int)SQG_TOPN_EARLY, (int)SQG_TOPN_EARLY_ITERS, (int)SQG_FULLSCAN_STRIDE,
           (int)SQG_SCORE_MODE, (int)SQG_REFINE_TOPM, (int)SQG_SQ8_STRIDE, (int)SQG_SAMPLE_DIMS, (int)SQG_MAX_WORDS,
           (int)SQG_RAW_KEEP, (int)SQG_WARMUP_ITERS, (double)SQG_COOL_RATIO, (int)SQG_MIN_BEAM, (int)SQG_EST_ITERS);
#else
    printf("SQG compile-time: ENABLE_SQG=0\n");
#endif
#if ENABLE_RABITQ
    printf("RaBitQ compile-time: ENABLE_RABITQ=1, RABITQ_USE_FACTORS=%d, RABITQ_TOPN=%d, RABITQ_TOPN_EARLY=%d, RABITQ_TOPN_EARLY_ITERS=%d, RABITQ_WARMUP_ITERS=%d, RABITQ_COOL_RATIO=%f, RABITQ_ACTIVE_ITERS=%d, RABITQ_MAX_WORDS=%d, RABITQ_RAW_KEEP=%d, RABITQ_MIN_BEAM=%d, RABITQ_EST_ITERS=%d, RABITQ_EARLY_STOP_STALL_ITERS=%d, RABITQ_EARLY_STOP_WARMUP_ITERS=%d, RABITQ_MAX_ITER_CAP=%d\n",
           (int)RABITQ_USE_FACTORS,
           (int)RABITQ_TOPN, (int)RABITQ_TOPN_EARLY, (int)RABITQ_TOPN_EARLY_ITERS,
           (int)RABITQ_WARMUP_ITERS, (double)RABITQ_COOL_RATIO, (int)RABITQ_ACTIVE_ITERS,
           (int)RABITQ_MAX_WORDS, (int)RABITQ_RAW_KEEP, (int)RABITQ_MIN_BEAM,
           (int)RABITQ_EST_ITERS,
           (int)RABITQ_EARLY_STOP_STALL_ITERS,
           (int)RABITQ_EARLY_STOP_WARMUP_ITERS,
           (int)RABITQ_MAX_ITER_CAP);
#else
    printf("RaBitQ compile-time: ENABLE_RABITQ=0\n");
#endif
    printf("Search compile-time: BITLEN=%d, SMALL_HASH_RESET_INTERVAL=%d, ENABLE_FRONTIER_DEDUP=%d, FRONTIER_DEDUP_INTERVAL=%d, FRONTIER_DEDUP_METHOD=%d, OUTPUT_TOPK_UNIQUE=%d, DGS_WARMUP_ITERS=%d, DGS_SIGNBIT_INVERT=%d, DGS_DIVERSITY_KEEP=%d, DGS_TOPN_EARLY=%d, DGS_TOPN_EARLY_ITERS=%d, DGS_RAW_KEEP=%d, CANDIDATE_SELECT_STRICT=%d, CANDIDATE_SELECT_STRICT_ITERS=%d, DGS_DECOUPLE_EXPAND_WIDTH=%d, DGS_DECOUPLED_EXPAND_WIDTH=%d, DGS_FORCE_EXPAND_WIDTH=%d, DGS_FORCE_WIDE_EXPAND=%d, DGS_TREE_OFF_MODE=%d, DGS_TREE_OFF_PATTERN=%d, DGS_TREE_START_ITERS=%d, DGS_TREE_MIN_ACTIVE_ITERS=%d, DGS_TREE_RELAX_TOPN=%d, DGS_TREE_WIDE_CAP=%d, DGS_TREE_REQUIRE_STALL=%d, DGS_TREE_MODE2_EARLY_WARMUP_ITERS=%d, DGS_TREE_STALL_THRESHOLD=%d, DGS_TREE_MAX_HITS=%d, SQ8_DIST_PRUNE_ENABLE=%d, NEIGHBOR_BUFFER_SIZE=%d\n",
           (int)BITLEN, (int)SMALL_HASH_RESET_INTERVAL,
           (int)ENABLE_FRONTIER_DEDUP, (int)FRONTIER_DEDUP_INTERVAL, (int)FRONTIER_DEDUP_METHOD,
           (int)OUTPUT_TOPK_UNIQUE, (int)DGS_WARMUP_ITERS, (int)DGS_SIGNBIT_INVERT, (int)DGS_DIVERSITY_KEEP,
           (int)DGS_TOPN_EARLY, (int)DGS_TOPN_EARLY_ITERS, (int)DGS_RAW_KEEP,
           (int)CANDIDATE_SELECT_STRICT, (int)CANDIDATE_SELECT_STRICT_ITERS,
           (int)DGS_DECOUPLE_EXPAND_WIDTH, (int)DGS_DECOUPLED_EXPAND_WIDTH,
           (int)DGS_FORCE_EXPAND_WIDTH, (int)DGS_FORCE_WIDE_EXPAND,
           (int)DGS_TREE_OFF_MODE, (int)DGS_TREE_OFF_PATTERN,
           (int)DGS_TREE_START_ITERS, (int)DGS_TREE_MIN_ACTIVE_ITERS, (int)DGS_TREE_RELAX_TOPN,
           (int)DGS_TREE_WIDE_CAP,
           (int)DGS_TREE_REQUIRE_STALL, (int)DGS_TREE_MODE2_EARLY_WARMUP_ITERS,
           (int)DGS_TREE_STALL_THRESHOLD, (int)DGS_TREE_MAX_HITS,
           (int)SQ8_DIST_PRUNE_ENABLE,
           (int)NEIGHBOR_BUFFER_SIZE);
    printf("Ghost compile-time: ENABLE_GHOST_STAGE=%d, GHOST_SCALE_FACTOR=%d, GHOST_MIN_NODES=%d\n",
           (int)ENABLE_GHOST_STAGE, (int)GHOST_SCALE_FACTOR, (int)GHOST_MIN_NODES);
    printf("Ghost runtime: GHOST_REUSE_STARTS=%d, GHOST_GRAPH_DEG=%d, GHOST_STAGE_M=%d, GHOST_STAGE_R=%d, GHOST_STAGE_ITERS=%d\n",
           (int)GHOST_REUSE_STARTS, (int)GHOST_GRAPH_DEG, (int)GHOST_STAGE_M, (int)GHOST_STAGE_R, (int)GHOST_STAGE_ITERS);
    std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
    cudaDeviceProp prop{};
    int dev = 0;
    cudaError_t dev_err = cudaGetDevice(&dev);
    if (dev_err != cudaSuccess) {
      printf("WARN: cudaGetDevice failed (%s), fallback dev=0.\n",
             cudaGetErrorString(dev_err));
      // Clear sticky error so later cuda calls can proceed normally.
      (void)cudaGetLastError();
      dev = 0;
    }
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, dev);
    if (prop_err != cudaSuccess) {
      printf("WARN: cudaGetDeviceProperties failed (%s), continue with unknown device info.\n",
             cudaGetErrorString(prop_err));
      // Clear sticky error so later cuda calls can proceed normally.
      (void)cudaGetLastError();
      std::strncpy(prop.name, "unknown", sizeof(prop.name) - 1);
      prop.name[sizeof(prop.name) - 1] = '\0';
      prop.multiProcessorCount = 0;
    }
    printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
    int numBlocksPerSM = 0;
    int blockSize = (int)num_threads;   // threads per block
    size_t dynamicSMemSize = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM,
                        BeamSearch<T>, blockSize, dynamicSMemSize);
    if (err != cudaSuccess) {
      printf("WARN: occupancy query failed (%s), continue without occupancy hint.\n",
             cudaGetErrorString(err));
      // Clear sticky error so later cuda calls can proceed normally.
      (void)cudaGetLastError();
      numBlocksPerSM = -1;
    }
    printf("Max active blocks per SM = %d\n", numBlocksPerSM);
  }
  auto& cache = cache_for(this);
  const bool cache_mismatch =
      (cache.d_data == nullptr) ||
      (cache.d_edges == nullptr) ||
#if ENABLE_DGS || ENABLE_RABITQ
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
    cache.starting_points_bytes = 0;
    cache.starting_points_query_ptr = nullptr;
    cache.starting_points_nq = 0;
    cache.starting_points_beam = 0;
    cache.ghost_ids_host.clear();
    cache.ghost_adj_host.clear();
    cache.ghost_deg = 0;
    cache.ghost_num = 0;
#endif
#if ENABLE_DGS || ENABLE_RABITQ
    cuda_free_if(cache.d_sign_bit);
    cache.sign_words = 0;
    cache.sign_words_per_edge = 0;
#endif
#if ENABLE_RABITQ
    cuda_free_if(cache.d_rabitq_triple);
    cuda_free_if(cache.d_rabitq_factor_dq);
    cuda_free_if(cache.d_rabitq_factor_vq);
    cuda_free_if(cache.d_rabitq_factor_dqvq);
    cuda_free_if(cache.d_rabitq_factor_dqvq_h2);
    cache.rabitq_factor_count = 0;
    cuda_free_if(cache.d_rabitq_packed_codes);
    cache.rabitq_packed_code_bytes = 0;
    cache.rabitq_packed_code_bytes_per_edge = 0;
    cuda_free_if(cache.d_rabitq_proj);
    cache.rabitq_proj_rows = 0;
    cache.rabitq_proj_dim = 0;
    cache.rabitq_proj_host.clear();
    cuda_free_if(cache.d_rabitq_queries_proj);
    cache.rabitq_queries_proj_bytes = 0;
    cache.rabitq_queries_proj_src_ptr = nullptr;
    cache.rabitq_queries_proj_nq = 0;
    cache.rabitq_queries_proj_dim = 0;
#endif
#if ENABLE_SQG
    cuda_free_if(cache.d_sqg_sign_bit);
    cache.sqg_sign_words = 0;
    cuda_free_if(cache.d_sqg_center);
    cache.sqg_center_dim = 0;
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

  std::vector<vid_t> starting_points;
#if ENABLE_GHOST_STAGE
  const bool can_reuse_ghost_starts =
#if GHOST_REUSE_STARTS
      (cache.d_starting_points != nullptr) &&
      (cache.starting_points_query_ptr == queries) &&
      (cache.starting_points_nq == nq) &&
      (cache.starting_points_beam == beam_sz);
#else
      false;
#endif

  if (cache.ghost_ids_host.empty()) {
    const size_t scale = (GHOST_SCALE_FACTOR > 0) ? (size_t)GHOST_SCALE_FACTOR : (size_t)1;
    size_t ghost_num = npoints / scale;
    if (ghost_num < (size_t)GHOST_MIN_NODES) ghost_num = (size_t)GHOST_MIN_NODES;
    if (ghost_num > npoints) ghost_num = npoints;
    const int ghost_deg = std::max(1, std::min((int)ghost_num - 1, (int)GHOST_GRAPH_DEG));

    cache.ghost_ids_host = sample_ghost_ids(npoints, ghost_num);
    cache.ghost_adj_host = build_ghost_adj_topk<T>(h_data, dim, cache.ghost_ids_host, ghost_deg);
    cache.ghost_deg = ghost_deg;
    cache.ghost_num = ghost_num;
    printf("Ghost stage: built ghost graph (nodes=%zu, deg=%d)\n", cache.ghost_num, cache.ghost_deg);
  }

  if (can_reuse_ghost_starts) {
    printf("Starting points (ghost): reuse cached seeds for current query batch\n");
  } else {
    fill_starting_points_ghost_stage_with_graph<T>(
        h_data, (size_t)npoints, dim, beam_sz, nq, queries,
        cache.ghost_ids_host, cache.ghost_adj_host, cache.ghost_deg,
        starting_points);
    {
      const int start_eval = std::min(K, beam_sz);
      std::vector<vid_t> sp;
      sp.reserve((size_t)start_eval);
      for (int i = 0; i < start_eval; i++) sp.push_back(starting_points[(size_t)i]);
      std::sort(sp.begin(), sp.end());
      auto it = std::unique(sp.begin(), sp.end());
      const int unique_cnt = (int)std::distance(sp.begin(), it);
      printf("Starting points (ghost,q0): eval=%d unique=%d duplicate_ratio=%.3f\n",
             start_eval, unique_cnt, 1.0 - (double)unique_cnt / (double)start_eval);
    }
  }
#else
  starting_points.resize((size_t)beam_sz);
  for (int i = 0; i < beam_sz; i++) starting_points[(size_t)i] = rand() % (vid_t)npoints;
  {
    const int start_eval = std::min(K, beam_sz);
    std::vector<vid_t> sp(starting_points.begin(), starting_points.begin() + start_eval);
    std::sort(sp.begin(), sp.end());
    auto it = std::unique(sp.begin(), sp.end());
    const int unique_cnt = (int)std::distance(sp.begin(), it);
    printf("Starting points: eval=%d unique=%d duplicate_ratio=%.3f\n",
           start_eval, unique_cnt, 1.0 - (double)unique_cnt / (double)start_eval);
  }
#endif
  const size_t start_cap =
#if ENABLE_GHOST_STAGE
      (size_t)BEAM_SIZE * (size_t)nq;
#else
      (size_t)BEAM_SIZE;
#endif
  const size_t start_cap_bytes = start_cap * sizeof(vid_t);
  if (cache.d_starting_points == nullptr || cache.starting_points_bytes < start_cap_bytes) {
    cuda_free_if(cache.d_starting_points);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_starting_points, start_cap_bytes));
    cache.starting_points_bytes = start_cap_bytes;
  }
  const size_t start_copy =
#if ENABLE_GHOST_STAGE
      (size_t)beam_sz * (size_t)nq;
#else
      (size_t)beam_sz;
#endif
#if ENABLE_GHOST_STAGE
  if (!can_reuse_ghost_starts) {
    CUDA_SAFE_CALL(cudaMemcpy(cache.d_starting_points, starting_points.data(),
                              start_copy * sizeof(vid_t), cudaMemcpyHostToDevice));
    cache.starting_points_query_ptr = queries;
    cache.starting_points_nq = nq;
    cache.starting_points_beam = beam_sz;
  }
#else
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_starting_points, starting_points.data(),
                            start_copy * sizeof(vid_t), cudaMemcpyHostToDevice));
#endif

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
  int sign_words_per_edge = 0;
  const float* d_rabitq_triple_x = nullptr;
  const float* d_rabitq_factor_dq = nullptr;
  const float* d_rabitq_factor_vq = nullptr;
  const float2* d_rabitq_factor_dqvq = nullptr;
  const half2* d_rabitq_factor_dqvq_h2 = nullptr;
  const uint8_t* d_rabitq_packed_codes = nullptr;
  int rabitq_packed_code_bytes_per_edge = 0;
  const float* d_rabitq_proj = nullptr;
  const float* d_rabitq_queries_proj = nullptr;
  const uint32_t* d_sqg_sign_bit = nullptr;
  const float* d_sqg_center = nullptr;
#if ENABLE_DGS || ENABLE_RABITQ
  // Upload (once) the sign-bit table to device.
  if (cache.d_sign_bit == nullptr) {
    const std::string primary_path = this->graph_prefix.empty()
                                       ? std::string("sign_bit.bin")
                                       : (this->graph_prefix + ".signbit.bin");
    const std::string ref_path = this->graph_prefix.empty()
                                   ? std::string("graph-d32.ref.signbit.bin")
                                   : (this->graph_prefix + ".ref.signbit.bin");
    const std::string fallback_path = "sign_bit.bin";
    const std::string fallback_ref_path = "graph-d32.ref.signbit.bin";
    const char* signbit_override_env = std::getenv("SIGNBIT_FILE");
    const char* signbit_override2_env = std::getenv("SIGNBIT_FILE_ALT");
    const int W = (dim + 31) >> 5;
    int sign_words_upload = W;
#if ENABLE_RABITQ && !ENABLE_DGS && (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
    int rabitq_score_words_host = (RABITQ_SCORE_WORDS > 0)
                                      ? (int)RABITQ_SCORE_WORDS
                                      : (int)RABITQ_SCORE_WORDS_FALLBACK;
    rabitq_score_words_host = max(1, rabitq_score_words_host);
    sign_words_upload = min(sign_words_upload, rabitq_score_words_host);
#endif
    const int sign_words_min = max(1, sign_words_upload);
    const size_t edge_count = (size_t)npoints * (size_t)max_degree;
    const size_t expected_words = edge_count * (size_t)W;
    const size_t upload_words = edge_count * (size_t)sign_words_min;

    std::vector<std::string> candidates;
    candidates.reserve(8);
    if (signbit_override_env && signbit_override_env[0] != '\0') {
      candidates.push_back(std::string(signbit_override_env));
    }
    if (signbit_override2_env && signbit_override2_env[0] != '\0' &&
        std::string(signbit_override2_env) != (signbit_override_env ? std::string(signbit_override_env) : std::string())) {
      candidates.push_back(std::string(signbit_override2_env));
    }
    candidates.push_back(primary_path);
    if (ref_path != primary_path) candidates.push_back(ref_path);
    if (fallback_path != primary_path && fallback_path != ref_path) candidates.push_back(fallback_path);
    if (fallback_ref_path != primary_path && fallback_ref_path != ref_path && fallback_ref_path != fallback_path) {
      candidates.push_back(fallback_ref_path);
    }

    std::string loaded_from;
    std::string loaded_format;
    std::vector<uint32_t> sign_bit =
        try_load_sign_bits_with_fallback(candidates, expected_words, loaded_from, loaded_format);
    if (sign_bit.size() != expected_words) {
      printf("Residual-sign: building sign-bit table on host (expected_words=%zu, got=%zu)\n",
             expected_words, sign_bit.size());
      printf("Residual-sign: primary sign-bit path = %s\n", primary_path.c_str());
      sign_bit = build_dir_sign_bits_host((const float*)h_data, (size_t)npoints, dim, this->edges, max_degree);
      if (sign_bit.size() == expected_words) {
        bool saved_primary = save_sign_bit(primary_path.c_str(), sign_bit.data(), sign_bit.size());
        printf("Residual-sign: save %s %s (%zu bytes)\n", primary_path.c_str(), saved_primary ? "OK" : "FAILED",
               sign_bit.size() * sizeof(uint32_t));
        if (!saved_primary && primary_path != fallback_path) {
          bool saved_fallback = save_sign_bit(fallback_path.c_str(), sign_bit.data(), sign_bit.size());
          printf("Residual-sign: save fallback %s %s\n", fallback_path.c_str(), saved_fallback ? "OK" : "FAILED");
        }
      }
    } else {
      printf("Residual-sign: loaded sign-bit table from %s (format=%s, %zu bytes)\n",
             loaded_from.c_str(), loaded_format.c_str(), sign_bit.size() * sizeof(uint32_t));
    }
    if (sign_bit.size() == expected_words) {
      std::vector<uint32_t> sign_bit_upload;
      const uint32_t* sign_src = sign_bit.data();
      if (sign_words_min < W) {
        sign_bit_upload.resize(upload_words);
        for (size_t e = 0; e < edge_count; ++e) {
          const uint32_t* src = sign_src + e * (size_t)W;
          uint32_t* dst = sign_bit_upload.data() + e * (size_t)sign_words_min;
          memcpy(dst, src, (size_t)sign_words_min * sizeof(uint32_t));
        }
        sign_src = sign_bit_upload.data();
        printf("Residual-sign: compacted sign-bit words/edge from %d to %d (bytes: %zu -> %zu)\n",
               W, sign_words_min, expected_words * sizeof(uint32_t), upload_words * sizeof(uint32_t));
      }
      printf("GPU cache: uploading residual sign-bit table (%zu bytes)\n", upload_words * sizeof(uint32_t));
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_sign_bit, upload_words * sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_sign_bit, sign_src,
                                upload_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
      cache.sign_words = upload_words;
      cache.sign_words_per_edge = sign_words_min;
    } else {
      printf("Residual-sign: sign-bit table unavailable.\n");
      cache.d_sign_bit = nullptr;
      cache.sign_words = 0;
      cache.sign_words_per_edge = 0;
    }
  }
  d_sign_bit = cache.d_sign_bit;
  sign_words_per_edge = cache.sign_words_per_edge;
#else
  if (PRUNE_CONFIG) {
    auto sign_bit = load_sign_bit("sign_bit.bin");
    size_t sign_bit_size = sign_bit.size();
    uint32_t *h_sign_bit = sign_bit.data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sign_bit, sign_bit_size * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit, sign_bit_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }
#endif
#if ENABLE_RABITQ && RABITQ_PACKED_CODE_ENABLE && \
    (((RABITQ_FASTSCAN_ENABLE != 0)) || \
     ((RABITQ_SCORE_MODE == 1) && (RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE != 0)))
  if (cache.d_rabitq_packed_codes == nullptr) {
    const std::string packed_primary_path = this->graph_prefix.empty()
                                              ? std::string("rabitq.bin")
                                              : (this->graph_prefix + ".rabitq.bin");
    const std::string packed_fallback_path = "graph-d32.rabitq.bin";
    const char* packed_override_env = std::getenv("RABITQ_PACKED_FILE");
    std::vector<std::string> candidates;
    candidates.reserve(4);
    if (packed_override_env && packed_override_env[0] != '\0') {
      candidates.push_back(std::string(packed_override_env));
    }
    candidates.push_back(packed_primary_path);
    if (packed_fallback_path != packed_primary_path) {
      candidates.push_back(packed_fallback_path);
    }

    const size_t expected_words = (size_t)npoints * (size_t)max_degree * 4;  // 16B/edge
    std::vector<uint32_t> packed_words;
    std::string packed_from;
    int packed_payload_loaded = -1;
    for (const auto& p : candidates) {
      auto v = load_read_bin_payload_block(p.c_str(), (int)RABITQ_PACKED_PAYLOAD_INDEX);
      if (v.size() == expected_words) {
        packed_words = std::move(v);
        packed_from = p;
        packed_payload_loaded = (int)RABITQ_PACKED_PAYLOAD_INDEX;
        break;
      }
      if ((int)RABITQ_PACKED_PAYLOAD_INDEX != 0) {
        v = load_read_bin_payload_block(p.c_str(), 0);
        if (v.size() == expected_words) {
          packed_words = std::move(v);
          packed_from = p;
          packed_payload_loaded = 0;
          break;
        }
      }
    }
    if (!packed_words.empty()) {
      const size_t packed_bytes = packed_words.size() * sizeof(uint32_t);
      printf("RaBitQ packed-code: loaded from %s payload=%d (%zu bytes)\n",
             packed_from.c_str(), packed_payload_loaded, packed_bytes);
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_packed_codes, packed_bytes));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_packed_codes, packed_words.data(),
                                packed_bytes, cudaMemcpyHostToDevice));
      cache.rabitq_packed_code_bytes = packed_bytes;
      cache.rabitq_packed_code_bytes_per_edge = 16;
    } else {
      printf("RaBitQ packed-code: unavailable (expected_words=%zu)\n", expected_words);
      cache.d_rabitq_packed_codes = nullptr;
      cache.rabitq_packed_code_bytes = 0;
      cache.rabitq_packed_code_bytes_per_edge = 0;
    }
  }
  d_rabitq_packed_codes = cache.d_rabitq_packed_codes;
  rabitq_packed_code_bytes_per_edge = cache.rabitq_packed_code_bytes_per_edge;
#endif
#if ENABLE_RABITQ && RABITQ_QUERY_PROJ_ENABLE
  if (cache.rabitq_proj_host.empty() && dim == 128) {
    const std::string proj_primary_path = this->graph_prefix.empty()
                                            ? std::string("rabitq.proj.bin")
                                            : (this->graph_prefix + ".rabitq.proj.bin");
    const std::string proj_fallback_path = "graph-d32.rabitq.proj.bin";
    const char* proj_override_env = std::getenv("RABITQ_PROJ_FILE");
    std::vector<std::string> candidates;
    candidates.reserve(4);
    if (proj_override_env && proj_override_env[0] != '\0') {
      candidates.push_back(std::string(proj_override_env));
    }
    candidates.push_back(proj_primary_path);
    if (proj_fallback_path != proj_primary_path) {
      candidates.push_back(proj_fallback_path);
    }
    int rows = 0, cols = 0;
    std::vector<float> proj_host;
    std::string proj_from;
    for (const auto& p : candidates) {
      auto v = load_proj_with_header(p.c_str(), rows, cols);
      if (!v.empty() && cols == 128 && rows >= (int)RABITQ_QUERY_PROJ_STEPS) {
        proj_host = std::move(v);
        proj_from = p;
        break;
      }
    }
    if (!proj_host.empty()) {
      const size_t proj_elems = (size_t)RABITQ_QUERY_PROJ_STEPS * 128u;
      const size_t proj_bytes = proj_elems * sizeof(float);
      printf("RaBitQ query-proj: loaded from %s (rows=%d, cols=%d, use_steps=%d)\n",
             proj_from.c_str(), rows, cols, (int)RABITQ_QUERY_PROJ_STEPS);
      cache.rabitq_proj_host.assign(proj_host.begin(), proj_host.begin() + proj_elems);
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_proj, proj_bytes));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_proj, proj_host.data(),
                                proj_bytes, cudaMemcpyHostToDevice));
      cache.rabitq_proj_rows = (int)RABITQ_QUERY_PROJ_STEPS;
      cache.rabitq_proj_dim = 128;
    } else {
      printf("RaBitQ query-proj: unavailable (need cols=128 rows>=%d)\n",
             (int)RABITQ_QUERY_PROJ_STEPS);
      cache.d_rabitq_proj = nullptr;
      cache.rabitq_proj_rows = 0;
      cache.rabitq_proj_dim = 0;
      cache.rabitq_proj_host.clear();
    }
  }
  d_rabitq_proj = cache.d_rabitq_proj;
  if (!cache.rabitq_proj_host.empty() && dim == 128) {
    const size_t qproj_bytes = (size_t)nq * 128u * sizeof(float);
    const bool need_rebuild =
        (cache.d_rabitq_queries_proj == nullptr) ||
        (cache.rabitq_queries_proj_bytes < qproj_bytes) ||
        (cache.rabitq_queries_proj_src_ptr != queries) ||
        (cache.rabitq_queries_proj_nq != nq) ||
        (cache.rabitq_queries_proj_dim != dim);
    if (need_rebuild) {
      std::vector<float> queries_proj_host;
      if (build_rabitq_projected_queries_host(
              queries, nq, dim, cache.rabitq_proj_host, queries_proj_host)) {
        if (cache.d_rabitq_queries_proj == nullptr ||
            cache.rabitq_queries_proj_bytes < qproj_bytes) {
          cuda_free_if(cache.d_rabitq_queries_proj);
          CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_queries_proj, qproj_bytes));
          cache.rabitq_queries_proj_bytes = qproj_bytes;
        }
        CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_queries_proj, queries_proj_host.data(),
                                  qproj_bytes, cudaMemcpyHostToDevice));
        cache.rabitq_queries_proj_src_ptr = queries;
        cache.rabitq_queries_proj_nq = nq;
        cache.rabitq_queries_proj_dim = dim;
      } else {
        cache.rabitq_queries_proj_src_ptr = nullptr;
        cache.rabitq_queries_proj_nq = 0;
        cache.rabitq_queries_proj_dim = 0;
      }
    }
    d_rabitq_queries_proj = cache.d_rabitq_queries_proj;
  }
#endif
#if ENABLE_RABITQ && RABITQ_USE_FACTORS
  if (cache.d_rabitq_triple == nullptr ||
#if RABITQ_USE_PACKED_DQVQ
#if RABITQ_USE_FP16_DQVQ
      cache.d_rabitq_factor_dqvq_h2 == nullptr) {
#else
      cache.d_rabitq_factor_dqvq == nullptr) {
#endif
#else
      cache.d_rabitq_factor_dq == nullptr ||
      cache.d_rabitq_factor_vq == nullptr) {
#endif
    const size_t factor_count = (size_t)npoints * (size_t)max_degree;
    const std::string triple_primary_path = this->graph_prefix.empty()
                                              ? std::string("rabitq.triple.f32.bin")
                                              : (this->graph_prefix + ".rabitq.triple.f32.bin");
    const std::string dq_primary_path = this->graph_prefix.empty()
                                          ? std::string("rabitq.factor_dq.f32.bin")
                                          : (this->graph_prefix + ".rabitq.factor_dq.f32.bin");
    const std::string vq_primary_path = this->graph_prefix.empty()
                                          ? std::string("rabitq.factor_vq.f32.bin")
                                          : (this->graph_prefix + ".rabitq.factor_vq.f32.bin");
    const std::string triple_fallback_path = "rabitq.triple.f32.bin";
    const std::string dq_fallback_path = "rabitq.factor_dq.f32.bin";
    const std::string vq_fallback_path = "rabitq.factor_vq.f32.bin";
    const char* triple_override_env = std::getenv("RABITQ_TRIPLE_FILE");
    const char* dq_override_env = std::getenv("RABITQ_FACTOR_DQ_FILE");
    const char* vq_override_env = std::getenv("RABITQ_FACTOR_VQ_FILE");

    auto load_factor_with_fallback = [&](const char* override_path,
                                         const std::string& primary,
                                         const std::string& fallback,
                                         const char* tag) {
      if (override_path && override_path[0] != '\0') {
        auto v = load_float_vector_raw(override_path);
        if (v.size() == factor_count) {
          printf("RaBitQ: loaded %s from %s\n", tag, override_path);
          return v;
        }
      }
      auto v = load_float_vector_raw(primary.c_str());
      if (v.size() == factor_count) {
        printf("RaBitQ: loaded %s from %s\n", tag, primary.c_str());
        return v;
      }
      if (fallback != primary) {
        v = load_float_vector_raw(fallback.c_str());
        if (v.size() == factor_count) {
          printf("RaBitQ: loaded %s from %s\n", tag, fallback.c_str());
          return v;
        }
      }
      return std::vector<float>{};
    };

    std::vector<float> triple_host =
        load_factor_with_fallback(triple_override_env, triple_primary_path, triple_fallback_path, "triple");
    std::vector<float> dq_host =
        load_factor_with_fallback(dq_override_env, dq_primary_path, dq_fallback_path, "factor_dq");
    std::vector<float> vq_host =
        load_factor_with_fallback(vq_override_env, vq_primary_path, vq_fallback_path, "factor_vq");

    if (triple_host.size() != factor_count ||
        dq_host.size() != factor_count ||
        vq_host.size() != factor_count) {
      printf("RaBitQ: building factor tables on host (count=%zu)\n", factor_count);
      build_rabitq_factors_host((const float*)h_data, (size_t)npoints, dim, this->edges, max_degree,
                                triple_host, dq_host, vq_host);
      if (triple_host.size() == factor_count &&
          dq_host.size() == factor_count &&
          vq_host.size() == factor_count) {
        const bool ok_triple = save_float_vector_raw(triple_primary_path.c_str(), triple_host.data(), triple_host.size());
        const bool ok_dq = save_float_vector_raw(dq_primary_path.c_str(), dq_host.data(), dq_host.size());
        const bool ok_vq = save_float_vector_raw(vq_primary_path.c_str(), vq_host.data(), vq_host.size());
        printf("RaBitQ: save triple=%s (%s), dq=%s (%s), vq=%s (%s)\n",
               triple_primary_path.c_str(), ok_triple ? "OK" : "FAILED",
               dq_primary_path.c_str(), ok_dq ? "OK" : "FAILED",
               vq_primary_path.c_str(), ok_vq ? "OK" : "FAILED");
      }
    }

    if (triple_host.size() == factor_count &&
        dq_host.size() == factor_count &&
        vq_host.size() == factor_count) {
      printf("GPU cache: uploading RaBitQ factors (%zu edges)\n", factor_count);
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_triple, factor_count * sizeof(float)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_triple, triple_host.data(),
                                factor_count * sizeof(float), cudaMemcpyHostToDevice));
#if RABITQ_USE_PACKED_DQVQ
#if RABITQ_USE_FP16_DQVQ
      std::vector<half2> dqvq_h2_host(factor_count);
      for (size_t i = 0; i < factor_count; ++i) {
        dqvq_h2_host[i] = __floats2half2_rn(dq_host[i], vq_host[i]);
      }
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_factor_dqvq_h2, factor_count * sizeof(half2)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_factor_dqvq_h2, dqvq_h2_host.data(),
                                factor_count * sizeof(half2), cudaMemcpyHostToDevice));
      cache.d_rabitq_factor_dqvq = nullptr;
#else
      std::vector<float2> dqvq_host(factor_count);
      for (size_t i = 0; i < factor_count; ++i) {
        dqvq_host[i] = make_float2(dq_host[i], vq_host[i]);
      }
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_factor_dqvq, factor_count * sizeof(float2)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_factor_dqvq, dqvq_host.data(),
                                factor_count * sizeof(float2), cudaMemcpyHostToDevice));
      cache.d_rabitq_factor_dqvq_h2 = nullptr;
#endif
      cache.d_rabitq_factor_dq = nullptr;
      cache.d_rabitq_factor_vq = nullptr;
#else
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_factor_dq, factor_count * sizeof(float)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_rabitq_factor_vq, factor_count * sizeof(float)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_factor_dq, dq_host.data(),
                                factor_count * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_rabitq_factor_vq, vq_host.data(),
                                factor_count * sizeof(float), cudaMemcpyHostToDevice));
#endif
      cache.rabitq_factor_count = factor_count;
    } else {
      printf("RaBitQ: factor tables unavailable; RaBitQ filter will be disabled.\n");
      cache.d_rabitq_triple = nullptr;
      cache.d_rabitq_factor_dq = nullptr;
      cache.d_rabitq_factor_vq = nullptr;
      cache.d_rabitq_factor_dqvq = nullptr;
      cache.d_rabitq_factor_dqvq_h2 = nullptr;
      cache.rabitq_factor_count = 0;
    }
  }
  d_rabitq_triple_x = cache.d_rabitq_triple;
  d_rabitq_factor_dq = cache.d_rabitq_factor_dq;
  d_rabitq_factor_vq = cache.d_rabitq_factor_vq;
  d_rabitq_factor_dqvq = cache.d_rabitq_factor_dqvq;
  d_rabitq_factor_dqvq_h2 = cache.d_rabitq_factor_dqvq_h2;
#endif
#if ENABLE_SQG
  const bool use_sqg_center = (SQG_SIGNBIT_MODE == 1);
  const bool sqg_use_fht = (SQG_USE_FHT == 1) && (dim == 128);
  const std::string sqg_tag_base = use_sqg_center ? std::string("sqg.mc") : std::string("sqg");
  const std::string sqg_tag = sqg_use_fht ? (sqg_tag_base + ".fht") : sqg_tag_base;
  const std::string sqg_local_prefix =
      use_sqg_center ? std::string("sqg_mc") : std::string("sqg");
  const std::string sqg_local_suffix = sqg_use_fht ? std::string("_fht") : std::string("");
  const std::string primary_path = this->graph_prefix.empty()
                                     ? (sqg_local_prefix + sqg_local_suffix + ".signbit.bin")
                                     : (this->graph_prefix + "." + sqg_tag + ".signbit.bin");
  const std::string fallback_path = sqg_local_prefix + sqg_local_suffix + ".signbit.bin";
  const std::string center_primary_path = this->graph_prefix.empty()
                                            ? (sqg_local_prefix + sqg_local_suffix + ".center.f32.bin")
                                            : (this->graph_prefix + "." + sqg_tag + ".center.f32.bin");
  const std::string center_fallback_path = sqg_local_prefix + sqg_local_suffix + ".center.f32.bin";
  const char* sqg_signbit_override_env = std::getenv("SQG_SIGNBIT_FILE");
  const char* sqg_center_override_env = std::getenv("SQG_CENTER_FILE");
  const int W = (dim + 31) >> 5;
  const size_t expected_words = (size_t)npoints * (size_t)W;
  std::vector<float> sqg_center_host;

  if (use_sqg_center && cache.d_sqg_center == nullptr) {
    if (sqg_center_override_env && sqg_center_override_env[0] != '\0') {
      sqg_center_host = load_float_vector_raw(sqg_center_override_env);
      if ((int)sqg_center_host.size() == dim) {
        printf("SQG: loaded center vector from %s\n", sqg_center_override_env);
      } else {
        sqg_center_host.clear();
      }
    }
    if (sqg_center_host.empty()) {
      sqg_center_host = load_float_vector_raw(center_primary_path.c_str());
      if (sqg_center_host.size() != (size_t)dim && center_fallback_path != center_primary_path) {
        sqg_center_host = load_float_vector_raw(center_fallback_path.c_str());
      }
      if ((int)sqg_center_host.size() == dim) {
        printf("SQG: loaded center vector from cache (%s)\n", center_primary_path.c_str());
      } else {
        sqg_center_host.clear();
      }
    }
    if (sqg_center_host.empty()) {
      printf("SQG: building mean-center vector on host (dim=%d)\n", dim);
      sqg_center_host = build_vector_mean_host((const float*)h_data, (size_t)npoints, dim);
      if ((int)sqg_center_host.size() == dim) {
        const bool saved_center = save_float_vector_raw(center_primary_path.c_str(), sqg_center_host.data(), sqg_center_host.size());
        printf("SQG: save center %s %s\n", center_primary_path.c_str(), saved_center ? "OK" : "FAILED");
      }
    }
    if ((int)sqg_center_host.size() == dim) {
      printf("GPU cache: uploading SQG center vector (%zu bytes)\n", (size_t)dim * sizeof(float));
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_sqg_center, (size_t)dim * sizeof(float)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_sqg_center, sqg_center_host.data(),
                                (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
      cache.sqg_center_dim = dim;
    } else {
      printf("SQG: center vector unavailable; fallback to raw sign threshold (0).\n");
      cache.d_sqg_center = nullptr;
      cache.sqg_center_dim = 0;
    }
  }

  if (cache.d_sqg_sign_bit == nullptr) {
    std::vector<std::string> candidates;
    candidates.reserve(4);
    if (sqg_signbit_override_env && sqg_signbit_override_env[0] != '\0') {
      candidates.push_back(std::string(sqg_signbit_override_env));
    }
    candidates.push_back(primary_path);
    if (fallback_path != primary_path) candidates.push_back(fallback_path);

    std::string loaded_from;
    std::string loaded_format;
    std::vector<uint32_t> sqg_sign =
        try_load_sign_bits_with_fallback(candidates, expected_words, loaded_from, loaded_format);
    if (sqg_sign.size() != expected_words) {
      printf("SQG: building vector sign-bit table on host (mode=%d, expected_words=%zu, got=%zu)\n",
             (int)SQG_SIGNBIT_MODE, expected_words, sqg_sign.size());
      if (use_sqg_center && sqg_center_host.empty()) {
        if (sqg_center_override_env && sqg_center_override_env[0] != '\0') {
          sqg_center_host = load_float_vector_raw(sqg_center_override_env);
        }
        if (sqg_center_host.size() != (size_t)dim) {
          sqg_center_host = load_float_vector_raw(center_primary_path.c_str());
        }
        if (sqg_center_host.size() != (size_t)dim) {
          sqg_center_host = build_vector_mean_host((const float*)h_data, (size_t)npoints, dim);
        }
      }
      const float* center_ptr =
          (use_sqg_center && sqg_center_host.size() == (size_t)dim) ? sqg_center_host.data() : nullptr;
      sqg_sign = build_vector_sign_bits_host((const float*)h_data, (size_t)npoints, dim, center_ptr);
      if (sqg_sign.size() == expected_words) {
        bool saved_primary = save_sign_bit(primary_path.c_str(), sqg_sign.data(), sqg_sign.size());
        printf("SQG: save %s %s (%zu bytes)\n", primary_path.c_str(), saved_primary ? "OK" : "FAILED",
               sqg_sign.size() * sizeof(uint32_t));
        if (!saved_primary && primary_path != fallback_path) {
          bool saved_fallback = save_sign_bit(fallback_path.c_str(), sqg_sign.data(), sqg_sign.size());
          printf("SQG: save fallback %s %s\n", fallback_path.c_str(), saved_fallback ? "OK" : "FAILED");
        }
      }
    } else {
      printf("SQG: loaded sign-bit table from %s (format=%s, %zu bytes)\n",
             loaded_from.c_str(), loaded_format.c_str(), sqg_sign.size() * sizeof(uint32_t));
    }

    if (sqg_sign.size() == expected_words) {
      printf("GPU cache: uploading SQG sign-bit table (%zu bytes)\n", expected_words * sizeof(uint32_t));
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_sqg_sign_bit, expected_words * sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.d_sqg_sign_bit, sqg_sign.data(),
                                expected_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
      cache.sqg_sign_words = expected_words;
    } else {
      printf("SQG: sign-bit table unavailable; SQG filter disabled.\n");
      cache.d_sqg_sign_bit = nullptr;
      cache.sqg_sign_words = 0;
    }
  }
  d_sqg_sign_bit = cache.d_sqg_sign_bit;
  d_sqg_center = cache.d_sqg_center;
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
                                                       cache.d_total_count_dc, cache.d_total_hops,
                                                       cache.d_starting_points, d_seed_map, d_sign_bit, sign_words_per_edge,
                                                       d_rabitq_packed_codes, rabitq_packed_code_bytes_per_edge,
                                                       d_rabitq_proj,
                                                       d_rabitq_queries_proj,
                                                       d_rabitq_triple_x, d_rabitq_factor_dq, d_rabitq_factor_vq,
                                                       d_rabitq_factor_dqvq,
                                                       d_rabitq_factor_dqvq_h2,
                                                       d_sqg_sign_bit, d_sqg_center, gg);
  CUDA_SAFE_CALL(cudaEventRecord(ev1));
  CUDA_SAFE_CALL(cudaEventSynchronize(ev1));
  float kernel_ms = 0.0f;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
  CUDA_SAFE_CALL(cudaEventDestroy(ev0));
  CUDA_SAFE_CALL(cudaEventDestroy(ev1));
  if (print_runtime_info) {
    printf("GPU kernel time (BeamSearch + sync): %.3f ms\n", kernel_ms);
  }
#endif

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_results, cache.d_results, (size_t)nq * (size_t)K * sizeof(vid_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, cache.d_total_count_dc, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_hops, cache.d_total_hops, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  if (d_seed_map) CUDA_SAFE_CALL(cudaFree(d_seed_map));
#if ENABLE_DGS || ENABLE_RABITQ
  // d_sign_bit is cached on device; do not free here.
#endif
#if ENABLE_RABITQ
  // RaBitQ factor tables are cached on device; do not free here.
#endif
#if ENABLE_SQG
  // d_sqg_sign_bit is cached on device; do not free here.
#endif
  if (print_runtime_info) {
    std::cout << "total # distance computation: " << total_count_dc << "\n";
    std::cout << "average # distance computation: " << total_count_dc / nq << "\n";
    std::cout << "total # hops: " << total_hops << "\n";
    std::cout << "average # hops: " << total_hops / nq << "\n";
  }
}

template class IndexGraph<float>;
