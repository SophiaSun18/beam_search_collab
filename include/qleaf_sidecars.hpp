#pragma once

// =============================================================================
// qleaf_sidecars.hpp — Q-LEAF host-side sidecar orchestration (single header).
//
// Consolidated from three legacy headers in the qleaf_ablation_ladder tree:
//   beam_gpu_routea_types.hpp     — kernel-arg PODs + host time / runtime helpers
//   beam_gpu_host_sidecars.hpp    — file IO, QG/RaBitQ payload helpers, Q6 prep
//   beam_gpu_routea_sidecars.hpp  — orchestration templates the host path calls
//
// Every helper here is `inline` or template; safe to include from multiple
// translation units. Only the formal Q-LEAF sidecar surface is kept:
// QG metadata + graph override, edge sign-bit / triple / factor_dq /
// factor_vq upload, q6 bitplane query-prep, and the matching release
// helpers. Pointwise sidecars and historical diagnostic scorer assets are
// not migrated.
//
// The sidecar functions remain free templates (not in any namespace) so
// existing call sites in IndexGraph<T>::search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto keep their
// unqualified names; this preserves the timing scope and the error
// messages observed in the formal smoke baseline.
// =============================================================================


// ── Section 1 (formerly beam_gpu_routea_types.hpp) ──────────────────────────

// =============================================================================
// beam_gpu_routea_types.hpp — Shared host types and tiny utilities used by
// the RouteA sidecar manager and beam_gpu_base.cu.
//
// Phase 3.1 of the Q-LEAF architecture hardening plan: previously
// `StableRabitqSidecars` / `RouteAQuerySidecars` / `RouteAQueryPrepTiming`
// (and a small set of host time / runtime / GPU prep helpers) lived inside
// the anonymous namespace in `beam_gpu_base.cu`, and
// `beam_gpu_routea_sidecars.hpp` was forced to be `#include`'d from inside
// that same anonymous namespace because it referenced those file-scope
// helpers. That coupling is removed by giving the sidecar manager a real
// public header it can depend on.
//
// All entries below have inline (or template) linkage and are safe to
// include from multiple translation units, even if the current build only
// has one TU (`beam_gpu_base.cu`) that consumes them.
// =============================================================================

#include "utils.cuh"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <string>

// ─── RouteA sidecar handles passed to the kernel ─────────────────────────
// Both structs are copy-cheap PODs that the kernel launcher fills in from
// the DeviceCache + sidecar manager. They MUST keep the same field order /
// names as the kernel's launch arguments expect.

struct StableRabitqSidecars {
  uint32_t* d_sign_bit = nullptr;
  int sign_words_per_edge = 0;
  const float* d_rabitq_triple = nullptr;
  const float* d_rabitq_factor_dq = nullptr;
  const float* d_rabitq_factor_vq = nullptr;
};

struct RouteAQuerySidecars {
  const uint32_t* d_bp = nullptr;
  const float* d_lower = nullptr;
  const float* d_width = nullptr;
  const int* d_sumq = nullptr;
  int score_words = 0;
  int score_dims = 0;
};

struct RouteAQueryPrepTiming {
  double host_ms = 0.0;
  double alloc_ms = 0.0;
  double upload_ms = 0.0;
  double gpu_ms = 0.0;
  double total_ms = 0.0;
};

// ─── Host time helpers ────────────────────────────────────────────────────
// Used by the sidecar manager to measure host_ms/alloc_ms/upload_ms inside
// `build_and_upload_routea_query_sidecars`. Kept here so any module that
// needs the same timing shape gets identical semantics.

using HostClock = std::chrono::steady_clock;
using HostTimePoint = HostClock::time_point;

inline HostTimePoint host_now() {
  return HostClock::now();
}

inline double host_elapsed_ms(HostTimePoint start) {
  return std::chrono::duration<double, std::milli>(HostClock::now() - start).count();
}

// ─── RouteA runtime mode (load once per process) ─────────────────────────
// The RouteA sidecar build is gated by REFLOW_APPROX_MODE (default = routea).
// `routea_runtime_enabled()` is consulted by `ensure_rabitq_sidecars` and
// `build_and_upload_routea_query_sidecars`; making it inline here lets the
// sidecar manager compile without depending on the anonymous namespace in
// beam_gpu_base.cu.

enum class RouteAFilterMode {
  kOff,
  kRouteA,
};

inline RouteAFilterMode routea_load_filter_mode() {
  const char* raw = std::getenv("REFLOW_APPROX_MODE");
  if (raw == nullptr || raw[0] == '\0') {
    return RouteAFilterMode::kRouteA;
  }
  std::string mode(raw);
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (mode == "off" || mode == "exact" || mode == "0") {
    return RouteAFilterMode::kOff;
  }
  return RouteAFilterMode::kRouteA;
}

inline bool routea_runtime_enabled() {
  return routea_load_filter_mode() == RouteAFilterMode::kRouteA;
}

// ─── GPU query-prep kernel ────────────────────────────────────────────────
// Forward declaration for the kernel template. The definition still lives
// in beam_gpu_base.cu because it depends on shared device-side constants
// (kStableRabitqMaxScoreDims, the s_values smem layout, etc.). The sidecar
// manager only needs a well-typed entry point.

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
    int* __restrict__ d_query_sumq);

// ── Section 2 (formerly beam_gpu_host_sidecars.hpp) ─────────────────────────

#include "index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// File-scope sign-bit IO helpers (formerly defined in
// search_bitonic/kernel.cuh). Pure host C++ — kept at file scope (not in
// namespace beam_gpu_host) so historical call sites that look them up by
// unqualified name keep working. `inline` (not `static inline`) so every
// TU sees the same definition the linker can merge.

inline std::vector<uint32_t> load_sign_bit(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamoff bytes = f.tellg();
    if (bytes <= 0) return {};
    if ((bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(static_cast<size_t>(bytes / sizeof(uint32_t)));
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    if (!f) return {};
    return buf;
}

inline std::vector<uint32_t> load_sign_bit_read_bin(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};
    int32_t n = 0;
    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!f || n <= 0 || dim <= 0) return {};
    const size_t words = static_cast<size_t>(n) * static_cast<size_t>(dim);
    std::vector<uint32_t> buf(words);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(words * sizeof(uint32_t)));
    if (!f) return {};
    return buf;
}

inline bool save_sign_bit(const char* filename, const uint32_t* data, size_t words) {
    std::ofstream f(filename, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(words * sizeof(uint32_t)));
    return static_cast<bool>(f);
}

namespace beam_gpu_host {

inline std::vector<uint32_t> build_dir_sign_bits_host(
    const float* data, size_t npoints, int dim,
    const std::vector<vid_t>& edges, int max_degree) {
  const int words_per_edge = (dim + 31) >> 5;
  const size_t total_edges = npoints * static_cast<size_t>(max_degree);
  std::vector<uint32_t> out(total_edges * static_cast<size_t>(words_per_edge));

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int64_t uu = 0; uu < static_cast<int64_t>(npoints); uu++) {
    const size_t u = static_cast<size_t>(uu);
    const float* uvec = data + u * static_cast<size_t>(dim);
    const size_t edge_base = u * static_cast<size_t>(max_degree);
    for (int j = 0; j < max_degree; j++) {
      const vid_t v = edges[edge_base + static_cast<size_t>(j)];
      uint32_t* dst = out.data() + (edge_base + static_cast<size_t>(j)) * static_cast<size_t>(words_per_edge);
      if (static_cast<size_t>(v) >= npoints) {
        std::memset(dst, 0, static_cast<size_t>(words_per_edge) * sizeof(uint32_t));
        continue;
      }
      const float* vvec = data + static_cast<size_t>(v) * static_cast<size_t>(dim);
      for (int w = 0; w < words_per_edge; w++) {
        uint32_t word = 0;
        const int base_dim = w << 5;
        for (int b = 0; b < 32; b++) {
          const int i = base_dim + b;
          if (i >= dim) {
            break;
          }
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

inline std::vector<uint32_t> load_read_bin_payload_block(
    const char* filename, int payload_index) {
  if (payload_index < 0) {
    return {};
  }
  std::ifstream f(filename, std::ios::binary);
  if (!f) {
    return {};
  }

  int32_t n = 0;
  int32_t dim = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
  f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
  if (!f || n <= 0 || dim <= 0) {
    return {};
  }

  const size_t words = static_cast<size_t>(n) * static_cast<size_t>(dim);
  const size_t payload_bytes = words * sizeof(uint32_t);
  const size_t data_off =
      sizeof(int32_t) * 2 + static_cast<size_t>(payload_index) * payload_bytes;
  f.seekg(static_cast<std::streamoff>(data_off), std::ios::beg);
  if (!f) {
    return {};
  }

  std::vector<uint32_t> out(words);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(payload_bytes));
  if (!f) {
    return {};
  }
  return out;
}

inline std::vector<float> load_float_vector_raw(const char* filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) {
    return {};
  }
  f.seekg(0, std::ios::end);
  std::streamoff bytes = f.tellg();
  if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(float))) != 0) {
    return {};
  }
  f.seekg(0, std::ios::beg);
  std::vector<float> out(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!f) {
    return {};
  }
  return out;
}

inline std::vector<uint32_t> load_u32_vector_raw(const char* filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) {
    return {};
  }
  f.seekg(0, std::ios::end);
  std::streamoff bytes = f.tellg();
  if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0) {
    return {};
  }
  f.seekg(0, std::ios::beg);
  std::vector<uint32_t> out(
      static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(uint32_t))));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!f) {
    return {};
  }
  return out;
}

inline const char* routea_bitplane_qg_file_env() {
  const char* path = std::getenv("ROUTEA_BITPLANE_QG_FILE");
  return (path && path[0] != '\0') ? path : nullptr;
}

inline size_t routea_next_pow2(size_t v) {
  size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

struct RouteABitplaneQGLayout {
  size_t padded_dim = 0;
  size_t score_words_u32 = 0;
  size_t bitcode_words_u64 = 0;
  size_t code_offset_f = 0;
  size_t factor_offset_f = 0;
  size_t neighbor_offset_f = 0;
  size_t row_offset_f = 0;
  size_t row_bytes = 0;
};

inline RouteABitplaneQGLayout routea_qg_layout(int dim, int degree) {
  RouteABitplaneQGLayout layout;
  layout.padded_dim = routea_next_pow2(static_cast<size_t>(dim));
  layout.score_words_u32 = layout.padded_dim / 32;
  layout.bitcode_words_u64 = layout.padded_dim / 64;
  layout.code_offset_f = static_cast<size_t>(dim);
  layout.factor_offset_f = layout.code_offset_f + layout.bitcode_words_u64 * 2 * static_cast<size_t>(degree);
  layout.neighbor_offset_f = layout.factor_offset_f + 3 * static_cast<size_t>(degree);
  layout.row_offset_f = layout.neighbor_offset_f + static_cast<size_t>(degree);
  layout.row_bytes = layout.row_offset_f * sizeof(float);
  return layout;
}

inline bool routea_read_bitplane_qg_payloads(
    const char* path,
    size_t num_nodes,
    int dim,
    int degree,
    std::vector<vid_t>* out_edges,
    std::vector<uint32_t>* out_sign_bits,
    std::vector<float>* out_triple,
    std::vector<float>* out_factor_dq,
    std::vector<float>* out_factor_vq,
    std::vector<float>* out_signs) {
  if (path == nullptr || num_nodes == 0 || dim <= 0 || degree <= 0) {
    return false;
  }
  const RouteABitplaneQGLayout layout = routea_qg_layout(dim, degree);
  if (layout.score_words_u32 == 0 || layout.bitcode_words_u64 == 0) {
    return false;
  }
  const size_t expected_bytes =
      sizeof(uint32_t) + num_nodes * layout.row_bytes + layout.padded_dim * sizeof(float);

  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    std::printf("RouteA QG: cannot open bitplane codebook %s\n", path);
    return false;
  }
  fin.seekg(0, std::ios::end);
  const std::streamoff actual_bytes = fin.tellg();
  if (actual_bytes != static_cast<std::streamoff>(expected_bytes)) {
    std::printf("RouteA QG: size mismatch for %s expected=%zu actual=%lld\n",
                path, expected_bytes, static_cast<long long>(actual_bytes));
    return false;
  }
  fin.seekg(0, std::ios::beg);

  uint32_t entry_point = 0;
  fin.read(reinterpret_cast<char*>(&entry_point), sizeof(uint32_t));
  if (!fin) return false;

  const size_t edge_count = num_nodes * static_cast<size_t>(degree);
  if (out_edges) out_edges->assign(edge_count, 0);
  if (out_sign_bits) out_sign_bits->assign(edge_count * layout.score_words_u32, 0);
  if (out_triple) out_triple->assign(edge_count, 0.0f);
  if (out_factor_dq) out_factor_dq->assign(edge_count, 0.0f);
  if (out_factor_vq) out_factor_vq->assign(edge_count, 0.0f);

  std::vector<char> row(layout.row_bytes);
  const size_t code_bytes = static_cast<size_t>(degree) * layout.score_words_u32 * sizeof(uint32_t);
  for (size_t node = 0; node < num_nodes; ++node) {
    fin.read(row.data(), static_cast<std::streamsize>(row.size()));
    if (!fin) {
      std::printf("RouteA QG: truncated row at node=%zu\n", node);
      return false;
    }
    const char* row_base = row.data();
    const char* code_base = row_base + layout.code_offset_f * sizeof(float);
    const float* factors = reinterpret_cast<const float*>(
        row_base + layout.factor_offset_f * sizeof(float));
    const vid_t* neighbors = reinterpret_cast<const vid_t*>(
        row_base + layout.neighbor_offset_f * sizeof(float));

    const size_t edge_base = node * static_cast<size_t>(degree);
    if (out_edges) {
      std::memcpy(out_edges->data() + edge_base, neighbors,
                  static_cast<size_t>(degree) * sizeof(vid_t));
    }
    if (out_sign_bits) {
      std::memcpy(out_sign_bits->data() + edge_base * layout.score_words_u32,
                  code_base, code_bytes);
    }
    if (out_triple) {
      std::memcpy(out_triple->data() + edge_base, factors,
                  static_cast<size_t>(degree) * sizeof(float));
    }
    if (out_factor_dq) {
      std::memcpy(out_factor_dq->data() + edge_base,
                  factors + static_cast<size_t>(degree),
                  static_cast<size_t>(degree) * sizeof(float));
    }
    if (out_factor_vq) {
      std::memcpy(out_factor_vq->data() + edge_base,
                  factors + 2 * static_cast<size_t>(degree),
                  static_cast<size_t>(degree) * sizeof(float));
    }
  }

  if (out_signs) {
    out_signs->assign(layout.padded_dim, 0.0f);
    fin.read(reinterpret_cast<char*>(out_signs->data()),
             static_cast<std::streamsize>(layout.padded_dim * sizeof(float)));
    if (!fin) {
      std::printf("RouteA QG: truncated signs block\n");
      return false;
    }
  }

  std::printf("RouteA QG: loaded bitplane codebook %s (entry=%u, nodes=%zu, dim=%d, padded_dim=%zu, degree=%d)\n",
              path, entry_point, num_nodes, dim, layout.padded_dim, degree);
  return true;
}

template <typename T>
inline bool routea_override_graph_from_bitplane_qg(IndexGraph<T>& index, const char* path) {
  if (path == nullptr || index.ntotal == 0 || index.d <= 0 || index.maxDeg <= 0) {
    return false;
  }
  std::vector<vid_t> qg_edges;
  const bool ok = routea_read_bitplane_qg_payloads(
      path, index.ntotal, index.d, index.maxDeg,
      &qg_edges, nullptr, nullptr, nullptr, nullptr, nullptr);
  if (!ok || qg_edges.size() != index.edges.size()) {
    std::printf("RouteA QG: graph override failed for %s\n", path);
    return false;
  }
  index.edges.swap(qg_edges);
  index.graph_prefix = std::string(path);
  std::printf("RouteA QG: graph neighbors overridden from bitplane codebook\n");
  return true;
}

inline std::vector<float> load_proj_with_header(
    const char* filename, int& out_rows, int& out_cols) {
  out_rows = 0;
  out_cols = 0;
  std::ifstream f(filename, std::ios::binary);
  if (!f) {
    return {};
  }

  int32_t rows = 0;
  int32_t cols = 0;
  f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
  f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
  if (!f || rows <= 0 || cols <= 0) {
    return {};
  }

  const size_t elem_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  std::vector<float> out(elem_count);
  f.read(reinterpret_cast<char*>(out.data()),
         static_cast<std::streamsize>(elem_count * sizeof(float)));
  if (!f) {
    return {};
  }
  out_rows = rows;
  out_cols = cols;
  return out;
}

inline bool save_float_vector_raw(const char* filename, const float* data, size_t count) {
  std::ofstream f(filename, std::ios::binary | std::ios::trunc);
  if (!f) {
    return false;
  }
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(count * sizeof(float)));
  return static_cast<bool>(f);
}

inline bool save_u32_vector_raw(const char* filename, const uint32_t* data, size_t count) {
  std::ofstream f(filename, std::ios::binary | std::ios::trunc);
  if (!f) {
    return false;
  }
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(count * sizeof(uint32_t)));
  return static_cast<bool>(f);
}

inline void routea_fwht_128_host(float* x) {
  for (int h = 1; h < 128; h <<= 1) {
    for (int i = 0; i < 128; i += (h << 1)) {
      for (int j = 0; j < h; ++j) {
        const float a = x[i + j];
        const float b = x[i + j + h];
        x[i + j] = a + b;
        x[i + j + h] = a - b;
      }
    }
  }
}

inline void routea_fwht_host(float* x, int n) {
  for (int h = 1; h < n; h <<= 1) {
    for (int i = 0; i < n; i += (h << 1)) {
      for (int j = 0; j < h; ++j) {
        const float a = x[i + j];
        const float b = x[i + j + h];
        x[i + j] = a + b;
        x[i + j + h] = a - b;
      }
    }
  }
}

inline bool routea_rotate_query_with_qg_signs_host(
    const float* query,
    int dim,
    const std::vector<float>& signs,
    int padded_dim,
    std::vector<float>& rotated) {
  if (query == nullptr || dim <= 0 || padded_dim < dim ||
      signs.size() < static_cast<size_t>(padded_dim)) {
    return false;
  }
  rotated.assign(static_cast<size_t>(padded_dim), 0.0f);
  for (int i = 0; i < dim; ++i) {
    rotated[static_cast<size_t>(i)] =
        query[static_cast<size_t>(i)] * signs[static_cast<size_t>(i)];
  }
  routea_fwht_host(rotated.data(), padded_dim);
  return true;
}

inline bool routea_apply_proj_steps_host(
    const float* src,
    int dim,
    const std::vector<float>& proj_steps,
    int rows,
    int cols,
    float* dst) {
  if (src == nullptr || dst == nullptr || dim <= 0) {
    return false;
  }
  std::copy(src, src + dim, dst);
  if (dim != 128 || rows <= 0 || cols != 128 ||
      proj_steps.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
    return false;
  }
  for (int st = 0; st < rows; ++st) {
    const float* proj_row = proj_steps.data() + static_cast<size_t>(st) * 128u;
    for (int i = 0; i < 128; ++i) {
      dst[i] *= proj_row[i];
    }
    routea_fwht_128_host(dst);
  }
  return true;
}

inline std::vector<float> build_vector_mean_host(
    const float* data,
    size_t npoints,
    int dim) {
  std::vector<float> mean(static_cast<size_t>(dim), 0.0f);
  if (data == nullptr || npoints == 0 || dim <= 0) {
    return mean;
  }

#ifdef _OPENMP
  std::vector<double> partial(static_cast<size_t>(dim), 0.0);
  #pragma omp parallel
  {
    std::vector<double> local(static_cast<size_t>(dim), 0.0);
    #pragma omp for schedule(static)
    for (int64_t uu = 0; uu < static_cast<int64_t>(npoints); ++uu) {
      const float* vec = data + static_cast<size_t>(uu) * static_cast<size_t>(dim);
      for (int d = 0; d < dim; ++d) {
        local[static_cast<size_t>(d)] += static_cast<double>(vec[d]);
      }
    }
    #pragma omp critical
    {
      for (int d = 0; d < dim; ++d) {
        partial[static_cast<size_t>(d)] += local[static_cast<size_t>(d)];
      }
    }
  }
  for (int d = 0; d < dim; ++d) {
    mean[static_cast<size_t>(d)] =
        static_cast<float>(partial[static_cast<size_t>(d)] / static_cast<double>(npoints));
  }
#else
  std::vector<double> sum(static_cast<size_t>(dim), 0.0);
  for (size_t u = 0; u < npoints; ++u) {
    const float* vec = data + u * static_cast<size_t>(dim);
    for (int d = 0; d < dim; ++d) {
      sum[static_cast<size_t>(d)] += static_cast<double>(vec[d]);
    }
  }
  for (int d = 0; d < dim; ++d) {
    mean[static_cast<size_t>(d)] =
        static_cast<float>(sum[static_cast<size_t>(d)] / static_cast<double>(npoints));
  }
#endif

  return mean;
}

struct RouteAQueryQ6Prepared {
  std::vector<uint8_t> qint;
  std::vector<uint16_t> lut;
  float lower = 0.0f;
  float upper = 0.0f;
  float width = 0.0f;
  float residual_norm2 = 0.0f;
  int sumq = 0;
};

inline bool routea_prepare_q6_from_values_host(
    const float* values,
    int dim,
    RouteAQueryQ6Prepared& out) {
  out = RouteAQueryQ6Prepared{};
  if (values == nullptr || dim <= 0) {
    return false;
  }

  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  float norm2 = 0.0f;
  for (int i = 0; i < dim; ++i) {
    const float v = values[static_cast<size_t>(i)];
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    norm2 = std::fmaf(v, v, norm2);
  }
  if (!std::isfinite(lo) || !std::isfinite(hi)) {
    return false;
  }

  out.lower = lo;
  out.upper = hi;
  out.width = (hi > lo) ? ((hi - lo) / 63.0f) : 0.0f;
  out.residual_norm2 = norm2;
  out.qint.assign(static_cast<size_t>(dim), 0u);
  out.sumq = 0;

  for (int i = 0; i < dim; ++i) {
    int qv = 0;
    if (out.width > 0.0f) {
      const float normalized =
          ((values[static_cast<size_t>(i)] - out.lower) / out.width) + 0.5f;
      qv = static_cast<int>(std::lround(normalized));
      qv = std::max(0, std::min(63, qv));
    }
    out.qint[static_cast<size_t>(i)] = static_cast<uint8_t>(qv);
    out.sumq += qv;
  }

  return true;
}

inline bool routea_prepare_q6_lut_from_values_host(
    const float* values,
    int dim,
    RouteAQueryQ6Prepared& out) {
  if (!routea_prepare_q6_from_values_host(values, dim, out)) {
    return false;
  }

  const int groups = (dim + 3) >> 2;
  out.lut.assign(static_cast<size_t>(groups) * 16u, 0u);
  for (int g = 0; g < groups; ++g) {
    uint16_t* lut_row = out.lut.data() + static_cast<size_t>(g) * 16u;
    lut_row[0] = 0u;
    for (int code = 1; code < 16; ++code) {
      const int lowbit = code & -code;
      const int bit = __builtin_ctz(lowbit);
      const int idx = (g << 2) + bit;
      const uint16_t add =
          (idx < dim) ? static_cast<uint16_t>(out.qint[static_cast<size_t>(idx)]) : 0u;
      lut_row[code] = static_cast<uint16_t>(lut_row[code - lowbit] + add);
    }
  }

  return true;
}

inline bool routea_prepare_raw_query_q6_lut_host(
    const float* query,
    int dim,
    RouteAQueryQ6Prepared& out) {
  if (query == nullptr || dim <= 0) {
    out = RouteAQueryQ6Prepared{};
    return false;
  }
  std::vector<float> work(static_cast<size_t>(dim), 0.0f);
  std::copy(query, query + dim, work.begin());
  return routea_prepare_q6_lut_from_values_host(work.data(), dim, out);
}

inline bool routea_prepare_query_q6_lut_host(
    const float* query,
    int dim,
    const std::vector<float>& centroid,
    const std::vector<float>& proj_steps,
    int proj_rows,
    int proj_cols,
    RouteAQueryQ6Prepared& out) {
  out = RouteAQueryQ6Prepared{};
  if (query == nullptr || dim <= 0 ||
      centroid.size() != static_cast<size_t>(dim)) {
    return false;
  }

  std::vector<float> work(static_cast<size_t>(dim), 0.0f);
  const bool used_proj = routea_apply_proj_steps_host(
      query, dim, proj_steps, proj_rows, proj_cols, work.data());
  if (!used_proj) {
    std::copy(query, query + dim, work.begin());
  }
  for (int i = 0; i < dim; ++i) {
    work[static_cast<size_t>(i)] -= centroid[static_cast<size_t>(i)];
  }
  return routea_prepare_q6_lut_from_values_host(work.data(), dim, out);
}

inline std::vector<uint32_t> try_load_sign_bits_with_fallback(
    const std::vector<std::string>& candidates,
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

}  // namespace beam_gpu_host

// ── Section 3 (formerly beam_gpu_routea_sidecars.hpp) ───────────────────────

// =============================================================================
// beam_gpu_routea_sidecars.hpp — Host-side RouteA sidecar orchestration.
//
// Template host helpers for QG metadata, edge sidecar, and per-query q6
// bitplane upload. As of Phase 3.2 of the architecture hardening plan this
// header is includable at top level: the shared host types and runtime
// helpers (`StableRabitqSidecars`,
// `RouteAQuerySidecars`, `RouteAQueryPrepTiming`, `HostTimePoint`,
// `host_now`, `host_elapsed_ms`, `routea_runtime_enabled`,
// `routea_query_prep_gpu_kernel<T>` decl) live in
// `beam_gpu_routea_types.hpp`. `save_sign_bit` is provided at file scope
// by `search_bitonic/kernel.cuh` (already included by callers).
//
// The sidecar functions remain free templates (not in any namespace) so
// existing call sites in `IndexGraph<T>::search` keep their unqualified
// names; this is intentional and preserves the timing scope and error
// messages observed in the formal smoke baseline.
// =============================================================================

#include "qleaf_host_cache.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


template <typename T>
void ensure_qg_metadata_loaded(
    beam_gpu_host::DeviceCache<T>& cache,
    size_t npoints,
    int dim,
    int max_degree) {
  // Loads QG-mode host-side metadata (cache.qg_signs and cache.qg_padded_dim)
  // WITHOUT touching the per-edge GPU sign-bit upload. Needed for any
  // routea path that uses bitplane_qg input where the edge sidecar GPU
  // upload is gated off but query rotation and score_dims still depend on
  // QG metadata.
  if (cache.qg_padded_dim > 0 && !cache.qg_signs.empty()) {
    return;
  }
  const char* qg_path = beam_gpu_host::routea_bitplane_qg_file_env();
  if (qg_path == nullptr) {
    return;
  }
  const int words_per_edge =
      (dim <= 0) ? 0
                 : static_cast<int>(beam_gpu_host::routea_qg_layout(dim, max_degree).score_words_u32);
  if (words_per_edge <= 0) {
    printf("RouteA QG metadata: invalid layout (dim=%d max_degree=%d)\n",
           dim, max_degree);
    return;
  }
  std::vector<float> signs;
  const bool ok = beam_gpu_host::routea_read_bitplane_qg_payloads(
      qg_path, npoints, dim, max_degree,
      nullptr, nullptr, nullptr, nullptr, nullptr, &signs);
  if (!ok || signs.empty()) {
    printf("RouteA QG metadata: failed to load signs from %s\n", qg_path);
    return;
  }
  cache.qg_signs.swap(signs);
  cache.qg_padded_dim = words_per_edge << 5;
  cache.routea_bitplane_qg_loaded = true;
  printf("RouteA QG metadata: loaded host signs (count=%zu) and qg_padded_dim=%d (no GPU edge upload)\n",
         cache.qg_signs.size(), cache.qg_padded_dim);
}

template <typename T>
void ensure_sign_bit_sidecar(
    const IndexGraph<T>* index,
    beam_gpu_host::DeviceCache<T>& cache,
    const T* host_data,
    size_t npoints,
    int dim,
    int max_degree) {
  if (cache.edge_sidecar.d_sign_bit == nullptr) {
    const char* qg_path = beam_gpu_host::routea_bitplane_qg_file_env();
    if (qg_path != nullptr) {
      std::vector<uint32_t> sign_bit;
      std::vector<float> signs;
      const bool ok = beam_gpu_host::routea_read_bitplane_qg_payloads(
          qg_path, npoints, dim, max_degree,
          nullptr, &sign_bit, nullptr, nullptr, nullptr, &signs);
      const int upload_words_per_edge = (dim <= 0) ? 0 :
          static_cast<int>(beam_gpu_host::routea_qg_layout(dim, max_degree).score_words_u32);
      const size_t upload_words = npoints * static_cast<size_t>(max_degree) *
                                  static_cast<size_t>(std::max(0, upload_words_per_edge));
      if (ok && upload_words_per_edge > 0 && sign_bit.size() == upload_words) {
        printf("RouteA QG: uploading bitplane codebook bits as sign table (%zu bytes, words/edge=%d)\n",
               upload_words * sizeof(uint32_t), upload_words_per_edge);
        CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_sign_bit,
                                  upload_words * sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_sign_bit, sign_bit.data(),
                                  upload_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
        cache.edge_sidecar.sign_words = upload_words;
        cache.edge_sidecar.sign_words_per_edge = upload_words_per_edge;
        cache.qg_signs.swap(signs);
        cache.qg_padded_dim = upload_words_per_edge << 5;
        cache.routea_bitplane_qg_loaded = true;
      } else {
        printf("RouteA QG: failed to load bitplane codebook bits from %s\n", qg_path);
        cache.edge_sidecar.d_sign_bit = nullptr;
        cache.edge_sidecar.sign_words = 0;
        cache.edge_sidecar.sign_words_per_edge = 0;
      }
      return;
    }
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
      CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_sign_bit, upload_words * sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_sign_bit, sign_src,
                                upload_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
      cache.edge_sidecar.sign_words = upload_words;
      cache.edge_sidecar.sign_words_per_edge = upload_words_per_edge;
    } else {
      printf("Residual-sign: sign-bit table unavailable.\n");
      cache.edge_sidecar.d_sign_bit = nullptr;
      cache.edge_sidecar.sign_words = 0;
      cache.edge_sidecar.sign_words_per_edge = 0;
    }
  }
}

template <typename T>
RouteAQuerySidecars routea_query_sidecars_from_cache(
    const beam_gpu_host::DeviceCache<T>& cache) {
  RouteAQuerySidecars sidecars;
  sidecars.d_bp = cache.query_prep.d_routea_query_bp;
  sidecars.d_lower = cache.query_prep.d_routea_query_lower;
  sidecars.d_width = cache.query_prep.d_routea_query_width;
  sidecars.d_sumq = cache.query_prep.d_routea_query_sumq;
  sidecars.score_words = cache.query_prep.routea_query_score_words;
  sidecars.score_dims = cache.query_prep.routea_query_score_dims;
  return sidecars;
}

inline bool routea_gpu_query_prep_runtime_enabled() {
#if ROUTEA_GPU_QUERY_PREP
  const char* raw = std::getenv("ROUTEA_GPU_QUERY_PREP_RUNTIME");
  return raw == nullptr || raw[0] == '\0' || raw[0] != '0';
#else
  return false;
#endif
}

template <typename T>
bool ensure_qg_signs_device(beam_gpu_host::DeviceCache<T>& cache) {
  if (cache.qg_signs.empty()) {
    return false;
  }
  const size_t count = cache.qg_signs.size();
  if (cache.d_qg_signs == nullptr || cache.qg_signs_count < count) {
    beam_gpu_host::cuda_free_if(cache.d_qg_signs);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.d_qg_signs, count * sizeof(float)));
    cache.qg_signs_count = count;
  }
  CUDA_SAFE_CALL(cudaMemcpy(cache.d_qg_signs, cache.qg_signs.data(),
                            count * sizeof(float), cudaMemcpyHostToDevice));
  return true;
}

template <typename T>
bool build_routea_query_sidecars_gpu(
    beam_gpu_host::DeviceCache<T>& cache,
    int nq,
    int dim,
    int score_dims,
    bool qg_mode,
    RouteAQueryPrepTiming* timing = nullptr) {
  if (!routea_gpu_query_prep_runtime_enabled() ||
      cache.d_queries == nullptr ||
      nq <= 0 || dim <= 0 ||
      score_dims <= 0 || score_dims > kStableRabitqMaxScoreDims) {
    return false;
  }
  if (qg_mode && !ensure_qg_signs_device(cache)) {
    return false;
  }

  const int score_words = (score_dims + 31) >> 5;
  const size_t bp_words = static_cast<size_t>(nq) * 6u * static_cast<size_t>(score_words);

  const HostTimePoint alloc_start = host_now();
  if (cache.query_prep.d_routea_query_bp == nullptr || cache.query_prep.routea_query_bp_words < bp_words) {
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_bp);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_bp,
                              bp_words * sizeof(uint32_t)));
    cache.query_prep.routea_query_bp_words = bp_words;
  }
  if (cache.query_prep.d_routea_query_lower == nullptr || cache.query_prep.routea_query_count < static_cast<size_t>(nq)) {
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_lower);
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_width);
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_sumq);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_lower,
                              static_cast<size_t>(nq) * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_width,
                              static_cast<size_t>(nq) * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_sumq,
                              static_cast<size_t>(nq) * sizeof(int)));
    cache.query_prep.routea_query_count = static_cast<size_t>(nq);
  }
  if (timing != nullptr) {
    timing->alloc_ms = host_elapsed_ms(alloc_start);
  }

  cudaEvent_t ev0, ev1;
  CUDA_SAFE_CALL(cudaEventCreate(&ev0));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1));
  CUDA_SAFE_CALL(cudaEventRecord(ev0));
  constexpr int kThreads = 256;
  routea_query_prep_gpu_kernel<T><<<nq, kThreads>>>(
      cache.d_queries, nq, dim, score_dims,
      qg_mode ? cache.d_qg_signs : nullptr,
      qg_mode ? 1 : 0,
      cache.query_prep.d_routea_query_bp,
      cache.query_prep.d_routea_query_lower,
      cache.query_prep.d_routea_query_width,
      cache.query_prep.d_routea_query_sumq);
  CUDA_SAFE_CALL(cudaEventRecord(ev1));
  CUDA_SAFE_CALL(cudaEventSynchronize(ev1));
  float gpu_ms = 0.0f;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&gpu_ms, ev0, ev1));
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaEventDestroy(ev0));
  CUDA_SAFE_CALL(cudaEventDestroy(ev1));

  cache.query_prep.routea_query_score_words = score_words;
  cache.query_prep.routea_query_score_dims = score_dims;
  if (timing != nullptr) {
    timing->gpu_ms = static_cast<double>(gpu_ms);
  }
  printf("RouteA query-prep: GPU q6 bitplanes (qg_mode=%d, score_dims=%d, score_words=%d, bytes=%zu, gpu_ms=%.3f)\n",
         qg_mode ? 1 : 0, score_dims, score_words, bp_words * sizeof(uint32_t),
         static_cast<double>(gpu_ms));
  return true;
}

template <typename T>
void build_and_upload_routea_query_sidecars(
    beam_gpu_host::DeviceCache<T>& cache,
    const T* queries,
    int nq,
    int dim,
    RouteAQueryPrepTiming* timing = nullptr) {
  const HostTimePoint total_start = host_now();
  auto clear_query_sidecars = [&]() {
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_bp);
    cache.query_prep.routea_query_bp_words = 0;
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_lower);
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_width);
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_sumq);
    cache.query_prep.routea_query_count = 0;
    cache.query_prep.routea_query_score_words = 0;
    cache.query_prep.routea_query_score_dims = 0;
  };
  if (!routea_runtime_enabled()) {
    clear_query_sidecars();
    if (timing != nullptr) {
      timing->total_ms = host_elapsed_ms(total_start);
    }
    return;
  }
  // Query-prep is needed whenever the LOCAL edge gate (GATE_MODE=1) or
  // NAV=8 (Sophia edge-residual RaBitQ-as-key) is active. For exact_w*
  // variants (qprep=0) we already returned above.
  if constexpr (!kRouteANeedsQueryPrep) {
    clear_query_sidecars();
    if (timing != nullptr) {
      timing->total_ms = host_elapsed_ms(total_start);
    }
    return;
  }
  if (nq <= 0 || queries == nullptr) {
    clear_query_sidecars();
    if (timing != nullptr) {
      timing->total_ms = host_elapsed_ms(total_start);
    }
    return;
  }
  if constexpr (kRouteANeedsEdgeSidecar) {
    if (cache.edge_sidecar.d_sign_bit == nullptr ||
        cache.edge_sidecar.d_rabitq_triple == nullptr ||
        cache.edge_sidecar.d_rabitq_factor_dq == nullptr ||
        cache.edge_sidecar.d_rabitq_factor_vq == nullptr) {
      printf("RouteA query-prep: edge sidecar required by GATE_MODE=%d but not loaded; skipping\n",
             (int)ROUTEA_GATE_MODE);
      clear_query_sidecars();
      if (timing != nullptr) {
        timing->total_ms = host_elapsed_ms(total_start);
      }
      return;
    }
  }

  const bool qg_mode = (beam_gpu_host::routea_bitplane_qg_file_env() != nullptr);
  const int score_dims = qg_mode ? cache.qg_padded_dim : dim;
  const int score_words = (score_dims + 31) >> 5;
  if (score_dims <= 0 || score_words <= 0 ||
      score_words > kStableRabitqScoreWords) {
    printf("RouteA query-prep: invalid score window score_dims=%d score_words=%d compile_words=%d\n",
           score_dims, score_words, kStableRabitqScoreWords);
    clear_query_sidecars();
    if (timing != nullptr) {
      timing->total_ms = host_elapsed_ms(total_start);
    }
    return;
  }
  if (qg_mode && cache.qg_signs.size() < static_cast<size_t>(score_dims)) {
    printf("RouteA query-prep: qg mode requested but signs are unavailable\n");
    clear_query_sidecars();
    if (timing != nullptr) {
      timing->total_ms = host_elapsed_ms(total_start);
    }
    return;
  }

  if (build_routea_query_sidecars_gpu(
          cache, nq, dim, score_dims, qg_mode, timing)) {
    if (timing != nullptr) {
      timing->total_ms = host_elapsed_ms(total_start);
    }
    return;
  }

  const HostTimePoint host_start = host_now();
  const size_t bp_words = static_cast<size_t>(nq) * 6u * static_cast<size_t>(score_words);
  std::vector<uint32_t> bp_host(bp_words, 0u);
  std::vector<float> lower_host(static_cast<size_t>(nq), 0.0f);
  std::vector<float> width_host(static_cast<size_t>(nq), 0.0f);
  std::vector<int> sumq_host(static_cast<size_t>(nq), 0);
  std::vector<float> rotated;

  for (int q = 0; q < nq; ++q) {
    beam_gpu_host::RouteAQueryQ6Prepared prep;
    const float* query_f = reinterpret_cast<const float*>(queries) +
                           static_cast<size_t>(q) * static_cast<size_t>(dim);
    bool ok = false;
    if (qg_mode) {
      ok = beam_gpu_host::routea_rotate_query_with_qg_signs_host(
          query_f, dim, cache.qg_signs, score_dims, rotated);
      if (ok) {
        ok = beam_gpu_host::routea_prepare_q6_from_values_host(
            rotated.data(), score_dims, prep);
      }
    } else {
      ok = beam_gpu_host::routea_prepare_q6_from_values_host(
          query_f, dim, prep);
    }
    if (!ok || prep.qint.size() != static_cast<size_t>(score_dims)) {
      printf("RouteA query-prep: failed for q=%d (dim=%d score_dims=%d qg_mode=%d)\n",
             q, dim, score_dims, qg_mode ? 1 : 0);
      clear_query_sidecars();
      if (timing != nullptr) {
        timing->host_ms = host_elapsed_ms(host_start);
        timing->total_ms = host_elapsed_ms(total_start);
      }
      return;
    }
    uint32_t* bp_q = bp_host.data() + static_cast<size_t>(q) * 6u * static_cast<size_t>(score_words);
    for (int i = 0; i < score_dims; ++i) {
      const int qv = static_cast<int>(prep.qint[static_cast<size_t>(i)]);
      for (int k = 0; k < 6; ++k) {
        if (((qv >> k) & 1) != 0) {
          bp_q[static_cast<size_t>(k) * static_cast<size_t>(score_words) +
               static_cast<size_t>(i >> 5)] |= (1u << (i & 31));
        }
      }
    }
    lower_host[static_cast<size_t>(q)] = prep.lower;
    width_host[static_cast<size_t>(q)] = prep.width;
    sumq_host[static_cast<size_t>(q)] = prep.sumq;
  }
  if (timing != nullptr) {
    timing->host_ms = host_elapsed_ms(host_start);
  }

  const HostTimePoint alloc_start = host_now();
  if (cache.query_prep.d_routea_query_bp == nullptr || cache.query_prep.routea_query_bp_words < bp_words) {
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_bp);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_bp,
                              bp_words * sizeof(uint32_t)));
    cache.query_prep.routea_query_bp_words = bp_words;
  }
  if (cache.query_prep.d_routea_query_lower == nullptr || cache.query_prep.routea_query_count < static_cast<size_t>(nq)) {
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_lower);
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_width);
    beam_gpu_host::cuda_free_if(cache.query_prep.d_routea_query_sumq);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_lower,
                              static_cast<size_t>(nq) * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_width,
                              static_cast<size_t>(nq) * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.query_prep.d_routea_query_sumq,
                              static_cast<size_t>(nq) * sizeof(int)));
    cache.query_prep.routea_query_count = static_cast<size_t>(nq);
  }
  if (timing != nullptr) {
    timing->alloc_ms = host_elapsed_ms(alloc_start);
  }

  const HostTimePoint upload_start = host_now();
  CUDA_SAFE_CALL(cudaMemcpy(cache.query_prep.d_routea_query_bp, bp_host.data(),
                            bp_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.query_prep.d_routea_query_lower, lower_host.data(),
                            static_cast<size_t>(nq) * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.query_prep.d_routea_query_width, width_host.data(),
                            static_cast<size_t>(nq) * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.query_prep.d_routea_query_sumq, sumq_host.data(),
                            static_cast<size_t>(nq) * sizeof(int), cudaMemcpyHostToDevice));
  if (timing != nullptr) {
    timing->upload_ms = host_elapsed_ms(upload_start);
  }
  cache.query_prep.routea_query_score_words = score_words;
  cache.query_prep.routea_query_score_dims = score_dims;
  printf("RouteA query-prep: uploaded q6 bitplanes (qg_mode=%d, score_dims=%d, score_words=%d, bytes=%zu)\n",
         qg_mode ? 1 : 0, score_dims, score_words, bp_words * sizeof(uint32_t));
  if (timing != nullptr) {
    timing->total_ms = host_elapsed_ms(total_start);
  }
}

template <typename T>
void ensure_factor_sidecars(
    const IndexGraph<T>* index,
    beam_gpu_host::DeviceCache<T>& cache,
    size_t npoints,
    int max_degree) {
  if (cache.edge_sidecar.d_rabitq_triple != nullptr &&
      cache.edge_sidecar.d_rabitq_factor_dq != nullptr &&
      cache.edge_sidecar.d_rabitq_factor_vq != nullptr) {
    return;
  }

  const size_t factor_count = npoints * static_cast<size_t>(max_degree);
  const char* qg_path = beam_gpu_host::routea_bitplane_qg_file_env();
  if (qg_path != nullptr) {
    std::vector<float> triple_host;
    std::vector<float> dq_host;
    std::vector<float> vq_host;
    std::vector<float> signs;
    const bool ok = beam_gpu_host::routea_read_bitplane_qg_payloads(
        qg_path, npoints, cache.dim, max_degree,
        nullptr, nullptr, &triple_host, &dq_host, &vq_host,
        cache.qg_signs.empty() ? &signs : nullptr);
    if (!ok || triple_host.size() != factor_count ||
        dq_host.size() != factor_count || vq_host.size() != factor_count) {
      printf("RouteA QG: factor tables unavailable from %s\n", qg_path);
      return;
    }
    if (!signs.empty()) {
      cache.qg_signs.swap(signs);
      cache.qg_padded_dim = static_cast<int>(beam_gpu_host::routea_qg_layout(cache.dim, max_degree).padded_dim);
    }
    printf("RouteA QG: uploading codebook factors (%zu edges)\n", factor_count);
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_rabitq_triple, factor_count * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_rabitq_factor_dq, factor_count * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_rabitq_factor_vq, factor_count * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_rabitq_triple, triple_host.data(),
                              factor_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_rabitq_factor_dq, dq_host.data(),
                              factor_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_rabitq_factor_vq, vq_host.data(),
                              factor_count * sizeof(float), cudaMemcpyHostToDevice));
    cache.edge_sidecar.rabitq_factor_count = factor_count;
    return;
  }

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
  std::string vq_from;
  std::vector<float> triple_host = load_factor_with_fallback(
      "RABITQ_TRIPLE_FILE", "rabitq.triple.f32.bin", ".rabitq.triple.f32.bin", "triple", triple_from);
  std::vector<float> dq_host = load_factor_with_fallback(
      "RABITQ_FACTOR_DQ_FILE", "rabitq.factor_dq.f32.bin", ".rabitq.factor_dq.f32.bin", "factor_dq", dq_from);
  std::vector<float> vq_host = load_factor_with_fallback(
      "RABITQ_FACTOR_VQ_FILE", "rabitq.factor_vq.f32.bin", ".rabitq.factor_vq.f32.bin", "factor_vq", vq_from);

  if (triple_host.size() != factor_count ||
      dq_host.size() != factor_count ||
      vq_host.size() != factor_count) {
    printf("RaBitQ: factor tables unavailable; current stable kernel cannot use CPU-style compensated approx.\n");
    return;
  }

  printf("GPU cache: uploading RaBitQ factors (%zu edges)\n", factor_count);
  CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_rabitq_triple, factor_count * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_rabitq_factor_dq, factor_count * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&cache.edge_sidecar.d_rabitq_factor_vq, factor_count * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_rabitq_triple, triple_host.data(),
                            factor_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_rabitq_factor_dq, dq_host.data(),
                            factor_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(cache.edge_sidecar.d_rabitq_factor_vq, vq_host.data(),
                            factor_count * sizeof(float), cudaMemcpyHostToDevice));
  cache.edge_sidecar.rabitq_factor_count = factor_count;
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
  // Edge-sidecar load+upload is gated on the LOCAL gate consuming it
  // (GATE_MODE=1) or NAV=8 reading it. For exact_w* variants the kernel
  // never reads the edge sidecar, so we skip both file I/O and GPU
  // malloc/memcpy entirely.
  if constexpr (kRouteANeedsEdgeSidecar) {
    ensure_sign_bit_sidecar(index, cache, host_data, npoints, dim, max_degree);
    ensure_factor_sidecars(index, cache, npoints, max_degree);
  } else {
    // QG metadata is still needed for query-prep shape even when the edge
    // sidecar is gated off (e.g. variants that load no per-edge sidecars
    // but still rotate the query). Local-gate QG runs get the same
    // metadata from ensure_sign_bit_sidecar above, avoiding a duplicate
    // QG file read.
    ensure_qg_metadata_loaded(cache, npoints, dim, max_degree);
  }

  sidecars.d_sign_bit = cache.edge_sidecar.d_sign_bit;
  sidecars.sign_words_per_edge = cache.edge_sidecar.sign_words_per_edge;
  sidecars.d_rabitq_triple = cache.edge_sidecar.d_rabitq_triple;
  sidecars.d_rabitq_factor_dq = cache.edge_sidecar.d_rabitq_factor_dq;
  sidecars.d_rabitq_factor_vq = cache.edge_sidecar.d_rabitq_factor_vq;
  return sidecars;
}
