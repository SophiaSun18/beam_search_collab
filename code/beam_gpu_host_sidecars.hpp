#pragma once

#include "index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

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

inline bool routea_prepare_q6_lut_from_values_host(
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
