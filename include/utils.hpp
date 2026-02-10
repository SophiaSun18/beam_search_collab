#pragma once

#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>
#include <cxxabi.h>
#include <unistd.h>
#include <sys/resource.h>
#include <array>

template <typename T>
void PREFETCH_VECTOR(int dim, T *vec) {
#ifdef USE_PF_VEC
  char* a = (char*)vec;
  int l = dim*sizeof(T)/64;
  for (int j = 0; j < l; j++) {
    __builtin_prefetch(a+64*j);
    //_mm_prefetch(a+64*j, _MM_HINT_T2);
    //if (j < 4) _mm_prefetch(a+64*j, _MM_HINT_T0);
    //else _mm_prefetch(a+64*j, _MM_HINT_T2);
  }
#endif
}

template <typename T = float>
T* read_vecs(const char* filename, size_t& n_out, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open fvecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    if (d <= 0 || d > 10000) {
        throw std::runtime_error("Invalid dimension read from bvecs header");
    }
    dim = d;
    //std::cout << "dim: " << dim << std::endl;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(T);
    input.seekg(0, std::ios::beg);
    size_t num = file_size / record_size;
    //std::cout << "num: " << num << std::endl;
    size_t total = num * dim;
    T* data;
    if ((dim*sizeof(T)) % 32 == 0)
      data = (T*)aligned_alloc(32, sizeof(T) * total); // for SIMD
    else data = new T[total];

    for (size_t i = 0; i < num; ++i) {
        int vec_dim = 0;
        input.read((char*)&vec_dim, sizeof(int));
        assert(vec_dim == d);
        input.read((char*)(data + i * d), sizeof(T) * d);
    }
    n_out = num;
    //std::cout << "nb: " << num << std::endl;
    return data;
}

template <typename T = float>
T* read_bin(const char* filename, size_t& n_out, size_t& d_out) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open bin file");
  int32_t num = 0, dim = 0;
  input.read(reinterpret_cast<char*>(&num), sizeof(int32_t));
  input.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
  if (num <= 0 || dim <= 0)
    throw std::runtime_error("Invalid dimensions read from file");
  size_t total = static_cast<size_t>(num) * dim;
  T *data;
  if ((dim*sizeof(T)) % 32 == 0)
    data = (T*)aligned_alloc(32, sizeof(T) * total); // for SIMD
  else data = new T[total];
  input.read(reinterpret_cast<char*>(data), sizeof(T) * total);
  //for (size_t i = 0; i < num; i++) in.read((char*)(data+i*dim), sizeof(T)*dim);
  if (!input) throw std::runtime_error("Error reading bin vector data");
  n_out = size_t(num);
  d_out = size_t(dim);
  input.close();
  return data;
}

template <typename T>
void write_bin(const char* filename, const T* data, size_t n, int d) {
  std::ofstream output(filename, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot open output ibin file");
  output.write(reinterpret_cast<const char*>(&n), sizeof(int));
  output.write(reinterpret_cast<const char*>(&d), sizeof(int));
  output.write(reinterpret_cast<const char*>(data), sizeof(T) * size_t(d) * n);
  if (!output) throw std::runtime_error("Error writing to ibin file");
  output.close();
}


template <typename T>
float compute_recall(const T* predicted, const int* groundtruth,
                     size_t nq, int topk, int gt_k) {
    size_t correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        for (int j = 0; j < topk; ++j) {
            int pred = int(predicted[i * topk + j]);
            for (int k = 0; k < gt_k; ++k) {
                int gt = groundtruth[i * gt_k + k];
                if (pred == gt) {
                    ++correct;
                    break;
                }
            }
        }
    }
    return float(correct) / (nq * topk);
}

#define LOWBIT(x) ((x) & (-(x)))

constexpr static std::array<int, 16> kPos = {
    3 /*0000*/,
    3 /*0001*/,
    2 /*0010*/,
    3 /*0011*/,
    1 /*0100*/,
    3 /*0101*/,
    2 /*0110*/,
    3 /*0111*/,
    0 /*1000*/,
    3 /*1001*/,
    2 /*1010*/,
    3 /*1011*/,
    1 /*1100*/,
    3 /*1101*/,
    2 /*1110*/,
    3 /*1111*/,
};

// Use Fast Hadamard Transform to rotate vector
class FHTRotator {
 private:
  int padded_dim;
  std::vector<float> rands;
 public:
  FHTRotator(int dim) : padded_dim(dim) {
    rands.resize(padded_dim);
    // assert(dim % 32 == 0); // todo: must be multiple of 32 for now
  }
  void init() { 
    std::uniform_int_distribution<int> bernoulli(0, 1);
    std::random_device rdd;
    std::mt19937_64 gen(rdd());
    for (int i = 0; i < padded_dim; ++i) {
      rands[i] = static_cast<float>((2 * bernoulli(gen)) - 1) /
                std::sqrt(static_cast<float>(padded_dim));
    }
  }
  inline void rotate(const float* __restrict__ raw_vec, 
                           float* __restrict__ rotated_vec) {
    for (int i = 0; i < padded_dim; ++i)
      rotated_vec[i] = raw_vec[i] * rands[i];
  }
  void load(std::ifstream &input) {
    // printf("Loading rotator data\n");
    input.read(reinterpret_cast<char*>(rands.data()), sizeof(float) * padded_dim);
    if (!input) throw std::runtime_error("Error reading rotator bin data");
    // else printf("rotator data loaded\n");
  }
  void store(std::ofstream &output) {
    // std::ofstream output(gFile, std::ios::binary | std::ios::app);
    // if (!output) throw std::runtime_error("Cannot open output rotator fbin file");
    output.write(reinterpret_cast<const char*>(&padded_dim), sizeof(int));
    output.write(reinterpret_cast<const char*>(rands.data()), sizeof(float) * padded_dim);
    if (!output) throw std::runtime_error("Error writing to rotator fbin file");
    // else printf("rotator data stored, padded_dim = %d\n", padded_dim);
    // output.close();
  }
};

template <typename T>
std::vector<uint32_t> compute_sign_vector(int dim, const T* vec1, const T* vec2) {
  int num = (dim + 31) / 32;
  std::vector<uint32_t> sign_vec(num, 0);
  for (int i = 0; i < dim; ++i) {
    if (vec1[i] > vec2[i]) sign_vec[i / 32] |= (1U << (31-(i % 32)));
  }
  return sign_vec;
}

inline void accumulate_impl(int length,
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result) {
}

template<typename T>
T approx_dist_impl(size_t num_points,
    T sqr_y, float width, float vl,
    const T* __restrict__ result,
    const T* __restrict__ triple_x,
    const T* __restrict__ fac_dq,
    const T* __restrict__ fac_vq) {
  T dist = 0;
  return dist;
}

constexpr static std::array<int, 16> kPerm0 = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
};

template <typename T, class TA>
static inline void get_column(const T* src, size_t rows, size_t cols, size_t row, size_t col, TA& dest) {
  size_t k = 0;
  size_t max_k = std::min(rows - row, dest.size());
  for (; k < max_k; ++k) {
    dest[k] = src[((k + row) * cols) + col];
  }
  if (k < dest.size()) {
    std::fill(dest.begin() + k, dest.end(), 0);
  }
}

#include <cstring>

static inline void pack_codes_helper(int dim, size_t ncode, const uint8_t* codes, uint8_t* blocks) {
    size_t ncode_pad = (ncode + 31) & ~31;  // size_t ncode_pad = ((ncode + 31) / 32) * 32;
    int num_codebook = dim / 4;
    std::memset(blocks, 0, ncode_pad * num_codebook / 2);

    uint8_t* codes2 = blocks;
    for (size_t blk = 0; blk < ncode_pad; blk += 32) {
        // enumerate i
        for (int i = 0; i < num_codebook; i += 2) {
            std::array<uint8_t, 32> col;
            std::array<uint8_t, 32> col_lo;
            std::array<uint8_t, 32> col_hi;
            get_column(codes, ncode, num_codebook / 2, blk, i / 2, col);
            for (int j = 0; j < 32; j++) {
                col_lo[j] = col[j] & 15;
                col_hi[j] = col[j] >> 4;
            }
            for (int j = 0; j < 16; j++) {
                auto val0 = col_lo[kPerm0[j]] | (col_lo[kPerm0[j] + 16] << 4);
                auto val1 = col_hi[kPerm0[j]] | (col_hi[kPerm0[j] + 16] << 4);
                codes2[j] = val0;
                codes2[j + 16] = val1;
            }
            codes2 += 32;
        }
    }
}

inline void pack_rabitq_codes(int ncode, int dim, const uint64_t* binary_code, uint8_t* blocks) {
    size_t ncode_pad = (ncode + 31) & ~31;  // size_t ncode_pad = ((ncode + 31) / 32) * 32;
    std::vector<uint8_t> binary_code_8bit(ncode_pad * dim / 8);
    std::memcpy(binary_code_8bit.data(), binary_code, ncode * dim / 64 * sizeof(uint64_t));
    // swap byte pairs (0, 7), (1, 6), (2, 5), (3, 4)
    for (int i = 0; i < ncode; ++i) {
        for (int j = 0; j < dim / 64; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                std::swap(binary_code_8bit[(i * dim / 8) + (8 * j) + k],
                          binary_code_8bit[(i * dim / 8) + (8 * j) + 8 - k - 1]);
            }
        }
    }
    // swap high and low nibbles within each byte
    for (size_t i = 0; i < size_t(ncode) * dim / 8; ++i) {
        uint8_t val = binary_code_8bit[i];
        uint8_t val_hi = (val >> 4);
        uint8_t val_lo = (val & 15);
        binary_code_8bit[i] = (val_lo << 4) | val_hi;
    }
    pack_codes_helper(dim, ncode, binary_code_8bit.data(), blocks);
}