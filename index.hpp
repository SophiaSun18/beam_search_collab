#pragma once
#include <sys/types.h>
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
typedef uint32_t vid_t;

enum DistanceMetric {
  METRIC_L2 = 0,
  METRIC_IP
 };

constexpr int BITS_PER_DIM = 32;

template <typename T>
class Index {
protected:
  size_t ntotal;  // number of vectors
  int d;          // vector dimension
  DistanceMetric metric;  // distance metric: l2 or inner product
  T* data_vectors;
  T* d_data_vectors;;
public:
  Index(size_t num, int dim, DistanceMetric m) : ntotal(num), d(dim), metric(m), data_vectors(nullptr) {};
  Index(size_t num, int dim) : Index(num, dim, 0) {};
  Index() : Index(0, 0, 0) {};
  void add(size_t nb, T* xb);
  void search(int nq, const T* xq, int k, T* result_vec, vid_t *result_index, int n);
  //void read_index(const char* filename);
  void write_index(const char* filename);
  void set_dim(int dim) { d = dim; }
  void set_data_ptr(T *xb) { assert(xb != nullptr); data_vectors = xb; }
  void set_d_data_ptr(T *d_xb) { assert(d_xb != nullptr); d_data_vectors = d_xb; }
  size_t get_ntotal() { return ntotal; }
  int get_dim() { return d; }
  T* get_data_ptr() { return data_vectors; }
  T* get_d_data_ptr() { return d_data_vectors; }
  void generate_sign_vectors();
  uint32_t* get_sign_vector(vid_t v);
};

template<typename T>
class IndexIVF : public Index<T> {
 public:
  int nclusters;
  std::vector<std::vector<vid_t>> clusters;
public:
  IndexIVF(size_t num, int dim = 0, DistanceMetric metric = METRIC_L2, int nc = 128) : 
    Index<T>(num, dim, metric), nclusters(nc) { }

  void search(int nq, const T* xq, int k, T* result_vec, vid_t *result_index, int nc = 16);
};

#include "utils.hpp"
#include "distance.hpp"
#define QG_BQUERY 6
#define RABITQ_BITS 64
typedef uint64_t rabitq_t; // code type 
class QGVector {
  private:
  const float* data_ = nullptr;
  std::vector<uint8_t> lut_;
  int padded_dim_ = 0;
  float width_ = 0;
  float lower_val_ = 0;
  float upper_val_ = 0;
  int sumq_ = 0;

 public:
  QGVector(const float* vec, int padded_dim, FHTRotator* rotator)
      : data_(vec)
      , lut_(padded_dim << 2)  // padded_dim / 4 * 16
      , padded_dim_(padded_dim) {
    std::vector<float> rd_vec(padded_dim_);
    rotator->rotate(data_, rd_vec.data());
    std::vector<uint8_t> byte_vec(padded_dim_);
    data_range(rd_vec.data(), padded_dim_, lower_val_, upper_val_);
    width_ = (upper_val_ - lower_val_) / ((1 << QG_BQUERY) - 1);
    float one_over_width = 1.0F / width_;
    for (int i = 0; i < padded_dim; ++i) {
        auto cur = uint8_t(std::lround(((rd_vec[i] - lower_val_) * one_over_width) + 0.5));
        byte_vec[i] = cur;
        sumq_ += cur;
    }
    // init lookup table
    uint8_t* LUT = lut_.data();
    uint8_t* byte_ptr = byte_vec.data();
    int num_codebook = padded_dim >> 2;
    for (int i = 0; i < num_codebook; ++i) {
      LUT[0] = 0;
      for (int j = 1; j < 16; ++j)
        LUT[j] = LUT[j - LOWBIT(j)] + byte_ptr[kPos[j]];
      LUT += 16;
      byte_ptr += 4;
    }
  }
  const float& width() const { return width_; }
  const float& lower_val() const { return lower_val_; }
  const int32_t& sumq() const { return sumq_; }
  const std::vector<uint8_t>& lut() const { return lut_; }
  const float* data() const { return data_; }
};

/**
 * @brief this Factor only for illustration, the true storage is continous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 */
struct Factor {
    float triple_x;   // Sqr of distance to src vertex u + 2 * x * x1 / x0
    float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
    float factor_vq;  // Factor of v_l * ||q_r||
};

#include "pqueue.hpp"
#include "hash_filter.hpp"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "VertexSet.hpp"

using namespace std;

template<typename T>
class IndexGraph : public Index<T> {
 public:
  int maxDeg;
  int BUILD_BEAM_WIDTH;
  int sign_bit_vector_size;
  int packed_vector_dim_size;
  double alpha;
  FHTRotator* rotator;
  std::vector<vid_t> edges;
  std::vector<uint32_t> sign_vectors;  // for pathweaver compression code
  std::vector<rabitq_t> rabitq_vectors;// for rabitq compression code
  std::vector<float> triple_xs;        // for rabitq compression code
  std::vector<float> factor_dq;        // for rabitq compression code
  std::vector<float> factor_vq;        // for rabitq compression code
public:
  IndexGraph(size_t num, int deg = 32, int dim = 0, DistanceMetric m = METRIC_L2, int beam_sz = 128, double alpha_ = 2) : 
      Index<T>(num, dim, m), maxDeg(deg), BUILD_BEAM_WIDTH(beam_sz), 
      sign_bit_vector_size(0), packed_vector_dim_size(0), alpha(alpha_), rotator(nullptr) { }

  VertexSet N(vid_t v) { return VertexSet(&edges[size_t(maxDeg)*size_t(v)], maxDeg, v); }
  void search(int nq, const T* xq, int k, T* result_vec, vid_t *result_index, int beam_sz = 128);
  vid_t* get_adj_list(vid_t v) { return &edges[v*maxDeg]; }

  void constructEdge(int64_t eid, vid_t dst) { edges[eid] = dst; }
  
  void allocateFrom(vid_t nv, int64_t ne) {
    assert(ne % nv == 0); // regular graph
    maxDeg = ne / nv;
    this->ntotal = nv;
    edges.resize(ne);
    std::fill(edges.begin(), edges.end(), 0);
  }

  void add(size_t nb, T* xb) {
    assert(xb);
    if (this->ntotal != 0) assert(this->ntotal == nb);
    else this->ntotal = nb;
    this->data_vectors = xb;
    this->build_vamana_graph(nb, xb);
  }

  void fastprefetch(vid_t v, _mm_hint hint = _MM_HINT_T0) const {
    #ifdef USE_PF_GRAPH
    int l = (max_degree+15)/16;
    vid_t *start = edges+(eidType) max_degree*v;
    for (int i = 0; i < l; i++) {
      __builtin_prefetch(start+16*i);
      //_mm_prefetch(start+16*i, hint);
    }
    #endif
  }
  // store the graph index in a file
  void write_index(const char* gFile) { write_bin<vid_t>(gFile, edges.data(), this->ntotal, maxDeg); }

  // compression code and factors
  int get_sign_bit_vector_size() { return sign_bit_vector_size; }
  int get_packed_vector_dim_size() { return packed_vector_dim_size; }
  uint32_t* get_sign_vectors_ptr() { return sign_vectors.data(); }
  std::vector<uint32_t>& get_sign_vectors() { return sign_vectors; }
  uint32_t* get_sign_vector(vid_t v, int i) { return &sign_vectors[v*sign_bit_vector_size + i*packed_vector_dim_size]; }
  void store_sign_vectors(const char* gFile) { write_bin<uint32_t>(gFile, sign_vectors.data(), this->ntotal, sign_bit_vector_size); }

  // RabitQ compression
  rabitq_t* get_rabitq_vector(vid_t v, int i) { return &rabitq_vectors[v*sign_bit_vector_size + i*packed_vector_dim_size]; } 
  float* get_triple_xs(vid_t v, int i) { return &triple_xs[v*maxDeg + i]; }
  float* get_factor_dq(vid_t v, int i) { return &factor_dq[v*maxDeg + i]; }
  float* get_factor_vq(vid_t v, int i) { return &factor_vq[v*maxDeg + i]; }
  void store_rabitq_vectors(const char* gFile) {
    write_bin<rabitq_t>(gFile, rabitq_vectors.data(), this->ntotal, sign_bit_vector_size);
    std::ofstream output(gFile, std::ios::binary | std::ios::app);
    if (!output) throw std::runtime_error("Cannot open output factor bin file");
    else printf("rabitq vector data stored, sign_bit_vector_size = %d\n", sign_bit_vector_size); 
    auto num = this->ntotal * size_t(maxDeg);
    output.write(reinterpret_cast<const char*>(triple_xs.data()), sizeof(float) * num);
    output.write(reinterpret_cast<const char*>(factor_dq.data()), sizeof(float) * num);
    output.write(reinterpret_cast<const char*>(factor_vq.data()), sizeof(float) * num);
    if (!output) throw std::runtime_error("Error writing to factor bin file");
    else printf("rabitq factor data stored\n");
    rotator->store(output);
    output.close();
  }

  void load_codebook(const char* gFile) {
    auto dim = this->d;
    auto degree = maxDeg;
    assert(dim > 0 && degree > 0);
    size_t nb = 0, width = 0;
    std::ifstream input(gFile, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open codebook file");
    input.read(reinterpret_cast<char*>(&nb), sizeof(int));
    input.read(reinterpret_cast<char*>(&width), sizeof(int));
    assert(nb == this->ntotal && nb > 0);
#ifdef PATHW
    packed_vector_dim_size = (dim + 31) / 32;
    assert(dim * degree % 32 == 0);
    sign_bit_vector_size = dim * degree / 32;
    assert(width == size_t(sign_bit_vector_size) && width > 0);
    sign_vectors.resize(nb * width);
    input.read(reinterpret_cast<char*>(sign_vectors.data()), sizeof(uint32_t) * nb * width);
    if (!input) throw std::runtime_error("Error reading sign_vectors bin data");
    // else printf("sign_vectors loaded\n");
#else
    packed_vector_dim_size = (dim + RABITQ_BITS - 1) / RABITQ_BITS;
    assert(dim * degree % RABITQ_BITS == 0);
    sign_bit_vector_size = dim * degree / RABITQ_BITS;
    assert(width == size_t(sign_bit_vector_size) && width > 0);
    rabitq_vectors.resize(nb * width);
    input.read(reinterpret_cast<char*>(rabitq_vectors.data()), sizeof(rabitq_t) * nb * width);
    if (!input) throw std::runtime_error("Error reading rabitq_vectors bin data");
    // else printf("rabitq_vectors loaded, width = %ld\n", width);

    triple_xs.resize(nb * degree);
    input.read(reinterpret_cast<char*>(triple_xs.data()), sizeof(float) * nb * degree);
    if (!input) throw std::runtime_error("Error reading triple_xs bin data");
    // else printf("triple_xs loaded\n");
    factor_dq.resize(nb * degree);
    input.read(reinterpret_cast<char*>(factor_dq.data()), sizeof(float) * nb * degree);
    if (!input) throw std::runtime_error("Error reading factor_dq bin data");
    // else printf("factor_dq loaded\n");
    factor_vq.resize(nb * degree);
    input.read(reinterpret_cast<char*>(factor_vq.data()), sizeof(float) * nb * degree);
    if (!input) throw std::runtime_error("Error reading factor_vq bin data");
    // else printf("factor_vq loaded\n");
    
    int padded_dim = 0;
    input.read(reinterpret_cast<char*>(&padded_dim), sizeof(int));
    // printf("dim = %d\n", padded_dim);
    if (padded_dim <= 0) throw std::runtime_error("Invalid dimensions read from file");
    rotator = new FHTRotator(padded_dim);
    rotator->load(input);
#endif

    input.close();
  }

  void generate_sign_vectors() {
    std::cout << "Generate sign vectors based on the graph structure\n";
    auto degree = maxDeg;
    auto dim = this->d;
    auto n = this->ntotal;
    auto dataset_ptr = this->data_vectors;
    packed_vector_dim_size = (dim + 31) / 32;
    sign_bit_vector_size = dim * degree / 32; // 32 is the word size (since we use uint32 type for sign-bit vectors)
    std::cout << "dim: " << dim << "\n";
    std::cout << "packed_vector_dim_size: " << packed_vector_dim_size << "\n";
    std::cout << "sign_bit_vector_size: " << sign_bit_vector_size << "\n";
    sign_vectors.resize(n*sign_bit_vector_size, 0);

    #pragma omp parallel for
    for (size_t u = 0; u < n; u++) {
      size_t base = u * sign_bit_vector_size;
      T* vec_u = dataset_ptr + u * dim;
      for (int j = 0; j < degree; j++) {
        size_t base_j = base + j * packed_vector_dim_size;
        auto v = edges[u * degree + j];
        T* vec_v = dataset_ptr + v * dim;
        for (int k = 0; k < dim; k++) {
          if (vec_v[k] > vec_u[k]) {
            sign_vectors[base_j + k / 32] |= (1U << (31 - (k % 32)));
          }
        }
      }
    }
  }
  uint8_t* get_packed_bytecode(vid_t i) {
    return reinterpret_cast<uint8_t*>(rabitq_vectors.data()+i*sign_bit_vector_size);
  }
  void generate_rabitq_vectors() {
    std::cout << "Generate rabitq vectors based on the graph structure\n";
    auto degree = maxDeg;
    auto dim = this->d;
    auto n = this->ntotal;
    auto dataset_ptr = this->data_vectors;
    packed_vector_dim_size = (dim + RABITQ_BITS - 1) / RABITQ_BITS;
    sign_bit_vector_size = dim * degree / RABITQ_BITS;
    std::cout << "dim: " << dim << "\n";
    std::cout << "packed_vector_dim_size: " << packed_vector_dim_size << "\n";
    std::cout << "sign_bit_vector_size: " << sign_bit_vector_size << "\n";
    rabitq_vectors.resize(n*sign_bit_vector_size, 0);
    rotator = new FHTRotator(dim);
    rotator->init();
  
    triple_xs.resize(n*degree);
    factor_dq.resize(n*degree);
    factor_vq.resize(n*degree);
    float fac_norm = 1.f / std::sqrt(float(dim));

    #pragma omp parallel for
    for (size_t u = 0; u < n; u++) {
      // size_t base = u * sign_bit_vector_size;
      // random rotation
      T* vec_u = dataset_ptr + u * dim;
      std::vector<T> rotated_vec_u(dim, 0);
      rotator->rotate(vec_u, &rotated_vec_u[0]);
      std::vector<rabitq_t> sign_bits(degree*packed_vector_dim_size,0);
      for (int j = 0; j < degree; j++) {
        // size_t base_j = base + j * packed_vector_dim_size;
        size_t base_j = j * packed_vector_dim_size;
        auto v = edges[u * degree + j]; // v is a neighbor of u
        int bin_x_sum = 0;
        float sum0 = 0.f, sum1 = 0.f;
        float norm_sq = 0.f;
        T* vec_v = dataset_ptr + v * dim;
        std::vector<T> rotated_vec_v(dim, 0);
        rotator->rotate(vec_v, &rotated_vec_v[0]);
        for (int k = 0; k < dim; k++) { 
          float signed_x = -1;
          auto r = rotated_vec_v[k] - rotated_vec_u[k];
          if (r > 0) { // residual > 0
            sign_bits[base_j + k/RABITQ_BITS] |= (1U << (RABITQ_BITS - 1 - (k % RABITQ_BITS)));
            signed_x = 1;
            bin_x_sum ++;
          }
          sum0 += r * signed_x * fac_norm;
          norm_sq += r * r;
          sum1 += rotated_vec_u[k] * signed_x;
        }

        float x_rotated_norm = std::sqrt(norm_sq);
        float fac_x0 = sum0 / x_rotated_norm;
        float fac_x1 = sum1 * fac_norm;

        double cur_x = x_rotated_norm;  // current dist 2 centroid
        double cur_x0 = fac_x0;         // current <o, o_bar>
        double cur_x1 = fac_x1;         // current <c, o_bar>
        long double x_x0 = static_cast<long double>(cur_x) / cur_x0;
        triple_xs[u*degree+j] = float((cur_x * cur_x) + (2 * x_x0 * cur_x1));
        factor_dq[u*degree+j] = float(-2 * x_x0 * fac_norm);
        factor_vq[u*degree+j] = float(-2 * x_x0 * fac_norm * (bin_x_sum * 2 - int(dim)));
      }
      // compute codes of RaBitQ, store at packed_code
      pack_rabitq_codes(degree, dim, sign_bits.data(), get_packed_bytecode(u));
    }
  }

  inline void build_vamana_graph(size_t nb, const T* xb) {
    int dim = this->d;
    int R = this->maxDeg;
    int L = this->BUILD_BEAM_WIDTH;
    bool use_ip = (this->metric == METRIC_IP);
    cout << "building vamana graph: R = " << R << " L = " 
         << L << " dim = " << dim << " nb = " << nb << "\n";
    auto distance = [&](int i,int j) {
      auto ptri = &xb[size_t(i)*size_t(dim)];
      auto ptrj = &xb[size_t(j)*size_t(dim)];
      if (use_ip) 
        return compute_ip_distance(dim, ptri, ptrj);
      else
        return compute_distance_squared(dim, ptri, ptrj);
    };
    auto euclidean_distance = [&](int i,int j) {
      return compute_distance_squared(dim, &xb[size_t(i)*size_t(dim)], &xb[size_t(j)*size_t(dim)]);
    };
    if (use_ip) cout << "inner product distance" << endl;

    //int L = R * 2; // beam width
    size_t batch_bound = 0.02 * nb;
    size_t start = 1;
    vector<vector<pair<float, vid_t> > > graph(nb);

    auto prune = [&](int u) {
      if (graph[u].size() >= size_t(R)) {
          sort(graph[u].begin(), graph[u].end());
          vector<pair<float,vid_t> > pruned;
          for (double a: {1.0,alpha}) {
              for (auto [d,v]: graph[u]) {
                  if (pruned.size() == size_t(R)) break;
                  bool bad = false;
                  if (use_ip) d = euclidean_distance(u,v);
                  for (auto [d2,w]: pruned) {
                      if (a*euclidean_distance(w,v) <= d) {
                          bad = true;
                          break;
                      }
                  }
                  if (!bad) pruned.emplace_back(d,v);
              }
          }
          graph[u] = pruned;
          graph[u].shrink_to_fit();
      }
    };

    mt19937 gen(1234);
    std::vector<vid_t> order(nb);
    for (size_t i = 0; i < nb; i++) order[i] = i;
    shuffle(order.begin(),order.end(),gen);

    //std::cout << "start beam search to build graph\n";
    while (start < nb) {
      size_t end = min(min(2*start, start+batch_bound), nb);
      //if (start > 100000) std::cout << start << " to " << end << "\n";
      #pragma omp parallel for
      for (size_t i = start; i < end; ++i) {
        auto hash_size = min((size_t) L*L, (size_t)nb);
        hash_filter<vid_t> is_visited(hash_size);
        new_pqueue_t<vid_t> S(L);
        auto dist = distance(order[0],order[i]);
        S.push(order[0], dist);
        is_visited.add(order[0]);
        vector<vid_t> keep;
        while (S.has_unexpanded()) {
          int idx = S.get_next_index();
          int u = S[idx];
          auto dui = S.get_dist(idx);
          graph[order[i]].emplace_back(dui,u);
          S.set_front_expanded();
          keep.clear();
          for (auto [d,v]: graph[u]) {
            if (is_visited.add(v)) {
              keep.push_back(v);
              char *a = (char*)(xb + dim*v);
              int l = dim*sizeof(T)/64;
              for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
            }
          }
          for (auto v: keep) {
            auto dist = distance(v,order[i]);
            if ((S.size() < L) || (dist < S.get_tail_dist())) {
              S.push(v,dist);
            }
          }
        }
        prune(order[i]);
      }
      // add reverse edges
      auto rev_edges = parlay::group_by_key(parlay::flatten(parlay::tabulate(end-start,[&](size_t i) {
        return parlay::map(graph[order[i+start]],[&](auto x) -> pair<vid_t,pair<float,vid_t>> { return make_pair(x.second,make_pair(x.first,order[i+start])); });
      })));
      parlay::parallel_for(0,rev_edges.size(),[&](size_t i) {
        int u = rev_edges[i].first;
        auto &nei = rev_edges[i].second;
        graph[u].insert(graph[u].end(),nei.begin(),nei.end());
        prune(u);
      });
      start = end;
    }
    this->allocateFrom(nb, int64_t(nb) * R);
    #pragma omp parallel for
    for (size_t i = 0; i < nb; ++i) {
      auto offset = (int64_t) i * R;
      //this->fixEndEdge(i, offset + R);
      assert(graph[i].size() <= size_t(R));
      for (size_t j = 0; j < graph[i].size(); ++j) 
        this->constructEdge(offset+j, graph[i][j].second);
      for (auto j = graph[i].size(); j < size_t(R); ++j) 
        this->constructEdge(offset+j, 0);
      
      // sanity check
      sort(graph[i].begin(),graph[i].end());
      for (size_t j = 1; j < graph[i].size(); j++) {
        assert(graph[i][j].second != graph[i][j-1].second);
        assert(size_t(graph[i][j].second) != i);
      }
    }
    //cout << "done generating graph\n";
  }

  void print_access_frequency(const std::vector<int>& access_count, const std::string& fname = "") const {
      std::cout << "\n==== Vector Access Frequency ====\n";
      int total_accesses = 0;
      int accessed = 0;
      int max_freq = 0;
      for (auto c : access_count) {
          total_accesses += c;
          if (c > 0) {
              accessed++;
              if (c > max_freq) max_freq = c;
          }
      }
      std::cout << "Total accesses: " << total_accesses << "\n";
      std::cout << "Unique vectors accessed: " << accessed << " / " << access_count.size() << "\n";
      std::cout << "Max access count: " << max_freq << "\n";

      // Histogram (optional)
      std::vector<int> histogram(max_freq + 1, 0);
      for (auto c : access_count) {
          if (c > 0) histogram[c]++;
      }
      std::cout << "\nAccess frequency histogram (access_count : #vectors):\n";
      std::ostream* out;
      std::ofstream fout;
      if (fname.empty()) {
          out = &std::cout;
      } else {
          fout.open(fname);
          if (!fout.is_open()) {
              std::cerr << "Error: Cannot open file " << fname << " for writing.\n";
              return;
          }
          out = &fout;
      }
      for (int i = 1; i <= max_freq; ++i) {
          if (histogram[i] > 0) {
              *out << i << " : " << histogram[i] << "\n";
          }
      }
      if (fout.is_open()) {
          fout.close();
      }

      // === Top 20% hot vectors ===
      std::vector<int> sorted_access = access_count;
      std::sort(sorted_access.begin(), sorted_access.end(), std::greater<int>());

      size_t topk = access_count.size() / 5;  // top 20%
      int access_top = 0;
      for (size_t i = 0; i < topk; ++i)
          access_top += sorted_access[i];

      float access_ratio = float(access_top) / total_accesses;
      size_t dim_bytes = this->d * sizeof(T);
      size_t storage_bytes = topk * dim_bytes;
      float storage_MB = float(storage_bytes) / (1024 * 1024);
      float storage_GB = float(storage_bytes) / (1024.0 * 1024 * 1024);

      std::cout << "\n==== Top 20% Vector Access Stats ====\n";
      std::cout << "Top 20% vectors (#" << topk << ") account for:\n";
      std::cout << "  - " << (access_ratio * 100.0f) << "% of total accesses\n";
      std::cout << "  - " << storage_MB << " MB (" << storage_GB << " GB) of storage\n";
  }
};

template <typename T>
IndexGraph<T>* read_graph_index(const char* gFile) {
    size_t nb = 0, degree = 0;
    vid_t* data = read_bin<vid_t>(gFile, nb, degree);
    IndexGraph<T> *idx_ptr = new IndexGraph<T>(nb, degree);
    std::cout << "loading graph nv = " << nb << " degree = " << degree << "\n";
    idx_ptr->edges.assign(data, data + nb * degree);
    return idx_ptr;
}
