// Q-LEAF standalone exact-baseline GPU host path.
//
// `IndexGraph<T>::search_gpu_base` runs the formal Q-LEAF kernel with all
// Q-LEAF sidecar / query-prep buffers set to nullptr. This produces an
// exact GPU beam-search baseline whenever the binary was compiled with an
// exact-only compile-time configuration:
//
//   -DROUTEA_GATE_MODE=0 -DNAV_MODE=0 -DQLEAF_ENABLE_ADAPTIVE=0
//
// In that configuration `kRouteANeedsEdgeSidecar` and
// `kRouteANeedsQueryPrep` are both false, so the kernel body skips every
// sidecar branch and the launch is a clean exact L2 beam search. If the
// binary was compiled with adaptive/RaBitQ enabled, this entry refuses to
// run with a clear error message — the caller should switch to
// `gpu_rabitq_bitplane_navl2_prune_es_spec_auto` mode (`search_gpu_rabitq_bitplane_navl2_prune_es_spec_auto`) instead.
//
// All Q-LEAF host-side sidecar plumbing (RaBitQ codebook upload, q6
// bitplane query prep, manifest printing) lives in beam_gpu_rabitq_bitplane_navl2_prune_es_spec_auto.cu. By
// design, this file owns no sidecar code paths and uses raw CUDA APIs
// directly so the exact baseline can be reviewed end-to-end in one file.

#include "index.hpp"
#include "search_rabitq_bitplane_navl2_prune_es_spec_auto.cuh"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

inline bool consume_runtime_log_once_base() {
  static bool logged = false;
  if (logged) return false;
  logged = true;
  return true;
}

}  // anonymous namespace

template <typename T>
void IndexGraph<T>::search_gpu_base(int nq, const T* queries, int K,
                                    vid_t* result_idx, int beam_sz) {
  // Compile-time gate: this entry is only meaningful when no Q-LEAF
  // sidecar branch is active in the kernel. If the binary was built with
  // adaptive / RaBitQ enabled the user should use gpu_rabitq_bitplane_navl2_prune_es_spec_auto instead.
  if (kRouteANeedsEdgeSidecar || kRouteAUseEdgeRabitqNav ||
      (ROUTEA_ADAPTIVE_GATE != 0)) {
    std::cerr
        << "search_gpu_base requires an exact-only compile-time configuration "
        << "(ROUTEA_GATE_MODE=0, NAV_MODE=0, QLEAF_ENABLE_ADAPTIVE=0). "
        << "Either recompile with those macros or switch to gpu_rabitq_bitplane_navl2_prune_es_spec_auto mode.\n";
    std::exit(2);
  }

  const int dim = this->d;
  const size_t npoints = this->ntotal;
  const int max_degree = this->maxDeg;
  assert(beam_sz > 0 && beam_sz <= BEAM_SIZE);
  assert(K > 0 && K <= beam_sz);

  const bool log_once = consume_runtime_log_once_base();
  if (log_once) {
    std::cout << "GPU exact baseline (search_gpu_base): K=" << K
              << ", nq=" << nq << ", dim=" << dim
              << ", npoints=" << npoints
              << ", beam_size=" << beam_sz << "\n";
  }

  // Device buffers (locally owned for the exact baseline).
  T* d_queries = nullptr;
  T* d_data = nullptr;
  vid_t* d_edges = nullptr;
  vid_t* d_results = nullptr;
  vid_t* d_starting_points = nullptr;
  gpu_long_t* d_total_count_dc = nullptr;
  gpu_long_t* d_total_hops = nullptr;

  const size_t queries_bytes = static_cast<size_t>(nq) * dim * sizeof(T);
  const size_t data_bytes = npoints * static_cast<size_t>(dim) * sizeof(T);
  const size_t edges_bytes =
      npoints * static_cast<size_t>(max_degree) * sizeof(vid_t);
  const size_t results_bytes =
      static_cast<size_t>(nq) * static_cast<size_t>(K) * sizeof(vid_t);
  const int starting_count = std::max(beam_sz, kCandidateBufferSize);
  const size_t starting_bytes =
      static_cast<size_t>(starting_count) * sizeof(vid_t);

  CUDA_SAFE_CALL(cudaMalloc(&d_queries, queries_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, queries_bytes,
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc(&d_data, data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, this->get_data_ptr(), data_bytes,
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc(&d_edges, edges_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_edges, this->edges.data(), edges_bytes,
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc(&d_results, results_bytes));

  std::vector<vid_t> starting_points(static_cast<size_t>(starting_count),
                                     MAX_INDEX);
  starting_points[0] = static_cast<vid_t>(0);
  CUDA_SAFE_CALL(cudaMalloc(&d_starting_points, starting_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points.data(),
                            starting_bytes, cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc(&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMalloc(&d_total_hops, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemset(d_total_count_dc, 0, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemset(d_total_hops, 0, sizeof(gpu_long_t)));

  GraphGPU<vid_t> gg(static_cast<vid_t>(npoints),
                     static_cast<vid_t>(max_degree), d_edges);

  const size_t num_threads = static_cast<size_t>(BLOCK_SIZE);
  const size_t num_blocks = static_cast<size_t>(nq);

  cudaEvent_t ev0, ev1;
  CUDA_SAFE_CALL(cudaEventCreate(&ev0));
  CUDA_SAFE_CALL(cudaEventCreate(&ev1));
  CUDA_SAFE_CALL(cudaEventRecord(ev0));
  BeamSearch<T><<<num_blocks, num_threads, 0>>>(
      K, beam_sz, nq, dim, npoints,
      d_queries, d_data, d_results,
      d_total_count_dc, d_total_hops,
      /* d_routea_profile_counters */ nullptr,
      d_starting_points,
      /* d_sign_bit          */ nullptr,
      /* sign_words_per_edge */ 0,
      /* d_rabitq_triple     */ nullptr,
      /* d_rabitq_factor_dq  */ nullptr,
      /* d_rabitq_factor_vq  */ nullptr,
      /* d_routea_query_bp   */ nullptr,
      /* score_words         */ 0,
      /* score_dims          */ 0,
      /* d_routea_query_lower */ nullptr,
      /* d_routea_query_width */ nullptr,
      /* d_routea_query_sumq  */ nullptr,
      gg);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaEventRecord(ev1));
  CUDA_SAFE_CALL(cudaEventSynchronize(ev1));

  float kernel_ms = 0.0f;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
  CUDA_SAFE_CALL(cudaEventDestroy(ev0));
  CUDA_SAFE_CALL(cudaEventDestroy(ev1));
  if (log_once) {
    printf("GPU kernel time (BeamSearch + sync, exact baseline): %.3f ms\n",
           kernel_ms);
  }

  // Copy results + counters back to host.
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, results_bytes,
                            cudaMemcpyDeviceToHost));
  gpu_long_t total_count_dc = 0;
  gpu_long_t total_hops = 0;
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, d_total_count_dc,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_hops, d_total_hops,
                            sizeof(gpu_long_t), cudaMemcpyDeviceToHost));

  if (log_once) {
    std::cout << "total # distance computation: " << total_count_dc << "\n";
    std::cout << "average # distance computation: "
              << (total_count_dc / nq) << "\n";
    std::cout << "total # hops: " << total_hops << "\n";
    std::cout << "average # hops: " << (total_hops / nq) << "\n";
  }

  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_edges));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_starting_points));
  CUDA_SAFE_CALL(cudaFree(d_total_count_dc));
  CUDA_SAFE_CALL(cudaFree(d_total_hops));
}

template void IndexGraph<float>::search_gpu_base(int, const float*, int, vid_t*, int);
