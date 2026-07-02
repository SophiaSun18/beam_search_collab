#pragma once

#include <cfloat>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

constexpr int RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE = 64;

template <typename VidT, typename ExpandedFilter, typename EstimateFn>
size_t rabitq_gpu_like_prepare_candidates(
    const VidT* neighbors,
    size_t degree,
    const ExpandedFilter& expanded,
    std::vector<std::pair<VidT, float>>& candidates,
    EstimateFn estimate_fn) {
  constexpr VidT invalid = std::numeric_limits<VidT>::max();
  candidates.assign(RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE, {invalid, FLT_MAX});

  size_t valid = 0;
  const size_t limit =
      degree < RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE
          ? degree
          : RABITQ_GPU_LIKE_CANDIDATE_BUFFER_SIZE;
  for (size_t i = 0; i < limit; ++i) {
    const VidT child = neighbors[i];
    if (child == invalid || expanded.contains(child)) {
      continue;
    }
    float dist = estimate_fn(child, i);
    if (dist < 0.0f) {
      dist = 0.0f;
    }
    candidates[i] = {child, dist};
    ++valid;
  }
  return valid;
}
