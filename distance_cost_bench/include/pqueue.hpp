#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <cfloat>
#include <cassert>
#include <limits>
#include <utility>

template <typename T>
class pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  int next_idx;
  std::vector<T> vid_queue;
  std::vector<float> distances;
  std::vector<uint8_t> expanded;
  std::vector<T> vid_queue2;
  std::vector<float> distances2;
  std::vector<uint8_t> expanded2;

public:
  pqueue_t(int L) {
    next_idx = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(2*L);
    distances.resize(2*L);
    expanded.resize(2*L);
    vid_queue2.resize(2*L);
    distances2.resize(2*L);
    expanded2.resize(2*L);
    std::fill(expanded.begin(), expanded.end(), 0);
  }
  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : distances[queue_size-1]; }
  float get_dist(int idx) const { return distances[idx]; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }

  void set_expanded(int idx) { expanded[idx] = 1; }
  void set_unexpanded(int idx) { expanded[idx] = 0; }

  void inc_next_index() { next_idx++; }

  bool split_queue(std::vector<pqueue_t> &lqs) {
    if (queue_size == 0) return false;
    int num_tasks = 0;
    size_t j = 0;
    for (int i = next_idx; i < queue_size; i++) {
      if (expanded[i]) continue;
      //auto loc = lqs[j].push(vid_queue[i], distances[i]);
      auto loc = lqs[j].insert(vid_queue[i], distances[i]);
      if (loc >= 0) {
        set_expanded(i);
        num_tasks ++;
      } else break;
      j ++; // round robin assignment
      if (j == lqs.size()) j = 0;
    }
    if (num_tasks == 0) return false;
    return true;
  }

  int merge_queues(std::vector<pqueue_t> &lqs) {
    // linear insertion
    // TODO: do 'merge' instead to improve performance
    for (size_t i = 0; i < lqs.size(); i++) {
      for (int j = 0; j < lqs[i].size(); j++) {
        auto loc = push(lqs[i][j], lqs[i].get_dist(j));
        if (loc == -2) { // Duplicate
          if (!lqs[i].is_expanded(j) && is_expanded(loc)) {
            set_unexpanded(loc);
            if (loc < next_idx) next_idx = loc;
          }
        } else if (loc >= 0) {
          if (lqs[i].is_expanded(j)) {
            set_expanded(loc);
            if (loc < next_idx) next_idx++;
          } else if (loc < next_idx) next_idx = loc;
        }
      }
    }
    return next_idx;
  }

  int insert(T vid, float dist) {
    if (queue_size == queue_capacity) return -1;
    vid_queue[queue_size] = vid;
    distances[queue_size] = dist;
    expanded[queue_size] = 0;
    return queue_size ++;
  }
  int push(std::pair<float, T> element) {
    auto vid = element.second;
    auto dist = element.first;
    return push(vid, dist);
  }
  int push(T vid, float dist) {
    //printf("pushing %d with dist %f into the queue\n", vid, dist);
    if (0 == queue_size) {
      vid_queue[0] = vid;
      queue_size = 1;
      distances[0] = dist;
      expanded[0] = 0; // not expanded yet
      return 0;
    }
    // Find the insert location
    auto queue_start = &distances[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc != queue_size) {
      if (vid == vid_queue[insert_loc]) { // Duplicate, skip
        return -2;
      }
      if (queue_size >= queue_capacity) { // Queue is full
        --queue_size;
      }
    } else { // insert_loc == queue_size, insert at the end?
      if (queue_size < queue_capacity) { // Queue is not full
        // Insert at the end
        vid_queue[insert_loc] = vid;
        distances[insert_loc] = dist;
        expanded[insert_loc] = 0; // not expanded yet
        ++queue_size;
        return insert_loc;
      } else { // Queue is full, skip
        return -1;
      }
    }
    // Add into queue
    auto num = queue_size - insert_loc;
    memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
            reinterpret_cast<char *>(&vid_queue[insert_loc]),
            num * sizeof(T));
    vid_queue[insert_loc] = vid;
    memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
            reinterpret_cast<char *>(&distances[insert_loc]),
            num * sizeof(float));
    distances[insert_loc] = dist;
    memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
            reinterpret_cast<char *>(&expanded[insert_loc]),
            num * sizeof(uint8_t));
    expanded[insert_loc] = 0; // not expanded yet
    ++queue_size;
    if (insert_loc < next_idx) next_idx = insert_loc;
    return insert_loc;
  }

  void batch_push(const std::vector<std::pair<float,T> > &ins) {
    if (ins.empty()) return;
    if (ins.size() == 1) {
      push(ins[0]);
      return;
    }

    int i = 0,j = 0,k = 0;
    while ((i < queue_size) && (j < ins.size())) {
      if (vid_queue[i] == ins[j].second) j++;
      else if (distances[i] < ins[j].first) {
        vid_queue2[k] = vid_queue[i];
        distances2[k] = distances[i];
        expanded2[k] = expanded[i];
        i++,k++;
      }
      else {
        vid_queue2[k] = ins[j].second;
        distances2[k] = ins[j].first;
        expanded2[k] = 0;
        j++,k++;
      }
    }
    while (i < queue_size) {
      vid_queue2[k] = vid_queue[i];
      distances2[k] = distances[i];
      expanded2[k] = expanded[i];
      i++,k++;
    }
    while (j < ins.size()) {
      vid_queue2[k] = ins[j].second;
      distances2[k] = ins[j].first;
      expanded2[k] = 0;
      j++,k++;
    }

    queue_size = std::min(k,queue_capacity);
    /*
    memmove(reinterpret_cast<char *>(&vid_queue[0]),
            reinterpret_cast<char *>(&vid_queue2[0]),
            queue_size * sizeof(T));
    memmove(reinterpret_cast<char *>(&distances[0]),
            reinterpret_cast<char *>(&distances2[0]),
            queue_size * sizeof(float));
    memmove(reinterpret_cast<char *>(&expanded[0]),
            reinterpret_cast<char *>(&expanded2[0]),
            queue_size * sizeof(uint8_t));
    */
    swap(vid_queue,vid_queue2);
    swap(distances,distances2);
    swap(expanded,expanded2);

    next_idx = 0;
  }

  T* fetch_unexpanded_nodes(int P) {
    T* nodes = new T[P];
    int count = 0;
    for (int i = 0; i < queue_size; i++) {
      if (expanded[i]) nodes[count++] = i;
      if (count == P) break;
    }
    return nodes;
  }
};

template <typename T>
class new_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  // maintain invariant that next_idx and next_idx2 are always the two smallest
  // unexpanded positions. If they don't exist, then they equal queue_size.
  int next_idx, next_idx2;
  void _check_invariant() {
    assert(queue_size <= queue_capacity);
    assert(next_idx <= queue_size);
    assert((next_idx == queue_size) || !expanded[next_idx]);
    assert(next_idx2 <= queue_size);
    assert((next_idx2 == queue_size) || !expanded[next_idx2]);
    assert(next_idx <= next_idx2);
    assert((next_idx == queue_size) || (next_idx < next_idx2));
  }
  std::vector<T> vid_queue;
  std::vector<float> distances;
  std::vector<uint8_t> expanded;

  static constexpr T gpu_invalid_index() {
    return std::numeric_limits<T>::max();
  }

  static constexpr T gpu_expanded_mask() {
    return static_cast<T>(0x80000000u);
  }

  static constexpr T gpu_node_mask() {
    return static_cast<T>(~gpu_expanded_mask());
  }

  static void gpu_swap_if_needed(float& k0, T& v0, float& k1, T& v1, bool asc) {
    if ((k0 != k1) && ((k0 < k1) != asc)) {
      std::swap(k0, k1);
      std::swap(v0, v1);
    }
  }

  template <size_t N>
  static void gpu_warp_merge(
      std::array<std::array<float, N>, 32>& keys,
      std::array<std::array<T, N>, 32>& vals,
      unsigned range) {
    if constexpr (N == 2) {
      if (range == 1) {
        for (unsigned lane = 0; lane < 32; ++lane) {
          const bool p = ((lane & 1) == 0);
          gpu_swap_if_needed(keys[lane][0], vals[lane][0],
                             keys[lane][1], vals[lane][1], p);
        }
        return;
      }

      const unsigned b = range;
      for (unsigned c = b / 2; c >= 1; c >>= 1) {
        auto old_keys = keys;
        auto old_vals = vals;
        for (unsigned lane = 0; lane < 32; ++lane) {
          const bool p = static_cast<bool>(lane & b) == static_cast<bool>(lane & c);
          const unsigned other = lane ^ c;
          for (size_t i = 0; i < N; ++i) {
            keys[lane][i] = old_keys[lane][i];
            vals[lane][i] = old_vals[lane][i];
            const float other_key = old_keys[other][i];
            const T other_val = old_vals[other][i];
            if ((keys[lane][i] != other_key) && ((keys[lane][i] < other_key) != p)) {
              keys[lane][i] = other_key;
              vals[lane][i] = other_val;
            }
          }
        }
        if (c == 1) break;
      }
      for (unsigned lane = 0; lane < 32; ++lane) {
        const bool p = ((lane & b) == 0);
        gpu_swap_if_needed(keys[lane][0], vals[lane][0],
                           keys[lane][1], vals[lane][1], p);
      }
    } else if constexpr (N == 4) {
      if (range == 1) {
        for (unsigned b = 2; b <= N; b <<= 1) {
          for (unsigned c = b / 2; c >= 1; c >>= 1) {
            for (unsigned lane = 0; lane < 32; ++lane) {
              for (size_t i = 0; i < N; ++i) {
                const size_t j = i ^ c;
                if (i >= j) continue;
                const unsigned line_id = static_cast<unsigned>(i) + (N * lane);
                const bool p = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                gpu_swap_if_needed(keys[lane][i], vals[lane][i],
                                   keys[lane][j], vals[lane][j], p);
              }
            }
            if (c == 1) break;
          }
        }
        return;
      }

      const unsigned b = range;
      for (unsigned c = b / 2; c >= 1; c >>= 1) {
        auto old_keys = keys;
        auto old_vals = vals;
        for (unsigned lane = 0; lane < 32; ++lane) {
          const bool p = static_cast<bool>(lane & b) == static_cast<bool>(lane & c);
          const unsigned other = lane ^ c;
          for (size_t i = 0; i < N; ++i) {
            keys[lane][i] = old_keys[lane][i];
            vals[lane][i] = old_vals[lane][i];
            const float other_key = old_keys[other][i];
            const T other_val = old_vals[other][i];
            if ((keys[lane][i] != other_key) && ((keys[lane][i] < other_key) != p)) {
              keys[lane][i] = other_key;
              vals[lane][i] = other_val;
            }
          }
        }
        if (c == 1) break;
      }
      for (unsigned c = N / 2; c >= 1; c >>= 1) {
        for (unsigned lane = 0; lane < 32; ++lane) {
          const bool p = ((lane & b) == 0);
          for (size_t i = 0; i < N; ++i) {
            const size_t j = i ^ c;
            if (i >= j) continue;
            gpu_swap_if_needed(keys[lane][i], vals[lane][i],
                               keys[lane][j], vals[lane][j], p);
          }
        }
        if (c == 1) break;
      }
    }
  }

  template <size_t N>
  static void gpu_warp_sort(
      std::array<std::array<float, N>, 32>& keys,
      std::array<std::array<T, N>, 32>& vals) {
    for (unsigned range = 1; range <= 32; range <<= 1) {
      gpu_warp_merge<N>(keys, vals, range);
    }
  }

  void recompute_next_indices() {
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int i = 0; i < queue_size; i++) {
      if (!expanded[i]) {
        if (next_idx == queue_size) next_idx = i;
        else { next_idx2 = i; break; }
      }
    }
  }

  void merge_candidates_gpu_like_bitonic128(std::vector<std::pair<T, float>>& candidates) {
    constexpr int kTopk = 128;
    constexpr int kCandidateBufferSize = 64;
    const bool first = (queue_size == 0);

    std::array<float, kTopk> topk_dist;
    std::array<T, kTopk> topk_idx;
    for (int i = 0; i < kTopk; ++i) {
      if (i < queue_size) {
        topk_dist[i] = distances[i];
        topk_idx[i] = expanded[i] ? (vid_queue[i] | gpu_expanded_mask()) : vid_queue[i];
      } else {
        topk_dist[i] = FLT_MAX;
        topk_idx[i] = gpu_invalid_index();
      }
    }

    std::array<float, kCandidateBufferSize> candidate_dist;
    std::array<T, kCandidateBufferSize> candidate_idx;
    candidate_dist.fill(FLT_MAX);
    candidate_idx.fill(gpu_invalid_index());
    const size_t n_candidates = std::min(candidates.size(), candidate_dist.size());
    for (size_t i = 0; i < n_candidates; ++i) {
      candidate_idx[i] = candidates[i].first;
      candidate_dist[i] = candidates[i].second;
    }

    std::array<std::array<float, 2>, 32> key_1;
    std::array<std::array<T, 2>, 32> val_1;
    for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned i = 0; i < 2; ++i) {
        const unsigned j = lane + (32 * i);
        key_1[lane][i] = candidate_dist[j];
        val_1[lane][i] = candidate_idx[j];
      }
    }
    gpu_warp_sort<2>(key_1, val_1);
    for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned i = 0; i < 2; ++i) {
        const unsigned j = (2 * lane) + i;
        candidate_dist[j] = key_1[lane][i];
        candidate_idx[j] = val_1[lane][i];
      }
    }

    std::array<std::array<float, 4>, 32> key_2;
    std::array<std::array<T, 4>, 32> val_2;
    for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned j = first ? (lane + (32 * i)) : ((4 * lane) + i);
        key_2[lane][i] = topk_dist[j];
        val_2[lane][i] = topk_idx[j];
      }
    }
    if (first) {
      gpu_warp_sort<4>(key_2, val_2);
    }

    for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned j = (4 * lane) + i;
        const unsigned k = kTopk - 1 - j;
        if (k >= kCandidateBufferSize) continue;
        if (key_2[lane][i] > candidate_dist[k]) {
          key_2[lane][i] = candidate_dist[k];
          val_2[lane][i] = candidate_idx[k];
        }
      }
    }
    gpu_warp_merge<4>(key_2, val_2, 32);

    for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned j = (4 * lane) + i;
        topk_dist[j] = key_2[lane][i];
        topk_idx[j] = val_2[lane][i];
      }
    }

    queue_size = 0;
    for (int i = 0; i < kTopk; ++i) {
      if (topk_idx[i] == gpu_invalid_index()) continue;
      vid_queue[queue_size] = topk_idx[i] & gpu_node_mask();
      distances[queue_size] = topk_dist[i];
      expanded[queue_size] = (topk_idx[i] & gpu_expanded_mask()) ? 1 : 0;
      queue_size++;
    }
    recompute_next_indices();
  }

public:
  new_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(L+1);
    distances.resize(L+1);
    expanded.resize(L+1);
    std::fill(expanded.begin(), expanded.end(), 0);
  }

  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  int get_next_index2() const { return next_idx2; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : distances[queue_size-1]; }
  float get_dist(int idx) const { return distances[idx]; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }

  bool has_unexpanded() { return next_idx < queue_size; }
  bool has_second_unexpanded() { return next_idx2 < queue_size; }
  void set_front_expanded() {
    expanded[next_idx] = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && expanded[next_idx2]) next_idx2++;
    //_check_invariant();
  }
  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > distances[queue_size-1]))
      return; // queue is full
    auto queue_start = &distances[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc < queue_size) {
      // if (vid_queue[insert_loc] == vid) return; // duplicate
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
              reinterpret_cast<char *>(&distances[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    distances[insert_loc] = dist;
    expanded[insert_loc] = 0;
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size:next_idx+1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
    //_check_invariant();
  }

  void merge_candidates_gpu_like(std::vector<std::pair<T, float>>& candidates) {
    if (candidates.empty()) return;
    if (queue_capacity == 128) {
      merge_candidates_gpu_like_bitonic128(candidates);
      return;
    }

    struct Entry {
      T vid;
      float dist;
      uint8_t is_expanded;
    };

    std::vector<Entry> current;
    current.reserve(queue_size);
    for (int i = 0; i < queue_size; i++) {
      current.push_back({vid_queue[i], distances[i], expanded[i]});
    }

    std::vector<std::pair<T, float>> valid_candidates;
    valid_candidates.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      if (candidate.first == gpu_invalid_index() || candidate.second == FLT_MAX) {
        continue;
      }
      valid_candidates.push_back(candidate);
    }
    if (valid_candidates.empty()) {
      return;
    }

    std::stable_sort(valid_candidates.begin(), valid_candidates.end(),
        [](const std::pair<T, float>& a, const std::pair<T, float>& b) {
          return a.second < b.second;
        });

    int old_i = 0;
    size_t cand_i = 0;
    int out_i = 0;
    while (out_i < queue_capacity &&
           (old_i < static_cast<int>(current.size()) || cand_i < valid_candidates.size())) {
      const bool take_old =
          cand_i >= valid_candidates.size() ||
          (old_i < static_cast<int>(current.size()) &&
           current[old_i].dist <= valid_candidates[cand_i].second);

      if (take_old) {
        vid_queue[out_i] = current[old_i].vid;
        distances[out_i] = current[old_i].dist;
        expanded[out_i] = current[old_i].is_expanded;
        old_i++;
      } else {
        vid_queue[out_i] = valid_candidates[cand_i].first;
        distances[out_i] = valid_candidates[cand_i].second;
        expanded[out_i] = 0;
        cand_i++;
      }
      out_i++;
    }

    queue_size = out_i;
    recompute_next_indices();
  }

  void push_update(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > distances[queue_size-1]))
      return;

    int existing_loc = -1;    // check for existing entry's location
    for (int i = 0; i < queue_size; i++) {
      if (vid_queue[i] == vid) { existing_loc = i; break; }
    }
    if (existing_loc < 0) return;

    if (distances[existing_loc] <= dist) return; // existing entry is already closer

    // find where the new entry should go
    auto queue_start = &distances[0];
    int insert_loc = std::lower_bound(queue_start, queue_start + existing_loc, dist) - queue_start;

    auto num = existing_loc - insert_loc;
    if (num > 0) {
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
              reinterpret_cast<char *>(&distances[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    distances[insert_loc] = dist;
    expanded[insert_loc] = 0;

    // Recompute next_idx/next_idx2 since the shift may have moved them
    recompute_next_indices();
    //_check_invariant();
  }
};

template <typename T>
struct pqueue_entry {
  T vid;
  float dist;
  uint8_t expanded;
};

// alternative: array of struct instead of struct of array
template <typename T>
class struct_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  int next_idx, next_idx2;

  void _check_invariant() {
    assert(queue_size <= queue_capacity);
    assert(next_idx <= queue_size);
    assert((next_idx == queue_size) || !entries[next_idx].expanded);
    assert(next_idx2 <= queue_size);
    assert((next_idx2 == queue_size) || !entries[next_idx2].expanded);
    assert(next_idx <= next_idx2);
    assert((next_idx == queue_size) || (next_idx < next_idx2));
  }

  std::vector<pqueue_entry<T>> entries;

public:
  struct_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    entries.resize(L + 1);
    for (auto& e : entries) e.expanded = 0;
  }

  T& operator[](size_t i) { return entries[i].vid; }
  const T& operator[](size_t i) const { return entries[i].vid; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  int get_next_index2() const { return next_idx2; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : entries[queue_size - 1].dist; }
  float get_dist(int idx) const { return entries[idx].dist; }
  bool is_expanded(int idx) const { return entries[idx].expanded == 1; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }

  bool has_unexpanded() { return next_idx < queue_size; }
  bool has_second_unexpanded() { return next_idx2 < queue_size; }

  void set_front_expanded() {
    entries[next_idx].expanded = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && entries[next_idx2].expanded) next_idx2++;
    //_check_invariant();
  }

  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > entries[queue_size - 1].dist))
      return;

    // binary search on dist field
    int lo = 0, hi = queue_size;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (entries[mid].dist < dist) lo = mid + 1;
      else hi = mid;
    }
    int insert_loc = lo;

    if (insert_loc < queue_size) {
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char*>(&entries[insert_loc + 1]),
              reinterpret_cast<char*>(&entries[insert_loc]),
              num * sizeof(pqueue_entry<T>));
    }
    entries[insert_loc] = {vid, dist, 0};
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size : next_idx + 1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
    //_check_invariant();
  }

  void push_update(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > entries[queue_size - 1].dist))
      return;

    int existing_loc = -1;
    for (int i = 0; i < queue_size; i++) {
      if (entries[i].vid == vid) { existing_loc = i; break; }
    }
    if (existing_loc < 0) return;

    if (entries[existing_loc].dist <= dist) return;

    // binary search up to existing_loc
    int lo = 0, hi = existing_loc;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (entries[mid].dist < dist) lo = mid + 1;
      else hi = mid;
    }
    int insert_loc = lo;

    auto num = existing_loc - insert_loc;
    if (num > 0) {
      memmove(reinterpret_cast<char*>(&entries[insert_loc + 1]),
              reinterpret_cast<char*>(&entries[insert_loc]),
              num * sizeof(pqueue_entry<T>));
    }
    entries[insert_loc] = {vid, dist, 0};

    // Recompute next_idx/next_idx2 since the shift may have moved them
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int k = 0; k < queue_size; k++) {
      if (!entries[k].expanded) {
        if (next_idx == queue_size) next_idx = k;
        else { next_idx2 = k; break; }
      }
    }
    //_check_invariant();
  }
};

template <typename T>
class both_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  // maintain invariant that next_idx and next_idx2 are always the two smallest
  // unexpanded positions. If they don't exist, then they equal queue_size.
  int next_idx, next_idx2;
  void _check_invariant() {
    assert(queue_size <= queue_capacity);
    assert(next_idx <= queue_size);
    assert((next_idx == queue_size) || !expanded[next_idx]);
    assert(next_idx2 <= queue_size);
    assert((next_idx2 == queue_size) || !expanded[next_idx2]);
    assert(next_idx <= next_idx2);
    assert((next_idx == queue_size) || (next_idx < next_idx2));
  }
  std::vector<T> vid_queue;
  std::vector<float> rabitq_dist;
  std::vector<float> l2_dist;
  std::vector<uint8_t> expanded;

public:
  both_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(L+1);
    rabitq_dist.resize(L+1);
    l2_dist.resize(L+1);
    expanded.resize(L+1);
    std::fill(l2_dist.begin(), l2_dist.end(), -1.0f);
    std::fill(expanded.begin(), expanded.end(), 0);
  }

  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  int get_next_index2() const { return next_idx2; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : rabitq_dist[queue_size-1]; }
  float get_dist(int idx) const { return rabitq_dist[idx]; }
  float get_l2_dist(int idx) const { return l2_dist[idx]; }
  void set_l2_dist(int idx, float dist) { l2_dist[idx] = dist; }
  bool has_l2_dist(int idx) const { return l2_dist[idx] >= 0.0f; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; next_idx2 = 0; }

  bool has_unexpanded() { return next_idx < queue_size; }
  bool has_second_unexpanded() { return next_idx2 < queue_size; }
  void recompute_front() {
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int i = 0; i < queue_size; i++) {
      if (!expanded[i]) {
        if (next_idx == queue_size) next_idx = i;
        else {
          next_idx2 = i;
          break;
        }
      }
    }
  }
  void set_front_expanded() {
    expanded[next_idx] = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && expanded[next_idx2]) next_idx2++;
    //_check_invariant();
  }
  void set_expanded(int idx) {
    if ((idx < 0) || (idx >= queue_size) || expanded[idx]) return;
    expanded[idx] = 1;
    recompute_front();
  }
  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > rabitq_dist[queue_size-1]))
      return; // queue is full
    auto queue_start = &rabitq_dist[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc < queue_size) {
      // if (vid_queue[insert_loc] == vid) return; // duplicate
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&rabitq_dist[insert_loc + 1]),
              reinterpret_cast<char *>(&rabitq_dist[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&l2_dist[insert_loc + 1]),
              reinterpret_cast<char *>(&l2_dist[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    rabitq_dist[insert_loc] = dist;
    l2_dist[insert_loc] = -1;
    expanded[insert_loc] = 0;
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size:next_idx+1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
    //_check_invariant();
  }
};
