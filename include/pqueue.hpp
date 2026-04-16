#pragma once

#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <cfloat>
#include <cassert>

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
    next_idx = queue_size;
    next_idx2 = queue_size;
    for (int k = 0; k < queue_size; k++) {
      if (!expanded[k]) {
        if (next_idx == queue_size) next_idx = k;
        else { next_idx2 = k; break; }
      }
    }
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
