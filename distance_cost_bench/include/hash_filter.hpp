#pragma once

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <stdint.h>
#include <string.h>
#include <cfloat>
#include <limits>

template <typename T>
class hash_filter {
private:
  int bits;
  int capacity;
  std::vector<T> seen;
  // std::hash<T> hasher;
  std::unordered_set<T> stl_hash_;
/*
parlay uses:
inline uint64_t hash64_2(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}
*/
  size_t total_inserts = 0;
  size_t total_collisions = 0;
  static constexpr T kEmptyValue = std::numeric_limits<T>::max();

public:
  hash_filter(int L) {
    int bit_size = 0;
    int tmp = L;
    while (tmp != 0) { bit_size++; tmp >>= 1; }
    bits = (bit_size + 4) / 2 + 3;
    capacity = 1 << bits;
    seen.resize(capacity);
    std::fill(seen.begin(), seen.end(), kEmptyValue);
  }
  int get_hash(T thing) {
    //return hasher(thing) & ((1 << bits)-1);
    return thing & ((1 << bits)-1);
  }

  bool contains(T thing) {
    T val = seen[get_hash(thing)];
    if (val == thing) return true;
    if (val == kEmptyValue) return false;
    return stl_hash_.find(thing) != stl_hash_.end();
  }
  bool add(T thing) {
    T& val = seen[get_hash(thing)];
    if (val == thing) return false;
    total_inserts++;
    if (val != kEmptyValue) total_collisions++;  // slot occupied by different ID, spill to overflow
    else { val = thing; return true; }
    stl_hash_.insert(thing);
    return true;
  }

  size_t get_overflow_size() const { return stl_hash_.size(); }
  size_t get_total_inserts() const { return total_inserts; }
  size_t get_total_collisions() const { return total_collisions; }
  double get_collision_ratio() const { return total_inserts > 0 ? (double)total_collisions / total_inserts : 0.0; }

  void clear() {
    std::fill(seen.begin(), seen.end(), kEmptyValue);
    stl_hash_.clear();
  }
};

inline constexpr int gpu_hash_bitlen_for_topk(int topk) {
  constexpr int kGpuHashBaseBitlen = 11;
  return (topk * 4 > 3 * (1 << kGpuHashBaseBitlen))
      ? (kGpuHashBaseBitlen + 1)
      : kGpuHashBaseBitlen;
}

template <typename T>
class gpu_like_hash_filter {
private:
  int bits_;
  size_t capacity_;
  std::vector<T> seen_;
  static constexpr T kEmptyValue = std::numeric_limits<T>::max();

  size_t get_hash(T key) const {
    const size_t mask = capacity_ - 1;
    return (static_cast<size_t>(key) ^ (static_cast<size_t>(key) >> bits_)) & mask;
  }

public:
  explicit gpu_like_hash_filter(int bitlen)
      : bits_(bitlen), capacity_(static_cast<size_t>(1) << bitlen),
        seen_(capacity_, kEmptyValue) {}

  int bitlen() const { return bits_; }
  size_t capacity() const { return capacity_; }

  bool contains(T key) const {
    size_t index = get_hash(key);
    const size_t mask = capacity_ - 1;
    for (size_t i = 0; i < capacity_; i++) {
      const T old = seen_[index];
      if (old == key) return true;
      if (old == kEmptyValue) return false;
      index = (index + 1) & mask;
    }
    return false;
  }

  bool insert(T key) {
    size_t index = get_hash(key);
    const size_t mask = capacity_ - 1;
    for (size_t i = 0; i < capacity_; i++) {
      T& old = seen_[index];
      if (old == key) return false;
      if (old == kEmptyValue) {
        old = key;
        return true;
      }
      index = (index + 1) & mask;
    }
    return false;
  }

  void clear() {
    std::fill(seen_.begin(), seen_.end(), kEmptyValue);
  }
};
