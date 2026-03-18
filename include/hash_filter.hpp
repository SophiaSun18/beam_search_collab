#pragma once

#include <vector>
#include <unordered_set>
#include <stdint.h>
#include <string.h>
#include <cfloat>

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

public:
  hash_filter(int L) {
    // bits = std::__lg(L)+1; // round up to power of 2
    // L = (1 << bits);
    // capacity = L;
    // seen.resize(L);
    // fill(seen.begin(),seen.end(),-1); 

    int bit_size = 0;
    int tmp = L;
    while (tmp != 0) { bit_size++; tmp >>= 1; }
    bits = (bit_size + 4) / 2 + 3;
    capacity = 1 << bits;
    seen.resize(capacity);
    fill(seen.begin(), seen.end(), -1);
  }
  int get_hash(T thing) {
    //return hasher(thing) & ((1 << bits)-1);
    return thing & ((1 << bits)-1);
  }
  
  // old methods (lossy, no collision handling)
  bool contains(T thing) {
    int h = get_hash(thing);
    return (seen[h] == thing);
  }
  bool add(T thing) { // returns true if successfully added
    int h = get_hash(thing);
    if (seen[h] == thing) return false;
    total_inserts++;
    if (seen[h] != -1) total_collisions++;  // slot occupied by different ID
    seen[h] = thing;
    return true;
  }

  // new methods (lossless, with stl_hash_ overflow)
  bool contains_new(T thing) {
    T val = seen[get_hash(thing)];
    if (val == thing) return true;
    if (val == -1) return false;
    return stl_hash_.find(thing) != stl_hash_.end();
  }
  bool add_new(T thing) {
    T& val = seen[get_hash(thing)];
    if (val == thing) return false;
    total_inserts++;
    if (val != -1) total_collisions++;  // slot occupied by different ID, spill to overflow
    else { val = thing; return true; }
    stl_hash_.insert(thing);
    return true;
  }

  size_t get_overflow_size() const { return stl_hash_.size(); }
  size_t get_total_inserts() const { return total_inserts; }
  size_t get_total_collisions() const { return total_collisions; }
  double get_collision_ratio() const { return total_inserts > 0 ? (double)total_collisions / total_inserts : 0.0; }

  void clear() {
    fill(seen.begin(), seen.end(), -1);
    stl_hash_.clear();
  }
};
