#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "filter_hash.cuh"

template <unsigned BX, unsigned BY>
__global__ void kernel_test_hash_filter(
    const uint32_t* g_insert_keys,
    const uint32_t* g_query_keys,
    unsigned num_insert,
    unsigned num_query,
    unsigned bitlen,
    uint32_t* g_insert_results,
    uint32_t* g_query_results)
{
    extern __shared__ char smem[];
    char* ptr = smem;
    HashFilter filter(ptr, bitlen);

    filter.init();
    __syncthreads();

    const unsigned warp = threadIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned inst = blockIdx.x * BY + warp;

    const uint32_t* insert_keys = g_insert_keys + static_cast<size_t>(inst) * num_insert;
    const uint32_t* query_keys = g_query_keys + static_cast<size_t>(inst) * num_query;
    uint32_t* insert_results = g_insert_results + static_cast<size_t>(inst) * num_insert;
    uint32_t* query_results = g_query_results + static_cast<size_t>(inst) * num_query;

    for (unsigned i = lane; i < num_insert; i += BX)
        insert_results[i] = filter.insert(insert_keys[i]);
    __syncthreads();

    for (unsigned i = lane; i < num_query; i += BX)
        query_results[i] = filter.contains(query_keys[i]);
}

template <unsigned BX, unsigned BY>
bool run_hash_filter_test(unsigned bitlen, unsigned seed) {
    constexpr unsigned INSTANCES = BY;
    const unsigned num_insert = (1u << bitlen) / 2;
    const unsigned num_query = num_insert * 2;

    std::mt19937 rng(seed);
    std::vector<uint32_t> h_insert_keys(static_cast<size_t>(INSTANCES) * num_insert);
    std::vector<uint32_t> h_query_keys(static_cast<size_t>(INSTANCES) * num_query);
    std::vector<uint32_t> expected_query(static_cast<size_t>(INSTANCES) * num_query, 0);
    std::vector<std::unordered_set<uint32_t>> inserted_sets(INSTANCES);

    for (unsigned inst = 0; inst < INSTANCES; ++inst) {
        auto& seen = inserted_sets[inst];
        seen.reserve(num_insert * 2);
        const size_t insert_base = static_cast<size_t>(inst) * num_insert;
        const size_t query_base = static_cast<size_t>(inst) * num_query;

        for (unsigned i = 0; i < num_insert; ++i) {
            uint32_t key = rng();
            if ((i % 5) == 0 && i > 0) key = h_insert_keys[insert_base + (i / 2)];
            h_insert_keys[insert_base + i] = key;
            seen.insert(key);
        }

        std::vector<uint32_t> uniques;
        uniques.reserve(seen.size());
        for (uint32_t key : seen) uniques.push_back(key);

        for (unsigned i = 0; i < num_query; ++i) {
            uint32_t key;
            if ((i % 2) == 0 && !uniques.empty()) {
                key = uniques[static_cast<size_t>(rng()) % uniques.size()];
                expected_query[query_base + i] = 1u;
            } else {
                do {
                    key = rng();
                } while (seen.count(key) != 0);
                expected_query[query_base + i] = 0u;
            }
            h_query_keys[query_base + i] = key;
        }
    }

    uint32_t* d_insert_keys = nullptr;
    uint32_t* d_query_keys = nullptr;
    uint32_t* d_insert_results = nullptr;
    uint32_t* d_query_results = nullptr;
    cudaMalloc(&d_insert_keys, h_insert_keys.size() * sizeof(uint32_t));
    cudaMalloc(&d_query_keys, h_query_keys.size() * sizeof(uint32_t));
    cudaMalloc(&d_insert_results, h_insert_keys.size() * sizeof(uint32_t));
    cudaMalloc(&d_query_results, expected_query.size() * sizeof(uint32_t));

    cudaMemcpy(d_insert_keys, h_insert_keys.data(), h_insert_keys.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_keys, h_query_keys.data(), h_query_keys.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_insert_results, 0, h_insert_keys.size() * sizeof(uint32_t));
    cudaMemset(d_query_results, 0, expected_query.size() * sizeof(uint32_t));

    const dim3 block(BX, BY);
    const size_t smem_size = HashFilter::shared_mem_bytes(bitlen, BY);
    kernel_test_hash_filter<BX, BY><<<1, block, smem_size>>>(
        d_insert_keys, d_query_keys, num_insert, num_query, bitlen,
        d_insert_results, d_query_results);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_insert_results(h_insert_keys.size());
    std::vector<uint32_t> h_query_results(expected_query.size());
    cudaMemcpy(h_insert_results.data(), d_insert_results, h_insert_results.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_results.data(), d_query_results, h_query_results.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_insert_keys);
    cudaFree(d_query_keys);
    cudaFree(d_insert_results);
    cudaFree(d_query_results);

    bool pass = true;
    for (unsigned inst = 0; inst < INSTANCES; ++inst) {
        const size_t insert_base = static_cast<size_t>(inst) * num_insert;
        std::unordered_map<uint32_t, uint32_t> success_count;
        success_count.reserve(inserted_sets[inst].size() * 2);
        for (unsigned i = 0; i < num_insert; ++i) {
            if (h_insert_results[insert_base + i] != 0) {
                ++success_count[h_insert_keys[insert_base + i]];
            }
        }
        if (success_count.size() != inserted_sets[inst].size()) {
            printf("  FAIL insert unique-count for instance %u: expected %zu got %zu\n",
                   inst, inserted_sets[inst].size(), success_count.size());
            pass = false;
            break;
        }
        for (uint32_t key : inserted_sets[inst]) {
            auto it = success_count.find(key);
            if (it == success_count.end() || it->second != 1) {
                printf("  FAIL insert multiplicity for instance %u key %u\n", inst, key);
                pass = false;
                break;
            }
        }
        if (!pass) break;
    }
    for (size_t i = 0; pass && i < expected_query.size(); ++i) {
        if (h_query_results[i] != expected_query[i]) {
            printf("  FAIL query at %zu: expected %u got %u\n", i, expected_query[i], h_query_results[i]);
            pass = false;
            break;
        }
    }
    return pass;
}

int main() {
    bool all_pass = true;
    unsigned seed = 500;

    struct TestCase {
        const char* name;
        bool (*fn)(unsigned, unsigned);
        unsigned bitlen;
    } tests[] = {
        {"hash_filter<256,1> bitlen=10", run_hash_filter_test<256, 1>, 10},
        {"hash_filter<32,4> bitlen=10", run_hash_filter_test<32, 4>, 10},
        {"hash_filter<32,8> bitlen=10", run_hash_filter_test<32, 8>, 10},
    };

    for (const auto& test : tests) {
        const bool pass = test.fn(test.bitlen, seed++);
        printf("[%s] %s\n", pass ? "PASS" : "FAIL", test.name);
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
