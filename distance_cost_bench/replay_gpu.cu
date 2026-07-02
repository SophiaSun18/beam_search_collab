// GPU replay microbenchmark over real access traces.
//
// One CUDA block replays one query's trace records. Exact L2 uses warp_distance,
// and RaBitQ expansion uses the same query_prepare_gpu + staged code/factor copy
// + scan_one_neighbor_lanes_gpu<32> shape as search_rabitq.cuh.
#define QG_BQUERY 6

#include "include/qg.hpp"
#include "include/utils.cuh"

#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

template <typename T>
static __device__ inline void replay_staged_copy_async_begin_or_sync(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        __pipeline_memcpy_async(dst + i, src + i, sizeof(T));
    }
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
#endif
}

static __device__ inline void replay_staged_copy_async_finish_or_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

__global__ void replay_trace_kernel(
    const AccessTraceRecord* __restrict__ records,
    const uint32_t* __restrict__ offsets,
    int nq,
    int dim,
    int degree,
    const float* __restrict__ queries,
    const float* __restrict__ base_vectors,
    const vidType* __restrict__ graph_edges,
    const float* __restrict__ qg_data,
    const float* __restrict__ qg_signs,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset,
    bool use_ip,
    float* __restrict__ sink_out) {
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const float* query_global = queries + static_cast<size_t>(query_id) * dim;

    extern __shared__ char shared_memory[];
    float* query_buffer = reinterpret_cast<float*>(shared_memory);
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(static_cast<float>(dim))));
    size_t code_words = padded_dim / 64;

    float* rotated_query = query_buffer + dim;
    uint8_t* quantized_query = reinterpret_cast<uint8_t*>(rotated_query + padded_dim);
    uint8_t* lut = quantized_query + padded_dim;
    char* staged_base = reinterpret_cast<char*>(lut + (padded_dim << 2));
    staged_base = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) &
        ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint64_t* staged_codes = reinterpret_cast<uint64_t*>(staged_base);
    float* staged_factors = reinterpret_cast<float*>(staged_codes + static_cast<size_t>(degree) * code_words);

    __shared__ QueryFactors qf;
    if (tid == 0) {
        qf.rotated_query = rotated_query;
        qf.quantized_query = quantized_query;
        qf.lut = lut;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
    }

    const bool has_qg = qg_data != nullptr && qg_signs != nullptr;
    if (has_qg) {
        for (int i = tid; i < dim; i += blockDim.x) {
            query_buffer[i] = query_global[i];
        }
        __syncthreads();
        query_prepare_gpu(query_buffer, qf, qg_signs, dim, static_cast<int>(padded_dim));
        pack_lut_gpu(static_cast<int>(padded_dim), qf.quantized_query, qf.lut);
    }
    __syncthreads();

    float sink = 0.0f;
    const uint32_t begin = offsets[query_id];
    const uint32_t end = offsets[query_id + 1];
    const int bytes_per_neighbor = static_cast<int>(padded_dim >> 3);

    for (uint32_t pos = begin; pos < end; ++pos) {
        const AccessTraceRecord rec = records[pos];
        const uint32_t mask = rec.active_mask;

        if (rec.op == ACCESS_TRACE_BASE_SEED_EXACT) {
            if (warp_id == 0) {
                sink += warp_distance(dim, query_global,
                                      base_vectors + static_cast<size_t>(rec.node_id) * dim,
                                      use_ip);
            }
            __syncthreads();
        } else if (rec.op == ACCESS_TRACE_BASE_EXPAND_EXACT) {
            const vidType* neighbors = graph_edges + static_cast<size_t>(rec.node_id) * degree;
            for (int rank = warp_id; rank < degree && rank < 32; rank += WARPS_PER_BLOCK) {
                if ((mask & (1u << rank)) == 0) continue;
                const vidType child = neighbors[rank];
                sink += warp_distance(dim, query_global,
                                      base_vectors + static_cast<size_t>(child) * dim,
                                      use_ip);
            }
            __syncthreads();
        } else if (rec.op == ACCESS_TRACE_RQ_SEED_EXACT) {
            if (warp_id == 0) {
                sink += warp_distance(dim, query_global,
                                      qg_data + static_cast<size_t>(rec.node_id) * row_offset,
                                      use_ip);
            }
            __syncthreads();
        } else if (rec.op == ACCESS_TRACE_RQ_EXPAND) {
            const float* current_row = qg_data + static_cast<size_t>(rec.node_id) * row_offset;
            const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
            const float* current_factors = current_row + factor_offset;

            replay_staged_copy_async_begin_or_sync(
                staged_codes, current_codes, static_cast<int>(static_cast<size_t>(degree) * code_words));
            replay_staged_copy_async_begin_or_sync(staged_factors, current_factors, 3 * degree);

            const float exact_dist = warp_distance(dim, query_global, current_row, use_ip);
            replay_staged_copy_async_finish_or_sync();
            __syncthreads();

            for (int rank = warp_id; rank < degree && rank < 32; rank += WARPS_PER_BLOCK) {
                if ((mask & (1u << rank)) == 0) continue;
                const int batch = rank / WARP_SIZE;
                const int neighbor_idx = rank & (WARP_SIZE - 1);
                const uint8_t* packed_codes_block =
                    reinterpret_cast<const uint8_t*>(staged_codes) +
                    static_cast<size_t>(batch) * padded_dim * 4;
                const float* triple_x = staged_factors + rank;
                const float* factor_dq = staged_factors + degree + rank;
                const float* factor_vq = staged_factors + 2 * degree + rank;
                sink += scan_one_neighbor_lanes_gpu<32>(
                    qf, packed_codes_block, neighbor_idx,
                    triple_x, factor_dq, factor_vq,
                    exact_dist, static_cast<int>(padded_dim), bytes_per_neighbor);
            }
            __syncthreads();
        } else if (rec.op == ACCESS_TRACE_RQ_CHILD_EXACT ||
                   rec.op == ACCESS_TRACE_RQ_EXTRA_EXACT) {
            const float* parent_row = qg_data + static_cast<size_t>(rec.node_id) * row_offset;
            const vidType* neighbors = reinterpret_cast<const vidType*>(parent_row + neighbor_offset);
            for (int rank = warp_id; rank < degree && rank < 32; rank += WARPS_PER_BLOCK) {
                if ((mask & (1u << rank)) == 0) continue;
                const vidType child = neighbors[rank];
                sink += warp_distance(dim, query_global,
                                      qg_data + static_cast<size_t>(child) * row_offset,
                                      use_ip);
            }
            __syncthreads();
        }
    }

    if (lane_id == 0) {
        atomicAdd(sink_out + query_id, sink);
    }
}

struct TraceCounts {
    uint64_t records = 0;
    uint64_t groups = 0;
    uint64_t exact_ops = 0;
    uint64_t est_ops = 0;
};

static inline uint32_t popcount_mask(uint32_t mask) {
    return static_cast<uint32_t>(__builtin_popcount(mask));
}

static TraceCounts count_trace_ops(const vector<AccessTraceRecord>& records) {
    TraceCounts c;
    c.records = records.size();
    for (const auto& r : records) {
        c.groups++;
        switch (r.op) {
            case ACCESS_TRACE_BASE_SEED_EXACT:
            case ACCESS_TRACE_RQ_SEED_EXACT:
                c.exact_ops++;
                break;
            case ACCESS_TRACE_BASE_EXPAND_EXACT:
            case ACCESS_TRACE_RQ_CHILD_EXACT:
            case ACCESS_TRACE_RQ_EXTRA_EXACT:
                c.exact_ops += popcount_mask(r.active_mask);
                break;
            case ACCESS_TRACE_RQ_EXPAND:
                c.exact_ops++;
                c.est_ops += popcount_mask(r.active_mask);
                break;
            default:
                break;
        }
    }
    return c;
}

static inline bool has_op(const vector<AccessTraceRecord>& records, uint32_t op) {
    for (const auto& r : records) {
        if (r.op == op) return true;
    }
    return false;
}

static inline bool has_base_ops(const vector<AccessTraceRecord>& records) {
    return has_op(records, ACCESS_TRACE_BASE_SEED_EXACT) ||
           has_op(records, ACCESS_TRACE_BASE_EXPAND_EXACT);
}

static inline bool has_rq_ops(const vector<AccessTraceRecord>& records) {
    return has_op(records, ACCESS_TRACE_RQ_SEED_EXACT) ||
           has_op(records, ACCESS_TRACE_RQ_EXPAND) ||
           has_op(records, ACCESS_TRACE_RQ_CHILD_EXACT) ||
           has_op(records, ACCESS_TRACE_RQ_EXTRA_EXACT);
}

static void print_replay_result(const TraceCounts& counts, int repeat, float total_ms) {
    const double total_ns = static_cast<double>(total_ms) * 1.0e6;
    const uint64_t repeated_groups = counts.groups * static_cast<uint64_t>(repeat);
    const uint64_t repeated_exact = counts.exact_ops * static_cast<uint64_t>(repeat);
    const uint64_t repeated_est = counts.est_ops * static_cast<uint64_t>(repeat);
    const uint64_t repeated_ops = repeated_exact + repeated_est;
    printf("GPU_REPLAY,trace_total,records=%llu,groups=%llu,exact_ops=%llu,est_ops=%llu,total_ns=%.4f,ns_per_group=%.4f,ns_per_op=%.4f\n",
           static_cast<unsigned long long>(counts.records),
           static_cast<unsigned long long>(repeated_groups),
           static_cast<unsigned long long>(repeated_exact),
           static_cast<unsigned long long>(repeated_est),
           total_ns,
           repeated_groups ? total_ns / static_cast<double>(repeated_groups) : 0.0,
           repeated_ops ? total_ns / static_cast<double>(repeated_ops) : 0.0);
}

int main(int argc, char** argv) {
    string base_file, query_file, graph_file, qg_codebook, trace_file;
    int degree = 32;
    int repeat = 10;
    int warmup = 3;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-base" && i + 1 < argc) base_file = argv[++i];
        else if (a == "-query" && i + 1 < argc) query_file = argv[++i];
        else if (a == "-graph" && i + 1 < argc) graph_file = argv[++i];
        else if (a == "-qg" && i + 1 < argc) qg_codebook = argv[++i];
        else if (a == "-trace" && i + 1 < argc) trace_file = argv[++i];
        else if (a == "-degree" && i + 1 < argc) degree = atoi(argv[++i]);
        else if (a == "-repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (a == "-warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown or incomplete arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (base_file.empty() || query_file.empty() || trace_file.empty()) {
        fprintf(stderr,
                "Usage: %s -base <base.fvecs> -query <query.fvecs> -trace <trace.csv>"
                " [-graph <graph.bin>] [-qg <qg.index>] [-degree 32]"
                " [-repeat 10] [-warmup 3]\n",
                argv[0]);
        return 1;
    }
    if (repeat < 1) repeat = 1;
    if (warmup < 0) warmup = 0;

    g_metric_type = infer_metric_from_dataset_path(base_file);
    const bool use_ip = (g_metric_type == METRIC_IP);

    size_t n_base = 0, dim_base = 0, n_query = 0, dim_query = 0;
    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    if (dim_base != dim_query) {
        throw runtime_error("base/query dim mismatch");
    }

    vector<AccessTraceRecord> records = read_access_trace_csv(trace_file);
    sort(records.begin(), records.end(),
         [](const AccessTraceRecord& a, const AccessTraceRecord& b) {
             if (a.query_id != b.query_id) return a.query_id < b.query_id;
             if (a.step != b.step) return a.step < b.step;
             return a.op < b.op;
         });

    uint32_t max_query_id = 0;
    vector<uint8_t> seen_queries(n_query, 0);
    size_t active_query_count = 0;
    for (const auto& r : records) {
        if (r.query_id >= n_query) {
            throw runtime_error("trace query_id exceeds query rows");
        }
        max_query_id = std::max(max_query_id, r.query_id);
        if (!seen_queries[r.query_id]) {
            seen_queries[r.query_id] = 1;
            active_query_count++;
        }
    }
    const size_t active_nq = records.empty() ? 0 : static_cast<size_t>(max_query_id) + 1;

    const bool need_base = has_base_ops(records);
    const bool need_rq = has_rq_ops(records);

    Graph graph;
    if (need_base) {
        if (graph_file.empty()) {
            throw runtime_error("base access trace replay requires -graph");
        }
        graph.load_graph_index(graph_file.c_str());
        degree = static_cast<int>(graph.degree);
    }

    unique_ptr<QuantizationGraph> qg;
    if (need_rq) {
        if (qg_codebook.empty()) {
            throw runtime_error("RaBitQ access trace replay requires -qg");
        }
        qg.reset(new QuantizationGraph(n_base, dim_base, degree, qg_codebook));
        qg->set_metric(g_metric_type);
        degree = qg->get_degree();
    }

    vector<uint32_t> offsets(active_nq + 1, 0);
    for (const auto& r : records) {
        offsets[static_cast<size_t>(r.query_id) + 1]++;
    }
    for (size_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i - 1];
    }
    vector<uint32_t> cursor = offsets;
    vector<AccessTraceRecord> ordered(records.size());
    for (const auto& r : records) {
        ordered[cursor[r.query_id]++] = r;
    }

    AccessTraceRecord* d_records = nullptr;
    uint32_t* d_offsets = nullptr;
    float* d_queries = nullptr;
    float* d_base = nullptr;
    vidType* d_graph = nullptr;
    float* d_qg_data = nullptr;
    float* d_qg_signs = nullptr;
    float* d_sink = nullptr;

    CUDA_SAFE_CALL(cudaMalloc(&d_records, ordered.size() * sizeof(AccessTraceRecord)));
    CUDA_SAFE_CALL(cudaMemcpy(d_records, ordered.data(),
                              ordered.size() * sizeof(AccessTraceRecord),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_offsets, offsets.data(),
                              offsets.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&d_queries, active_nq * dim_base * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_queries, query_vectors,
                              active_nq * dim_base * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&d_base, n_base * dim_base * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_base, base_vectors,
                              n_base * dim_base * sizeof(float),
                              cudaMemcpyHostToDevice));
    if (need_base) {
        CUDA_SAFE_CALL(cudaMalloc(&d_graph, graph.edges.size() * sizeof(vidType)));
        CUDA_SAFE_CALL(cudaMemcpy(d_graph, graph.edges.data(),
                                  graph.edges.size() * sizeof(vidType),
                                  cudaMemcpyHostToDevice));
    }
    size_t row_offset = dim_base;
    size_t neighbor_offset = 0;
    size_t code_offset = 0;
    size_t factor_offset = 0;
    if (need_rq) {
        row_offset = qg->get_row_offset();
        neighbor_offset = qg->get_neighbor_offset();
        code_offset = qg->get_code_offset();
        factor_offset = qg->get_factor_offset();
        CUDA_SAFE_CALL(cudaMalloc(&d_qg_data, qg->get_data_bytes()));
        CUDA_SAFE_CALL(cudaMemcpy(d_qg_data, qg->get_data_ptr(), qg->get_data_bytes(),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc(&d_qg_signs, qg->get_signs_bytes()));
        CUDA_SAFE_CALL(cudaMemcpy(d_qg_signs, qg->get_signs_ptr(), qg->get_signs_bytes(),
                                  cudaMemcpyHostToDevice));
    }
    CUDA_SAFE_CALL(cudaMalloc(&d_sink, active_nq * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(d_sink, 0, active_nq * sizeof(float)));

    const size_t padded_dim = 1ULL << static_cast<size_t>(ceil(log2(static_cast<double>(dim_base))));
    const size_t code_words = padded_dim / 64;
    const size_t shared_bytes = need_rq
        ? (dim_base * sizeof(float) +
           padded_dim * sizeof(float) +
           padded_dim * sizeof(uint8_t) +
           (padded_dim << 2) * sizeof(uint8_t) +
           alignof(uint64_t) +
           static_cast<size_t>(degree) * code_words * sizeof(uint64_t) +
           3ULL * degree * sizeof(float))
        : 0;

    if (need_rq) {
        CUDA_SAFE_CALL(cudaFuncSetAttribute(
            replay_trace_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    auto launch = [&]() {
        replay_trace_kernel<<<static_cast<int>(active_nq), BLOCK_SIZE, shared_bytes>>>(
            d_records, d_offsets, static_cast<int>(active_nq),
            static_cast<int>(dim_base), degree,
            d_queries, d_base, d_graph, d_qg_data, d_qg_signs,
            row_offset, neighbor_offset, code_offset, factor_offset,
            use_ip, d_sink);
        CUDA_SAFE_CALL(cudaGetLastError());
    };

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        launch();
    }
    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&total_ms, start, stop));

    TraceCounts counts = count_trace_ops(records);
    printf("GPU replay trace: records=%zu queries=%zu metric=%s repeat=%d warmup=%d\n",
           records.size(), active_query_count, metric_name(g_metric_type), repeat, warmup);
    print_replay_result(counts, repeat, total_ms);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    CUDA_SAFE_CALL(cudaFree(d_records));
    CUDA_SAFE_CALL(cudaFree(d_offsets));
    CUDA_SAFE_CALL(cudaFree(d_queries));
    CUDA_SAFE_CALL(cudaFree(d_base));
    if (d_graph) CUDA_SAFE_CALL(cudaFree(d_graph));
    if (d_qg_data) CUDA_SAFE_CALL(cudaFree(d_qg_data));
    if (d_qg_signs) CUDA_SAFE_CALL(cudaFree(d_qg_signs));
    CUDA_SAFE_CALL(cudaFree(d_sink));
    return 0;
}
