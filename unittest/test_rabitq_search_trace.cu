#include <iostream>
#include <vector>
#include <cfloat>
#include <cstdint>
#include <algorithm>
#include <unordered_set>

#include "data_loader.hpp"
#include "cpu_graph.hpp"
#include "gpu_graph.cuh"
#include "gpu_search_rabitq.cuh"
#include "cpu_search_rabitq.hpp"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err_));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

// ── GPU kernel with per-iteration trace ─────────────────────────────────────

// Use the same warp structure: NY=1, SharedBitonicTopK<32,1>, REG=1
using TraceTopK = SharedBitonicTopK<32, 1>;
static constexpr unsigned TRACE_REG = 1;
static constexpr int TRACE_NUM_BITS = RABITQ_BQUERY;

static __host__ __device__ size_t fastscan_trace_smem_bytes(int dim)
{
    int pdim = rabitq_padded_dim(dim);
    int wpp = pdim / 32;
    size_t size = alignof(float) - 1;
    size += pdim * sizeof(float);
    size = (size + alignof(uint32_t) - 1) & ~(alignof(uint32_t) - 1);
    size += static_cast<size_t>(TRACE_NUM_BITS) * wpp * sizeof(uint32_t);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += sizeof(float);
    size += sizeof(float);
    size += sizeof(int32_t);
    return size;
}

__global__ void kernel_rabitq_traced(
    GPUGraphView graph, GPURabitqView rabitq, GPURabitqSearchCfg cfg,
    int dim,
    const float* __restrict__ d_query,
    vid_t* __restrict__ d_results, int K,
    vid_t* __restrict__ d_trace_nodes,
    float* __restrict__ d_trace_dists,
    int* __restrict__ d_num_iters)
{
    extern __shared__ char smem[];
    char* ptr = smem;

    const unsigned lane = threadIdx.x;
    const unsigned max_degree = graph.degree();
    const int wpp = rabitq.words_per_plane();

    // Construct search state (mirrors GPURabitqSearch internals)
    TraceTopK topk(ptr, cfg.topk_size);
    HashFilter hf(ptr, cfg.filter_capacity);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    float* query_buf = reinterpret_cast<float*>(ptr);
    ptr += dim * sizeof(float);

    uint32_t* parent_buf = reinterpret_cast<uint32_t*>(ptr);
    ptr += sizeof(uint32_t);

    uint32_t* cand_idx = reinterpret_cast<uint32_t*>(ptr);
    ptr += max_degree * sizeof(uint32_t);

    float* cand_dist = reinterpret_cast<float*>(ptr);
    ptr += max_degree * sizeof(float);

    char* fastscan_smem = ptr;
    ptr += fastscan_trace_smem_bytes(dim);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + alignof(uint32_t) - 1)
            & ~(static_cast<uintptr_t>(alignof(uint32_t)) - 1));
    uint32_t* staged_codes = reinterpret_cast<uint32_t*>(ptr);
    ptr += static_cast<size_t>(max_degree) * wpp * sizeof(uint32_t);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    float* staged_factors = reinterpret_cast<float*>(ptr);

    // Load query
    for (unsigned i = lane; i < static_cast<unsigned>(dim); i += 32)
        query_buf[i] = d_query[i];

    // Init candidates
    for (unsigned i = lane; i < max_degree; i += 32) {
        cand_idx[i]  = UINT_MAX;
        cand_dist[i] = FLT_MAX;
    }
    hf.init();
    __syncwarp();

    // Seed entry point
    {
        float d = l2_warp(dim, query_buf, graph.vect(cfg.entry_point).data());
        if (lane == 0) {
            cand_idx[0]  = cfg.entry_point;
            cand_dist[0] = d;
            hf.insert(cfg.entry_point);
        }
    }
    __syncwarp();

    // Prepare RabitQFastScan
    char* fsp = fastscan_smem;
    RabitqQuantizer<32> quantizer(rabitq.signs(), 0);
    RabitQFastScan<32, TRACE_NUM_BITS> fs(fsp, query_buf, dim, quantizer);
    __syncwarp();

    int actual_iters = 0;

    for (int iter = 0; iter < cfg.max_iterations; iter++) {
        // Hash reset
        if ((iter + 1) % cfg.reset_interval == 0) {
            hf.init();
            __syncwarp();
            for (unsigned i = lane; i < topk.capacity(); i += 32) {
                uint32_t node = topk.id(i);
                if (node != UINT_MAX) hf.insert(node);
            }
        }
        __syncwarp();

        // Merge candidates
        {
            float d[1] = {(lane < max_degree) ? cand_dist[lane] : FLT_MAX};
            uint32_t x[1] = {(lane < max_degree) ? cand_idx[lane] : UINT_MAX};
            topk.merge(d, x);
        }

        // Pick next node
        uint32_t pbuf[1];
        pbuf[0] = UINT_MAX;
        unsigned found = topk.template pick<1>(pbuf, 1);
        uint32_t picked = UINT_MAX;
        {
            const bool has_pick = found > 0 && pbuf[0] != UINT_MAX;
            const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
            if (ballot != 0) {
                const int src_lane = __ffs(ballot) - 1;
                picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
            }
        }
        if (lane == 0 && found > 0)
            parent_buf[0] = picked;
        __syncwarp();

        if (found == 0) break;

        vid_t current_node = parent_buf[0];

        // TRACE: record expanded node
        if (lane == 0) {
            d_trace_nodes[iter] = current_node;
            actual_iters = iter + 1;
        }

        // Expand: filter neighbors
        Span<const vid_t> nbrs = graph.nbrs(current_node);
        for (unsigned i = lane; i < max_degree; i += 32) {
            uint32_t child = nbrs[i];
            if (child != UINT_MAX && hf.insert(child) == 0)
                child = UINT_MAX;
            cand_idx[i]  = child;
            cand_dist[i] = FLT_MAX;
        }
        __syncwarp();

        // Load codes/factors
        const uint32_t* src_codes = rabitq.codes(current_node);
        int code_count = static_cast<int>(max_degree) * wpp;
        for (int i = static_cast<int>(lane); i < code_count; i += 32)
            staged_codes[i] = src_codes[i];
        const float* src_factors = rabitq.factors(current_node);
        int factor_count = static_cast<int>(3 * max_degree);
        for (int i = static_cast<int>(lane); i < factor_count; i += 32)
            staged_factors[i] = src_factors[i];
        __syncwarp();

        // L2 to expanding node
        float exact_dist = l2_warp(dim, query_buf, graph.vect(current_node).data());

        // TRACE: record exact dist
        if (lane == 0)
            d_trace_dists[iter] = exact_dist;

        // Distance estimation
        int num_batches = (max_degree + 32 - 1) / 32;
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_begin = batch * 32;
            int active = min(32, static_cast<int>(max_degree) - batch_begin);

            const uint32_t* batch_codes = staged_codes + batch_begin * wpp;

            fs.dot(batch_codes, active,
                   staged_factors + batch_begin, static_cast<int>(max_degree),
                   exact_dist, cand_dist + batch_begin);

            if (lane < static_cast<unsigned>(active)) {
                if (cand_idx[batch_begin + lane] == UINT_MAX)
                    cand_dist[batch_begin + lane] = FLT_MAX;
            }
        }
        __syncwarp();
    }

    // Final merge
    {
        float d[1] = {(lane < max_degree) ? cand_dist[lane] : FLT_MAX};
        uint32_t x[1] = {(lane < max_degree) ? cand_idx[lane] : UINT_MAX};
        topk.merge(d, x);
    }

    topk.flush(cand_dist, d_results, K);

    if (lane == 0)
        *d_num_iters = actual_iters;
}

// ── CPU search with per-iteration trace ─────────────────────────────────────

struct CPUTraceResult {
    std::vector<vid_t> expanded_nodes;
    std::vector<float> expanded_dists;
    std::vector<vid_t> results;
};

CPUTraceResult cpu_rabitq_search_trace(
    const CPUGraph& graph, const CPURabitqData& rq,
    const float* query, int K, unsigned topk_size,
    vid_t entry_point, int max_iterations)
{
    CPURabitqSearchCfg cfg;
    cfg.topk_size = topk_size;
    cfg.entry_point = entry_point;
    cfg.max_iterations = max_iterations;

    CPURabitqSearch searcher(graph, rq, cfg);

    CPUTraceResult trace;
    int dim  = graph.dim;
    int pdim = rq.padded_dim;
    int degree = graph.degree;
    int wpp = rq.words_per_plane;
    constexpr int BQUERY = 6;

    // rotate query
    std::vector<float> q_rot(pdim);
    for (int d = 0; d < dim; ++d)
        q_rot[d] = query[d] * rq.signs[d];
    for (int d = dim; d < pdim; ++d)
        q_rot[d] = 0.0f;
    for (int len = 1; len < pdim; len <<= 1)
        for (int i = 0; i < pdim; i += len << 1)
            for (int j = 0; j < len; ++j) {
                float u = q_rot[i + j], v = q_rot[i + j + len];
                q_rot[i + j]       = u + v;
                q_rot[i + j + len] = u - v;
            }

    float low = q_rot[0], high = q_rot[0];
    for (int d = 1; d < pdim; ++d) {
        low  = std::min(low,  q_rot[d]);
        high = std::max(high, q_rot[d]);
    }
    float w     = (high - low) / ((1 << BQUERY) - 1);
    float inv_w = 1.0f / w;

    std::vector<uint8_t> quant(pdim);
    int32_t sum_q = 0;
    for (int d = 0; d < pdim; ++d) {
        float x = (q_rot[d] - low) * inv_w;
        int q_val = static_cast<int>(floorf(x + 0.5f));
        if (q_val < 0) q_val = 0;
        if (q_val > (1 << BQUERY) - 1) q_val = (1 << BQUERY) - 1;
        quant[d] = static_cast<uint8_t>(q_val);
        sum_q += q_val;
    }

    auto l2_exact = [&](vid_t node) {
        const auto vec = graph.vect(node);
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = query[d] - vec[d];
            sum += diff * diff;
        }
        return sum;
    };

    auto estimate = [&](vid_t node, int nbr_idx, float exact_dist) {
        const float* factors = rq.factors.data()
            + static_cast<size_t>(node) * 3 * degree;
        float tx  = factors[nbr_idx];
        float fdq = factors[degree + nbr_idx];
        float fvq = factors[2 * degree + nbr_idx];
        if (tx == FLT_MAX) return FLT_MAX;

        const uint32_t* nbr_codes = rq.codes.data()
            + static_cast<size_t>(node) * degree * wpp
            + nbr_idx * wpp;

        uint32_t raw_sum = 0;
        for (int d = 0; d < pdim; ++d) {
            if ((nbr_codes[d / 32] >> (d % 32)) & 1)
                raw_sum += quant[d];
        }

        float ip = static_cast<float>(
            (static_cast<int32_t>(raw_sum) << 1) - sum_q);
        float est = tx + exact_dist + fdq * w * ip + fvq * low;
        if (!std::isfinite(est)) return FLT_MAX;
        return est;
    };

    struct BeamEntry { float dist; vid_t node; bool expanded; };
    std::vector<BeamEntry> beam;
    beam.reserve(topk_size + degree);
    std::unordered_set<vid_t> seen;
    seen.reserve(topk_size * 4);

    beam.push_back({l2_exact(entry_point), entry_point, false});
    seen.insert(entry_point);

    auto beam_sort = [](const BeamEntry& a, const BeamEntry& b) {
        return (a.dist != b.dist) ? a.dist < b.dist : a.node < b.node;
    };

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::sort(beam.begin(), beam.end(), beam_sort);
        auto it = std::find_if(beam.begin(), beam.end(),
                               [](const BeamEntry& e) { return !e.expanded; });
        if (it == beam.end()) break;

        vid_t current = it->node;
        float current_dist = it->dist;
        it->expanded = true;
        trace.expanded_nodes.push_back(current);
        trace.expanded_dists.push_back(current_dist);

        float exact_dist = l2_exact(current);
        auto nbrs = graph.nbrs(current);
        for (int i = 0; i < degree; ++i) {
            vid_t child = nbrs[i];
            if (child == UINT32_MAX) continue;
            if (!seen.insert(child).second) continue;
            float est = estimate(current, i, exact_dist);
            beam.push_back({est, child, false});
        }

        if (beam.size() > topk_size) {
            std::sort(beam.begin(), beam.end(), beam_sort);
            beam.resize(topk_size);
        }
    }

    std::sort(beam.begin(), beam.end(), beam_sort);
    trace.results.resize(K);
    for (int i = 0; i < K; ++i)
        trace.results[i] = (i < static_cast<int>(beam.size()))
            ? beam[i].node : UINT32_MAX;
    return trace;
}

// ── entry point ─────────────────────────────────────────────────────────────

static vid_t compute_entry_point(const CPUGraph& graph) {
    size_t n = graph.num_nodes;
    int dim = graph.dim;
    const float* data = graph.vectors.data();
    std::vector<float> centroid(dim, 0.0f);
    float inv = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; i++) {
        const float* row = data + i * dim;
        for (int d = 0; d < dim; d++) centroid[d] += row[d] * inv;
    }
    vid_t best = 0;
    float best_dist = FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        const float* row = data + i * dim;
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = row[d] - centroid[d];
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best = static_cast<vid_t>(i); }
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph_file> <base_file> <query_file> <codebook_file>"
                  << " [topk_size] [max_iterations] [filter_capacity] [reset_interval]"
                  << std::endl;
        return 1;
    }

    std::string graph_file    = argv[1];
    std::string base_file     = argv[2];
    std::string query_file    = argv[3];
    std::string codebook_file = argv[4];

    unsigned topk_size       = (argc > 5) ? std::stoi(argv[5]) : 256;
    int      max_iterations  = (argc > 6) ? std::stoi(argv[6]) : 64;
    unsigned filter_capacity = (argc > 7) ? std::stoi(argv[7]) : 1024;
    int      reset_interval  = (argc > 8) ? std::stoi(argv[8]) : 16;
    int      K = 10;

    CPUGraph cpu_graph;
    cpu_graph.load(graph_file, base_file);
    int dim = cpu_graph.dim;
    int degree = cpu_graph.degree;
    size_t npoints = cpu_graph.num_nodes;

    size_t nq = 0, query_dim = 0;
    float* h_queries = read_fvecs(query_file.c_str(), nq, query_dim);

    CPURabitqData cpu_rq = load_rabitq_codebook(codebook_file, npoints, dim, degree);
    vid_t entry_point = (cpu_rq.entry_point != UINT32_MAX)
                      ? cpu_rq.entry_point : compute_entry_point(cpu_graph);

    std::cout << "Data: " << npoints << " points, dim=" << dim
              << ", degree=" << degree << ", queries=" << nq
              << ", entry=" << entry_point << std::endl;
    std::cout << "Config: topk=" << topk_size << ", max_iter=" << max_iterations
              << ", filter_cap=" << filter_capacity
              << ", reset_interval=" << reset_interval << std::endl << std::endl;

    // CPU trace
    const float* query0 = h_queries;
    CPUTraceResult cpu_trace = cpu_rabitq_search_trace(
        cpu_graph, cpu_rq, query0, K, topk_size, entry_point, max_iterations);

    std::cout << "=== CPU trace (" << cpu_trace.expanded_nodes.size()
              << " iterations) ===" << std::endl;
    for (size_t i = 0; i < cpu_trace.expanded_nodes.size() && i < 64; ++i)
        printf("  iter %3zu: expand node %8u  (est_dist=%.4f)\n",
               i, cpu_trace.expanded_nodes[i], cpu_trace.expanded_dists[i]);

    std::cout << "\n  CPU results (top-" << K << "): ";
    for (int i = 0; i < K; ++i) printf("%u ", cpu_trace.results[i]);
    std::cout << std::endl << std::endl;

    // GPU trace
    GPUGraph gpu_graph(cpu_graph);
    GPURabitq gpu_rq(cpu_rq);

    float* d_query;
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_query, query0, dim * sizeof(float), cudaMemcpyHostToDevice));

    vid_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, K * sizeof(vid_t)));

    vid_t* d_trace_nodes;
    float* d_trace_dists;
    int*   d_num_iters;
    CUDA_CHECK(cudaMalloc(&d_trace_nodes, max_iterations * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_trace_dists, max_iterations * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_iters, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_trace_nodes, 0xFF, max_iterations * sizeof(vid_t)));
    CUDA_CHECK(cudaMemset(d_trace_dists, 0, max_iterations * sizeof(float)));

    GPURabitqSearchCfg gpu_cfg;
    gpu_cfg.topk_size       = topk_size;
    gpu_cfg.filter_capacity = filter_capacity;
    gpu_cfg.entry_point     = entry_point;
    gpu_cfg.max_iterations  = max_iterations;
    gpu_cfg.reset_interval  = reset_interval;

    // Compute shared mem (same layout as the traced kernel)
    size_t smem_bytes = GPURabitqSearch<1, TraceTopK, TRACE_REG, HashFilter>::shared_mem_bytes(
        gpu_cfg, dim, degree);

    CUDA_CHECK(cudaFuncSetAttribute(kernel_rabitq_traced,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    dim3 block(32, 1);
    kernel_rabitq_traced<<<1, block, smem_bytes>>>(
        gpu_graph.view(), gpu_rq.view(), gpu_cfg, dim,
        d_query, d_results, K,
        d_trace_nodes, d_trace_dists, d_num_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    std::vector<vid_t> gpu_trace_nodes(max_iterations);
    std::vector<float> gpu_trace_dists(max_iterations);
    int gpu_num_iters = 0;

    CUDA_CHECK(cudaMemcpy(gpu_trace_nodes.data(), d_trace_nodes,
        max_iterations * sizeof(vid_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_trace_dists.data(), d_trace_dists,
        max_iterations * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gpu_num_iters, d_num_iters, sizeof(int),
        cudaMemcpyDeviceToHost));

    std::vector<vid_t> gpu_results(K);
    CUDA_CHECK(cudaMemcpy(gpu_results.data(), d_results, K * sizeof(vid_t), cudaMemcpyDeviceToHost));

    std::cout << "=== GPU trace (" << gpu_num_iters << " iterations) ===" << std::endl;
    for (int i = 0; i < gpu_num_iters && i < 64; ++i)
        printf("  iter %3d: expand node %8u  (exact_l2=%.4f)\n",
               i, gpu_trace_nodes[i], gpu_trace_dists[i]);

    std::cout << "\n  GPU results (top-" << K << "): ";
    for (int i = 0; i < K; ++i) printf("%u ", gpu_results[i]);
    std::cout << std::endl << std::endl;

    // Side-by-side
    std::cout << "=== Side-by-side comparison ===" << std::endl;
    int max_show = std::min({static_cast<int>(cpu_trace.expanded_nodes.size()),
                            gpu_num_iters, 64});
    printf("  %4s | %10s %12s | %10s %12s | %s\n",
           "iter", "CPU node", "CPU dist", "GPU node", "GPU dist", "match?");
    printf("  %s\n", std::string(70, '-').c_str());

    int first_diverge = -1;
    for (int i = 0; i < max_show; ++i) {
        vid_t cn = cpu_trace.expanded_nodes[i];
        float cd = cpu_trace.expanded_dists[i];
        vid_t gn = gpu_trace_nodes[i];
        float gd = gpu_trace_dists[i];
        bool match = (cn == gn);
        printf("  %4d | %10u %12.4f | %10u %12.4f | %s\n",
               i, cn, cd, gn, gd, match ? "OK" : "DIFFER");
        if (!match && first_diverge < 0) first_diverge = i;
    }

    if (first_diverge >= 0)
        printf("\n  First divergence at iteration %d\n", first_diverge);
    else if (max_show > 0)
        printf("\n  All %d compared iterations match!\n", max_show);

    // Result overlap
    std::unordered_set<vid_t> cpu_set(cpu_trace.results.begin(), cpu_trace.results.end());
    int overlap = 0;
    for (int i = 0; i < K; ++i)
        if (cpu_set.count(gpu_results[i])) overlap++;
    printf("\n  Result overlap: %d/%d\n", overlap, K);

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_trace_nodes));
    CUDA_CHECK(cudaFree(d_trace_dists));
    CUDA_CHECK(cudaFree(d_num_iters));
    delete[] h_queries;

    return (first_diverge >= 0) ? 1 : 0;
}
