#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cstdint>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include "data_loader.hpp"
#include "cpu_graph.hpp"
#include "gpu_graph.cuh"
#include "gpu_search_base.cuh"
#include "gpu_search_warp.cuh"
#include "gpu_search_rabitq.cuh"

using json = nlohmann::json;

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess)                                            \
        throw std::runtime_error(std::string(__FILE__) + ":" +          \
            std::to_string(__LINE__) + " " + #call + " failed: " +      \
            cudaGetErrorString(err_));                                  \
} while (0)

static float compute_recall(const vid_t* predicted, const int* groundtruth,
                             size_t nq, int topk, int gt_k) {
    size_t correct = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<int> gt_set;
        gt_set.reserve(static_cast<size_t>(gt_k));
        for (int k = 0; k < gt_k; k++) {
            gt_set.insert(groundtruth[i * gt_k + k]);
        }
        std::unordered_set<int> seen;
        seen.reserve(static_cast<size_t>(topk));
        for (int j = 0; j < topk; j++) {
            int pred = static_cast<int>(predicted[i * topk + j]);
            if (!seen.insert(pred).second) continue;
            if (gt_set.count(pred) != 0) correct++;
        }
    }
    return static_cast<float>(correct) / (nq * topk);
}

// ── entry point computation ─────────────────────────────────────────────────

static vid_t compute_entry_point(const CPUGraph& graph) {
    size_t n = graph.num_nodes;
    int dim = graph.dim;
    const float* data = graph.vectors.data();

    std::vector<float> centroid(dim, 0.0f);
    float inv = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; i++) {
        const float* row = data + i * dim;
        for (int d = 0; d < dim; d++)
            centroid[d] += row[d] * inv;
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
        if (dist < best_dist) {
            best_dist = dist;
            best = static_cast<vid_t>(i);
        }
    }
    return best;
}

// ── kernels ─────────────────────────────────────────────────────────────────

static constexpr unsigned CLI_BLOCK_SIZE = 256;

template <unsigned BX>
__global__ void kernel_base_search(
    GPUGraphView graph, GPUBaseSearchCfg cfg,
    int nq, int dim,
    const float* __restrict__ d_queries,
    vid_t* __restrict__ d_results, int K)
{
    int qid = blockIdx.x;
    if (qid >= nq) return;

    extern __shared__ char smem[];
    char* ptr = smem;
    GPUBaseSearch<BX, SharedBitonicTopK<BX, 1>, 1, HashFilter> searcher(ptr, cfg, graph);
    __syncthreads();
    searcher.search(d_queries + qid * dim, d_results + qid * K, K);
}

template <unsigned NY, typename TopK, unsigned REG, typename Filter = HashFilter>
__global__ void kernel_warp_search(
    GPUGraphView graph, GPUBaseSearchCfg cfg,
    int nq, int dim,
    const float* __restrict__ d_queries,
    vid_t* __restrict__ d_results, int K)
{
    int qid = blockIdx.x * NY + threadIdx.y;
    if (qid >= nq) return;

    extern __shared__ char smem[];
    char* ptr = smem;
    GPUWarpSearch<NY, TopK, REG, Filter> searcher(ptr, cfg, graph);
    __syncwarp();
    searcher.search(d_queries + qid * dim, d_results + qid * K, K);
}

template <unsigned NY, typename TopK, unsigned REG, typename Filter = HashFilter>
__global__ void kernel_rabitq_search(
    GPUGraphView graph, GPURabitqView rabitq, GPURabitqSearchCfg cfg,
    int nq, int dim,
    const float* __restrict__ d_queries,
    vid_t* __restrict__ d_results, int K)
{
    int qid = blockIdx.x * NY + threadIdx.y;
    if (qid >= nq) return;

    extern __shared__ char smem[];
    char* ptr = smem;
    GPURabitqSearch<NY, TopK, REG, Filter> searcher(ptr, cfg, graph, rabitq);
    __syncwarp();
    searcher.search(d_queries + qid * dim, d_results + qid * K, K);
}

// ── run functions ───────────────────────────────────────────────────────────

template <unsigned NY, typename TopK, unsigned REG, typename Filter = HashFilter>
static void run_warp_impl(const GPUGraph& gpu_graph, const GPUBaseSearchCfg& cfg,
                          int nq, int dim, int K,
                          const float* d_queries, vid_t* d_results)
{
    size_t smem = GPUWarpSearch<NY, TopK, REG, Filter>::shared_mem_bytes(cfg, dim, gpu_graph.degree());
    int num_blocks = (nq + NY - 1) / NY;
    dim3 block(WARP_SIZE, NY);

    CUDA_CHECK(cudaFuncSetAttribute((kernel_warp_search<NY, TopK, REG, Filter>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem)));

    constexpr int NRUNS = 5;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NRUNS; r++) {
        kernel_warp_search<NY, TopK, REG, Filter><<<num_blocks, block, smem>>>(
            gpu_graph.view(), cfg, nq, dim, d_queries, d_results, K);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count() / NRUNS;
    std::cout << "  time: " << elapsed * 1000.0 << " ms"
              << "  qps: " << nq / elapsed << std::endl;
}

// Auto-dispatch REGX: register topk if topk_size fits (64/128/256/512), else shared memory.
template <unsigned NY, unsigned REG, unsigned REGX, typename Filter>
static void run_warp_regx(const GPUGraph& gpu_graph, const GPUBaseSearchCfg& cfg,
                          int nq, int dim, int K,
                          const float* d_queries, vid_t* d_results)
{
    if constexpr (REGX >= REG && REGX % REG == 0) {
        run_warp_impl<NY, RegBitonicTopK<REGX, REG>, REG, Filter>(
                gpu_graph, cfg, nq, dim, K, d_queries, d_results);
    } else {
        // REGX too small for this REG — fall back to shared memory.
        run_warp_impl<NY, SharedBitonicTopK<32, REG>, REG, Filter>(
                gpu_graph, cfg, nq, dim, K, d_queries, d_results);
    }
}

// Dispatch REGX from topk_size for a given REG.
template <unsigned NY, unsigned REG, typename Filter>
static void run_warp_reg(const GPUGraph& gpu_graph, const GPUBaseSearchCfg& cfg,
                         int nq, int dim, int K,
                         const float* d_queries, vid_t* d_results)
{
    unsigned regx = cfg.topk_size / 32;
    if (regx * 32 == cfg.topk_size) {
        switch (regx) {
            case  2: 
                run_warp_regx<NY, REG,  2, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
                return;
            case  4: 
                run_warp_regx<NY, REG,  4, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
                return;
            case  8: 
                run_warp_regx<NY, REG,  8, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
                return;
            case 16: 
                run_warp_regx<NY, REG, 16, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
                return;
            default: break;
        }
    }
    run_warp_impl<NY, SharedBitonicTopK<32, REG>, REG, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
}

// Auto-dispatch REG from graph degree: smallest power-of-2 where 32*REG >= degree.
template <unsigned NY, typename Filter = HashFilter>
static void run_warp(const GPUGraph& gpu_graph, const GPUBaseSearchCfg& cfg,
                     int nq, int dim, int K,
                     const float* d_queries, vid_t* d_results)
{
    unsigned degree = gpu_graph.degree();
    unsigned reg = 1;
    while (32 * reg < degree) reg *= 2;
    switch (reg) {
        case  1: 
            run_warp_reg<NY,  1, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        case  2: 
            run_warp_reg<NY,  2, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        case  4: 
            run_warp_reg<NY,  4, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        case  8: 
            run_warp_reg<NY,  8, Filter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        default:
            std::cerr << "degree " << degree << " too large (max 256)" << std::endl;
            exit(1);
    }
}

template <unsigned BX>
static void run_base_impl(const GPUGraph& gpu_graph, const GPUBaseSearchCfg& cfg,
                           int nq, int dim, int K,
                           const float* d_queries, vid_t* d_results)
{
    size_t smem = GPUBaseSearch<BX, SharedBitonicTopK<BX, 1>, 1, HashFilter>::shared_mem_bytes(cfg, dim, gpu_graph.degree());
    int num_blocks = nq;

    CUDA_CHECK(cudaFuncSetAttribute(kernel_base_search<BX>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem)));

    constexpr int NRUNS = 5;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NRUNS; r++) {
        kernel_base_search<BX><<<num_blocks, BX, smem>>>(
            gpu_graph.view(), cfg, nq, dim, d_queries, d_results, K);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count() / NRUNS;
    std::cout << "  time: " << elapsed * 1000.0 << " ms"
              << "  qps: " << nq / elapsed << std::endl;
}

// Auto-dispatch BX: largest power-of-2 <= topk_size, capped at 256.
static void run_base(const GPUGraph& gpu_graph, const GPUBaseSearchCfg& cfg,
                     int nq, int dim, int K,
                     const float* d_queries, vid_t* d_results)
{
    unsigned bx = 32;
    while (bx * 2 <= cfg.topk_size && bx < 256) bx *= 2;
    switch (bx) {
        case  32: 
            run_base_impl< 32>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        case  64: 
            run_base_impl< 64>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        case 128: 
            run_base_impl<128>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
        case 256: 
            run_base_impl<256>(gpu_graph, cfg, nq, dim, K, d_queries, d_results); 
            break;
    }
}

template <unsigned NY, typename TopK, unsigned REG, typename Filter = HashFilter>
static void run_rabitq_impl(const GPUGraph& gpu_graph, const GPURabitq& gpu_rabitq,
                            const GPURabitqSearchCfg& cfg,
                            int nq, int dim, int K,
                            const float* d_queries, vid_t* d_results)
{
    size_t smem = GPURabitqSearch<NY, TopK, REG, Filter>::shared_mem_bytes(
        cfg, dim, gpu_graph.degree());
    int num_blocks = (nq + NY - 1) / NY;
    dim3 block(WARP_SIZE, NY);

    CUDA_CHECK(cudaFuncSetAttribute(
        (kernel_rabitq_search<NY, TopK, REG, Filter>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    constexpr int NRUNS = 5;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NRUNS; r++) {
        kernel_rabitq_search<NY, TopK, REG, Filter><<<num_blocks, block, smem>>>(
            gpu_graph.view(), gpu_rabitq.view(), cfg, nq, dim,
            d_queries, d_results, K);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count() / NRUNS;
    std::cout << "  time: " << elapsed * 1000.0 << " ms"
              << "  qps: " << nq / elapsed << std::endl;
}

// Auto-dispatch REG from graph degree (same logic as warp search).
template <unsigned NY, typename Filter = HashFilter>
static void run_rabitq(const GPUGraph& gpu_graph, const GPURabitq& gpu_rabitq,
                       const GPURabitqSearchCfg& cfg,
                       int nq, int dim, int K,
                       const float* d_queries, vid_t* d_results)
{
    unsigned degree = gpu_graph.degree();
    unsigned reg = 1;
    while (32 * reg < degree) reg *= 2;
    switch (reg) {
        case 1:
            run_rabitq_impl<NY, SharedBitonicTopK<32, 1>, 1, Filter>(
                gpu_graph, gpu_rabitq, cfg, nq, dim, K, d_queries, d_results);
            break;
        case 2:
            run_rabitq_impl<NY, SharedBitonicTopK<32, 2>, 2, Filter>(
                gpu_graph, gpu_rabitq, cfg, nq, dim, K, d_queries, d_results);
            break;
        case 4:
            run_rabitq_impl<NY, SharedBitonicTopK<32, 4>, 4, Filter>(
                gpu_graph, gpu_rabitq, cfg, nq, dim, K, d_queries, d_results);
            break;
        case 8:
            run_rabitq_impl<NY, SharedBitonicTopK<32, 8>, 8, Filter>(
                gpu_graph, gpu_rabitq, cfg, nq, dim, K, d_queries, d_results);
            break;
        default:
            fprintf(stderr, "degree %u too large (max 256)\n", degree);
            exit(1);
    }
}

// ── main ────────────────────────────────────────────────────────────────────

int main() {
    // parse single JSON object from stdin
    json j;
    try {
        j = json::parse(std::cin, nullptr, true, true);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }

    // ── data config ─────────────────────────────────────────────────────
    if (!j.contains("data")) {
        std::cerr << "Missing \"data\" key" << std::endl;
        return 1;
    }
    auto& jd = j["data"];
    std::string graph_file    = jd.at("graph");
    std::string base_file     = jd.at("base");
    std::string query_file    = jd.at("query");
    std::string gt_file       = jd.at("gt");
    std::string codebook_file = jd.value("codebook", "");

    // load CPU data
    CPUGraph cpu_graph;
    cpu_graph.load(graph_file, base_file);
    int dim = cpu_graph.dim;
    size_t npoints = cpu_graph.num_nodes;

    size_t nq = 0, query_dim = 0;
    float* h_queries = read_fvecs(query_file.c_str(), nq, query_dim);

    size_t gt_nq = 0, gt_k_sz = 0;
    int* h_gt = read_ivecs(gt_file.c_str(), gt_nq, gt_k_sz);
    int gt_k = static_cast<int>(gt_k_sz);

    // build GPU graph
    GPUGraph gpu_graph(cpu_graph);
    vid_t entry_point = compute_entry_point(cpu_graph);

    // upload queries
    float* d_queries;
    CUDA_CHECK(cudaMalloc(&d_queries, nq * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, nq * dim * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "data: " << npoints << " points, dim=" << dim
              << ", degree=" << cpu_graph.degree
              << ", queries=" << nq
              << ", entry=" << entry_point << std::endl;

    // ── search config ───────────────────────────────────────────────────
    if (!j.contains("search")) {
        std::cerr << "Missing \"search\" key" << std::endl;
        return 1;
    }
    auto& js = j["search"];
    std::string mode = js.value("mode", "base");
    int K            = js.value("K", 100);

    // allocate results
    vid_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, nq * K * sizeof(vid_t)));
    std::vector<vid_t> h_results(nq * K);

    std::cout << "search: mode=" << mode << " K=" << K << std::endl;

    // ── dispatch ────────────────────────────────────────────────────────
    if (mode == "rabitq") {
        if (codebook_file.empty()) {
            std::cerr << "rabitq mode requires \"codebook\" in data config" << std::endl;
            return 1;
        }
        CPURabitqData cpu_rq = load_rabitq_codebook(
            codebook_file, npoints, dim, cpu_graph.degree);
        GPURabitq gpu_rq(cpu_rq);

        vid_t rq_entry = (cpu_rq.entry_point != UINT32_MAX)
                       ? cpu_rq.entry_point : entry_point;

        GPURabitqSearchCfg rq_cfg;
        rq_cfg.topk_size      = js.value("topk_size", 256u);
        rq_cfg.filter_capacity = js.value("filter_capacity", 1024u);
        rq_cfg.entry_point    = rq_entry;
        rq_cfg.max_iterations = js.value("max_iterations", 1024);
        rq_cfg.reset_interval = js.value("reset_interval", 16);

        run_rabitq<1>(gpu_graph, gpu_rq, rq_cfg, nq, dim, K, d_queries, d_results);
    } else {
        GPUBaseSearchCfg cfg;
        cfg.topk_size      = js.value("topk_size", 256u);
        cfg.filter_capacity = js.value("filter_capacity", 1024u);
        cfg.entry_point    = entry_point;
        cfg.max_iterations = js.value("max_iterations", 1024);
        cfg.reset_interval = js.value("reset_interval", 16);

        if (cfg.topk_size < static_cast<unsigned>(K)) {
            std::cerr << "topk_size (" << cfg.topk_size
                      << ") must be >= K (" << K << ")" << std::endl;
            return 1;
        }

        if (mode == "base") {
            run_base(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "warp1") {
            run_warp<1>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "warp2") {
            run_warp<2>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "warp4") {
            run_warp<4>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "warp8") {
            run_warp<8>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "bwarp1") {
            run_warp<1, BloomFilter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "bwarp2") {
            run_warp<2, BloomFilter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "bwarp4") {
            run_warp<4, BloomFilter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else if (mode == "bwarp8") {
            run_warp<8, BloomFilter>(gpu_graph, cfg, nq, dim, K, d_queries, d_results);
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            return 1;
        }
    }

    // ── results ─────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, nq * K * sizeof(vid_t), cudaMemcpyDeviceToHost));
    float recall = compute_recall(h_results.data(), h_gt, nq, K, gt_k);
    std::cout << "recall@" << K << ": " << recall << std::endl;

    // cleanup
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_results));
    delete[] h_queries;
    delete[] h_gt;

    return 0;
}
