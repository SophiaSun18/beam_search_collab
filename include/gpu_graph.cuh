#pragma once

#include "cpu_graph.hpp"
#include <cuda_runtime.h>

#define GPU_GRAPH_CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

/**
 *  @struct Span
 *  @brief  Lightweight non-owning view over a contiguous array.
 *
 *  Device-compatible alternative to std::span. Stores only a pointer
 *  and a length (16 bytes on 64-bit), no vtable.
 *
 *  @tparam T  Element type (may be const-qualified).
 */
template <typename T>
struct Span {
    T*     ptr;
    size_t len;

    __device__ __host__ T&       operator[](size_t i)       { return ptr[i]; }
    __device__ __host__ const T& operator[](size_t i) const { return ptr[i]; }
    __device__ __host__ size_t   size()               const { return len; }
    __device__ __host__ T*       data()                     { return ptr; }
    __device__ __host__ const T* data()               const { return ptr; }
};

/**
 *  @class GPUGraph
 *  @brief Device-resident graph. Owns CUDA allocations for vectors and edges.
 *
 *  Constructed from a CPUGraph by copying its data to the device.
 *  Non-copyable; freed on destruction.
 */
class GPUGraph {
public:
    /**
     * @brief Allocate device memory and copy graph data from host.
     *
     * @param cpu  Source graph whose vectors and edges are uploaded.
     */
    explicit GPUGraph(const CPUGraph& cpu);
    /** @brief Free device-side vectors and edges. */
    ~GPUGraph();

    /**
     * @brief Return the neighbor list of a node as a device-side span.
     *
     * @param node  Node index, must be < num_nodes.
     * @return Span of length degree pointing into device edge memory.
     */
    __device__ __host__ Span<vid_t>       nbrs(vid_t node);
    /**
     * @brief Return the feature vector of a node as a device-side span.
     *
     * @param node  Node index, must be < num_nodes.
     * @return Span of length dim pointing into device vector memory.
     */
    __device__ __host__ Span<const float> vect(vid_t node) const;

    __host__ __device__ int dim()    const { return dim_; }
    __host__ __device__ int degree() const { return degree_; }

    GPUGraph(const GPUGraph&) = delete;
    GPUGraph& operator=(const GPUGraph&) = delete;

private:
    float* d_vects_ = nullptr;
    vid_t* d_edges_   = nullptr;
    size_t num_nodes_  = 0;
    int    dim_        = 0;
    int    degree_     = 0;
};

// ── out-of-line implementations ─────────────────────────────────────────────

inline GPUGraph::GPUGraph(const CPUGraph& cpu)
    : num_nodes_(cpu.num_nodes), dim_(cpu.dim), degree_(cpu.degree)
{
    size_t vec_bytes  = num_nodes_ * dim_ * sizeof(float);
    size_t edge_bytes = num_nodes_ * degree_ * sizeof(vid_t);

    GPU_GRAPH_CUDA_CHECK(cudaMalloc(&d_vects_, vec_bytes));
    GPU_GRAPH_CUDA_CHECK(cudaMemcpy(d_vects_, cpu.vectors.data(),
                                    vec_bytes, cudaMemcpyHostToDevice));

    GPU_GRAPH_CUDA_CHECK(cudaMalloc(&d_edges_, edge_bytes));
    GPU_GRAPH_CUDA_CHECK(cudaMemcpy(d_edges_, cpu.edges.data(),
                                    edge_bytes, cudaMemcpyHostToDevice));

    GPU_GRAPH_CUDA_CHECK(cudaDeviceSynchronize());
}

inline GPUGraph::~GPUGraph() {
    cudaFree(d_vects_);
    cudaFree(d_edges_);
}

__device__ __host__ inline Span<vid_t> GPUGraph::nbrs(vid_t node) {
    return {d_edges_ + static_cast<size_t>(node) * degree_,
            static_cast<size_t>(degree_)};
}

__device__ __host__ inline Span<const float> GPUGraph::vect(vid_t node) const {
    return {d_vects_ + static_cast<size_t>(node) * dim_,
            static_cast<size_t>(dim_)};
}
