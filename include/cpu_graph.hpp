#pragma once

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using vid_t = uint32_t;

/**
 *  @class CPUGraph
 *  @brief Host-resident graph storing per-node feature vectors and edges.
 *
 *  Supports construction from raw data, stream serialization, and
 *  loading from legacy file pairs (.bin graph + .fvecs vectors).
 */
class CPUGraph {
public:
    size_t              num_nodes = 0;
    int                 dim       = 0;
    int                 degree    = 0;
    std::vector<float>  vectors;   // num_nodes * dim
    std::vector<vid_t>  edges;     // num_nodes * degree

    CPUGraph() = default;

    /**
     * @brief Construct from raw data by taking ownership of vectors and edges.
     *
     * @param num_nodes  Number of nodes in the graph.
     * @param dim        Dimensionality of each feature vector.
     * @param degree     Fixed out-degree (neighbors per node).
     * @param vectors    Feature vectors, size must equal num_nodes * dim.
     * @param edges      Adjacency list, size must equal num_nodes * degree.
     */
    CPUGraph(size_t num_nodes, int dim, int degree,
             std::vector<float> vectors, std::vector<vid_t> edges);

    /**
     * @brief Deserialize from own binary format.
     *
     * Reads a 3-int header (num_nodes, dim, degree), then vectors,
     * then edges. The stream must have been written by save().
     *
     * @param in  Binary input stream positioned at the start of the header.
     */
    explicit CPUGraph(std::istream& in);

    /**
     * @brief Serialize to own binary format.
     *
     * Writes a 3-int header (num_nodes, dim, degree), then vectors,
     * then edges. Readable by the stream constructor.
     *
     * @param out  Binary output stream.
     */
    void save(std::ostream& out) const;

    /**
     * @brief Load from a legacy file pair.
     *
     * Reads graph topology from a .bin file (int32 num_nodes, int32 degree,
     * flat vid_t array) and feature vectors from a .fvecs file (per-vector
     * int32 dim prefix followed by dim floats). Overwrites current contents.
     *
     * @param graph_bin   Path to the graph .bin file.
     * @param data_fvecs  Path to the base vectors .fvecs file.
     */
    void load(const std::string& graph_bin, const std::string& data_fvecs);

    /**
     * @brief Return a pointer to the neighbor list of a node.
     *
     * @param node  Node index, must be < num_nodes.
     * @return Pointer to degree contiguous vid_t entries.
     */
    const vid_t* neighbors(vid_t node) const;

    /**
     * @brief Return a pointer to the feature vector of a node.
     *
     * @param node  Node index, must be < num_nodes.
     * @return Pointer to dim contiguous floats.
     */
    const float* vector(vid_t node) const;
};

// ── out-of-line implementations ─────────────────────────────────────────────

inline CPUGraph::CPUGraph(size_t num_nodes, int dim, int degree,
                          std::vector<float> vectors, std::vector<vid_t> edges)
    : num_nodes(num_nodes), dim(dim), degree(degree),
      vectors(std::move(vectors)), edges(std::move(edges)) {}

inline CPUGraph::CPUGraph(std::istream& in) {
    int32_t h[3];
    in.read(reinterpret_cast<char*>(h), sizeof(h));
    num_nodes = static_cast<size_t>(h[0]);
    dim       = h[1];
    degree    = h[2];

    vectors.resize(num_nodes * dim);
    in.read(reinterpret_cast<char*>(vectors.data()),
            num_nodes * dim * sizeof(float));

    edges.resize(num_nodes * degree);
    in.read(reinterpret_cast<char*>(edges.data()),
            num_nodes * degree * sizeof(vid_t));
}

inline void CPUGraph::save(std::ostream& out) const {
    int32_t h[3] = {static_cast<int32_t>(num_nodes), dim, degree};
    out.write(reinterpret_cast<const char*>(h), sizeof(h));
    out.write(reinterpret_cast<const char*>(vectors.data()),
              num_nodes * dim * sizeof(float));
    out.write(reinterpret_cast<const char*>(edges.data()),
              num_nodes * degree * sizeof(vid_t));
}

inline void CPUGraph::load(const std::string& graph_bin,
                           const std::string& data_fvecs) {
    // ── graph edges (.bin: int32 n, int32 deg, flat vid_t array) ──
    {
        std::ifstream f(graph_bin, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open graph file: " + graph_bin);

        int32_t n32, d32;
        f.read(reinterpret_cast<char*>(&n32), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&d32), sizeof(int32_t));
        num_nodes = static_cast<size_t>(n32);
        degree    = d32;

        edges.resize(num_nodes * degree);
        f.read(reinterpret_cast<char*>(edges.data()),
               num_nodes * degree * sizeof(vid_t));

        std::cout << "Loaded graph: " << num_nodes << " nodes, degree "
                  << degree << " from " << graph_bin << "\n";
    }

    // ── base vectors (.fvecs: per-vector int32 dim + dim floats) ──
    {
        std::ifstream f(data_fvecs, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open data file: " + data_fvecs);

        int dim_check;
        f.read(reinterpret_cast<char*>(&dim_check), sizeof(int));

        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t n = file_size / (sizeof(int) + dim_check * sizeof(float));
        assert(n == num_nodes);
        dim = dim_check;

        vectors.resize(num_nodes * dim);
        f.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num_nodes; ++i) {
            int d;
            f.read(reinterpret_cast<char*>(&d), sizeof(int));
            assert(d == dim);
            f.read(reinterpret_cast<char*>(vectors.data() + i * dim),
                   dim * sizeof(float));
        }

        std::cout << "Loaded vectors: " << num_nodes << " x " << dim
                  << " from " << data_fvecs << "\n";
    }
}

inline const vid_t* CPUGraph::neighbors(vid_t node) const {
    return edges.data() + static_cast<size_t>(node) * degree;
}

inline const float* CPUGraph::vector(vid_t node) const {
    return vectors.data() + static_cast<size_t>(node) * dim;
}
