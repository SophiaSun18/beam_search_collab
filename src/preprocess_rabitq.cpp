#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <omp.h>

#include "../include/utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " -base <base.fvecs> -graph <graph.bin> -out <output.bin> -sign <signs.bin>\n";
        return 1;
    }

    std::string base_file, graph_file, out_file, signs_file;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-base" && i + 1 < argc) base_file = argv[++i];
        else if (arg == "-graph" && i + 1 < argc) graph_file = argv[++i];
        else if (arg == "-out" && i + 1 < argc) out_file = argv[++i];
        else if (arg == "-sign" && i + 1 < argc) signs_file = argv[++i];
    }

    if (base_file.empty() || graph_file.empty() || out_file.empty() || signs_file.empty()) {
        std::cerr << "Error: -base, -graph, -out, and -sign are required.\n";
        return 1;
    }

    size_t n_base, dim;
    float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim);
    std::cout << "Loaded " << n_base << " base vectors, dim=" << dim << "\n";

    assert(dim % 64 == 0);

    Graph g;
    // g.load_graph_index(graph_file.c_str());
    g.load_qg_index(graph_file.c_str(), n_base, dim);
    std::cout << "Loaded graph with " << g.n << " nodes, degree=" << g.degree << "\n";

    // Initialize the random signs table
    size_t padded_dim = 1ULL << (size_t)std::ceil(std::log2(dim));
    // std::vector<float> signs(padded_dim);
    float scale = 1.0f / std::sqrt(static_cast<float>(padded_dim));
    
    // hardcoded sign table for now to match SymQG
    std::vector<float> signs = {
        0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, 
        0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 
        -0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
        0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 
        0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 
        -0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 
        0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 
        0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
        0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 
        -0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
        -0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, 
        -0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, 0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
        0.0883883f, 0.0883883f, 0.0883883f, -0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f, 0.0883883f
    };

    // std::vector<float> signs = {
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, 
    //     -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f, -0.0883883f
    // };

    // std::uniform_int_distribution<int> bernoulli(0, 1);
    // std::random_device rdd;
    // std::mt19937_64 gen(rdd());
    // for (size_t i = 0; i < padded_dim; ++i) {
    //     signs[i] = static_cast<float>((2 * bernoulli(gen)) - 1) /
    //                 std::sqrt(static_cast<float>(padded_dim));
    // }

    // Config the table: [bitcode_words][triple_x][factor_dq][factor_vq]
    int bitcode_words = dim / 64; 
    int factor_words = 2; // 3 floats = 12 bytes, fits in 16 bytes (2 uint64_t)
    int words_per_neighbor = bitcode_words + factor_words;
    int words_per_node = g.degree * words_per_neighbor;

    std::vector<uint64_t> table(g.n * words_per_node, 0);    // store all data in 64 bit words

    // Main preprocessing loop
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t u = 0; u < g.n; u++) {
        alignas(64) float u_rotated[padded_dim] = {};
        for (size_t i = 0; i < padded_dim; i++) {
            u_rotated[i] = base_vectors[u * dim + i];
        }
        rotate_vector(u_rotated, u_rotated, signs.data(), dim, padded_dim);

        const uint32_t* neighbors = g.get_neighbors(u);
        for (size_t i = 0; i < g.degree; i++) {
            uint32_t v = neighbors[i];
           alignas(64) float v_rotated[padded_dim] = {};
            for (size_t i = 0; i < padded_dim; i++) {
                v_rotated[i] = base_vectors[v * dim + i];
            }
            rotate_vector(v_rotated, v_rotated, signs.data(), dim, padded_dim);

            // Compute residual
            alignas(64) float residual[dim] = {};
            float sqr_r = 0.0f; 
            for (size_t d = 0; d < dim; d++) {
                residual[d] = v_rotated[d] - u_rotated[d];
                sqr_r += residual[d] * residual[d];
            }
            float r_norm = std::sqrt(sqr_r) + 1e-10f;

            // Compute binary code and store into the table
            uint64_t* neighbor_data = &table[u * words_per_node + i * words_per_neighbor];
            int bin_x_sum = 0;
            for (size_t d = 0; d < dim; d++) {
                if (residual[d] > 0.0f) {
                    neighbor_data[d / 64] |= (1ULL << (d % 64));
                    bin_x_sum++;
                }
            }

            // in SymQG: pack_codes to prepare for fastscan
            // currently skipped for simplicity

            float fac_norm = 1.F / std::sqrt(static_cast<float>(dim));

            // Compute signed quantized vectors, fac_x0 and fac_x1
            float fac_x0 = 0.0f;
            float fac_x1 = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float signed_val = (residual[d] > 0.0f) ? 1.0f : -1.0f;
                fac_x0 += residual[d] * signed_val * fac_norm;
                fac_x1 += u_rotated[d] * signed_val;
            }
            fac_x0 /= r_norm;
            fac_x1 *= fac_norm;

            // Compute 3 factors and store into the table
            long double x_x0 = static_cast<long double>(r_norm) / fac_x0;
            float* f_ptr = reinterpret_cast<float*>(neighbor_data + bitcode_words);
            f_ptr[0] = static_cast<float>((r_norm * r_norm) + (2 * x_x0 * fac_x1));   // triple_x
            f_ptr[1] = static_cast<float>(-2 * x_x0 * fac_norm);    // factor_dq
            f_ptr[2] = static_cast<float>(f_ptr[1] * (bin_x_sum * 2 - static_cast<int>(dim)));  // factor_vq
        }
    }

    write_bin<uint64_t>(out_file.c_str(), table.data(), g.n, words_per_node);
    write_bin<float>(signs_file.c_str(), signs.data(), padded_dim, 1);
    std::cout << "Wrote rabitQ table to " << out_file << "\n";
    std::cout << "Wrote signs to " << signs_file << "\n";

    delete[] base_vectors;
    return 0;
}
