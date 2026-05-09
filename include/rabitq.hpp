#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using vid_t = uint32_t;

static constexpr int RABITQ_BQUERY = 6;

/**
 *  @struct CPURabitqData
 *  @brief  Host-side RAbitQ quantization data (pure data, no methods).
 *
 *  Stores binary codes, factors, and sign table separately from graph
 *  topology (which lives in CPUGraph).  Codes are 1-bit per dimension:
 *  each neighbor has words_per_plane consecutive uint32_t words, where
 *  bit j of word w represents dimension w*32+j.
 */
struct CPURabitqData {
    size_t                num_nodes       = 0;
    int                   dim             = 0;
    int                   padded_dim      = 0;
    int                   degree          = 0;
    int                   words_per_plane = 0;   // padded_dim / 32
    vid_t                 entry_point     = UINT32_MAX;
    std::vector<uint32_t> codes;                 // [num_nodes * degree * words_per_plane]
    std::vector<float>    factors;               // [num_nodes * 3 * degree]
    std::vector<float>    signs;                 // [padded_dim]
};

/**
 *  Load RAbitQ data from a codebook file.
 *
 *  Codebook layout per node:
 *    [raw_vector (dim floats)]
 *    [codes (words_per_plane * degree uint32_t, stored as float-sized slots)]
 *    [factors (3 * degree floats: triple_x | factor_dq | factor_vq)]
 *    [neighbor_IDs (degree uint32 as floats)]
 *
 *  Only codes, factors, and signs are extracted.  Vectors and edges
 *  are expected to come from CPUGraph / GPUGraph.
 */
inline CPURabitqData load_rabitq_codebook(
    const std::string& path, size_t num_nodes, int dim, int degree)
{
    CPURabitqData d;
    d.num_nodes       = num_nodes;
    d.dim             = dim;
    d.degree          = degree;
    d.padded_dim      = 1;
    while (d.padded_dim < dim) d.padded_dim <<= 1;
    d.words_per_plane = d.padded_dim / 32;

    // codebook row layout (units of float/uint32_t — same size)
    size_t code_off   = static_cast<size_t>(dim);
    size_t factor_off = code_off + static_cast<size_t>(d.words_per_plane) * degree;
    size_t nbr_off    = factor_off + 3 * static_cast<size_t>(degree);
    size_t row_off    = nbr_off + static_cast<size_t>(degree);

    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open codebook: " + path);

    fin.read(reinterpret_cast<char*>(&d.entry_point), sizeof(uint32_t));

    d.codes.resize(num_nodes * static_cast<size_t>(degree) * d.words_per_plane);
    d.factors.resize(num_nodes * 3 * static_cast<size_t>(degree));
    d.signs.resize(static_cast<size_t>(d.padded_dim));

    std::vector<float> row(row_off);
    for (size_t i = 0; i < num_nodes; i++) {
        fin.read(reinterpret_cast<char*>(row.data()),
                 static_cast<std::streamsize>(row_off * sizeof(float)));

        std::memcpy(d.codes.data() + i * static_cast<size_t>(degree) * d.words_per_plane,
                    row.data() + code_off,
                    static_cast<size_t>(degree) * d.words_per_plane * sizeof(uint32_t));

        std::memcpy(d.factors.data() + i * 3 * static_cast<size_t>(degree),
                    row.data() + factor_off,
                    3 * static_cast<size_t>(degree) * sizeof(float));
    }

    fin.read(reinterpret_cast<char*>(d.signs.data()),
             static_cast<std::streamsize>(d.padded_dim * sizeof(float)));

    std::cout << "Loaded rabitq codebook: " << d.num_nodes << " nodes, dim=" << d.dim
              << ", degree=" << d.degree << ", padded_dim=" << d.padded_dim
              << " (entry=" << d.entry_point << ")\n";

    return d;
}
