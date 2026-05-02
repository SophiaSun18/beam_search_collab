#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

#include "../include/cpu_graph.hpp"

// ── Walsh-Hadamard Transform (in-place, power-of-2 length) ─────────────────

static void wht_inplace(float* data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

// ── Rotate: element-wise sign flip then WHT ────────────────────────────────

static void rotate_vector(const float* src, float* dst,
                           const float* signs, int dim, int padded_dim) {
    for (int i = 0; i < dim; ++i)
        dst[i] = src[i] * signs[i];
    for (int i = dim; i < padded_dim; ++i)
        dst[i] = 0.0f;
    wht_inplace(dst, padded_dim);
}

// ── Fastscan encode: write one 4-bit nibble for (lane, codebook) ───────────

static void fastscan_encode(uint8_t* packed, int lane_id,
                             int codebook_idx, uint8_t nibble) {
    int pair_idx     = codebook_idx >> 1;
    int nibble_group = codebook_idx & 1;
    int lane16       = lane_id & 15;
    int byte_idx     = ((lane16 & 7) << 1) | (lane16 >> 3);
    int offset       = pair_idx * 32 + nibble_group * 16 + byte_idx;
    if (lane_id < 16)
        packed[offset] = static_cast<uint8_t>(
            (packed[offset] & 0xf0) | (nibble & 0x0f));
    else
        packed[offset] = static_cast<uint8_t>(
            (packed[offset] & 0x0f) | ((nibble & 0x0f) << 4));
}

// ── Compute 4-bit nibble from binary code for one codebook group ───────────

static uint8_t binary_to_nibble(const uint8_t* bits, int codebook_idx) {
    int base = codebook_idx * 4;
    uint8_t n = 0;
    // bit layout matches fastscan LUT: bit3→dim+0, bit2→dim+1, bit1→dim+2, bit0→dim+3
    if (bits[base + 0]) n |= 8;
    if (bits[base + 1]) n |= 4;
    if (bits[base + 2]) n |= 2;
    if (bits[base + 3]) n |= 1;
    return n;
}

// ── Find medoid (node closest to centroid) ─────────────────────────────────

static vid_t find_medoid(const CPUGraph& g) {
    int dim = g.dim;
    std::vector<double> centroid(dim, 0.0);
    for (size_t i = 0; i < g.num_nodes; ++i) {
        auto v = g.vect(static_cast<vid_t>(i));
        for (int d = 0; d < dim; ++d)
            centroid[d] += v[d];
    }
    for (int d = 0; d < dim; ++d)
        centroid[d] /= static_cast<double>(g.num_nodes);

    vid_t best = 0;
    double best_dist = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        vid_t local_best = 0;
        double local_best_dist = std::numeric_limits<double>::max();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < g.num_nodes; ++i) {
            auto v = g.vect(static_cast<vid_t>(i));
            double dist = 0.0;
            for (int d = 0; d < dim; ++d) {
                double diff = v[d] - centroid[d];
                dist += diff * diff;
            }
            if (dist < local_best_dist) {
                local_best_dist = dist;
                local_best = static_cast<vid_t>(i);
            }
        }
        #pragma omp critical
        {
            if (local_best_dist < best_dist) {
                best_dist = local_best_dist;
                best = local_best;
            }
        }
    }
    return best;
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string base_file, graph_file, out_file;
    uint32_t seed = 12345;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-base"  && i + 1 < argc) base_file  = argv[++i];
        else if (arg == "-graph" && i + 1 < argc) graph_file = argv[++i];
        else if (arg == "-out"   && i + 1 < argc) out_file   = argv[++i];
        else if (arg == "-seed"  && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
    }

    if (base_file.empty() || graph_file.empty() || out_file.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " -base <base.fvecs> -graph <graph.bin> -out <codebook>"
                  << " [-seed <uint32>]\n";
        return 1;
    }

    // ── load graph ──────────────────────────────────────────────────────
    CPUGraph g;
    g.load(graph_file, base_file);

    int dim        = g.dim;
    int degree     = g.degree;
    size_t n       = g.num_nodes;
    int padded_dim = 1;
    while (padded_dim < dim) padded_dim <<= 1;
    int bitcode_words  = padded_dim / 64;
    int num_codebooks  = padded_dim / 4;
    int batches_per_node = (degree + 31) / 32;

    std::cout << "dim=" << dim << " padded_dim=" << padded_dim
              << " degree=" << degree << " n=" << n
              << " bitcode_words=" << bitcode_words << "\n";

    // ── generate random signs ───────────────────────────────────────────
    std::vector<float> signs(padded_dim);
    {
        std::mt19937_64 gen(seed);
        std::uniform_int_distribution<int> coin(0, 1);
        float scale = 1.0f / std::sqrtf(static_cast<float>(padded_dim));
        for (int i = 0; i < padded_dim; ++i)
            signs[i] = static_cast<float>(2 * coin(gen) - 1) * scale;
    }

    // ── find entry point (medoid) ───────────────────────────────────────
    vid_t entry_point = find_medoid(g);
    std::cout << "Entry point (medoid): " << entry_point << "\n";

    // ── per-node codebook row layout (in floats) ────────────────────────
    //   [raw_vector: dim]
    //   [packed_codes: bitcode_words * 2 * degree]  (uint64 reinterpreted as pairs of float)
    //   [factors: 3 * degree]                       (triple_x | factor_dq | factor_vq)
    //   [neighbor_IDs: degree]                      (uint32 cast to float)
    size_t code_off   = static_cast<size_t>(dim);
    size_t factor_off = code_off + static_cast<size_t>(bitcode_words) * 2 * degree;
    size_t nbr_off    = factor_off + 3 * static_cast<size_t>(degree);
    size_t row_floats = nbr_off + static_cast<size_t>(degree);

    std::cout << "Row layout: raw_vec=" << code_off << " codes=" << (factor_off - code_off)
              << " factors=" << (nbr_off - factor_off) << " nbrs=" << degree
              << " total=" << row_floats << " floats\n";

    // ── allocate output buffer ──────────────────────────────────────────
    std::vector<float> rows(n * row_floats, 0.0f);

    // ── main preprocessing loop ─────────────────────────────────────────
    std::cout << "Quantizing edges...\n";
    #pragma omp parallel
    {
        std::vector<float> u_rot(padded_dim);
        std::vector<float> v_rot(padded_dim);
        std::vector<float> residual(padded_dim);
        std::vector<uint8_t> bits(padded_dim);

        #pragma omp for schedule(dynamic, 64)
        for (size_t u = 0; u < n; ++u) {
            float* row = rows.data() + u * row_floats;

            // raw vector
            auto u_vec = g.vect(static_cast<vid_t>(u));
            std::memcpy(row, u_vec.data(), dim * sizeof(float));

            // rotate center
            rotate_vector(u_vec.data(), u_rot.data(), signs.data(), dim, padded_dim);

            auto nbrs = g.nbrs(static_cast<vid_t>(u));

            // pointer to packed codes region (as bytes)
            uint8_t* codes_ptr = reinterpret_cast<uint8_t*>(row + code_off);
            // zero out codes region
            std::memset(codes_ptr, 0,
                        static_cast<size_t>(bitcode_words) * degree * sizeof(uint64_t));

            // pointer to factors region: [triple_x * degree][factor_dq * degree][factor_vq * degree]
            float* fac_triple = row + factor_off;
            float* fac_dq     = fac_triple + degree;
            float* fac_vq     = fac_dq + degree;

            // pointer to neighbor IDs region
            float* nbr_ids = row + nbr_off;

            for (int i = 0; i < degree; ++i) {
                vid_t v = nbrs[i];

                // store neighbor ID as float
                uint32_t nbr_u32 = v;
                float nbr_f;
                std::memcpy(&nbr_f, &nbr_u32, sizeof(float));
                nbr_ids[i] = nbr_f;

                if (v == UINT32_MAX) {
                    fac_triple[i] = FLT_MAX;
                    fac_dq[i]     = 0.0f;
                    fac_vq[i]     = 0.0f;
                    continue;
                }

                // rotate neighbor
                auto v_vec = g.vect(v);
                rotate_vector(v_vec.data(), v_rot.data(), signs.data(), dim, padded_dim);

                // compute residual = v_rot - u_rot (only first dim components)
                float sqr_r = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    residual[d] = v_rot[d] - u_rot[d];
                    sqr_r += residual[d] * residual[d];
                }
                float r_norm = std::sqrt(sqr_r) + 1e-10f;

                // binary quantize the residual
                int bin_sum = 0;
                for (int d = 0; d < padded_dim; ++d) {
                    bits[d] = (d < dim && residual[d] > 0.0f) ? 1 : 0;
                    bin_sum += bits[d];
                }

                // fastscan pack: find which batch of 32 this neighbor belongs to
                int batch = i / 32;
                int lane  = i % 32;
                uint8_t* batch_ptr = codes_ptr
                    + batch * padded_dim * 4;   // 4 bytes per dim per batch of 32

                for (int cb = 0; cb < num_codebooks; ++cb) {
                    uint8_t nibble = binary_to_nibble(bits.data(), cb);
                    fastscan_encode(batch_ptr, lane, cb, nibble);
                }

                // compute factors
                float fac_norm = 1.0f / std::sqrt(static_cast<float>(dim));
                float fac_x0 = 0.0f;
                float fac_x1 = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float sign_val = (residual[d] > 0.0f) ? 1.0f : -1.0f;
                    fac_x0 += residual[d] * sign_val * fac_norm;
                    fac_x1 += u_rot[d] * sign_val;
                }
                fac_x0 /= r_norm;
                fac_x1 *= fac_norm;

                long double x_x0 = static_cast<long double>(r_norm) / fac_x0;

                fac_triple[i] = static_cast<float>(
                    static_cast<long double>(r_norm) * r_norm + 2.0L * x_x0 * fac_x1);
                fac_dq[i] = static_cast<float>(-2.0L * x_x0 * fac_norm);
                fac_vq[i] = fac_dq[i] * static_cast<float>(2 * bin_sum - dim);
            }

            if ((u & 0xfffff) == 0 && u > 0)
                std::cout << "  " << u << " / " << n << " nodes done\n";
        }
    }

    // ── write codebook file ─────────────────────────────────────────────
    std::cout << "Writing codebook to " << out_file << "\n";
    {
        std::ofstream fout(out_file, std::ios::binary);
        if (!fout) {
            std::cerr << "Cannot open output file: " << out_file << "\n";
            return 1;
        }

        // header: entry point
        fout.write(reinterpret_cast<const char*>(&entry_point), sizeof(uint32_t));

        // per-node rows
        fout.write(reinterpret_cast<const char*>(rows.data()),
                   static_cast<std::streamsize>(n * row_floats * sizeof(float)));

        // footer: signs table
        fout.write(reinterpret_cast<const char*>(signs.data()),
                   static_cast<std::streamsize>(padded_dim * sizeof(float)));
    }

    std::cout << "Done. Codebook: " << out_file
              << " (" << (sizeof(uint32_t) + n * row_floats * sizeof(float)
                          + padded_dim * sizeof(float))
              << " bytes)\n";

    return 0;
}
