#include <iostream>
#include <cstring>
#include <string>

#include "../include/utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " -base <base.fvecs> -graph <graph.bin> -out <output.bin>\n";
        return 1;
    }

    std::string base_file, graph_file, out_file;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-base" && i + 1 < argc) base_file = argv[++i];
        else if (arg == "-graph" && i + 1 < argc) graph_file = argv[++i];
        else if (arg == "-out" && i + 1 < argc) out_file = argv[++i];
    }

    size_t n_base, dim;
    float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim);
    std::cout << "Loaded " << n_base << " base vectors, dim=" << dim << "\n";

    Graph g;
    g.load_graph_index(graph_file.c_str());

    size_t packed_dim = (dim + 31) / 32;
    size_t sign_bit_vector_size = g.degree * packed_dim;
    std::vector<uint32_t> table(g.n * sign_bit_vector_size, 0);

    std::cout << "Computing sign-bit table: packed_dim=" << packed_dim
              << ", sign_bit_vector_size=" << sign_bit_vector_size << "\n";

    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t u = 0; u < g.n; u++) {
        const vidType* neighbors = g.get_neighbors(u);
        for (size_t j = 0; j < g.degree; j++) {
            vidType v = neighbors[j];
            uint32_t* out = table.data() + u * sign_bit_vector_size + j * packed_dim;
            get_sign_bit_vec(base_vectors + u * dim, base_vectors + v * dim, out, dim);
        }
    }

    write_bin<uint32_t>(out_file.c_str(), table.data(), g.n, sign_bit_vector_size);
    std::cout << "Wrote sign-bit table to " << out_file << "\n";

    delete[] base_vectors;
    return 0;
}
