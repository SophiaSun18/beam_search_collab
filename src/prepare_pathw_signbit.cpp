#include <iostream>
#include <memory>
#include <string>

#include "../include/utils.h"

using namespace std;

namespace {

void print_usage(const char* prog) {
    cerr << "Usage:\n"
         << prog
         << " -base <base.fvecs> -graph <graph_index> -out <signbit.bin>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    string base_file;
    string graph_file;
    string out_file;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-base" && i + 1 < argc) {
            base_file = argv[++i];
        } else if (arg == "-graph" && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (arg == "-out" && i + 1 < argc) {
            out_file = argv[++i];
        } else {
            cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (base_file.empty() || graph_file.empty() || out_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    size_t n_base = 0;
    size_t dim = 0;
    unique_ptr<float[]> base_vectors(read_fvecs(base_file.c_str(), n_base, dim));

    Graph g;
    g.load_graph_index(graph_file.c_str());

    if (g.n != n_base) {
        cerr << "Error: base vectors and graph node count mismatch: n_base=" << n_base
             << ", graph_n=" << g.n << "\n";
        return 1;
    }

    const size_t packed_dim = (dim + 31) / 32;
    const size_t signbit_width = g.degree * packed_dim;
    unique_ptr<uint32_t[]> signbit_table(new uint32_t[n_base * signbit_width]);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t node = 0; node < n_base; ++node) {
        const float* node_vec = base_vectors.get() + node * dim;
        const vidType* neighbors = g.get_neighbors(static_cast<int>(node));
        uint32_t* node_out = signbit_table.get() + node * signbit_width;
        for (size_t nbr_idx = 0; nbr_idx < g.degree; ++nbr_idx) {
            const float* nbr_vec = base_vectors.get() + static_cast<size_t>(neighbors[nbr_idx]) * dim;
            get_sign_bit_vec(node_vec, nbr_vec, node_out + nbr_idx * packed_dim, dim);
        }
    }

    write_bin<uint32_t>(out_file.c_str(), signbit_table.get(), n_base, signbit_width);
    cout << "Wrote pathw signbit file: " << out_file << "\n";
    cout << "  nodes=" << n_base << ", dim=" << dim
         << ", degree=" << g.degree << ", packed_dim=" << packed_dim << "\n";
    return 0;
}
