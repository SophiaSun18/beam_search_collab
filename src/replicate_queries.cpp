#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "data_loader.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: replicate_queries <input.fvecs|ivecs> <output> <repeats> <format>\n"
                  << "  format: fvecs | ivecs\n";
        return 1;
    }

    const char* input  = argv[1];
    const char* output = argv[2];
    int repeats        = std::atoi(argv[3]);
    std::string fmt    = argv[4];

    if (repeats <= 0) {
        std::cerr << "repeats must be positive\n";
        return 1;
    }

    if (fmt == "fvecs") {
        size_t n = 0, d = 0;
        float* data = read_fvecs(input, n, d);
        int dim = static_cast<int>(d);
        size_t total = n * static_cast<size_t>(repeats);

        std::vector<float> tiled(total * d);
        for (int r = 0; r < repeats; r++)
            std::copy(data, data + n * d, tiled.data() + r * n * d);

        write_fvecs(output, tiled.data(), total, dim);
        std::cout << n << " x " << repeats << " = " << total
                  << " vectors (dim=" << dim << ") -> " << output << "\n";
        delete[] data;
    } else if (fmt == "ivecs") {
        size_t n = 0, d = 0;
        int* data = read_ivecs(input, n, d);
        int dim = static_cast<int>(d);
        size_t total = n * static_cast<size_t>(repeats);

        std::vector<int> tiled(total * d);
        for (int r = 0; r < repeats; r++)
            std::copy(data, data + n * d, tiled.data() + r * n * d);

        write_ivecs(output, tiled.data(), total, dim);
        std::cout << n << " x " << repeats << " = " << total
                  << " vectors (dim=" << dim << ") -> " << output << "\n";
        delete[] data;
    } else {
        std::cerr << "Unknown format: " << fmt << " (use fvecs or ivecs)\n";
        return 1;
    }

    return 0;
}
