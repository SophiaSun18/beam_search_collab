#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

inline float* read_fvecs(const char* filename, size_t& n, size_t& d) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot open ") + filename);
    int dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    n = file_size / (dim * sizeof(float) + sizeof(int));
    d = dim;
    float* data = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        int dd;
        in.read(reinterpret_cast<char*>(&dd), sizeof(int));
        in.read(reinterpret_cast<char*>(data + i * d), d * sizeof(float));
    }
    return data;
}

inline int* read_ivecs(const char* filename, size_t& n, size_t& d) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot open ") + filename);
    int dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    n = file_size / ((dim + 1) * sizeof(int));
    d = dim;
    int* data = new int[n * d];
    for (size_t i = 0; i < n; i++) {
        int dd;
        in.read(reinterpret_cast<char*>(&dd), sizeof(int));
        in.read(reinterpret_cast<char*>(data + i * d), d * sizeof(int));
    }
    return data;
}

inline void write_fvecs(const char* filename, const float* data, size_t n, int dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error(std::string("Cannot open ") + filename);
    for (size_t i = 0; i < n; i++) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(data + i * dim),
                  dim * sizeof(float));
    }
}

inline void write_ivecs(const char* filename, const int* data, size_t n, int dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error(std::string("Cannot open ") + filename);
    for (size_t i = 0; i < n; i++) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(data + i * dim),
                  dim * sizeof(int));
    }
}
