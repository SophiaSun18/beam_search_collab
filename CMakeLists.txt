cmake_minimum_required(VERSION 3.24)
project(beam_search_collab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include project root so "include/..." paths work from src/
include_directories(${CMAKE_SOURCE_DIR})

# ── CPU targets ──────────────────────────────────────────────────────────────

set(CPU_FLAGS -Wall -Wextra -Ofast -march=native -fpic -ftree-vectorize -fexceptions -DEIGEN_DONT_PARALLELIZE -mavx512f)

find_package(OpenMP REQUIRED)

add_executable(cpu_base src/beam_cpu_base.cpp)
target_compile_options(cpu_base PRIVATE ${CPU_FLAGS})
target_link_libraries(cpu_base PRIVATE OpenMP::OpenMP_CXX rt)

add_executable(cpu_pathw src/beam_cpu_pathw.cpp)
target_compile_options(cpu_pathw PRIVATE ${CPU_FLAGS})
target_link_libraries(cpu_pathw PRIVATE OpenMP::OpenMP_CXX rt)

add_executable(cpu_rabitq src/beam_cpu_rabitq.cpp)
target_compile_options(cpu_rabitq PRIVATE ${CPU_FLAGS})
target_link_libraries(cpu_rabitq PRIVATE OpenMP::OpenMP_CXX rt)

add_executable(preprocess_rabitq src/preprocess_rabitq.cpp)
target_compile_options(preprocess_rabitq PRIVATE ${CPU_FLAGS})
target_link_libraries(preprocess_rabitq PRIVATE OpenMP::OpenMP_CXX rt)

# ── GPU targets ──────────────────────────────────────────────────────────────

add_executable(gpu_search src/main.cu src/beam_gpu_base.cu src/beam_gpu_rabitq.cu)
target_compile_options(gpu_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)

add_executable(gpu_microbenchmark src/gpu_microbenchmark.cu)
target_compile_options(gpu_microbenchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)

# ── Output directory ─────────────────────────────────────────────────────────

set_target_properties(
    cpu_base cpu_pathw cpu_rabitq preprocess_rabitq
    gpu_search gpu_microbenchmark
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
