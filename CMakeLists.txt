cmake_minimum_required(VERSION 3.24)
project(beam_search_collab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include project root so "include/..." paths work from src/
include_directories(${CMAKE_SOURCE_DIR})

# ── Third-party interface libraries ──────────────────────────────────────────

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/3rd)

# ── GPU targets ──────────────────────────────────────────────────────────────

add_executable(gpu_cli src/gpu_cli.cu)
target_link_libraries(gpu_cli PRIVATE nlohmann_json)
target_compile_options(gpu_cli PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)

# add_executable(gpu_microbenchmark src/gpu_microbenchmark.cu)
# target_compile_options(gpu_microbenchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)

# ── Tests ────────────────────────────────────────────────────────────────────

enable_testing()

add_executable(test_topk_merge unittest/test_topk_merge.cu)
target_compile_options(test_topk_merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_merge COMMAND test_topk_merge)

add_executable(test_bitonic_sort unittest/test_bitonic_sort.cu)
target_compile_options(test_bitonic_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_bitonic_sort COMMAND test_bitonic_sort)

add_executable(test_topk_no_dup unittest/test_topk_no_dup.cu)
target_compile_options(test_topk_no_dup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_no_dup COMMAND test_topk_no_dup)

add_executable(test_bitonic_sort_full unittest/test_bitonic_sort_full.cu)
target_compile_options(test_bitonic_sort_full PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_bitonic_sort_full COMMAND test_bitonic_sort_full)


# ── Output directory ─────────────────────────────────────────────────────────

# set_target_properties(
#     gpu_search gpu_microbenchmark
#     PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
# )
