cmake_minimum_required(VERSION 3.24)
project(beam_search_collab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

# ── Third-party interface libraries ──────────────────────────────────────────

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/3rd)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# ── CLI targets ──────────────────────────────────────────────────────────────

add_executable(build_graph src/build_graph.cpp)
target_compile_options(build_graph PRIVATE -O3)
target_link_libraries(build_graph PRIVATE OpenMP::OpenMP_CXX)

add_executable(cpu_cli src/cpu_cli.cpp)
target_include_directories(cpu_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cpu_cli PRIVATE nlohmann_json)
target_compile_options(cpu_cli PRIVATE -O3)

add_executable(replicate_queries src/replicate_queries.cpp)
target_include_directories(replicate_queries PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(replicate_queries PRIVATE -O3)

add_executable(gpu_cli src/gpu_cli.cu)
target_include_directories(gpu_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_cli PRIVATE nlohmann_json)
target_compile_options(gpu_cli PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)

# ── Tests ────────────────────────────────────────────────────────────────────

enable_testing()

add_executable(test_topk_merge unittest/test_topk_merge.cu)
target_include_directories(test_topk_merge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_merge COMMAND test_topk_merge)

add_executable(test_bitonic_sort unittest/test_bitonic_sort.cu)
target_include_directories(test_bitonic_sort PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_bitonic_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_bitonic_sort COMMAND test_bitonic_sort)

add_executable(test_topk_no_dup unittest/test_topk_no_dup.cu)
target_include_directories(test_topk_no_dup PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_no_dup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_no_dup COMMAND test_topk_no_dup)

add_executable(test_hash_filter unittest/test_hash_filter.cu)
target_include_directories(test_hash_filter PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_hash_filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_hash_filter COMMAND test_hash_filter)

add_executable(test_bitonic_sort_full unittest/test_bitonic_sort_full.cu)
target_include_directories(test_bitonic_sort_full PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_bitonic_sort_full PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_bitonic_sort_full COMMAND test_bitonic_sort_full)

add_executable(test_reg_topk_merge unittest/test_reg_topk_merge.cu)
target_include_directories(test_reg_topk_merge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_reg_topk_merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_reg_topk_merge COMMAND test_reg_topk_merge)

add_executable(test_reg_topk_no_dup unittest/test_reg_topk_no_dup.cu)
target_include_directories(test_reg_topk_no_dup PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_reg_topk_no_dup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_reg_topk_no_dup COMMAND test_reg_topk_no_dup)

# ── Profiling ───────────────────────────────────────────────────────────────

add_executable(prof_hash_filter unittest/prof_hash_filter.cu)
target_include_directories(prof_hash_filter PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_hash_filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(prof_l2 unittest/prof_l2.cu)
target_include_directories(prof_l2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_l2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(prof_topk unittest/prof_topk.cu)
target_include_directories(prof_topk PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_topk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
