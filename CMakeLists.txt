cmake_minimum_required(VERSION 3.24)

# Use nvcc with GCC 14 as CUDA host compiler (GCC 16's libstdc++ is incompatible with nvcc)
set(GCC14_ROOT "$ENV{HOME}/.local/opt/gcc14/usr")
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "${GCC14_ROOT}/bin/g++")
    set(CMAKE_CUDA_HOST_COMPILER "${GCC14_ROOT}/bin/g++")
endif()

project(beam_search_collab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures (e.g. 80;90, native, all)")
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

add_executable(build_rabitq src/build_rabitq.cpp)
target_include_directories(build_rabitq PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(build_rabitq PRIVATE -O3)
target_link_libraries(build_rabitq PRIVATE OpenMP::OpenMP_CXX)

add_executable(gpu_cli src/gpu_cli.cu)
target_include_directories(gpu_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_cli PRIVATE nlohmann_json)
target_compile_options(gpu_cli PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=--warn-on-local-memory-usage>)

add_executable(gpu_cli_sophia src/gpu_cli_sophia.cu)
target_include_directories(gpu_cli_sophia PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_cli_sophia PRIVATE nlohmann_json)
target_compile_options(gpu_cli_sophia PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# ── Tests ────────────────────────────────────────────────────────────────────

enable_testing()

add_executable(test_topk_shared_merge unittest/test_topk_shared_merge.cu)
target_include_directories(test_topk_shared_merge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_shared_merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_shared_merge COMMAND test_topk_shared_merge)

add_executable(test_topk_shared_bitonic_sort unittest/test_topk_shared_bitonic_sort.cu)
target_include_directories(test_topk_shared_bitonic_sort PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_shared_bitonic_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_shared_bitonic_sort COMMAND test_topk_shared_bitonic_sort)

add_executable(test_topk_shared_bitonic_sort_full unittest/test_topk_shared_bitonic_sort_full.cu)
target_include_directories(test_topk_shared_bitonic_sort_full PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_shared_bitonic_sort_full PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_shared_bitonic_sort_full COMMAND test_topk_shared_bitonic_sort_full)

add_executable(test_topk_shared_no_dup unittest/test_topk_shared_no_dup.cu)
target_include_directories(test_topk_shared_no_dup PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_shared_no_dup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_shared_no_dup COMMAND test_topk_shared_no_dup)

add_executable(test_topk_reg_bitonic_sort unittest/test_topk_reg_bitonic_sort.cu)
target_include_directories(test_topk_reg_bitonic_sort PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_reg_bitonic_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_reg_bitonic_sort COMMAND test_topk_reg_bitonic_sort)

add_executable(test_topk_reg_bitonic_sort_full unittest/test_topk_reg_bitonic_sort_full.cu)
target_include_directories(test_topk_reg_bitonic_sort_full PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_reg_bitonic_sort_full PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_reg_bitonic_sort_full COMMAND test_topk_reg_bitonic_sort_full)

add_executable(test_topk_reg_merge unittest/test_topk_reg_merge.cu)
target_include_directories(test_topk_reg_merge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_reg_merge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_reg_merge COMMAND test_topk_reg_merge)

add_executable(test_topk_reg_pick_order unittest/test_topk_reg_pick_order.cu)
target_include_directories(test_topk_reg_pick_order PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_reg_pick_order PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_reg_pick_order COMMAND test_topk_reg_pick_order)

add_executable(test_topk_reg_no_dup unittest/test_topk_reg_no_dup.cu)
target_include_directories(test_topk_reg_no_dup PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_topk_reg_no_dup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_topk_reg_no_dup COMMAND test_topk_reg_no_dup)

add_executable(test_filter_hash unittest/test_filter_hash.cu)
target_include_directories(test_filter_hash PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_filter_hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_filter_hash COMMAND test_filter_hash)

add_executable(test_fastscan unittest/test_fastscan.cu)
target_include_directories(test_fastscan PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_fastscan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_test(NAME test_fastscan COMMAND test_fastscan)

# ── Profiling ───────────────────────────────────────────────────────────────

add_executable(prof_filter_hash unittest/prof_filter_hash.cu)
target_include_directories(prof_filter_hash PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_filter_hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(prof_l2 unittest/prof_l2.cu)
target_include_directories(prof_l2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_l2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(prof_topk_shared unittest/prof_topk_shared.cu)
target_include_directories(prof_topk_shared PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_topk_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(prof_topk_reg unittest/prof_topk_reg.cu)
target_include_directories(prof_topk_reg PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(prof_topk_reg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(test_rabitq_search_trace unittest/test_rabitq_search_trace.cu)
target_include_directories(test_rabitq_search_trace PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_rabitq_search_trace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

add_executable(test_rabitq_dedup_cpu unittest/test_rabitq_dedup_cpu.cpp)
target_include_directories(test_rabitq_dedup_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_rabitq_dedup_cpu PRIVATE -O3)
