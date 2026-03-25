#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>

// Stable mainline external entry.
// External users include only this file. Internally it fans into:
// - `search_bitonic/defaults.cuh` as the single config file
// - `search_bitonic/kernel.cuh` as the single kernel main file
#include "search_bitonic/defaults.cuh"
#include "search_bitonic/kernel.cuh"
