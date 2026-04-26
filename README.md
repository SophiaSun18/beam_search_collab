# GPU Beam Search

GPU-accelerated graph-based approximate nearest neighbor search with beam search.

## Build

Requires CUDA 12+ and a C++20 compiler. Tested on sm_89 (RTX 4090).

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces:

- `gpu_cli` — unified search CLI (JSON config from stdin)
- `test_*` — unit tests

## Usage

`gpu_cli` reads a single JSON object from stdin with `"data"` and `"search"` keys:

```bash
./build/gpu_cli <<EOF
{
    "data": {
        "graph": "path/to/graph.bin",
        "base":  "path/to/base.fvecs",
        "query": "path/to/query.fvecs",
        "gt":    "path/to/groundtruth.ivecs"
    },
    "search": {
        "mode": "base",
        "K": 100,
        "topk_size": 256,
        "bitlen": 10,
        "max_iterations": 1024,
        "reset_interval": 16
    }
}
EOF
```

Configs can be composed from separate files:

```bash
./build/gpu_cli <<EOF
{ "data": $(cat data.json), "search": $(cat search.json) }
EOF
```

### Search modes

| Mode | Description |
|------|-------------|
| `base` | Block-level search. 256 threads per query. All warps compute L2 in parallel. |
| `warp4` | Warp-level search. 4 queries per block. Each warp runs independently with async prefetch. |
| `warp8` | Warp-level search. 8 queries per block. |

### Search config fields

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `"base"` | Search mode: `base`, `warp4`, `warp8` |
| `K` | `100` | Number of nearest neighbors to return |
| `topk_size` | `256` | Beam buffer size (must be >= block size for base, >= 32*REG for warp) |
| `bitlen` | `10` | Hash filter size = 2^bitlen entries |
| `max_iterations` | `1024` | Maximum beam search iterations |
| `reset_interval` | `16` | Forgettable hash table reset interval |

## Graph files

The CLI expects pre-built graph files in the ScaleANN `.bin` format:

- **graph.bin** — `int32 num_nodes`, `int32 degree`, flat `uint32[]` adjacency
- **base.fvecs** — standard fvecs format (per-vector `int32 dim` + `dim` floats)
- **query.fvecs** — same format
- **groundtruth.ivecs** — standard ivecs format

Graph building is not yet integrated. Use [ScaleANN](https://github.com/scalable-anns/ScaleANN) to build graphs, then point `gpu_cli` at the output files.

## Tests

```bash
cd build && ctest
```

| Test | What it covers |
|------|---------------|
| `test_bitonic_sort` | Warp-level bitonic sort (ascending/descending) |
| `test_bitonic_sort_full` | Full bitonic sort from unsorted input |
| `test_topk_merge` | SharedBitonicTopK merge correctness |
| `test_topk_no_dup` | Interleaved pick/merge yields no duplicate node IDs |

## Project structure

### Active (new architecture)

| File | Description |
|------|-------------|
| `include/l2.cuh` | L2 distance primitives (`l2_warp_distance`) |
| `include/topk.cuh` | `SharedBitonicTopK` — sorted priority queue with merge/pick/flush |
| `include/hash_filter.cuh` | `HashFilter` — linear-probing hash set for visited nodes |
| `include/gpu_graph.cuh` | `GPUGraph` (owning) + `GPUGraphView` (immutable, kernel-safe) |
| `include/cpu_graph.hpp` | `CPUGraph` — host-side graph with file I/O |
| `include/gpu_base_search.cuh` | `GPUBaseSearch` — block-level L2 beam search |
| `include/gpu_warp_search.cuh` | `GPUWarpSearch` — warp-level L2 beam search with async prefetch |
| `src/gpu_cli.cu` | Unified CLI binary |
| `3rd/nlohmann/json.hpp` | Vendored JSON parser (v3.11.3) |

### Legacy (not yet migrated)

| File | Description |
|------|-------------|
| `include/search_bitonic.cuh` | Old block-level L2 kernel (replaced by `GPUBaseSearch`) |
| `include/search_rabitq.cuh` | RAbitQ beam search kernel — not yet migrated to new architecture |
| `include/search_tensor_rabitq.cuh` | Tensor-core RAbitQ kernel — not yet migrated |
| `include/utils.cuh` | Legacy utilities — rabitq distance estimation, query preparation, etc. |
| `include/utils.h` | Host-side utilities — file I/O, recall, distance functions |
| `include/index.hpp` | Legacy `IndexGraph` with file loading (used by old `main.cu`) |
| `include/qg.hpp` | `QuantizationGraph` for RAbitQ |
| `src/main.cu` | Old CLI entry point (uses `IndexGraph::search`) |
| `src/beam_gpu_base.cu` | Old host-side launcher for `BeamSearch` kernel |
| `src/beam_gpu_rabitq.cu` | Host-side launcher for RAbitQ kernels |
| `src/gpu_microbenchmark.cu` | Microbenchmarks for individual GPU primitives |

### Not yet integrated

- **Graph building** — Vamana graph construction currently requires [ScaleANN](https://github.com/scalable-anns/ScaleANN). Integrating the builder would improve reproducibility.
- **RAbitQ search** — The quantized search kernels (`search_rabitq.cuh`, `search_tensor_rabitq.cuh`) work but haven't been refactored into the new `GPUBaseSearch`/`GPUWarpSearch` architecture.
- **Microbenchmarks** — `gpu_microbenchmark.cu` is functional but not included in the default build.
