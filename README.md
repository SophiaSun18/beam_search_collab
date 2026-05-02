# GPU Beam Search

GPU-accelerated graph-based approximate nearest neighbor search.

## Build

Requires CUDA 12+ and a C++20 compiler. Tested on sm_89 (RTX 4090).

```bash
cmake -B build
cmake --build build -j$(nproc)
```

This produces:

- `build_graph` — host-side Vamana graph builder
- `build_rabitq` — RAbitQ codebook builder (quantizes graph edges)
- `gpu_cli` — GPU search CLI (JSON config from stdin)
- `cpu_cli` — CPU search CLI (same JSON format, reference implementation)
- `test_*` / `prof_*` — unit tests and profiling binaries

## Data Preparation

### HDF5 to project binaries

`script/unpack.py` converts ANN-benchmarks style HDF5 files into the formats consumed by the CLIs:

```bash
uv run script/unpack.py sift-128-euclidean.hdf5 data/sift
```

Generated files:

- `base.fvecs` from dataset `train`
- `query.fvecs` from dataset `test`
- `groundtruth.ivecs` from dataset `neighbors`
- optional `groundtruth_distances.npy` from dataset `distances`

### Build a graph

```bash
./build/build_graph \
  --base data/sift/base.fvecs \
  --graph data/sift/graph.bin \
  --degree 32 \
  --beam-width 128 \
  --alpha 2.0
```

Options:

- `--metric l2|ip` selects the build distance metric
- `--seed` controls insertion order shuffling

### Build a RAbitQ codebook

`build_rabitq` quantizes each edge in the graph for approximate distance estimation:

```bash
./build/build_rabitq \
  -base data/sift/base.fvecs \
  -graph data/sift/graph.bin \
  -out data/sift/codebook.bin \
  -seed 12345
```

The output codebook is required by the `rabitq` search mode. It contains per-edge binary codes in fastscan-packed format, distance estimation factors, and the random sign table.

## Usage

Both `gpu_cli` and `cpu_cli` read a single JSON object from stdin with `"data"` and `"search"` keys:

```bash
./build/gpu_cli <<EOF
{
    "data": {
        "graph": "path/to/graph.bin",
        "base":  "path/to/base.fvecs",
        "query": "path/to/query.fvecs",
        "gt":    "path/to/groundtruth.ivecs",
        "codebook": "path/to/codebook.bin"
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

#### gpu_cli

| Mode | Description |
|------|-------------|
| `base` | Block-level L2 search. 256 threads per query. |
| `warp1` | Warp-level L2 search. 1 query per block. |
| `warp2` | Warp-level L2 search. 2 queries per block. |
| `warp4` | Warp-level L2 search. 4 queries per block. |
| `warp8` | Warp-level L2 search. 8 queries per block. |
| `bwarp1`..`bwarp8` | Same as `warp*` but with Bloom filter instead of hash filter. |
| `rabitq` | Block-level RAbitQ search with fastscan. Requires `codebook` in data config. |

#### cpu_cli

| Mode | Description |
|------|-------------|
| `base` | CPU beam search with exact L2 distances. |
| `rabitq` | CPU beam search with RAbitQ distance estimation. Requires `codebook` in data config. |

### Data config fields

| Field | Required | Description |
|-------|----------|-------------|
| `graph` | yes | Path to graph `.bin` file |
| `base` | yes | Path to base vectors `.fvecs` file |
| `query` | yes | Path to query vectors `.fvecs` file |
| `gt` | yes | Path to groundtruth `.ivecs` file |
| `codebook` | rabitq only | Path to RAbitQ codebook file |

### Search config fields

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `"base"` | Search mode (see tables above) |
| `K` | `100` | Number of nearest neighbors to return |
| `topk_size` | `256` | Beam buffer size |
| `bitlen` | `10` | Hash/Bloom filter size = 2^bitlen entries (gpu_cli only) |
| `max_iterations` | `1024` | Maximum beam search iterations |
| `reset_interval` | `16` | Forgettable hash table reset interval (gpu_cli only) |

## File formats

- **graph.bin** — `int32 num_nodes`, `int32 degree`, flat `uint32[]` adjacency
- **base.fvecs** — standard fvecs (per-vector `int32 dim` + `dim` floats)
- **query.fvecs** — same format
- **groundtruth.ivecs** — standard ivecs format
- **codebook.bin** — `uint32 entry_point`, per-node rows of `[raw_vector | packed_codes | factors | neighbor_IDs]`, then `float[padded_dim]` sign table

## Tests

```bash
cd build && ctest
```

| Test | What it covers |
|------|---------------|
| `test_bitonic_sort` | Warp-level bitonic sort (ascending/descending) |
| `test_bitonic_sort_full` | Full bitonic sort from unsorted input |
| `test_bitonic_sort_reg` | Register-only bitonic sort |
| `test_bitonic_sort_full_reg` | Full register-only bitonic sort |
| `test_topk_merge` | SharedBitonicTopK merge correctness |
| `test_topk_no_dup` | Interleaved pick/merge yields no duplicate node IDs |
| `test_reg_topk_merge` | RegBitonicTopK merge correctness |
| `test_reg_topk_no_dup` | Register topk no-duplicate check |
| `test_reg_topk_pick_order` | Register topk pick ordering |
| `test_hash_filter` | Hash filter insert/contains correctness |

## Project structure

| File | Description |
|------|-------------|
| `src/build_graph.cpp` | Host-side Vamana graph builder |
| `src/build_rabitq.cpp` | RAbitQ codebook builder (per-edge quantization) |
| `src/gpu_cli.cu` | GPU search CLI |
| `src/cpu_cli.cpp` | CPU search CLI |
| `src/replicate_queries.cpp` | Query replication utility |
| `include/cpu_graph.hpp` | `CPUGraph` — host-side graph with file I/O |
| `include/cpu_base_search.hpp` | `CPUBaseSearch` — CPU beam search with exact L2 |
| `include/cpu_rabitq_search.hpp` | `CPURabitqSearch` — CPU beam search with RAbitQ estimation |
| `include/data_loader.hpp` | fvecs/ivecs read/write utilities |
| `include/gpu_graph.cuh` | `GPUGraph` (owning) + `GPUGraphView` (kernel-safe) |
| `include/gpu_base_search.cuh` | `GPUBaseSearch` — block-level L2 beam search |
| `include/gpu_warp_search.cuh` | `GPUWarpSearch` — warp-level L2 beam search |
| `include/gpu_rabitq_search.cuh` | `GPURabitqSearch` — block-level RAbitQ beam search with fastscan |
| `include/rabitq.hpp` | `CPURabitqData` — host-side RAbitQ codebook loader |
| `include/rabitq.cuh` | `GPURabitq` (owning) + `GPURabitqView` (kernel-safe) + `RabitqQuery` |
| `include/l2.cuh` | L2 distance primitives |
| `include/topk.cuh` | Top-K dispatch header |
| `include/topk_shared.cuh` | `SharedBitonicTopK` — shared memory priority queue |
| `include/topk_reg.cuh` | `RegBitonicTopK` — register-only priority queue |
| `include/filter_hash.cuh` | `HashFilter` — linear-probing hash set for visited nodes |
| `include/filter_bloom.cuh` | `BloomFilter` — Bloom filter alternative |
| `include/visit_filter.cuh` | `VisitFilter` concept + includes both filter implementations |
| `3rd/nlohmann/json.hpp` | Vendored JSON parser (v3.11.3) |
