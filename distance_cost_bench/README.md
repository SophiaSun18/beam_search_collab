# Distance Cost Benchmarks

This directory contains standalone microbenchmarks for measuring the cost of
distance evaluation in graph-based ANN search.

The main goal is to compare exact L2 distance evaluation with RaBitQ distance
estimation on CPU and GPU. The numbers are used to isolate the distance-compute
part of search cost from other overheads such as beam maintenance, queue
updates, neighbor loading, and synchronization.

## Benchmark Types

There are two benchmark paths.

### Primitive Distance Benchmarks

These benchmarks run repeated distance computations directly:

- `distance_cpu`
- `distance_gpu`

They measure the basic per-evaluation latency of exact L2 and RaBitQ distance
estimation under a controlled loop or kernel.

### Access-Trace Replay Benchmarks

These benchmarks use access traces collected from real search runs. The search
program first records which distance operations happen for each query, then the
replay program executes only those distance operations again.

Trace exporters:

- `trace_cpu_base`: CPU exact-distance graph search trace exporter
- `trace_cpu_rabitq`: CPU RaBitQ graph search trace exporter
- `trace_gpu`: GPU exact-distance and RaBitQ graph search trace exporter

Trace replay programs:

- `replay_cpu`
- `replay_gpu`

Each trace row records the query id, search step, operation type, accessed node,
and active neighbor mask. Replaying these traces gives a distance-cost view that
matches the real search access pattern while excluding most non-distance search
logic.

## Remote Runs

The `telerun/` scripts package this directory, compile the CPU and GPU binaries
on the remote GPU machine, export access traces, and replay them.

Example:

```bash
DATASETS=sift RUN_TRACE_REPLAY=1 ./telerun/run_distance_cost_telerun.sh
```

The scripts are intended for datasets already available on the remote machine,
such as SIFT, GIST, and Deep1M.
