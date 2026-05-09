# GPUWarpSearch Variant Benchmark (ef90a8d)

## Machine

- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM, compute capability 8.9
- **CPU**: Intel Core i7-14700HX, 20 cores / 28 threads, up to 5.5 GHz
- **RAM**: 32 GB
- **OS**: Arch Linux, kernel 6.19.14-arch1-1
- **Driver**: 595.58.03

## Dataset

SIFT-1M: 1,000,000 base vectors, 10,000 queries, dim=128, graph degree=32, K=100

## Results (topk_size=256)

| Mode   | NY | REG | TopK buf | Time (ms) | QPS     | Recall@100 |
|--------|----|-----|----------|-----------|---------|------------|
| warp12 | 1  | 2   | 64       | 179.8     | 55,628  | 0.9682     |
| warp22 | 2  | 2   | 64       | 182.9     | 54,684  | 0.9679     |
| warp42 | 4  | 2   | 64       | 188.9     | 52,944  | 0.9677     |
| warp21 | 2  | 1   | 32       | 191.4     | 52,257  | 0.9678     |
| warp11 | 1  | 1   | 32       | 194.2     | 51,504  | 0.9682     |
| warp44 | 4  | 4   | 128      | 199.2     | 50,196  | 0.9677     |
| warp14 | 1  | 4   | 128      | 200.7     | 49,816  | 0.9682     |
| warp24 | 2  | 4   | 128      | 205.1     | 48,756  | 0.9679     |
| warp82 | 8  | 2   | 64       | 230.9     | 43,313  | 0.9676     |
| warp84 | 8  | 4   | 128      | 257.9     | 38,778  | 0.9676     |
| warp48 | 4  | 8   | 256      | 285.7     | 34,998  | 0.9679     |
| warp88 | 8  | 8   | 256      | 343.5     | 29,113  | 0.9679     |

## Results (topk_size=128)

| Mode   | NY | REG | TopK buf | Time (ms) | QPS     | Recall@100 |
|--------|----|-----|----------|-----------|---------|------------|
| warp22 | 2  | 2   | 64       | 69.4      | 144,081 | 0.9494     |
| warp12 | 1  | 2   | 64       | 70.9      | 141,020 | 0.9511     |
| warp11 | 1  | 1   | 32       | 75.0      | 133,279 | 0.9510     |
| warp14 | 1  | 4   | 128      | 75.9      | 131,709 | 0.9511     |

## Observations

- REG=2 (topk merge buffer = 64) is consistently the fastest across all NY values.
- Fewer warps per block generally improves throughput at topk_size=256 (warp12 > warp22 > warp42 > warp82).
- At topk_size=128, warp22 slightly beats warp12 -- less shared memory per warp lets NY=2 benefit from warp-level parallelism.
- Recall is nearly identical (~0.968 at topk_size=256, ~0.951 at topk_size=128) regardless of NY/REG.
- Higher REG hurts throughput without improving recall at these beam sizes.
