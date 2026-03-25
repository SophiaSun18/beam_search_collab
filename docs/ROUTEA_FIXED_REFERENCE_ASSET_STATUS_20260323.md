# Route A Fixed-Reference Asset Status

## What Was Built

已在远端真实数据上生成了一版 **projected fixed-reference** Route A 资产。

远端 build 记录：

- `research_engine_reflow_routea_mainline_20260323/results/routea_node_build_sift10m_20260323_105616.telerun.log`
- `research_engine_reflow_routea_mainline_20260323/results/routea_node_build_sift10m_20260323_105616.summary.txt`

关键结果：

- dataset: `sift10m`
- graph: `/home/sunzeai/proj/.data/sift10m/graph-d32`
- output prefix: `/home/sunzeai/proj/.data/sift10m/graph-d32.projfixed`
- projection file: `/home/sunzeai/proj/.data/sift10m/graph-d32.rabitq.proj.bin`
- build time: about `11.06 sec`
- save status: `OK`

---

## Produced Files

远端产物路径：

- `/home/sunzeai/proj/.data/sift10m/graph-d32.projfixed.routea.centroid.f32.bin`
- `/home/sunzeai/proj/.data/sift10m/graph-d32.projfixed.routea.node.packed.bin`
- `/home/sunzeai/proj/.data/sift10m/graph-d32.projfixed.routea.node.meta.txt`

这套资产的契约是：

- reference object: global centroid
- geometry space: projected space
- signbit object: `sign(x_proj - centroid_proj)`
- sidecar record: `[sign words][triple][factor_dq]`
- `factor_vq`: not stored; derive later from `factor_dq` and sign sum

---

## Loader Skeleton

Route A host 侧已经接入了这套资产的加载骨架，但默认不会自动生效。

当前显式启用方式：

```bash
export ROUTEA_NODE_PREFIX=/home/sunzeai/proj/.data/sift10m/graph-d32.projfixed
```

对应 loader 位置：

- `research_engine_reflow_routea_mainline_20260323/code/beam_gpu_base.cu`

当前行为：

- 如果设置了 `ROUTEA_NODE_PREFIX`，host 会尝试加载 `.routea.*` 资产进 GPU cache
- 但当前 search kernel 还没有消费这套资产
- 因此，这一步目前只是把 host/cache 层打通

---

## Why This Matters

这一步解决的是之前最关键的契约问题：

- 旧资产：`edge residual (v-u)`
- 新资产：`node residual to fixed centroid`

只有新契约下，下面这条 CPU/GPU 分工才数学上成立：

- CPU: `query -> proj -> quantize -> LUT`
- GPU: `node signbit + triple + factor_dq -> approx -> shortlist -> exact`

---

## Next Kernel Work

下一步不应该再回去碰旧 residual 主线，而应该直接沿这套新资产继续：

1. host 侧把 query 预处理结果上传到 GPU
2. kernel 改成消费 fixed-reference node sidecars
3. approx 排序改为 fixed-reference Route A 路径
4. exact / frontier merge 先保持保守语义

最先要接的三个量是：

- centroid
- per-query LUT
- per-node signbit / triple / factor_dq
