# Route A Mainline

这个目录是从
`research_engine_reflow_stable_mainline_20260318`
复制出来的独立实验线。

目的只有一个：

- 冻结当前 stable 目录
- 在这个新目录里允许更大幅度落地 Route A
- 不再受“必须只做小修小改”的约束

## 边界

冻结目录：
- `research_engine_reflow_stable_mainline_20260318`

后续所有更大改动：
- 只在 `research_engine_reflow_routea_mainline_20260323` 里做

这意味着：

- stable 目录继续保留当前可验证基线
- 这个目录允许重写 query prep、重排 sidecar contract、改 shortlist / exact admission 语义
- 两条线不再混用

## 目录

- `code/search_bitonic.cuh`：唯一外部入口
- `code/search_bitonic/defaults.cuh`：唯一配置文件
- `code/search_bitonic/kernel.cuh`：唯一 kernel 主文件
- `code/beam_gpu_base.cu`：host 侧加载、query/sidecar/launch
- `tools/rabitq_audit.hpp`：公式/旋转/asset 审计
- `tools/rabitq_signbit_build.hpp`：raw sign-bit 构建
- `build/`：本地构建产物
- `results/`：这个实验线自己的结果
- `run_routea_telerun.sh`：这个实验线唯一远端入口

## 起始快照

这个目录复制时对应的 stable 基线是：

- `QPS=94495.6`
- `RECALL=0.97457`
- `AVG_DC=3612`
- `AVG_HOPS=48`
- `KERNEL_MS=10.488`

对应结果文件在冻结目录里：

- `research_engine_reflow_stable_mainline_20260318/results/stable_sift10m_b300_20260323_082037.summary.txt`

## 当前设计约束

这条实验线仍然从当前主线三件套出发：

- 一个外部入口
- 一个配置文件
- 一个 kernel 主文件

但这里不再把“维持 stable 形状”当成第一优先级。
这里的第一优先级变成：

1. 建立更强但语义自洽的 Route A 查询路径
2. 重新设计 approx / exact 的职责边界
3. 用远端 `telerun` 做闭环验证

## 本地编译

```bash
cd code
make -j1 sm89
```


## Route A Asset Build

新增了一个固定参考系的 Route A 资产构建模式：

```bash
cd code
make -j1 sm89
./../build/beam_search_gpu <data.fvecs> <graph_prefix> --build-routea-node-sidecar
```

如果要用投影 sidecar：

```bash
./../build/beam_search_gpu <data.fvecs> <graph_prefix>   --build-routea-node-sidecar   --routea-use-proj   --routea-proj-file <graph_prefix>.rabitq.proj.bin
```

默认输出文件：

- `<prefix>.routea.centroid.f32.bin`
- `<prefix>.routea.node.packed.bin`
- `<prefix>.routea.node.meta.txt`

其中 `packed.bin` 的每个节点记录布局为：

- `[sign words][triple][factor_dq]`

`factor_vq` 不单独落盘，约定为：

- `factor_vq = factor_dq * sign_sum(signbit)`

## 远端运行

```bash
./run_routea_telerun.sh
```

默认仍跑 `sift10m + beam=300`，但结果会写进这个目录自己的 `results/`，不再污染冻结的 stable 目录。
