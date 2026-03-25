# Stable Mainline Structure

这份目录现在按第一性原理收成稳定主线的最小阅读结构，不再保留 helper umbrella、`*.inc.cuh` 和其它中间层。

## 1. 先看哪三个文件

稳定搜索主线只需要先看三处：

1. `code/search_bitonic.cuh`
2. `code/search_bitonic/defaults.cuh`
3. `code/search_bitonic/kernel.cuh`

对应关系很直接：

- `code/search_bitonic.cuh`
  - 唯一外部入口
  - 只做头文件引入
- `code/search_bitonic/defaults.cuh`
  - 唯一配置文件
  - 只放稳定主线仍然真实生效的宏和常量
- `code/search_bitonic/kernel.cuh`
  - 唯一 kernel 主文件
  - 同时承载 helper 例程和 `BeamSearch` 主体

也就是说，`search_bitonic/` 子目录里现在只有两个内部文件：

- `code/search_bitonic/defaults.cuh`
- `code/search_bitonic/kernel.cuh`

非主线工具已经移出 `code/`，放到：

- `tools/rabitq_audit.hpp`
- `tools/rabitq_signbit_build.hpp`

## 2. 现在为什么这样收

之前那套结构虽然已经把外部入口、配置和 kernel 主体分开了，但内部仍然残留了两层不必要的拆分：

- `helpers.cuh -> helpers_select/helpers_distance/helpers_hash/helpers_signbit_io`
- `kernel.cuh -> kernel_setup.inc/kernel_loop.inc/kernel_tail.inc`

对稳定主线来说，这些层并不提供真实的算法分叉，只是在增加跳转路径。当前只有一条执行形状，所以最自然的结构就是：

- 配置单独放一处
- 所有 kernel 相关代码顺序放一处

这比“一个 kernel 再拆多个 include 片段”更符合当前目录的真实复杂度。

## 3. `kernel.cuh` 里现在怎么读

`code/search_bitonic/kernel.cuh` 现在按顺序组织成两大段：

1. helper 例程
2. `BeamSearch` kernel

helper 例程本身也按使用顺序排：

1. frontier / expander 选择
2. candidate 选择与 merge
3. exact / partial-L2 距离
4. hash 插入策略
5. sign-bit 文件 IO

然后直接进入：

- `BeamSearch(...)`

kernel 主体内部也不再跨文件跳转，而是顺序展开：

1. block/query setup
2. shared state 初始化
3. 初始 frontier seed
4. 主循环
5. 结果写回和统计

现在读稳定主线时，不再需要在 `helpers_*` 和 `kernel_*.inc` 之间反复切换。

## 4. 配置文件里还剩什么

`code/search_bitonic/defaults.cuh` 只保留两类东西：

- build shape：
  - `BEAM_SIZE`
  - `SEARCH_WIDTH`
  - `BITLEN`
- 稳定主线固定常量：
  - `HASH_INSERT_MAX_PROBES`
  - `HASH_INSERT_LIMIT_START_FILL`
  - `RABITQ_EARLY_STOP_STALL_ITERS`
  - `RABITQ_EARLY_STOP_WARMUP_ITERS`
  - `RABITQ_MAX_ITER_CAP`
  - `RABITQ_STALL_IMPROVE_EPS_ABS`
  - `RABITQ_STALL_IMPROVE_EPS_REL`
  - `REFLOW_TOPN`
  - `REFLOW_POP_BATCH`
  - `REFLOW_ANCHOR_EXACT`

以及当前稳定主线需要的少量 `inline constexpr` 缓冲尺寸常量。

现在稳定目录不再通过更多宏去描述“可能存在的分支”；它只描述当前真正运行的主线。

## 5. host 侧怎么读

稳定主线的 host orchestration 还是看：

- `code/beam_gpu_base.cu`

建议阅读顺序：

1. `code/beam_gpu_base.cu`
2. `code/search_bitonic.cuh`
3. `code/search_bitonic/defaults.cuh`
4. `code/search_bitonic/kernel.cuh`

如果只关心 kernel 执行形状，直接从：

1. `code/search_bitonic/defaults.cuh`
2. `code/search_bitonic/kernel.cuh`

开始就够了。

## 6. 工具模式现在在哪里

程序入口 `code/main.cu` 仍然保留三种模式：

- stable benchmark
- RabitQ audit
- raw sign-bit build

但后两者的实现已经不再占据主线代码区，而是移到了：

- `tools/rabitq_audit.hpp`
- `tools/rabitq_signbit_build.hpp`

所以现在的 `code/` 更接近“稳定主线 + host/runtime 骨架”，而不是“主线和工具实现混在一起”。

## 7. 当前结构达成了什么

当前这次收缩以后，稳定主线的搜索代码已经满足：

- 一个外部入口
- 一个配置文件
- 一个 kernel 主文件

同时还保持：

- `code/` 里只放主线源码和它的 host/runtime 骨架
- `build/` 里放本地产物和 runner 缓存
- 远端仍然通过 `run_stable_telerun.sh` 做权威验证

## 8. 当前已验证基线

最近稳定验证结果：

- `results/stable_sift10m_b300_20260323_032128.summary.txt`

指标：

```text
CONFIG: qps=88978.1 recall=0.97677 avg_dc=5319 avg_hops=47 kernel_ms=11.134
```

后续只要继续沿这条稳定主线改代码，就继续用：

- 本地 `make -j1 sm89`
- 远端 `./run_stable_telerun.sh`

来判断有没有真实回退。
