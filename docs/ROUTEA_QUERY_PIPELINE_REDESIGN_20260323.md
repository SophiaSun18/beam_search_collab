# Route A Query Pipeline Redesign

## Goal

在 `research_engine_reflow_routea_mainline_20260323` 里，按更激进的 Route A 架构推进：

1. CPU 负责离线图侧 sidecar 构建。
2. GPU 查询路径只保留真正热循环里必须存在的工作。
3. 继续复用现有 beam/frontier/exact merge 主干，不在第一刀就重写整条搜索语义。

---

## What Is Reused

保留并复用当前主线里已经稳定的部分：

- `BeamSearch` 的 frontier / candidate / merge 主干
- 当前 `filter-then-refine` 的整体节奏
- 当前 exact distance 和 partial-L2 路径
- 现有 signbit / triple / factor sidecar 加载器

第一阶段不直接做：

- 全程 approx 驱动 frontier
- 直接缩 exact budget
- projected-space 资产切换

---

## Critical Contract Fix

用户给出的目标架构里，有一处需要修正：

> 纯 `CPU per-query LUT` 只在图侧编码不依赖当前 expander 时才成立。

而当前已经审计过的资产契约不是这个形状。当前资产是：

- signbit: `sign(v - u)`
- triple / factor_dq / factor_vq: 也匹配 raw residual `v - u`

也就是说，当前图侧 sidecar 的参考中心是 **当前扩展点 `u`**，不是一个固定的全局中心。

因此，真正进入近似公式的查询量必须是：

- `q_r = q - u`

而不是单纯的：

- `q`
- `q_rot`

这意味着：

1. 当前资产契约下，不能把完整的 residual LUT 预先在 CPU 上按 `query-only` 一次性做好。
2. 如果要完全实现“CPU per-query LUT + GPU 直接查表”，就必须先把图侧 sidecar 契约改成固定参考系，而不是当前的 edge-residual 参考系。
3. 在继续复用现有边残差资产的前提下，第一阶段应当把可下放到 CPU 的工作和必须保留在 GPU 的工作重新划清。

---

## Corrected Route A Shape

### Phase A: CPU Offline Build

CPU 离线图构建仍然负责：

- 旋转
- 量化
- signbit
- triple
- factor_dq
- packed code

但这些 sidecar 必须与实际 kernel 使用的参考系严格一致。

在当前 Route A 第一阶段，仍然先保持：

- raw residual sidecars

而不直接切 projected-space。

### Phase B: Host Runtime Prep

Host 运行时负责：

- 加载 sidecars
- 把分散的 sidecar 重排为更适合 GPU 热路径的连续布局

第一阶段的目标布局是每条边一个连续记录：

- `sign_words[4]`
- `triple`
- `factor_dq`

`factor_vq` 不再单独存储。它可以在 kernel 里由：

- `factor_dq`
- `popcount(sign_words)`

恢复出来。

这一步的收益是：

1. 把当前分散的多数组访问收束成一次连续边记录读取。
2. 为后续 LUT / stronger approx 路径提供稳定的热数据契约。

### Phase C: GPU Query Path

GPU 热循环中，第一阶段仍然保留当前 beam/frontier 主干。

但近似评分路径改造成：

1. 每个 lane 连续读取单条边的 approx sidecar。
2. 使用该边 sidecar 的 sign words 做 cheap score。
3. 使用 `triple + factor_dq + derived factor_vq` 生成更接近 CPU RaBitQ 的 compensated approx。
4. 继续走 filter-then-refine。

第一阶段先不把 exact admission 语义改成更激进版本。

---

## Implementation Order

### Step 1

把边侧 sidecar 收束成单一连续结构：

- `StableRabitqEdgeApprox24`

包含：

- 4 个 sign words
- 1 个 triple
- 1 个 factor_dq

### Step 2

在 kernel 里改成从 `StableRabitqEdgeApprox24` 读取近似数据，而不是分别读：

- `d_sign_bit`
- `d_rabitq_triple`
- `d_rabitq_factor_dq`
- `d_rabitq_factor_vq`

### Step 3

保留当前 shortlist/exact/merge 主干，只替换 approx score 的输入契约。

### Step 4

只有在 Step 1-3 通过本地编译和远端回归后，才进入：

- per-expander residual query descriptor
- LUT
- packed fastscan

---

## Non-Goals For This First Landing

这一轮明确不做：

1. 直接把用户草案里的 `CPU per-query LUT` 原样照搬进当前 edge-residual 资产契约。
2. 直接切 projected-space sidecars。
3. 直接把 `exact_limit` 大幅压缩。
4. 用兼容层同时维持很多旧分支。

---

## Decision

本轮 Route A 的第一刀是：

> 先把边侧近似数据契约做对并做紧，再在这个契约上继续往 LUT / stronger compensated approx 推进。

这是当前最接近用户目标架构、同时又不违反现有 edge-residual 数学契约的落地方式。
