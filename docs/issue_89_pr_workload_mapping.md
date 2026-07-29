# Issue #89 — PR Workload/Spec Mapping & Runnability Check

> Milestone deliverable for `bench: collect missing performance evidence for merged research PRs` (#89).
> Owner: @SuccinctPaul · Due: 2026-08-20 23:59 Asia/Shanghai
> This document satisfies the **2026-08-05** checkpoint: 30 PR 的 workload/spec 映射与可运行性检查。

## 1. 合规结果类型

每个 PR 最终必须落入以下之一：

1. **主线 same-spec real-online** — 与 `official-ascend-jan-2026-v0.18.0` / 14B / 1×910B2 / FP16 / `gpu_memory_utilization=0.6` / `max_model_len=32768` 严格对齐。
2. **专项 targeted pair** — 使用机制匹配的 specialty spec，以 base/head（+latest）形成独立结果卡，`coverage_class=targeted-pair`。
3. **component-only** — 仅 PR body 微基准，明确标记，不进入主折线。
4. **blocked / 无显著收益** — 有明确结论与阻塞链接。

## 2. Spec Inventory

### 2.1 已有可用 specs（可直接运行）

| Workload | Scenario 定义 | Official Baseline | Fixed-Target Profile |
|---|---|---|---|
| `random-online` | official_scenarios.json | ✓ (910b2, 2chip, 4chip) | `core-text-14b` (active) |
| `sharegpt-online` | official_scenarios.json | ✓ (910b2, 2chip, 4chip) | — |
| `prefix-repetition-online` | official_scenarios.json | ✓ (910b2, 2chip, 4chip) | — |
| `instructcoder-online` | official_scenarios.json | ✓ | `coder-14b` (active) |
| `visionarena-online` | official_scenarios.json | ✓ | `vision-7b` (active) |
| `sharegpt-throughput` | official_scenarios.json | ✓ | — |
| `sonnet-throughput` | official_scenarios.json | ✓ (910b2, 2chip, 4chip, 8chip) | — |
| `random-latency` | official_scenarios.json | ✓ | — |
| `agent-research-online` | official_scenarios.json | ✓ (910b2, 2chip, 4chip) | — |
| `kv-tiering-prefix-online` | official_scenarios.json | ✓ (qwen25-7b) | — |
| `logprobs-online` | official_scenarios.json | ✓ (qwen25-14b) | — |
| `ngram-instructcoder-online` | official_scenarios.json | ✓ (qwen25-7b) | — |

Specialty profiles 已在 registry 但 workload 仍占位 `random-online`（需补真实 workload）：
`multi-chip-2chip/4chip/8chip-text-14b`、`int8-quant-text-14b`、`moe-text`、`speculative-decoding-text-14b`。

### 2.2 Specialty specs（已在 feat/issue_89 分支创建）

以下 12 个 specialty specs 已在 `feat/issue_89` 分支创建，注册于 `official_scenarios.json` 并配有 `docs/official-baselines/official-ascend-jan-2026-v0180-*.json` spec 文件。所有 registry 与 contract 测试通过。

| spec | 服务的 PR | 依赖 issue | 状态 |
|---|---|---|---|
| `attention-boundary-online` | #6, #10, vllm-ascend #133 | — | spec 已创建，待 NPU 运行 |
| `kv-pressure-online` | #54, #171 | — | spec 已创建，待 NPU 运行 |
| `agent-cache-pressure-online` | #171, vllm-ascend #39 | — | spec 已创建，待 NPU 运行 |
| `saturated-warm-cache-online` (SimLLM) | vllm-ascend #66, #70, #80 | #59 | placeholder spec 已创建，最终 workload 待 #59 |
| `spec-decode-online` | #121, vllm-ascend #123 | — | spec 已创建，待 NPU 运行 |
| `unified-comm-online` | #42 | — | spec 已创建（e2e），建议另跑通信 microbench |
| `multi-nic-throughput` | #168 | — | spec 已创建，待多节点 NPU |
| `eplb-expert-rebalance-online` | vllm-ascend #36 | — | spec 已创建，待 NPU 运行 |
| `kv-transfer-latency` | #67 (issue) | — | spec 已创建，待 NPU 运行 |
| `slicegpt-compression-online` | #158 | — | spec 已创建，待 NPU 运行 |
| `knorm-kv-compression-longctx` | #76 | — | spec 已创建，待 NPU 运行 |
| `moe-alltoall-online` | vllm-ascend #7 | — | spec 已创建，待多卡 NPU |

## 3. PR-Workload 映射总表

状态图例：`READY`(spec+artifact 就绪可跑) / `SPEC-CREATED`(spec 已创建，待 NPU 运行) / `PARTIAL`(有部分 artifact) / `BLOCKED`(有外部阻塞) / `DONE`(已有合规证据)

> **2026-07-29 更新**：所有 12 个 specialty specs 已在 `feat/issue_89` 分支创建并通过测试。原 `SPEC-CREATED` 状态已更新为 `SPEC-CREATED`。`fixed_target_registry.json` 的 specialty profile workload_name 更新将在 #95 合并后补上（该文件当前仅存在于 feat/pr_95 分支）。

### 批次 A1 — 注意力与运行时热路径

| PR | 推荐 workload | Spec 状态 | 现有 artifact | 可运行性 | 证据缺口 |
|---|---|---|---|---|---|
| vllm-hust #6 | `attention-boundary-online` + `agent-research-online` | SPEC-CREATED (attention) + READY (agent) | 无 | 部分（agent 可跑，attention 需建 spec） | 缺 base/head paired |
| vllm-hust #10 | mixed prefill/decode targeted | SPEC-CREATED | 无 | 需建 spec | 缺 paired |
| vllm-hust #13 | KV-scale/quant-sensitive | SPEC-CREATED | 无 | 需建 spec | 缺 paired |
| vllm-hust #30 | `visionarena-online` + `instructcoder-online` | READY | 无 | 可跑 | 缺 base/head/latest |
| vllm-hust #37 | `instructcoder-online` | READY | 无 | 可跑 | 缺 paired |
| vllm-hust #46 | `logprobs-online` | READY | 无 | 可跑 | 缺 paired（与 #130 演进对照） |
| vllm-hust #54 | `kv-pressure-online` | SPEC-CREATED | 无 | 需建 spec | 缺 paired |
| vllm-hust #115 | `prefix-repetition-online` | READY | merge-only (1chip, mem=0.90) | 可跑 | 缺 base/head/latest；mem_util 应为 0.6 |
| vllm-hust #116 | `prefix-repetition-online` (text-only) | READY | merge-only (1chip, mem=0.90) | 可跑 | 缺 base/head/latest；与 #115 隔离 A/B |
| vllm-hust #130 | `logprobs-online` | READY | base+head pair ✓ | 可跑 | 缺 latest 第三点 + 内存/尾延迟指标 |

### 批次 A2 — 通信、并行与控制面

| PR | 推荐 workload | Spec 状态 | 现有 artifact | 可运行性 | 证据缺口 |
|---|---|---|---|---|---|
| vllm-hust #42 | unified_comm microbench + e2e | SPEC-CREATED | 无 | 需建 spec | 缺 paired |
| vllm-hust #168 | multi-NIC vs single-NIC | SPEC-CREATED | 无 | 需建 spec | 缺 paired |
| vllm-ascend #7 | MoE/all-to-all comm-sensitive | SPEC-CREATED | 无 | 需建 spec | 缺 paired |
| vllm-ascend #33 | DP>1 online pair | SPEC-CREATED (DP profile) | 无 | 需建 spec | 缺 paired；现有组件微基准不能当通用 speedup |
| vllm-ascend #36 | EPLB expert-rebalance | SPEC-CREATED | 无 | 需建 spec | 缺 paired |

### 批次 B1 — KV、缓存与状态管理

| PR | 推荐 workload | Spec 状态 | 现有 artifact | 可运行性 | 证据缺口 |
|---|---|---|---|---|---|
| vllm-hust #49 | KV offload pressure | SPEC-CREATED | merge-only (多 workload) | 需建 spec | 缺 offload disabled/enabled paired |
| vllm-hust #76 | Knorm KV compression long-ctx | SPEC-CREATED | 无 | 需建 spec（或复用 prefix-repetition） | 缺 paired + 质量/误差 |
| vllm-hust #80 | multi-node prefix-repetition | READY (单卡) / 需多节点 | merge-only | BLOCKED by #97 (启动失败) | 缺多节点 paired；单卡仅无回归证据 |
| vllm-hust #124 | `kv-tiering-prefix-online` | READY | base+head pair ✓ (62.4→78.4) | 可跑 | 缺 latest 第三点 + canonical 重跑 |
| vllm-hust #161 | KV INT8 (FP16 vs INT8) | READY (registry int8 profile, 但 workload 占位) | 无 | 需补 int8 真实 workload | 缺 paired + HBM/质量 |
| vllm-hust #171 | BidKV `agent-cache-pressure-online` | SPEC-CREATED | 无 | 需建 spec | 缺 default vs BidKV paired |
| vllm-ascend #28 | KV transfer debug-token | SPEC-CREATED | 无 | 需建 spec | 缺 component + online sanity |
| vllm-ascend #39 | BidKV utility victim selection | SPEC-CREATED | 无 | 需建 spec（与 #171 统一 comparison_id） | 缺 preemption/eviction/SLO |
| vllm-ascend #66 | SimLLM saturated warm-cache | SPEC-CREATED (#59) | head-only (多 workload) | 需建 spec | 缺官方 base/head |
| vllm-ascend #70 | SimLLM robustness | SPEC-CREATED (#59) | head-only (多 workload) | 需建 spec | 与 #66/#80 隔离演进对照 |
| vllm-ascend #80 | SimLLM performance path | SPEC-CREATED (#59) | merge-only | 需建 spec | 缺 cache hit/prefill reduction/TTFT/P99 |

### 批次 B2 — 推测解码、压缩与 KV 传输

| PR | 推荐 workload | Spec 状态 | 现有 artifact | 可运行性 | 证据缺口 |
|---|---|---|---|---|---|
| vllm-hust #121 | EAGLE spec-decode | SPEC-CREATED (registry profile 占位) | 无 | 需建 spec-decode workload | 缺 target/draft/acceptance rate |
| vllm-ascend #123 | QkNorm/RoPE shape (配 #121) | SPEC-CREATED | base-only (ambiguous) | 需建 spec | 缺 head + 正确性证据 |
| vllm-hust #158 | SliceGPT dense vs compressed | SPEC-CREATED | 无 | 需建 spec | 缺吞吐/显存/PPL |
| issue #67 | KV transfer bandwidth/latency | SPEC-CREATED | 无 | 需建 spec | 缺 microbench + online migration |

### 独立优化仓库（不并入主线，各自 canonical result card）

| 仓库 | 推荐 workload | 状态 |
|---|---|---|
| `vllm-hust-bidkv` | KV 压力/驱逐/SLO/吞吐 | 需独立 result card |
| `vllm-ascend-hust-diffspec` | target/draft/acceptance/吞吐时延 | 需独立 result card |
| `vllm-ascend-quant-hust` | 精度/显存/吞吐曲线 | 需独立 result card |
| `vllm-ascend-hust-LatchMoE` | MoE offload/图兼容/专家命中 | 需独立 result card |
| `adaptive-selector-plugin` | 选择准确率/切换开销/端到端 | 需独立 result card |

## 4. 汇总统计

| 分类 | 数量 |
|---|---|
| 已有完整 base+head paired | 2 (#124, #130) |
| 仅有 merge/single 点 | 8 (#41, #49, #66, #70, #80-ascend, #115, #116, #123-base) |
| 无任何 artifact | ~18 |
| spec 就绪可立即跑（GPU 待排） | 7 (#30, #37, #46, #115, #116, #124-latest, #130-latest) |
| 需先创建 specialty spec（非 GPU） | 12 个 spec 缺口，覆盖 ~15 个 PR |
| 外部阻塞 | #80 (by #97), SimLLM 系 (by #59) |

## 5. 非可运行性发现（需修复）

1. **#115/#116 现有 artifact 使用 `gpu_memory_utilization=0.90`**，违反主线 0.6 默认。重跑时必须用 0.6/32768。
2. **#124 base/head 使用 Qwen2.5-7B-Instruct**，非 14B 主线。作为 specialty card 可接受，但需在 artifact 明确标记 `coverage_class=targeted-pair`，不得混入 14B 主折线。
3. **#133 attention-boundary artifact 已存在但无规范 spec**，需补建 `attention-boundary-online` scenario 并重新走 canonical 聚合。
4. **#135 ngram 仅有 head 无 base**，paired 不完整。
5. SimLLM 系列 (#66/#70/#80-ascend) 现有 artifact 用通用 workload，不能证明 SimLLM 机制收益，必须等 #59 saturated warm-cache spec。

### 5.1 trend schema 对 "latest 第三点" 的建模方式（无需改代码）

`trend-coverage/v1` schema 的 `point_role` enum 为 `["baseline", "head", "checkpoint", null]`，
**没有 `"latest"` 角色**。issue #89 要求的 "base/head/latest 三点" 应按以下方式建模，无需修改 schema：

- **comparison_id `prX-merge`**: `point_role=baseline` (fork point) + `point_role=head` (merge commit) — 证明 PR 本身的增益。
- **comparison_id `prX-current`**: `point_role=baseline` (fork point 或 merge commit) + `point_role=head` (current main) — 证明增益在 main 上持续。

即 "latest" 不是新角色，而是第二个 targeted-pair 的 head。`coverage_class=targeted-pair` 已支持此用法。
`aggregate_results.py` 的 `method="latest"`（取最高 repeat_index）与此无关，不要混淆。

## 6. GPU/NPU 依赖任务清单（暂 skip，待资源）

以下任务需要 910B2 NPU 资源，当前 skip 并 tracking：

### P0 — spec 就绪，NPU 一就绪即可跑（2026-08-12 目标）

| 任务 | PR | workload | 需要的点 |
|---|---|---|---|
| 补 latest 第三点 | #124 | kv-tiering-prefix-online | latest (current main) |
| 补 latest 第三点 | #130 | logprobs-online | latest + 内存/尾延迟 |
| base/head/latest | #115 | prefix-repetition-online (0.6/32768) | 3 点 ×3 reps |
| base/head/latest | #116 | prefix-repetition-online (text-only, 0.6/32768) | 3 点 ×3 reps |
| base/head/latest | #30 | visionarena-online + instructcoder-online | 2 workload × 3 点 |
| base/head/latest | #37 | instructcoder-online | 3 点 ×3 reps |
| base/head/latest | #46 | logprobs-online | 3 点 ×3 reps |

### P1 — 需先建 spec（非 GPU），再跑（2026-08-16 目标）

| 任务 | PR | 阻塞 spec |
|---|---|---|
| base/head | #6 | attention-boundary-online |
| base/head | #54, #171 | kv-pressure-online |
| base/head | #121, #123 | spec-decode-online |
| base/head | #161 | int8 真实 workload |
| base/head | #66, #70, #80-ascend | saturated-warm-cache (#59) |
| base/head | #42, #168, vllm-ascend #7/#33/#36 | 通信类 specs |

### P2 — 外部阻塞

| 任务 | 阻塞 |
|---|---|
| #80 (vllm-hust) 多节点 | #97 启动失败 |
| #158 SliceGPT | 需 model-compression specialty |

## 7. 验收标准跟踪

- [ ] 30 个 PR 均有明确证据状态和可追溯链接（本表已建立映射，待 artifact 落地）
- [ ] 有效结果能自动映射到 repo/PR/commit pair/workload/配置
- [ ] 主线只连接 same-spec 有序版本点；targeted pair 独立卡片
- [ ] 独立优化仓库均有结果入口
- [ ] #95 merge gate 能阻止未来无合规 evidence 的性能 PR 合并（已实现，待 #95 合并）
