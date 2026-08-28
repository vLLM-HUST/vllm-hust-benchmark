# issue #105 配套：Benchmark 设计与实现说明

> Legacy design: the GitHub Actions benchmark and publication chain described here is retired. New
> PR validation is performed by the external fixed-machine dataset service.

> 这份文档配套 [issue #105 需求分析](../.trae/documents/issue_105_%E9%9C%80%E6%B1%82%E5%88%86%E6%9E%90.md) 阅读。
>
> 需求文档讲"要做什么"，这份文档讲"这套系统是怎么搭的、数据怎么流、为什么这么设计"。目的是让做可行性分析的同事能快速看懂实现，评估风险。
>
> **说人话原则**：能用一句白话说清楚的，不用两句；能画数据流的，不写长段落。

______________________________________________________________________

## 1. 这套系统是干什么的（一句话）

**一个把 vLLM-hust 的性能跑分变成可信、可比、可公开发布的排行榜的中间层。**

它解决三个问题：

1. **怎么跑**：调起 vllm-hust 的 benchmark，跑出原始结果
1. **怎么比**：用固定靶（fixed-target）契约保证不同 PR 之间能比
1. **怎么发**：把结果聚合成排行榜 snapshot，同步到 GitHub / HF / website / PPT

issue #105 主要管第 2 和第 3 个问题里的"可信度"。

______________________________________________________________________

## 2. 仓库布局：三个仓库怎么分工

```
vllm-hust-benchmark      ← 你现在在这个仓库（编排层 + 数据源）
    ├── src/vllm_hust_benchmark/   ← Python 包
    ├── scripts/                    ← 运行脚本
    ├── docs/official-baselines/   ← 官方 baseline spec
    ├── submissions/                ← 原始 benchmark 产物
    └── leaderboard-data/snapshots/ ← 排行榜 snapshot（权威数据源）

vllm-hust                ← 兄弟仓库（真正的 benchmark 实现）
    ├── benchmarks/*.py
    └── .buildkite/performance-benchmarks/

vllm-hust-website        ← 兄弟仓库（网站）
    └── scripts/aggregate_results.py  ← 聚合脚本
```

**关键原则**（来自 [README.md](../README.md)）：

- `vllm-hust-benchmark` 保持"薄"，不重新实现属于 `vllm-hust` 的 benchmark 运行时逻辑
- 结果 shaping 和网站发布可以留在这里，因为它们是跨仓库编排问题
- **benchmark 仓库的 `submissions/` 和 `leaderboard-data/snapshots/` 是权威数据源**，website 仓库只是镜像

______________________________________________________________________

## 3. 数据流：从一次跑分到上排行榜

### 3.1 整体数据流（说人话版）

```
[跑 benchmark]
    │  vllm-hust 的 benchmark 脚本跑出原始结果
    ▼
[导出 submission artifact]
    │  export-leaderboard-artifact 把原始结果转成标准 leaderboard entry
    │  写到 submissions/<run-id>/run_leaderboard.json
    ▼
[聚合 aggregate]
    │  aggregate_to_website 调用 website 仓库的 aggregate_results.py
    │  把所有 submissions/ 下的 run_leaderboard.json 聚合成：
    │    - leaderboard_single.json（单卡榜）
    │    - leaderboard_multi.json（多卡榜）
    │    - leaderboard_compare.json（对比榜）
    ▼
[admission gate 检查]  ← issue #105 的核心
    │  扫描聚合后的 snapshot，剔除不符合固定靶的 entry
    │  原地修改 leaderboard_single.json / leaderboard_multi.json
    ▼
[发布 publish]
    │  同步到 GitHub（本仓库 main 分支）
    │  同步到 HF dataset（push-to-hf.yml workflow）
    │  同步到 website（website 读 GitHub/HF/local）
    │  同步到 PPT source
```

### 3.2 关键设计点

1. **GitHub 是第一新鲜源**：website 的读取优先级是 `github -> hf -> local`，所以 GitHub 仓库的 snapshot 必须先更新
1. **HF 是分发镜像**：给读不到 GitHub 的消费者用
1. **website-local 不是权威**：不能让它盖过更新的 GitHub/HF 数据
1. **不手改最终 JSON**：所有 snapshot 都从 canonical submission + aggregator 重建

______________________________________________________________________

## 4. fixed_target_registry：判断对错的尺子

### 4.1 它是什么

一个 JSON
文件（[fixed_target_registry.json](../src/vllm_hust_benchmark/data/fixed_target_registry.json)），定义了"官方固定靶"。它是所有判定的**唯一事实来源**。

### 4.2 它的结构（说人话）

```
registry
├── schema_version: "fixed-target-registry/v1"
└── profiles: [           ← 一组靶子定义
    {
      target_id:          "official-ascend-jan-2026-v0.18.0"  ← 哪个 campaign
      target_version:     "Official Ascend Jan 2026"          ← campaign 版本标签
      profile_name:       "core-text-14b"                    ← 靶子名字
      model:              "Qwen/Qwen2.5-14B-Instruct"        ← 模型
      hardware_chip_model: "910B2"                           ← 硬件
      chip_count:         1                                  ← 卡数
      model_precision:    "FP16"                             ← 精度
      tensor_parallel_size: 1                                ← 并行
      gpu_memory_utilization: 0.6                            ← 显存（active 必填）
      max_model_len:      32768                              ← 上下文（active 必填）
      workload_name:      "random-online"                    ← workload
      status:             "active"                           ← active/specialty/retired
    },
    ...
]
```

### 4.3 三种 status 的区别

| status      | 含义               | 是否校验显存/上下文               |
| ----------- | ------------------ | --------------------------------- |
| `active`    | 主线靶子，必须对齐 | 是（必须有且数值相等）            |
| `specialty` | 专项靶子，不进主线 | 否（直接 disposition=specialty）  |
| `retired`   | 退役靶子，不能再上 | 否（直接 disposition=quarantine） |

### 4.4 profile 怎么匹配（find_matching_profile）

[fixed_target_registry.py:89-130](../src/vllm_hust_benchmark/fixed_target_registry.py#L89-L130)
的匹配逻辑：

```
对 registry 里每个 profile：
    1. 模型对得上？（profile.model 在 entry 的 repo_id/canonical_id/name 里）
    2. 硬件型号对得上？（910B2 == 910B2）
    3. 卡数对得上？（1 == 1）
    4. 精度对得上？（FP16 == FP16）
    5. workload 对得上？（如果 entry 的 workload 在 registry 里有，才校验）
    全部对上 → 返回这个 profile
    否则 → 继续下一个
都匹配不到 → 返回 None（非官方 entry，不归这个 gate 管）
```

**注意**：`gpu_memory_utilization` 和 `max_model_len` **不参与匹配**，只参与匹配后的校验。这就是为什么会出现"匹配上了但配置漂移"的情况。

______________________________________________________________________

## 5. admission gate：数据进排行榜前的闸门

### 5.1 它在数据流的哪个位置

在
`aggregate_to_website`（[integration.py:519-643](../src/vllm_hust_benchmark/integration.py#L519-L643)）里，**在外部聚合脚本写完
snapshot 之后**：

```
aggregate_to_website()
    │
    ├─ 1. 检查 excluded submission（leaderboard-exclusions.json）
    ├─ 2. 检查 admission failures（STATUS 文件、临时目录等）
    ├─ 3. 检查 superseded coexistence conflicts（新旧共存冲突）
    ├─ 4. 调用 website 仓库的 aggregate_results.py 聚合
    │
    └─ 5. ★ admission gate（issue #105 核心）：
         ├─ 加载 fixed_target_registry
         ├─ _scan_fixed_target_misaligned_entries：扫描不对齐的 entry
         └─ _quarantine_misaligned_snapshot_entries：原地剔除
```

### 5.2 扫描逻辑（\_scan_fixed_target_misaligned_entries）

[integration.py:1909-2063](../src/vllm_hust_benchmark/integration.py#L1909-L2063) 的逻辑：

```
对 leaderboard_single.json 和 leaderboard_multi.json 里每条 entry：
    1. find_matching_profile(entry, registry)
       ├─ None → 跳过（非官方 entry，不归这个 gate 管）
       └─ 找到 profile → 继续

    2. 根据 profile.status 判定：
       ├─ specialty → disposition=specialty，reason=specialty_without_contract
       ├─ retired   → disposition=quarantine，reason=retired_target
       └─ active   → 校验两个字段：

            对 (gpu_memory_utilization, max_model_len)：
            ├─ 字段不在 server 参数里 → reason=missing_xxx → quarantine
            ├─ 数值不等（用 _fixed_target_numeric_equal）→ reason=config_drift → quarantine
            └─ 都对 → keep（不出现在 misaligned 列表里）
```

### 5.3 剔除逻辑（\_quarantine_misaligned_snapshot_entries）

[integration.py:2066-2105](../src/vllm_hust_benchmark/integration.py#L2066-L2105)：

```
收集所有 misaligned 的 entry_id
对 leaderboard_single.json 和 leaderboard_multi.json：
    读 → 过滤掉 misaligned 的 entry → 写回
```

**关键**：只改 snapshot 文件，**不动 submissions/ 下的原始 artifact**。这就是"不物理删除"的实现。

### 5.4 数值比较为什么要用 \_fixed_target_numeric_equal

[integration.py:1885-1890](../src/vllm_hust_benchmark/integration.py#L1885-L1890)：

```python
def _fixed_target_numeric_equal(left, right):
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right
```

因为 `0.6`（float）和 `"0.6"`（string）和 `0.60` 都应该算相等。直接 `==` 会因为类型不同误判。

______________________________________________________________________

## 6. workload_config_contract：每条数据的契约校验

### 6.1 它和 admission gate 的区别

| 维度     | admission gate                   | workload_config_contract                             |
| -------- | -------------------------------- | ---------------------------------------------------- |
| 检查什么 | 模型/硬件/精度/显存/上下文对不对 | workload 配置完不完整、对不对                        |
| 什么时候 | 聚合后扫 snapshot                | 聚合前验 submission                                  |
| 触发条件 | 匹配到固定靶 profile             | official spec entry（spec_id 以 official- 前缀开头） |

### 6.2 它校验什么（[workload_config_contract.py](../src/vllm_hust_benchmark/workload_config_contract.py)）

```
对 official spec entry（spec_id 以 official-ascend-jan-2026-v0.18.0- 开头）：
    1. metadata.workload_config_contract == "explicit-effective/v1"
    2. metadata.submitted_at 必须有
    3. 如果 submitted_at >= 2026-07-29：
       metadata.target_id 和 metadata.target_version 必须有且在 registry 里
    4. workload 对象必须含 name/input_length/output_length/batch_size/concurrent_requests/dataset
    5. server 参数必须含 gpu_memory_utilization + max_model_len
    6. client 参数按 scenario 要求含对应字段（no_stream / gpu_memory_utilization）
    7. server 的 gpu_memory_utilization 必须 == 0.6（文本）/ 0.6（vision）
    8. server 的 max_model_len 必须 == 32768（文本）/ 30720（vision）
    9. workload 的 input/output/batch/concurrency 必须和 client 实际值一致
```

### 6.3 BREAKING change：移除 grandfathering

[integration.py:2172-2202](../src/vllm_hust_benchmark/integration.py#L2172-L2202) 的
`_validate_entry_workload_contract`：

```python
# 旧逻辑：submitted_at 早于 2026-07-24 的 official entry 可以跳过
# 新逻辑（BREAKING）：require_official=True 时，所有 official entry 都要查
if not must_validate and require_official:
    if is_official_workload_contract_entry(payload):
        must_validate = True
```

人话：**历史遗留 entry 不能再靠"我跑得早"来绕过检查。** 这是 issue #105 的硬约束，在
[test_historical_legacy_not_bypassed](../tests/test_fixed_target_admission.py#L164-L178) 里有测试。

______________________________________________________________________

## 7. 发布链路：数据怎么同步到四处

### 7.1 四处同步的优先级

```
GitHub (vllm-hust-benchmark@main)
    │  第一新鲜源（权威）
    │  push-to-hf.yml workflow 监听 submissions/** 变化
    ▼
HuggingFace dataset (intellistream/vllm-hust-benchmark-results)
    │  分发镜像
    │  给读不到 GitHub 的消费者用
    ▼
website (vllm-hust-website)
    │  读取优先级：github -> hf -> local
    │  local 不能盖过更新的 GitHub/HF
    ▼
PPT source
    成果页/canonical 数据源
```

### 7.2 sync_submission_to_huggingface 的流程

[integration.py:2234-2497](../src/vllm_hust_benchmark/integration.py#L2234-L2497)：

```
1. 校验 submission 目录（_validate_formal_submission_sources）
   ├─ 排除 PR preview artifact
   ├─ 排除 leaderboard exclusion 命中的
   └─ 校验 workload_config_contract（BREAKING：无 grandfathering）

2. 下载 HF 上已有的历史 submission，合并新 submission

3. 调用 aggregate_to_website 重新聚合
   └─ 内部跑 admission gate

4. 校验聚合结果（validate_aggregated_leaderboard_outputs）
   ├─ 不能所有 tab 只有 vllm-hust（要有 baseline）
   └─ hard-constraint scope key 不能丢

5. 上传到 HF：
   ├─ 删除被 exclusion 的旧 repo path
   ├─ 上传新的 snapshot 文件
   └─ 上传新的 submission 文件
```

### 7.3 snapshot 冻结（freeze_snapshot.py）

[freeze_snapshot.py](../scripts/freeze_snapshot.py) 在清理前冻结一份：

```
input:  leaderboard-data/snapshots/leaderboard_single.json + leaderboard_multi.json
output: pre_cleanup_freeze.json
        ├─ schema_version: "freeze-snapshot/v1"
        ├─ leaderboard_single_checksum: sha256:...
        ├─ leaderboard_multi_checksum: sha256:...
        ├─ entry_ids: [所有 entry_id 列表]
        └─ source_commit: git HEAD
```

这是回滚的锚点。当前已冻结：199 entry_ids，source_commit `0a26e778`。

______________________________________________________________________

## 8. admission report：怎么生成诊断报告

### 8.1 generate_admission_report.py 的流程

[generate_admission_report.py](../scripts/generate_admission_report.py)：

```
input:
  --snapshot  leaderboard_single.json
  --registry  fixed_target_registry.json
  --output    admission_report.json

中途逻辑：
  对 snapshot 里每条 entry：
    1. find_matching_profile
    2. 判定 disposition（和 admission gate 一样的逻辑）：
       ├─ 无匹配 → keep（非官方）
       ├─ specialty → specialty
       ├─ retired → quarantine
       └─ active → 校验字段 → keep / quarantine
    3. 记录 actual_config / required_config / missing_fields / drift_fields / reason

output:
  admission_report.json
    ├─ schema_version: "admission-report/v1"
    ├─ generated_at
    ├─ registry_version
    └─ entries: [每条的处置记录]
```

### 8.2 disposition 四种值

| disposition  | 含义   | 什么时候                           |
| ------------ | ------ | ---------------------------------- |
| `keep`       | 留下   | 对齐的 / 非官方的                  |
| `quarantine` | 隔离   | 缺字段 / 配置漂移 / retired        |
| `specialty`  | 归专项 | 匹配到 specialty profile           |
| `rerun`      | 补跑   | （目前代码里没自动产生，人工标记） |

______________________________________________________________________

## 9. 测试覆盖：6 个场景

[test_fixed_target_admission.py](../tests/test_fixed_target_admission.py) 覆盖：

| 测试                                              | 场景                          | 预期                       |
| ------------------------------------------------- | ----------------------------- | -------------------------- |
| `test_missing_gpu_memory_utilization_quarantined` | 缺显存字段                    | quarantine                 |
| `test_config_drift_quarantined`                   | 显存 0.9（应 0.6）            | quarantine                 |
| `test_wrong_profile_quarantined`                  | vision 用了 32768（应 30720） | quarantine（config_drift） |
| `test_specialty_without_contract`                 | 2 卡 entry                    | specialty                  |
| `test_aligned_entry_kept`                         | 完全对齐                      | 不出现在 misaligned        |
| `test_retired_target_quarantined`                 | 匹配 retired profile          | quarantine                 |
| `test_historical_legacy_not_bypassed`             | 历史遗留 + 缺字段             | quarantine（BREAKING）     |
| `test_quarantine_misaligned_snapshot_entries`     | 剔除后只留 aligned            | 原地过滤                   |

______________________________________________________________________

## 10. issue #105 的实现就绪度总览

| 组件                                | 状态                            | 说明                                                                                                                                                               |
| ----------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| fixed_target_registry               | ✅ 已实现                       | 3 active + 6 specialty profile                                                                                                                                     |
| admission gate（scan + quarantine） | ✅ 已实现                       | 在 aggregate_to_website 里集成                                                                                                                                     |
| workload_config_contract            | ✅ 已实现                       | BREAKING：无 grandfathering                                                                                                                                        |
| generate_admission_report           | ✅ 已实现                       | dry run: keep=24/quarantine=175/specialty=0                                                                                                                        |
| freeze_snapshot                     | ✅ 已实现                       | 已冻结 pre_cleanup_freeze.json                                                                                                                                     |
| 测试覆盖                            | ✅ 8 个场景测试                 | 96 passed                                                                                                                                                          |
| **#95 merge gate**                  | ⚠️ **已有需求文档，代码未实现** | 需求文档见 [.trae/documents/issue_95_需求分析.md](../.trae/documents/issue_95_%E9%9C%80%E6%B1%82%E5%88%86%E6%9E%90.md)，判定逻辑可复用本仓库的 admission gate 代码 |
| **官方骨架重建（服务端执行）**      | ⏳ 待执行                       | phase 2，需服务端 NPU                                                                                                                                              |
| **PR paired reruns**                | ⏳ 待执行                       | phase 4，依赖 #95 或替代标准                                                                                                                                       |
| **四处同步收口**                    | ⏳ 待执行                       | GitHub/HF/website/PPT checksum 对齐                                                                                                                                |

______________________________________________________________________

## 11. 关键风险点（给可行性分析参考）

### 11.1 设计上的风险

1. **profile 匹配不校验显存/上下文**：`find_matching_profile` 只用 model/hardware/precision/chip_count/workload
   匹配，显存和上下文是匹配后才校验。如果某条 entry 的 model/hardware/precision/workload 都对但显存漂移，会被正确 quarantine；但如果
   workload_name 在 registry 里不存在，会跳过 workload 校验直接匹配——需要确认这是不是预期行为。

1. **数值比较的容错**：`_fixed_target_numeric_equal` 用 `float()` 比较，`0.6` 和 `"0.6"` 会判等。如果未来出现 `0.6000001`
   这种浮点漂移，会被判 drift——需要确认浮点精度边界。

1. **specialty 不校验显存/上下文**：匹配到 specialty profile 直接 disposition=specialty，不校验显存/上下文。如果 specialty
   的显存也漂移了，目前不会被发现——需要确认 specialty 是否需要单独的契约。

### 11.2 流程上的风险

1. **admission gate 是聚合后扫 snapshot**：如果聚合脚本本身有 bug 写错了数据，admission gate 只能挡固定靶不对的，挡不了聚合逻辑错误。

1. **#95 merge gate 未实现**：PR evidence 的"可比性"目前只能靠 admission gate（fixed-target 对齐），但 fixed-target
   对齐不等于"可比"——两条 entry 可能都对齐了固定靶，但用了不同的 client 参数（如 request_rate），目前 workload_config_contract 会校验部分
   client 参数，但不全面。

1. **四处同步的原子性**：GitHub/HF/website/PPT 四处同步不是原子的，中间可能出现不一致窗口。`pre_cleanup_freeze.json`
   是回滚锚点，但回滚后需要重新触发 HF/website 同步。

______________________________________________________________________

## 附录 A：数据结构速查

### A.1 leaderboard entry 的关键字段

```
entry
├── entry_id: "uuid"
├── engine: "vllm-hust" / "vllm"
├── engine_version: "0.7.3"
├── model:
│   ├── repo_id: "Qwen/Qwen2.5-14B-Instruct"
│   ├── canonical_id: "hf:Qwen/..."
│   └── precision: "FP16"
├── hardware:
│   ├── chip_model: "910B2"
│   └── chip_count: 1
├── workload:
│   ├── name: "random-online"
│   ├── input_length: 1024
│   └── output_length: 256
├── same_spec:
│   ├── spec_id: "official-ascend-jan-2026-v0.18.0-..."
│   ├── scenario: "random-online"
│   ├── resolved_server_parameters:
│   │   ├── gpu_memory_utilization: 0.6    ← admission gate 校验
│   │   └── max_model_len: 32768           ← admission gate 校验
│   └── resolved_client_parameters: {...}
├── metadata:
│   ├── submitted_at: "2026-07-25T..."
│   ├── workload_config_contract: "explicit-effective/v1"  ← contract 校验
│   ├── target_id: "official-ascend-jan-2026-v0.18.0"     ← 2026-07-29 后必填
│   ├── target_version: "Official Ascend Jan 2026"         ← 2026-07-29 后必填
│   ├── runtime_provenance:
│   │   ├── engine: {commit: "..."}
│   │   └── plugin: {commit: "..."}
│   └── supersedes: "old-entry-id" (可选)
└── cluster:
    └── node_count: 1
```

### A.2 disposition 判定决策树

```
entry 进入 admission gate
    │
    ├─ find_matching_profile == None?
    │   └─ YES → keep（非官方，不归这个 gate 管）
    │
    ├─ profile.status == "specialty"?
    │   └─ YES → specialty
    │
    ├─ profile.status == "retired"?
    │   └─ YES → quarantine（retired_target）
    │
    └─ profile.status == "active"
        │
        ├─ gpu_memory_utilization 不在 server 参数?
        │   └─ YES → quarantine（missing_gpu_memory_utilization）
        │
        ├─ max_model_len 不在 server 参数?
        │   └─ YES → quarantine（missing_max_model_len）
        │
        ├─ gpu_memory_utilization 数值不等?
        │   └─ YES → quarantine（config_drift）
        │
        ├─ max_model_len 数值不等?
        │   └─ YES → quarantine（config_drift）
        │
        └─ 都对 → keep
```
