# vLLM-HUST 官方固定靶

> 本文件由 official specs 自动生成，请勿手工修改。

- Registry version: `1.3.6`
- Effective from: `2026-08-18`

`registry_version` 表示本次生成的 registry 快照版本；每个 target 的 `target_version` 表示该 target 不可变的执行契约版本。无关
target 的更新不应 改变它。生产端应记录 canonical 的 `target_contract_id` 和 `target_contract_version` 字段。

公开排行榜固定靶与 3B 快速门禁是不同契约；provisional 记录不得作为公开成果对比。

| 用途 | 状态 | Profile | Workload | 模型 | 硬件 | 精度 | 显存比例 | 最大长度 | Spec | | --- | --- | --- | --- | --- |
--- | --- | --- | --- | --- | | public-leaderboard | active | code | instructcoder-online |
Qwen/Qwen2.5-Coder-14B-Instruct | 910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json)
| | public-leaderboard | active | core-text | agent-research-online | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | prefix-repetition-online | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | 0.9 | 32768 |
[`official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | random-latency | Qwen/Qwen2.5-14B-Instruct | 910B2 × 1
| FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-random-latency-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-random-latency-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | random-online | Qwen/Qwen2.5-14B-Instruct | 910B2 × 1
| FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | sharegpt-online | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | sharegpt-throughput | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | sonnet-throughput | Qwen/Qwen2.5-14B-Instruct | 910B2
× 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-910b2.json)
| | public-leaderboard | active | multimodal | visionarena-online | Qwen/Qwen2.5-VL-7B-Instruct |
910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-visionarena-online-qwen25-vl-7b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-visionarena-online-qwen25-vl-7b-910b2.json)
| | perfgate | provisional | perfgate-code | instructcoder-online | Qwen/Qwen2.5-Coder-3B-Instruct |
910B2 × 1 | BF16 | — | — |
[`perfgate-ascend-instructcoder-online-qwen25-coder-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-instructcoder-online-qwen25-coder-3b-910b2.json)
| | perfgate | provisional | perfgate-multimodal | visionarena-online | Qwen/Qwen2.5-VL-3B-Instruct
| 910B2 × 1 | BF16 | — | — |
[`perfgate-ascend-visionarena-online-qwen25-vl-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-visionarena-online-qwen25-vl-3b-910b2.json)
| | perfgate | provisional | perfgate-text | agent-research-online | Qwen/Qwen2.5-3B-Instruct |
910B2 × 1 | BF16 | — | — |
[`perfgate-ascend-agent-research-online-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-agent-research-online-qwen25-3b-910b2.json)
| | perfgate | provisional | perfgate-text | prefix-repetition-online | Qwen/Qwen2.5-3B-Instruct |
910B2 × 1 | BF16 | — | 1280 |
[`perfgate-ascend-prefix-repetition-online-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-prefix-repetition-online-qwen25-3b-910b2.json)
| | perfgate | provisional | perfgate-text | random-latency | Qwen/Qwen2.5-3B-Instruct | 910B2 × 1 |
BF16 | — | 1280 |
[`perfgate-ascend-random-latency-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-random-latency-qwen25-3b-910b2.json)
| | perfgate | provisional | perfgate-text | random-online | Qwen/Qwen2.5-3B-Instruct | 910B2 × 1 |
BF16 | — | 256 |
[`perfgate-ascend-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json)
| | perfgate | provisional | perfgate-text | sharegpt-online | Qwen/Qwen2.5-3B-Instruct | 910B2 × 1
| BF16 | — | — |
[`perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json)
| | perfgate | provisional | perfgate-text | sharegpt-throughput | Qwen/Qwen2.5-3B-Instruct | 910B2
× 1 | BF16 | — | — |
[`perfgate-ascend-sharegpt-throughput-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-sharegpt-throughput-qwen25-3b-910b2.json)
| | perfgate | provisional | perfgate-text | sonnet-throughput | Qwen/Qwen2.5-3B-Instruct | 910B2 ×
1 | BF16 | — | — |
[`perfgate-ascend-sonnet-throughput-qwen25-3b-910b2.json`](../docs/official-baselines/perfgate-ascend-sonnet-throughput-qwen25-3b-910b2.json)
| | specialty | provisional | multi-chip | agent-research-online-2chip | Qwen/Qwen2.5-14B-Instruct |
910B2 × 2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-2chip-910b2.json)
| | specialty | provisional | multi-chip | agent-research-online-4chip | Qwen/Qwen2.5-14B-Instruct |
910B2 × 4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-4chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-4chip-910b2.json)
| | specialty | provisional | multi-chip | eplb-expert-rebalance-online | Qwen/Qwen2.5-14B-Instruct
| 910B2 × 2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-eplb-expert-rebalance-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-eplb-expert-rebalance-online-qwen25-14b-910b2.json)
| | specialty | provisional | multi-chip | kv-transfer-latency | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-kv-transfer-latency-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-kv-transfer-latency-qwen25-14b-910b2.json)
| | specialty | provisional | multi-chip | moe-alltoall-online | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-moe-alltoall-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-moe-alltoall-online-qwen25-14b-910b2.json)
| | specialty | provisional | multi-chip | multi-nic-throughput | Qwen/Qwen2.5-14B-Instruct | 910B2
× 4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-multi-nic-throughput-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-multi-nic-throughput-qwen25-14b-910b2.json)
| | specialty | provisional | multi-chip | prefix-repetition-online-2chip |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-2chip-910b2.json)
| | specialty | provisional | multi-chip | prefix-repetition-online-4chip |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-4chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-4chip-910b2.json)
| | specialty | provisional | multi-chip | random-online-2chip | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-random-online-qwen25-14b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-2chip-910b2.json)
| | specialty | provisional | multi-chip | random-online-4chip | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-random-online-qwen25-14b-4chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-4chip-910b2.json)
| | specialty | provisional | multi-chip | sharegpt-online-2chip | Qwen/Qwen2.5-14B-Instruct | 910B2
× 2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-2chip-910b2.json)
| | specialty | provisional | multi-chip | sharegpt-online-4chip | Qwen/Qwen2.5-14B-Instruct | 910B2
× 4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-4chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-4chip-910b2.json)
| | specialty | provisional | multi-chip | sonnet-throughput-2chip | Qwen/Qwen2.5-14B-Instruct |
910B2 × 2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-2chip-910b2.json)
| | specialty | provisional | multi-chip | sonnet-throughput-4chip | Qwen/Qwen2.5-14B-Instruct |
910B2 × 4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-4chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-4chip-910b2.json)
| | specialty | provisional | multi-chip | sonnet-throughput-8chip | Qwen/Qwen2.5-14B-Instruct |
910B2 × 8 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-8chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-8chip-910b2.json)
| | specialty | provisional | multi-chip | unified-comm-online | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
2 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-unified-comm-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-unified-comm-online-qwen25-14b-910b2.json)
| | specialty | provisional | multi-chip | unified-comm-online-4chip | Qwen/Qwen2.5-14B-Instruct |
910B2 × 4 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-unified-comm-online-qwen25-14b-4chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-unified-comm-online-qwen25-14b-4chip-910b2.json)
| | specialty | provisional | specialty | kv-tiering-prefix-online | Qwen/Qwen2.5-7B-Instruct |
910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-kv-tiering-prefix-online-qwen25-7b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-kv-tiering-prefix-online-qwen25-7b-910b2.json)
| | specialty | provisional | specialty | ngram-instructcoder-online | Qwen/Qwen2.5-7B-Instruct |
910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-ngram-instructcoder-online-qwen25-7b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-ngram-instructcoder-online-qwen25-7b-910b2.json)
| | specialty | provisional | specialty-text | agent-cache-pressure-online |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-agent-cache-pressure-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-agent-cache-pressure-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | attention-boundary-online | Qwen/Qwen2.5-14B-Instruct
| 910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-attention-boundary-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-attention-boundary-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | instructcoder-online | Qwen/Qwen2.5-14B-Instruct |
910B3 × 1 | FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-instructcoder-online-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-instructcoder-online-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | knorm-kv-compression-longctx |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-knorm-kv-compression-longctx-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-knorm-kv-compression-longctx-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | kv-pressure-online | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | 0.45 | — |
[`official-ascend-jan-2026-v0180-kv-pressure-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-kv-pressure-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | logprobs-online | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-logprobs-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-logprobs-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | prefix-repetition-online | Qwen/Qwen2.5-14B-Instruct
| 910B3 × 1 | FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-prefix-repetition-online-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-prefix-repetition-online-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | random-latency | Qwen/Qwen2.5-14B-Instruct | 910B3 ×
1 | FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-random-latency-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-random-latency-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | random-online | Qwen/Qwen2.5-14B-Instruct | 910B3 × 1
| FP16 | 0.6 | 32768 |
[`specialty-ascend-current-random-online-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-current-random-online-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | random-online | Qwen/Qwen2.5-14B-Instruct | 910B3 × 1
| FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-random-online-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-random-online-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | saturated-warm-cache-online |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-saturated-warm-cache-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-saturated-warm-cache-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | sharegpt-online | Qwen/Qwen2.5-14B-Instruct | 910B3 ×
1 | FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-sharegpt-online-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-sharegpt-online-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | sharegpt-throughput | Qwen/Qwen2.5-14B-Instruct |
910B3 × 1 | FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-sharegpt-throughput-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-sharegpt-throughput-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | slicegpt-compression-online |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-slicegpt-compression-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-slicegpt-compression-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | sonnet-throughput | Qwen/Qwen2.5-14B-Instruct | 910B3
× 1 | FP16 | 0.6 | 32768 |
[`specialty-ascend-full-graph-parallel-inplace-sonnet-throughput-qwen25-14b-910b3.json`](../docs/official-baselines/specialty-ascend-full-graph-parallel-inplace-sonnet-throughput-qwen25-14b-910b3.json)
| | specialty | provisional | specialty-text | spec-decode-online | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-spec-decode-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-spec-decode-online-qwen25-14b-910b2.json)
| | specialty | provisional | specialty-text | unified-comm-online-1chip | Qwen/Qwen2.5-14B-Instruct
| 910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-unified-comm-online-qwen25-14b-1chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-unified-comm-online-qwen25-14b-1chip-910b2.json)
|

机器可读快照：[`leaderboard-data/official-targets.json`](../leaderboard-data/official-targets.json)
