# vLLM-HUST 官方固定靶

> 本文件由 official specs 自动生成，请勿手工修改。

- Registry version: `1.8.0`
- Effective from: `2026-08-03`

公开排行榜固定靶与 3B 快速门禁是不同契约；provisional 记录不得作为公开成果对比。

| 用途 | 状态 | Profile | Workload | 模型 | 硬件 | 精度 | 显存比例 | 最大长度 | Spec | | --- | --- | --- | --- | --- |
--- | --- | --- | --- | --- | | public-leaderboard | active | code | instructcoder-online |
Qwen/Qwen2.5-Coder-14B-Instruct | 910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json)
| | public-leaderboard | active | core-text | agent-research-online | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json)
| | public-leaderboard | active | core-text | prefix-repetition-online | Qwen/Qwen2.5-14B-Instruct |
910B2 × 1 | FP16 | 0.6 | 32768 |
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
| | public-leaderboard | active | production-trace | burstgpt-production-replay |
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 910B2 × 2 | BF16 | 0.92 | 131072 |
[`official-ascend-jan-2026-v0221rc1-burstgpt-production-replay-deepseek-r1-distill-qwen32b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0221rc1-burstgpt-production-replay-deepseek-r1-distill-qwen32b-2chip-910b2.json)
| | public-leaderboard | active | production-trace | tracelab-coding-agent-replay |
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 910B2 × 2 | BF16 | 0.92 | 131072 |
[`official-ascend-jan-2026-v0221rc1-tracelab-coding-agent-replay-deepseek-r1-distill-qwen32b-2chip-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0221rc1-tracelab-coding-agent-replay-deepseek-r1-distill-qwen32b-2chip-910b2.json)
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
| | specialty | active | simllm-warm-cache | simllm-random-online-warm-cache |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-simllm-random-online-warm-cache-qwen25-14b-1chip-910b2.json`](../docs/official-baselines/official-simllm-random-online-warm-cache-qwen25-14b-1chip-910b2.json)
| | specialty | active | simllm-warm-cache | simllm-saturated-throughput-warm-cache |
Qwen/Qwen2.5-14B-Instruct | 910B2 × 1 | FP16 | 0.6 | 32768 |
[`official-simllm-saturated-throughput-warm-cache-qwen25-14b-1chip-910b2.json`](../docs/official-baselines/official-simllm-saturated-throughput-warm-cache-qwen25-14b-1chip-910b2.json)
| | specialty | provisional | specialty | kv-tiering-prefix-online | Qwen/Qwen2.5-7B-Instruct |
910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-kv-tiering-prefix-online-qwen25-7b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-kv-tiering-prefix-online-qwen25-7b-910b2.json)
| | specialty | provisional | specialty | ngram-instructcoder-online | Qwen/Qwen2.5-7B-Instruct |
910B2 × 1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-ngram-instructcoder-online-qwen25-7b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-ngram-instructcoder-online-qwen25-7b-910b2.json)
| | specialty | provisional | specialty-text | logprobs-online | Qwen/Qwen2.5-14B-Instruct | 910B2 ×
1 | FP16 | — | — |
[`official-ascend-jan-2026-v0180-logprobs-online-qwen25-14b-910b2.json`](../docs/official-baselines/official-ascend-jan-2026-v0180-logprobs-online-qwen25-14b-910b2.json)
|

机器可读快照：[`leaderboard-data/official-targets.json`](../leaderboard-data/official-targets.json)
