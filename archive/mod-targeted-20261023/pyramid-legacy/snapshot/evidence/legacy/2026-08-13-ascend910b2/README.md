# Legacy Ascend 910B2 result — 2026-08-13

This is public **historical evidence**, not a result from the current
`import_only` package or the refreshed host main branches. It records the final
matched result from the archived integration and provides a reproducible
baseline for future exact-head comparison.

## Suggested one-line summary

> Historical exact-head Qwen2.5-14B testing on one Ascend 910B2 at 7168 input / 512 output tokens and concurrency 32 reduced active KV ownership from 56 to 8 blocks per request (1.125 GiB released), increased throughput by 41.38%, and reduced mean TTFT by 30.98%; the tradeoff was 50.50% higher TPOT, 14.78% higher E2E latency, and a 2.415-point Qwen LongBench mean QA-F1 drop.

The sentence must retain the word “Historical” until the active extension is
tested from its own pushed exact head.

## Identity

| Item | Value |
| --- | --- |
| Core | `4861aab3af39e721c1b5a8b27b72c4f6bebda888` |
| Ascend | `a0a2ffffce472220061c3e9696defa6b8ddbe63a` |
| Hardware | One physical Ascend 910B2, card 6 |
| Runtime | CANN 9.0.0, Python 3.11.15, torch 2.9.0, torch-npu 2.9.0 |
| Model | Local `Qwen2.5-14B-Instruct`, BF16 |
| Execution | EAGER, V1 runner, TP=1, PP=1 |
| KV/cache | block size 128, prefix caching enabled, memory utilization 0.8 |
| Scheduler | chunked prefill enabled, async scheduling enabled |
| Limits | max model length 8192, max sequences 32, max batched tokens 2048 |

Model identity:

- `config.json` SHA-256:
  `0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e`
- `model.safetensors.index.json` SHA-256:
  `46627f7a9e851d9cede6a7a4c49999082148ced9c22d2b9cfaaa5ba8b65dc68f`
- `tokenizer.json` SHA-256:
  `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`

The original local paths in `provenance.json` identify the evidence-producing
workspace; they are not portable installation instructions.

## Provider configuration

```json
{
  "schema_version": 1,
  "provider": "pyramidkv_ascend",
  "provider_config": {
    "max_capacity_prompt": 512,
    "min_compression_prompt_tokens": 4096,
    "window_size": 8,
    "kernel_size": 7,
    "pooling": "maxpool",
    "beta": 20,
    "kv_cache_granularity": "kv_head",
    "gqa_score_aggregation": "mean",
    "merge": null
  }
}
```

Prompts of at most 4096 tokens remained on the ordinary path. Real boundary
checks recorded no compression plan at 1024 or 4096 tokens and one plan at
4097 tokens.

## Workloads and method

Baseline and enabled modes used the same Core, Ascend, model, card, server
arguments, and workload seeds. The six server lifecycles were alternated in
this order:

```text
disabled:1, enabled:1, disabled:2, enabled:2, disabled:3, enabled:3
```

The disabled command omitted only `--kv-cache-compression-config`. The enabled
server used the provider JSON above and otherwise the equivalent arguments:

```bash
vllm serve /path/to/Qwen2.5-14B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --enforce-eager \
  --generation-config vllm \
  --seed 0 \
  --kv-cache-compression-config '<provider JSON above>'
```

Short workload:

```bash
vllm bench serve \
  --backend vllm \
  --model /path/to/Qwen2.5-14B-Instruct \
  --tokenizer /path/to/Qwen2.5-14B-Instruct \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 48 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0.0 \
  --request-rate 1 \
  --seed 0 \
  --ignore-eos
```

Long workload:

```bash
vllm bench serve \
  --backend vllm \
  --model /path/to/Qwen2.5-14B-Instruct \
  --tokenizer /path/to/Qwen2.5-14B-Instruct \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 32 \
  --random-input-len 7168 \
  --random-output-len 512 \
  --random-range-ratio 0.0 \
  --request-rate inf \
  --max-concurrency 32 \
  --seed 0 \
  --ignore-eos
```

Every table value below is the mean over three independent server lifecycles.

## Performance result

| Workload | Mode | Throughput (req/s) | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024/256, rate 1 | disabled | 0.6783 | 314.6 | 91.3 | 23,599.8 |
| 1024/256, rate 1 | enabled | 0.6759 | 323.6 | 93.2 | 24,084.8 |
| 7168/512, c=32 | disabled | 0.2048 | 40,295.8 | 101.0 | 91,924.6 |
| 7168/512, c=32 | enabled | 0.2896 | 27,811.3 | 152.1 | 105,513.7 |

Enabled relative to disabled:

| Workload | Throughput | Mean TTFT | Mean TPOT | Mean E2E |
| --- | ---: | ---: | ---: | ---: |
| 1024/256, rate 1 | -0.36% | +2.88% | +2.04% | +2.06% |
| 7168/512, c=32 | **+41.38%** | **-30.98%** | +50.50% | +14.78% |

This is a long-context capacity/throughput result, not a general latency or
leaderboard improvement. The TPOT and E2E regressions are part of the result
and must be displayed with the throughput and TTFT gains.

## Capacity and operational behavior

- Active request ownership changed from 56 source blocks to 8 destination
  blocks for each 7168-token request.
- Each Qwen BF16 block occupied 24 MiB across all 48 layers; releasing 48
  blocks made 1.125 GiB of active KV capacity available per request.
- Peak reported KV-cache usage changed from 99.93% to 42.42%.
- Mean preemptions per lifecycle changed from 2 to 0.
- All six lifecycles ended with zero waiting requests and zero final KV usage.
- HBM process peak stayed near the fixed pool allocation, 81% disabled versus
  82% enabled, so this evidence does not claim lower total process allocation.

Prefix source blocks can remain immutable and cached. They are not counted as
active request ownership reduction unless the Core transaction releases them
from that request.

## Correctness and quality

The archived exact pair recorded 112 CPU tests passed with one skipped, five
real-NPU A2 tests passed, and real-NPU Llama/Qwen eager and
`FULL_DECODE_ONLY` end-to-end tests passed. Repeated greedy responses matched.
Those test logs remain in the archived working evidence and are summarized
here; the public JSON files below contain the matched performance and LongBench
aggregates.

LongBench used the fixed first 50 samples from `narrativeqa`, `qasper`, and
`2wikimqa`. The historical gate required mean QA-F1 drop no greater than 3 and
each task drop no greater than 5.

| Model | Disabled mean | Enabled mean | Mean drop | Maximum task drop | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama-3-8B-Instruct | 29.0309 | 28.2223 | 0.8086 | 1.4870 | Pass |
| Qwen2.5-14B-Instruct | 36.8439 | 34.4289 | 2.4150 | 4.2791 | Pass |

These small fixed samples are regression gates, not claims about complete
LongBench performance.

## Public evidence and integrity

- [Provenance](provenance.json)
- [Performance aggregate](performance-eager-prefix-aggregate.json)
- [Quality aggregate](quality-summary.json)

The files are byte-identical copies of the retained final evidence:

| File | SHA-256 |
| --- | --- |
| `provenance.json` | `84b4a0fb8ce2d92f6b8fb49f22df650f7127f56c55f43e69ff987e1ffb034385` |
| `performance-eager-prefix-aggregate.json` | `65a17d6bde2d350d2d4410e7fbe813d25f2c74d0b879372518efc50ee8556424` |
| `quality-summary.json` | `1a7c05d896149c08e350ad9489e72d91bff6b2d27ef31f27cb658dba794a3392` |

One environment exception is preserved in provenance:
`VLLM_ASCEND_TORCH_PREFLIGHT=0` bypassed a standalone fixed 20-second Core
probe that timed out on this host. It did not bypass engine validation: the
real NPU tests allocated tensors, each lifecycle reached `/health`, completed
inference, drained requests/KV usage, and returned to its HBM baseline. The
end-to-end tests separately ran with the Core preflight enabled.

## Current status

The current repository package remains `import_only`; refreshed Core and
Ascend main do not yet export the transactional compression host contract.
Before an active alpha or current-result card, repeat the relevant tests from
the pushed exact heads of the active Core, Ascend, and provider artifacts.
