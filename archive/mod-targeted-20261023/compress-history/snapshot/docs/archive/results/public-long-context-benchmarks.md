# Public Long-Context Benchmarks

English | [简体中文](public-long-context-benchmarks.zh.md)

> **Historical archive.** These 2026-09-20 public experiments remain visible,
> including negative performance results, but are not the current 0.8 Frontier
> cohorts. See [Frontier benchmarking](frontier-benchmarking-0.9.md).

Updated: 2026-09-20. This record supplements the synthetic commissioning test
with three public long-context/KV scenarios. It is a same-host engineering
comparison, not formal V4.6 official-baseline acceptance or the 30-minute A3
stability test.

Standard release testing is frozen to Qwen2.5-14B-Instruct by the acceptance
requirements. Qwen3.5-35B-A3B is supplementary adaptation evidence for a
related hybrid architecture. Earlier Qwen2.5-Coder runs remain in the history
of the [HTML benchmark leaderboard](../../results/benchmark-leaderboard.html).

## Scenarios and frozen inputs

| Scenario | Purpose | Evaluated scope | Input tokens | Prepared SHA-256 |
| --- | --- | ---: | ---: | --- |
| LongBench-v2 | Long-context multiple-choice reasoning | 116 eligible short-stratum cases | 10,171–31,869 | `06255143...b538be6c` |
| LongBench `passage_retrieval_en` | Passage retrieval | 200/200 cases | 10,412–15,670 | `cc015530...98a2cb6` |
| LongBench `qasper` | Scientific-paper QA | 93/200 cases at or above the 5,120-token pressure floor | 5,168–22,074 | `20b04612...b8a6da94` |

LongBench-v2 is pinned to Hugging Face revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`; its downloaded `data.json`
hash is `15d61c22...604c7fe2`. The LongBench archive is pinned to revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`; `data.zip` hashes to
`cb45b11a...857f7f64`.

No accepted prompt is truncated. The service limit is 32,768 tokens, so the
387 LongBench-v2 medium/long-stratum cases outside that scope are recorded in
the generated unsupported file rather than shortened. Qasper cases below
5,120 tokens are excluded because they do not exercise the original 4K
compression threshold. `--limit 0` means all eligible cases, not a sample.

The downloader verifies the pins and hashes before extraction. LongBench-v2's
data card declares Apache-2.0. LongBench combines source datasets with their
own terms, so the scripts download it from upstream but neither the package nor
this repository redistributes benchmark text. Users remain responsible for
the terms of each source dataset.

## Reproduction

Download and prepare with the exact model tokenizer:

```bash
MODEL=/path/to/Qwen2.5-14B-Instruct
python scripts/kvcompress_benchmark_data.py download \
  --source all --root .benchmarks/datasets

python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-v2 --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-v2.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-passage-retrieval-en --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-passage-retrieval-en.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-qasper --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-qasper.jsonl
```

Models such as Qwen3.5 that enable reasoning by default must add
`--disable-thinking` to all three `prepare` commands. The option passes
`enable_thinking=false` to the chat template and freezes
`chat_template_enable_thinking: false` in the request-set manifest. Do not mix
a request set rendered with the default thinking template into a non-thinking
B0/B1 comparison.

The frozen B0 is the same plugin and host with a 32,768-token
budget/32,896-token threshold, so no request can compress. The 0.6 candidate
reuses that B0 because the host, model, tokenizer, datasets, and service flags
are unchanged. B1 uses [the example configuration](../../../examples/triattention.json):
8,192-token budget, 1,024-token recompute window, 512 recent protected tokens,
8,192-token scoring chunks, layer stride 8, and a 64-token minimum requested
output before compression is worthwhile.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve "$MODEL" \
  --served-model-name qwen2.5-14b --dtype float16 \
  --block-size 128 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 4 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm
```

Run each prepared file against each service arm, changing paths and labels as
needed:

```bash
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-v2.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-14b \
  --result .benchmarks/results/b1-longbench-v2.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 0 \
  --server-log .benchmarks/logs/b1.log

# passage_retrieval_en has max_output_tokens=32 and must exercise the bypass:
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-passage-retrieval-en.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-14b \
  --result .benchmarks/results/b1-passage-retrieval-en.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 0 \
  --server-log .benchmarks/logs/b1.log \
  --compression-expectation forbidden

python scripts/kvcompress_benchmark_score.py \
  --baseline .benchmarks/results/b0-longbench-v2.json \
  --plugin .benchmarks/results/b1-longbench-v2.json \
  --quality-tolerance-pp 1 \
  --output .benchmarks/results/paired-longbench-v2.json
```

The server must use `--generation-config vllm`. The runner fixes
`temperature=0`, `top_p=1`, `top_k=-1`, `min_p=0`, presence/frequency
penalties 0, `repetition_penalty=1`, `n=1`, beam search off, an empty stop
list, `seed=0`, and `add_special_tokens=true`. It saves per-request
predictions, TTFT/TPOT/E2E, package versions,
device samples, server-reported token counts, and scheduler/worker compression
evidence. The scorer implements the official LongBench-v2 answer extraction,
LongBench retrieval score, and LongBench QA F1 normalization. A B1 result fails
if a request fails, prompt token counts reveal truncation, the declared
compression expectation is violated, an acknowledgement is missing, or quality
drops by more than one percentage point. The default expectation is `required`;
the short-output retrieval run declares `forbidden` to prove the bypass rather
than silently accepting an uncompressed run.

## Results

### Standard model: Qwen2.5-14B-Instruct

Model: `Qwen/Qwen2.5-14B-Instruct`, FP16. Each service uses one Ascend 910B2.
Each arm is one cold-service run, unlimited arrival rate, concurrency 4, with
the frozen server and client sampling contract. B0 is the same-host
no-compression engineering control described above.

| Scenario | B0 → B1 quality | Requests / compression | Request throughput | Mean TTFT | Mean TPOT | Mean E2E | Physical blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 40.52 → 39.66 accuracy (−0.86 pp) | 116 / 116 | **+7.78%** | +4.00% | **−4.34%** | **−6.99%** | −61.96% |
| Passage retrieval | 98.75 → 98.75 (0.00 pp) | 200 / 0 (bypassed) | **+2.41%** | +0.36% | **−4.13%** | **−2.33%** | n/a |
| Qasper | 43.60 → 43.36 F1 (−0.24 pp) | 93 / 11 | **+1.35%** | **−2.94%** | **−1.08%** | **−1.41%** | −38.73% on compressed cases |

All 818 arm-requests completed with zero failures or silent truncations. B1
produced 127 scheduler commits and 127 worker acknowledgements. Across
compressed cases, 20,666 source blocks became 8,128 destination blocks, a
60.67% reduction. All three quality changes remain within the frozen 1 pp gate.

The workload boundary is material. The 10K–16K retrieval task requests at most
32 output tokens, so the 64-token gate bypasses scoring and copy work; all 200
cases show zero commits/acknowledgements. Qasper is mostly below B1's 9,216-token
threshold, so only 11 cases compress. Each arm has only one run, and tests on
other NPUs ran concurrently; performance deltas are exploratory engineering
evidence. Quality, request integrity, and compression transactions are the
primary conclusions.

Machine-readable evidence is in the
[Qwen2.5 public benchmark summary](../../results/evidence/kvcompress-working-tree-20260920-qwen25-public-summary.json).

### Supplementary model: Qwen3.5-35B-A3B

Qwen3.5 uses BF16, TP=2, eager execution, and a non-thinking chat template;
each service uses two Ascend 910B2 devices. It is not the standard acceptance
model.

| Scenario | B0 → B1 quality | Requests / compression | Request throughput | Mean TTFT | Mean TPOT | Mean E2E | Physical blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 47.27 → 47.27 accuracy (0.00 pp) | 110 / 110 | −5.83% | +5.66% | +4.44% | +4.97% | −63.76% |
| Passage retrieval | 100.00 → 100.00 (0.00 pp) | 200 / 0 (bypassed) | −2.39% | +1.65% | +3.10% | +2.54% | n/a |
| Qasper | 51.07 → 50.64 F1 (−0.43 pp) | 93 / 11 | −1.28% | +5.43% | +9.69% | +2.26% | −42.11% on compressed cases |

All 806 arm-requests completed with zero failures or silent truncations. B1
produced 121 scheduler commits and 242 per-rank worker acknowledgements under
TP=2. Every quality gate passed, but this host requires eager execution and the
explicit legacy GDN ABI bridge, and performance did not beat B0. The result
demonstrates a working hybrid-state, TP synchronization, and transaction path.
The bypassed retrieval run does not establish compressed needle retention.
Machine-readable evidence is in the
[Qwen3.5 public benchmark summary](../../results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json).

## Remaining limits

- The performance comparison has one run per arm rather than three cold runs,
  and other NPUs on the host carried concurrent test load.
- B0 is a same-host no-compression engineering control, not the prescribed
  official V4.6 host baseline.
- LongBench-v2 medium and long strata require a service/model with a context
  limit above 32K and remain untested here.
- Qwen3.5 is supplementary adaptation evidence; standard release tests remain
  on Qwen2.5-14B-Instruct.
- This is not the 30-minute/six-window A3 stability test; A3 is recorded
  separately.
