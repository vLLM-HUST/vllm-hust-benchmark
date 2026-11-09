# PyramidKV Qwen3.5 TP2 C128: requested 456704-token pool

**One OFF/ON pair on 2026-11-04 Asia/Shanghai; Eager diagnostic, excluded from official
leaderboards.** Host timestamps in the originals use UTC (2026-11-03). The user requested
Qwen3.5-35B-A3B, TP2/C128, with 456,704 native effective token slots, continuing the existing
7168-input/512-output comparison. Both arms actually use **457,113 slots**, the nearest achievable
capacity under the unchanged hybrid cache layout.

## Capacity and comparison boundary

The only serving-argument change from the previous accepted Qwen3.5 C128 pair is
`--num-gpu-blocks-override 558`. Native capacity is `int(blocks / 10 * 8192)` for this
configuration: 557 blocks give 456,294 slots, and 558 give 457,113. The requested value falls
between them; the chosen value is 409 slots higher (+0.08955%). It is not an exact 456,704-slot run.
Both startup and final native metrics must confirm 558 physical blocks, 2048-token logical pages,
457,113 effective slots and max-length concurrency 55.8. Multiplying hybrid physical blocks by
logical page size would give the wrong capacity. The capacity formula is pinned in vLLM's
`get_kv_cache_capacity` and `get_max_concurrency_for_kv_cache_config` at the core commit below.

Both arms use the same two 64 GiB Ascend910B2 cards, physical 1/5 (PCI 01:00.0/41:00.0). The
previous Qwen3.5 C128 experiment used physical 1/6. Consequently cross-run comparisons include a
physical-card substitution and do not isolate a pool-size-only causal effect. The current OFF/ON
comparison is matched within the same device pair.

## Results

| Metric                               |    OFF |     ON |  Change |
| ------------------------------------ | -----: | -----: | ------: |
| Output throughput (tokens/s)         | 151.54 | 166.74 | +10.03% |
| Mean TTFT (s)                        | 121.88 |  93.60 | -23.21% |
| Median TTFT (s)                      |  74.43 |  79.70 |  +7.08% |
| p95 TTFT (s)                         | 272.29 | 246.26 |  -9.56% |
| Mean TPOT (ms)                       | 390.77 | 408.33 |  +4.49% |
| Mean request E2E (s)                 | 321.57 | 302.25 |  -6.01% |
| Peak active KV (%)                   |  99.82 |  99.82 |  +0.00% |
| Peak simultaneous two-card HBM (GiB) |  96.05 |  96.22 |  +0.18% |
| Exact lifecycle preemptions          |    141 |     98 |       — |

Output throughput changed by +10.03%, mean TTFT by -23.21%, mean TPOT by +4.49%, and mean E2E by
-6.01%. These metrics answer different questions: batch throughput, first-token waiting,
subsequent-token spacing, and total per-request time. One observation per arm does not estimate
run-to-run variance or establish task-quality preservation.

| Native effective capacity | Physical cards | OFF tokens/s | ON tokens/s | ON throughput change |
| ------------------------- | -------------- | -----------: | ----------: | -------------------: |
| 644,710 tokens            | 1/6 (previous) |       175.78 |      187.12 |               +6.45% |
| 457,113 tokens            | 1/5 (current)  |       151.54 |      166.74 |              +10.03% |

The
[previous C128 report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8afbf04a57963dcefd5c9f1bb54556a023173740/reports/pyramidkv-pressure-20261103)
is immutable; its analysis is included with SHA256
`3a2eae1a384e7575fb27847d99dbd2bf929c323aea64259f07f042080c47b70a`. The larger pool was not rerun on
the current cards.

## Frozen settings and validation

Qwen3.5-35B-A3B, TP2/PP1, BF16, text-only, Eager/no MTP, 128 maximum sequences, context 8192,
batch-token budget 4096, utilization 0.8, requested kernel block size 128, resolved hybrid logical
page size 2048, Mamba cache `align`, seed 0, prefix caching, chunked prefill, async scheduling, and
`generation-config=vllm` are retained. Each fresh server handles 48 short 1024/256 requests at rate
1 before the target cohort of 128 simultaneous 7168/512 requests; one separate first-prompt
readiness request precedes each cohort. Total: 352 measured requests and 4 readiness requests. Only
one OFF/ON pair.

The request file, client, algorithm config and candidate wheel are byte-identical to the previous
accepted Qwen3.5 C128 experiment. The client uses `TCPConnector(limit=0)` so the client connection
limit does not cap 128 requests or starve monitoring. Request SHA256:
`cff89e9ce8fb72911b4f35ad3ecd3c821e2f5dbf4395dd367bc7af2a5ad38d4e`. `lineage.json` maps all reused
inputs to their prior names. `pressure-coverage.json` checks 128 server-side running-plus-waiting
requests before the first completion and reports telemetry errors, sampling gaps and request launch
spread.

Candidate wheel 0.4.0rc2.dev0, source `c8742d038a1a5680c6de21dbd126b647a5fd3944`; core
`d0f22d2bda562156e4dbf433ce645e1769b4f804`, Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`.
Runtime: CANN 9.1, Torch 2.10.0+cpu, torch-npu 2.10.0.post4, transformers 5.14.1. Installed metadata
versions vLLM 0.23.0+empty and Ascend 0.23.0.post1 differ from the prepared ABI selector
`VLLM_VERSION=0.25.1`. Clean native source commits, runtime versions, environment hash, and all 28
model file size/mtime pins were checked before freezing. Previously computed full model hashes are
reused only after those stat checks. CPU quota is 8 cores, host memory limit 64 GiB. Startup-only
checkpoint cache eviction is recorded; none is performed during inference.

PyramidKV settings remain capacity 512, beta 20, window 8, kernel 7, maxpool, KV-head/mean GQA, no
merge, and minimum prompt length 4096. The model has 10 full-attention and 30 linear-attention
layers. A 7168-token admitted prompt compresses from 4 logical pages to 1 (2048 physical tokens);
layer-specific selections are distinct from physical allocation. Every observed scheduler commit is
checked against acknowledgements from both TP ranks.

**Frozen full-compression acceptance was not met.** The original analyzer expected 129 commits
including readiness; only 128 are present (127/128 target requests plus readiness), with 256
matching TP-rank acknowledgements. Target request `long-109` completed successfully but has no
commit. Its 6144 cached tokens do not prove the cause: other requests with the same cached-token
count did compress. A resumed prefill crossing the initial prompt boundary is a code-supported
hypothesis, not a confirmed per-request explanation. The retained trace cannot establish why this
request missed compression or continuous compression state after preemption. This is a comparison of
the enabled plugin policy, not uniform compression of all 128 requests.

`analysis-amendment.json` and `frozen-analysis-failure.log` preserve the failed original criterion.
Versioned `analyze_v2.py`/`audit_v2.py` recompute every request, including `long-109`, validate
observed transaction integrity, and explicitly report incomplete coverage. They change offline
reporting only; no frozen scripts, runtime, raw evidence or measurement parameters were modified. No
repeat experiment was run. Mutation tests check rejection of missing rank ACKs,
duplicate/unknown/malformed commits and incorrect geometry.

Contract SHA256: `e6f69e5485f11035abf5bd165dc319b7c6b004989a09460ff5159eb9e9179365`. No plugin
algorithm or runtime source was changed.

## Memory and timing definitions

This hybrid layout uses 10 shared tensor slots, each with 2,121,728 bytes per physical block. Thus
558 blocks allocate 11.026153564 GiB of cache tensors per rank, compared with 15.551223755 GiB at
787 blocks. The pool includes the hybrid layout's padded attention and state pages. Worker
`Available KV cache memory` is a profiled budget, not actual tensor allocation after the override.
Total HBM is separately sampled and includes weights, workspace and runtime allocations.

KV usage is active block occupancy inside the configured pool. HBM sum is the maximum simultaneous
two-card sample, not the sum of separate per-card maxima. Metrics are sampled about every 0.2
seconds, HBM about every 2 seconds; transient peaks may be missed. Exact preemption counts span the
fresh server lifecycle including readiness and the short prelude; cohort sampled lower bounds remain
separately labeled in `analysis.json`.

TTFT ends at the first choice event, including an empty choice. E2E ends at the last choice,
excluding usage/DONE. TPOT divides the interval after the first choice by output tokens minus one.
Throughput divides completed output tokens by cohort wall time. Raw per-request latencies, median
and p95 are retained. File writes and monitoring add common overhead. Synthetic performance does not
measure answer quality.

## Originals and reproduction

The capsule contains frozen inputs, launchers, wheel, model/runtime identities, complete server
logs, raw SSE, timestamped events, results, telemetry, pool confirmations and cleanup receipts. The
verifier checks exact request payloads and token usage, SSE/event agreement, recomputed
timing/aggregates, TP-rank commits, pool identity, actual concurrency, zero final active/waiting
requests and freed devices. Known shutdown-only native errors are checked separately from serving
errors.

`SHA256SUMS` covers public files; `members.json.gz` covers every archived member. With Python 3.12
or newer, CPU only and no runtime/model dependencies:

```sh
python verify.py
```

The verifier safely extracts originals, verifies all member hashes, recomputes analysis and
coverage, and byte-compares them to the public JSON. Pinned public files are downloaded again and
verified after upload.
