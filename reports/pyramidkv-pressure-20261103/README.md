# PyramidKV pressure test at 64 and 128 concurrent requests

**Single-round Eager diagnostic; excluded from official performance leaderboards.** The user
requested doubling Qwen2.5 concurrency from 32 to 64 and quadrupling Qwen3.5 concurrency from 32 to
128\. Each model has one fresh-process OFF/ON pair. Both request concurrency and the server sequence
limit increase together. The automatic KV pool is retained. Different concurrency between models
means this measures their response to pressure, not the isolated effect of model choice.

Both OFF arms reached practical pool saturation: Qwen2.5 peaked at 100% with 3 lifecycle
preemptions; Qwen3.5 at 99.62% with 1. Enabling PyramidKV removed those preemptions and increased
batch output throughput by 18.94% and 6.45%, respectively. Mean TTFT improved by 19.29% and 4.00%.
The tradeoff remains: TPOT increased by 21.90% and 7.38%, and mean request E2E increased by 8.87%
and 3.83%. Faster completion of the entire batch therefore does not imply lower average per-request
latency.

At C32, enabling the same candidate reduced throughput by 17.25% and 3.68%, with zero baseline
preemptions. This pressure test demonstrates that the observed benefit depends on workload pressure.
It does not attribute the difference to a specific kernel or establish repeatability from a single
pair.

## Qwen2.5-14B-Instruct C64

| Metric                               |        OFF |         ON |  Change |
| ------------------------------------ | ---------: | ---------: | ------: |
| Output throughput (tokens/s)         |     189.38 |     225.24 | +18.94% |
| Mean TTFT (ms)                       |  39,603.80 |  31,964.89 | -19.29% |
| Mean TPOT (ms)                       |     167.58 |     204.27 | +21.90% |
| Mean end to end (ms)                 | 125,236.71 | 136,347.79 |  +8.87% |
| Peak active KV (%)                   |     100.00 |      26.35 | -73.65% |
| Peak simultaneous two-card HBM (MiB) | 107,404.00 | 107,708.00 |  +0.28% |
| Peak running requests                |      51.00 |      64.00 | +25.49% |
| Peak waiting requests                |      58.00 |      59.00 |  +1.72% |

| Arm | Native blocks | Logical page | Effective token capacity | Rank0 KV budget GiB | Exact lifecycle preemptions |
| --- | ------------: | -----------: | -----------------------: | ------------------: | --------------------------: |
| OFF |          2916 |          128 |                   373248 |               34.18 |                           3 |
| ON  |          2916 |          128 |                   373248 |               34.17 |                           0 |

## Qwen3.5-35B-A3B C128

| Metric                               |        OFF |         ON |  Change |
| ------------------------------------ | ---------: | ---------: | ------: |
| Output throughput (tokens/s)         |     175.78 |     187.12 |  +6.45% |
| Mean TTFT (ms)                       |  88,519.66 |  84,981.47 |  -4.00% |
| Mean TPOT (ms)                       |     381.87 |     410.07 |  +7.38% |
| Mean end to end (ms)                 | 283,655.49 | 294,524.95 |  +3.83% |
| Peak active KV (%)                   |      99.62 |      82.95 | -16.73% |
| Peak simultaneous two-card HBM (MiB) | 107,649.00 | 107,788.00 |  +0.13% |
| Peak running requests                |     112.00 |     128.00 | +14.29% |
| Peak waiting requests                |     121.00 |     122.00 |  +0.83% |

| Arm | Native blocks | Logical page | Effective token capacity | Rank0 KV budget GiB | Exact lifecycle preemptions |
| --- | ------------: | -----------: | -----------------------: | ------------------: | --------------------------: |
| OFF |           787 |         2048 |                   644710 |               15.56 |                           1 |
| ON  |           787 |         2048 |                   644710 |               15.56 |                           0 |

Median TTFT increased by 12.77% for Qwen2.5 and 6.93% for Qwen3.5. The mean improvement does not
describe every part of the latency distribution.

## Configuration and relationship to the previous run

The
[previous C32 comparison](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/6c136e521fa6ca09fc89a2d70d4ac40946fab48f/reports/pyramidkv-model-comparison-20261103)
used the same model weights, plugin wheel, native runtime, physical Ascend910B2 cards 1/6, TP2,
BF16, Eager, no MTP, 7168 input tokens and 512 forced output tokens. This run keeps max model length
8192, batch-token budget 4096, utilization 0.8, APC, async scheduling, chunked prefill, seed 0 and
generation-config=vllm. Qwen3.5 retains native Mamba align; Qwen2.5 has no Mamba state. The HTTP
timeout increases from 600 to 1800 seconds in both arms to allow pressure queueing/recomputation;
successful-request metric definitions stay unchanged.

Candidate source `c8742d038a1a5680c6de21dbd126b647a5fd3944` and wheel 0.4.0rc2.dev0 are reused byte
for byte. Source PR35 has since merged without changing the measured runtime code. This is not a
measurement of the published 0.4.0rc1 wheel. CANN 9.1, Torch 2.10.0+cpu, torch-npu 2.10.0.post4 and
transformers 5.14.1 are unchanged. Native core `d0f22d2bda562156e4dbf433ce645e1769b4f804` and Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989` remain pinned. Installed vLLM metadata is 0.23.0+empty,
Ascend 0.23.0.post1; VLLM_VERSION=0.25.1 selects the prepared ABI branch, rather than identifying
the installed version.

PyramidKV retains capacity 512, beta 20, window 8, kernel 7, maxpool, KV-head/mean GQA, no merge,
and prompt admission above 4096 tokens. Dense allocation after compression is 1024 physical tokens;
hybrid logical alignment requires 2048. Layer-specific selected lengths remain distinct from the
physical allocation.

The original common vocabulary and seed 0 generator extend the long cohort to 128 unique literal
prompts. Whole prompts encode to exactly 7168 tokens under both tokenizers. The first 32 are
byte-identical to the previous run; Qwen2.5 uses the first 64 of Qwen3.5's 128. No old request is
duplicated to inflate load. OFF/ON inputs within a model are identical. Request files are pinned in
contract SHA256 `c6a8d0606bb1061e55ccb4cdc9e942a47a7b7f40e60f6e67a301dfc6f1bf05e9`.

Each lifecycle retains 48 short 1024/256 requests at rate 1, then 64 or 128 long requests sent
simultaneously. One separate readiness request precedes each cohort; the long readiness reuses its
first prompt, identically in both arms. Thus 576 measured and 8 readiness requests are verified,
including 384 target long requests. Each model has one accepted pair. An initial Qwen3.5 OFF attempt
was aborted when the default aiohttp connector capped active HTTP requests at 100 and blocked
metrics polling. Its full partial evidence is retained and excluded from performance aggregates.
connection-limit-amendment.json freezes the fix: TCPConnector(limit=0), applied identically to both
accepted Qwen3.5 arms. A local behavioral test reproduces 100/128 arrivals with the original client
and verifies 128 held requests plus live metrics with the corrected client. Qwen2.5 C64 never
reached the original 100-connection limit and was not repeated. Original contract and amended
contract `03a2592aca7ca9de81979824383832660ad49c8aa636c62967af5428a95c0c13` are both preserved; no
model, request or algorithm setting changed during this correction. The client fix and local HTTP
regression test are merged in
[source PR36](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/36). Startup-only file
cache eviction is recorded to keep checkpoint loading within the 64 GiB host cgroup; no eviction
occurs during inference.

## How to interpret the metrics

KV usage measures active blocks within the preallocated pool. Reclaiming blocks increases room for
other requests; it does not release that entire allocation back to device HBM. Dense and hybrid
native pool layouts differ; the hybrid reported effective token capacity cannot be divided by its
logical page size to recover physical block count. Pool sizes above come directly from native
cache_config_info metrics.

Preemption totals are exact final counters across each fresh lifecycle, including its readiness and
short prelude. The JSON also exposes sampled cohort lower bounds, which may miss an unsampled tail.
Running/waiting peaks and KV usage are sampled about every 0.2s; two-card HBM about every 2s. HBM
sum is the maximum simultaneous sampled sum, not the sum of separate per-card maxima.

TTFT ends at the first streamed choice, including an empty choice; E2E ends at the last choice,
excluding usage/DONE. TPOT is the interval after the first choice divided by output tokens minus
one. Throughput is completed output tokens divided by cohort wall time. Per-request records and
cohort mean/p50/p95 are preserved. Synchronous artifact writes and telemetry add common
instrumentation overhead. A single pair provides no run-to-run variance estimate, and synthetic
throughput does not establish task-quality parity. Differences against C32 also include additional
unique requests and scheduling effects, not just resident KV demand.

## Raw evidence and verification

The four accepted lifecycles retain every server log, raw SSE stream, timestamped event, request
result, telemetry sample, final native metric and shutdown receipt. Frozen launchers, inputs,
package and model identities are also preserved. Model weights are referenced by hashes/stat pins
rather than duplicated. For these accepted lifecycles, every request must match 7168/512 or 1024/256
usage, both TP ranks must acknowledge admitted scheduler commits, final running/waiting counters
must be zero, and both devices must be free after shutdown. Known native output-handler exceptions
after explicit shutdown are separated from serving errors; no serving error is accepted.

The aborted Qwen3.5 attempt has its own archive: it preserves partial streams, 128 canceled long
request results, successful prelude/readiness records, interruption and cleanup receipts. It has no
final-metrics.prom and none of its requests enter the accepted performance aggregates. Its archived
bytes are integrity-checked, not asserted to be successful completed requests.

Run with Python 3.12 or newer, CPU only:

```sh
python verify.py
```

The verifier checks SHA256SUMS, every archived member, safe extraction, exact request
identities/usage, raw SSE/event correspondence, independent metric recomputation, TP-rank
acknowledgements, native pools and cleanup. Its recomputed analysis must match analysis.json byte
for byte. pressure-coverage.json additionally verifies that all 64 or 128 requests were visible as
running plus waiting on the server before any completed, and records telemetry gaps/errors. The
public archives are also read back after upload. No model installation is needed for offline
verification.
