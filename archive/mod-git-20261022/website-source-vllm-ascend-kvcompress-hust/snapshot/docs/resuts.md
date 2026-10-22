# Current Benchmark Results

English | [简体中文](resuts.zh.md)

This document is the single public summary of benchmark results for the current
repository version. The README intentionally contains no A/B result table.
Commands and reporting rules are maintained separately in the
[benchmarking guide](benchmarking.md).

## Status

The current implementation has passed functional, capacity, and serving
validation on one Ascend 910B2 in ACL graph mode. Compression was active in all
TriAttention runs, every request completed, and the tested 8192-token prompts
were reduced from 64 physical blocks to 16.

The main outcome is workload-dependent:

- long-context total-token throughput improved by **10.25%**;
- high-concurrency KV-pressure throughput improved by **4.68%**;
- the short prefix workload was effectively unchanged at **+0.01%**;
- the short random workload retained **-1.25%** fixed overhead; and
- the 64-case long-context recall check changed from 64/64 to 63/64, a
  **1.5625 percentage-point** decrease.

The quality decrease is material: the current 2048-token budget does not meet a
hypothetical guardrail of at most one percentage point on this 64-case check.
These results therefore support engineering validation, not a universal
performance or quality claim.

## Validated Configuration

| Item | Value |
| --- | --- |
| Device | One Ascend 910B2, 64 GiB HBM |
| Model | Qwen2.5-Coder-14B-Instruct |
| Execution | ACL graph (`FULL_AND_PIECEWISE` capture/replay), normal torch-npu startup preflight |
| KV layout | Separate BF16/FP16 K/V, block size 128 |
| Maximum model length | 12288 |
| Async scheduling / prefix caching | Disabled / disabled |
| Independent Knorm compressor | Disabled |
| TriAttention budget | 2048 tokens |
| Recompute / protected recent window | 128 / 128 tokens |
| Score layer stride | 4 |

Baseline and compression services used the same model, device, execution mode,
KV pool, request set, concurrency, and sampling configuration. The compression
configuration was the intended service-side difference.

## End-to-End A/B Results

All values are from matched single runs. Total throughput includes prompt and
output tokens. Latency values are means in milliseconds. Deltas are
TriAttention relative to baseline.

| Scenario | Requests x actual input/output | Total tok/s, baseline to compression | Delta | TTFT, baseline to compression | TPOT, baseline to compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix repetition | 4 x 2560/256, c=1 | 336.77 to 336.82 | **+0.01%** | 381.27 to 372.04 | 30.91 to 30.94 |
| Random | 2 x 2560/32, c=1 | 1522.03 to 1502.95 | **-1.25%** | 358.53 to 367.05 | 30.82 to 31.24 |
| Long context | 8 x 8192/256, c=4 | 2215.61 to 2442.73 | **+10.25%** | 1770.26 to 1791.57 | 47.86 to 42.27 |
| KV pressure | 16 x 8192/64, c=16 | 5787.01 to 6058.10 | **+4.68%** | 10497.88 to 10592.61 | 156.12 to 161.26 |

The four compression workloads completed 30/30 requests with zero failures.
Each matched pair persisted identical input and output lengths. The prefix and
long-context clients used the benchmark snapshot's generic 256-token output
length; the table reports persisted lengths rather than the smaller dedicated
override supplied to the client.

## Capacity and Stateful Compression

For every 8192-token prompt, the committed plan changed the physical cache from
64 blocks to 16 blocks and returned 48 blocks. At block size 128, this is an
8192-to-2048 reduction in physical prompt KV, or **75%**. The semantic length
remained 8192.

Repeated compression was also observed on one request:

- at semantic length 2560, 20 physical blocks became 16 and 4 blocks were
  returned;
- after decode reached semantic length 2688, 17 physical blocks again became
  16 and one more block was returned.

This verifies that the current runtime is not limited to one final-prefill
compression transaction.

The allocator reserves the KV pool at service startup, so returned blocks do
not normally reduce process-level HBM shown by `npu-smi`. Returned block counts,
committed physical tokens, and active KV-pool occupancy are the relevant
capacity signals.

## Kernel Microbenchmarks

The following NPU timings compare the previous generic hot paths with the
current optimized paths:

| Hot path | Previous | Current | Speedup |
| --- | ---: | ---: | ---: |
| K/V materialization | 0.582 ms | 0.330 ms | 1.76x |
| Paged-key scoring | 1.258 ms | 0.728 ms | 1.73x |

The offset tensor operation itself measured 0.055 ms. The decode-path benefit
comes mainly from avoiding a per-step CPU-to-NPU offset refresh and a second
full slot-mapping computation.

Microbenchmarks isolate kernels and do not predict end-to-end gains by
themselves. The A/B table above is the serving-level result.

## Correctness and Quality

- Python test suite: **44 passed**.
- Ruff checks: passed.
- `git diff --check`: passed for the validated implementation snapshot.
- Ascend kernel compilation and numerical smoke test: passed within the
  configured tolerance against the generic scoring path.
- Long-context recall check: baseline **64/64 (100%)**; compression
  **63/64 (98.4375%)**.

All-layer scoring, the generic scoring implementation, and the specialized
kernel reproduced the same missed case. The observed quality difference is
therefore not hidden as kernel numerical error. The current default keeps
four-layer-stride scoring for its serving performance. Workloads that require a
stricter quality guardrail should evaluate a larger KV budget, different
calibration statistics, or an adaptive policy.

## Interpretation Boundaries

These are current single-device engineering results for one model and one set
of workloads. They do not establish:

- cross-model or cross-platform speedup;
- lower process-level allocated HBM;
- production quality at every KV budget;
- multi-device, async, speculative, hybrid/MLA, or quantized-KV support; or
- formal third-party acceptance.

Any future result update should replace this summary rather than append a dated
history. Dated commands, raw output, failed attempts, machine paths, and
intermediate analysis belong under `docs/dev/`.
