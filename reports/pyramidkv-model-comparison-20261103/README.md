# PyramidKV model comparison: shared TP2 eager profile

**Diagnostic Eager measurements, excluded from official performance leaderboards.** This report
compares the effect of enabling the independent PyramidKV candidate within each model, then compares
those changes between models. It does not isolate architecture from parameter count, MoE, weight
footprint or cache layout, and it does not establish task-quality parity.

Candidate source:
[`c8742d038a1a5680c6de21dbd126b647a5fd3944`](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/tree/c8742d038a1a5680c6de21dbd126b647a5fd3944).
Measured wheel: `0.4.0rc2.dev0`; this is not the published `0.4.0rc1` wheel. The preceding
[Qwen2.5 TP1 reproduction](../pyramidkv-qwen25-matched-20261103/) used a different prompt bank and
2048 batch budget; absolute differences against that run must not be called a TP-only effect.

## Frozen comparison

Both models use the same two physical Ascend910B2 cards1/6 (visible0/1), BF16, TP2, Eager/NONE, no
MTP, APC/async/chunked prefill, seed0, context8192, 32 maximum sequences, 4096 batch-token budget,
automatic utilization0.8, text-only requests and `generation-config=vllm`. Qwen3.5 alone requires
Mamba `align`; dense Qwen2.5 has no Mamba state. Requested kernel pages are128 tokens; Qwen3.5
retains native2048-token logical cache alignment.

The runtime is the user-approved current CANN9.1, Torch2.10.0+cpu, torch-npu2.10.0.post4,
transformers5.14.1. Installed metadata says vLLM0.23.0+empty and Ascend0.23.0.post1;
`VLLM_VERSION=0.25.1` selects the prepared ABI branch. Pinned core
`d0f22d2bda562156e4dbf433ce645e1769b4f804` and Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989` are
preserved in the capsule. Both arms use the same explicit host worker.

Compression remains512/beta20/window8/kernel7/maxpool, KV-head/mean GQA, no merge, and admission
above4096 prompt tokens. At7168, dense materialization allocates 1024 physical tokens (8 private
pages), while hybrid alignment allocates2048 (1 logical page). Layer-specific retained lengths
remain distinct.

All four arms receive identical literal synthetic English-word prompts. Whole prompts were verified
to encode to exactly1024 or7168 tokens under both local tokenizers before freezing. Requests SHA256:
`6a9d0dccc06241424d9d4cd8ea6801779844f53864f8a8158809b64649f8a672`. Each fresh lifecycle has48
short1024/256 requests at rate1, followed by32 long7168/512 requests at rate infinity/concurrency32.
One separate first-prompt readiness request precedes each cohort; no additional warmups.
`ignore_eos=true` forces exact output length. All320 measured requests and8 readiness requests are
independently verified from raw SSE and timestamped events.

The original contract planned three pairs per model. While the first round was running, the user
explicitly requested only one pair per model. The original contract remains intact;
`single-round-amendment.json` records the four selected lifecycles. These are single observations,
with change calculated directly from OFF and ON metrics; no run-to-run variance estimate is
available. No model, request or runtime parameters were changed, and no selective retry occurred.
The native pool was not shrunk to manufacture pressure. Startup-only file-cache eviction events are
retained; no eviction occurs during measured requests.

## Qwen2.5-14B-Instruct

| Metric                               |        OFF |         ON |  Change |
| ------------------------------------ | ---------: | ---------: | ------: |
| Output throughput (tokens/s)         |     182.04 |     150.64 | -17.25% |
| TTFT (ms)                            |  13,764.59 |  15,555.65 | +13.01% |
| TPOT (ms)                            |     142.81 |     174.06 | +21.89% |
| End-to-end (ms)                      |  86,738.26 | 104,501.88 | +20.48% |
| Peak active KV usage (%)             |      65.87 |      14.27 | -78.33% |
| Peak simultaneous two-card HBM (MiB) | 107,404.00 | 107,548.00 |  +0.13% |

Individual fresh-process runs:

| Run          | Output tokens/s |   TTFT ms | TPOT ms | Peak active KV |
| ------------ | --------------: | --------: | ------: | -------------: |
| qwen25-off-1 |          182.04 | 13,764.59 |  142.81 |       0.658662 |
| qwen25-on-1  |          150.64 | 15,555.65 |  174.06 |       0.142710 |

Native cache configuration by arm:

- OFF: 2916 native blocks, logical block size 128; effective context capacity 373248 tokens; rank0
  available KV budget [34.18] GiB (rounded host log).
- ON: 2916 native blocks, logical block size 128; effective context capacity 373248 tokens; rank0
  available KV budget [34.18] GiB (rounded host log).

## Qwen3.5-35B-A3B

| Metric                               |        OFF |         ON |  Change |
| ------------------------------------ | ---------: | ---------: | ------: |
| Output throughput (tokens/s)         |     111.51 |     107.40 |  -3.68% |
| TTFT (ms)                            |  17,182.87 |  17,665.35 |  +2.81% |
| TPOT (ms)                            |     233.71 |     243.29 |  +4.10% |
| End-to-end (ms)                      | 136,606.45 | 141,985.65 |  +3.94% |
| Peak active KV usage (%)             |      29.26 |      21.88 | -25.22% |
| Peak simultaneous two-card HBM (MiB) | 107,444.00 | 107,460.00 |  +0.01% |

Individual fresh-process runs:

| Run          | Output tokens/s |   TTFT ms | TPOT ms | Peak active KV |
| ------------ | --------------: | --------: | ------: | -------------: |
| qwen35-off-1 |          111.51 | 17,182.87 |  233.71 |       0.292621 |
| qwen35-on-1  |          107.40 | 17,665.35 |  243.29 |       0.218830 |

Native cache configuration by arm:

- OFF: 787 native blocks, logical block size 2048; effective context capacity 644710 tokens; rank0
  available KV budget [15.55] GiB (rounded host log).
- ON: 787 native blocks, logical block size 2048; effective context capacity 644710 tokens; rank0
  available KV budget [15.56] GiB (rounded host log).

## Interpretation and metric boundaries

In this single round, enabling PyramidKV reduced active KV occupancy for both models but reduced
output throughput: Qwen2.5 by17.25%, Qwen3.5 by3.68%. Both OFF arms had zero lifecycle preemptions
and substantial free pool capacity. Unlike the earlier TP1 run, this workload therefore offered no
preemption to remove. Lower occupancy did not compensate for the enabled path's total cost. This
identifies the observed tradeoff, not a profiling attribution to a specific kernel. A single round
cannot establish the repeatability of these differences.

The models differ in more than name: Qwen2.5 has48 full-attention layers; Qwen3.5 has10
full-attention layers plus30 linear-attention layers. PyramidKV materializes the full-attention KV,
while native linear state remains present. Hybrid logical alignment also raises the physical
allocation floor. These properties explain why the same nominal algorithm budget does not mean the
same fraction of whole-model cache can be reclaimed.

KV usage is active block occupancy inside a preallocated pool. Lower occupancy does not imply a
matching reduction in device HBM. HBM is sampled on both cards; the sum peak is the maximum
simultaneous sampled sum, not the sum of independent per-card maxima. Native `num_gpu_blocks` is
recorded directly. Hybrid reported context-token capacity is an effective capacity and cannot be
divided by logical block size to recover physical block count.

TTFT ends at the first streamed choice event, including an empty choice; E2E ends at the last
choice, excluding usage/DONE bookkeeping. TPOT divides the interval after the first choice by output
tokens minus one. Cohort throughput uses actual completed output tokens divided by cohort wall time.
Per-process mean/p50/p95 and all request-level data are preserved.

Metrics are sampled about every0.2s, HBM about every2s. Peaks can miss events between samples.
`sampled_preemptions_lower_bound` may omit the unsampled tail of a cohort; it is not an exact cohort
total. Exact final counters for each fresh lifecycle, including readiness and both cohorts, are
`{'qwen25-off-1': 0.0, 'qwen25-on-1': 0.0, 'qwen35-off-1': 0.0, 'qwen35-on-1': 0.0}`. Synchronous
completed-request artifact writes are common instrumentation overhead.

Both TP ranks acknowledge every admitted long-request scheduler commit; ON has 33 such transactions
per lifecycle (readiness plus32 measured long requests). Every lifecycle ends with zero
active/waiting requests and both devices free. The full ledger includes unsuccessful preflights:
initial text-only setup, 2048-budget hybrid serialization, and an inapplicable Mamba flag on dense
Qwen2. Those were corrected before the frozen formal series; no failed formal run was silently
discarded. One out-of-scope startup was canceled after the fourth lifecycle, before readiness or
requests; its stop receipt and cleanup logs are also preserved. The complete preflight summaries and
logs are preserved. All four services emitted the same native
`AsyncLLM output_handler`/`EngineDeadError` trace only after explicit shutdown and engine-manager
completion. The original frozen analyzer rejected these teardown messages. Its unchanged source is
retained; `analyze_verified.py` changes only that error-boundary check, with negative tests that
still reject serving errors and other shutdown errors. Raw requests, timing, metric formulas,
transaction checks and original logs remain unchanged.

## Originals and offline verification

`raw-common.tar.gz` contains frozen inputs, package/source identities, controller, analysis,
preflights and validation receipts. Each `raw-<lifecycle>.tar.gz` contains that lifecycle's service,
streaming, metrics and shutdown originals. `members.json.gz` hashes every member; `SHA256SUMS`
hashes all public files. Model weights are referenced by hashes/stat pins and are not duplicated
here.

Run with Python3.12 or newer, CPU only, no model/runtime installation:

```sh
python verify.py
```

The verifier checks outer and member hashes, safely extracts the complete capsule, recomputes every
measured/readiness request and aggregate, validates rank commits/pools/cleanup, and byte-compares
the resulting analysis. Public GitHub download/readback is checked after upload. Full model hash
receipts were reused only after matching current file sizes and nanosecond mtimes; that reuse is
explicitly recorded in model identity files.
