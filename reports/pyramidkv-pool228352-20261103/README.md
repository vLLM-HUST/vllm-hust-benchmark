# PyramidKV Qwen2.5 TP2 C64 with 228352 token cache capacity

**One OFF/ON pair, Eager diagnostic, excluded from official leaderboards.** The user requested
retaining TP2/C64, prompts, batch budget and plugin version while reducing both arms' native
effective KV capacity from 373,248 to 228,352 token slots. The only serving-argument change is
`--num-gpu-blocks-override 1784`: both arms verify 1784 native blocks, 128-token pages and 228,352
reported effective token capacity before sending any measured request, and verify the same native
configuration after completion.

The current container exposes physical cards 2/3 instead of the prior run's 1/6. Both are 64 GiB
Ascend 910B2 cards and the user explicitly approved this substitution. The current OFF/ON pair uses
the same two cards; comparisons with the previous pool size also include this device change and
cannot establish a strict pool-size-only causal effect.

Enabling PyramidKV in this pair increased output throughput by 56.20%, reduced mean TTFT by 46.88%
and median TTFT by 62.58%, and changed peak active KV occupancy from 100% to 43.07%. Exact lifecycle
preemptions fell from 5 to 0. Mean TPOT increased 35.43%, while mean request E2E changed by −0.57%,
effectively similar in this single observation. The result therefore combines higher batch
throughput and earlier first tokens with slower subsequent tokens.

| Metric                               |       OFF |        ON |  Change |
| ------------------------------------ | --------: | --------: | ------: |
| Output throughput (tokens/s)         |    143.41 |    224.01 | +56.20% |
| Mean TTFT (s)                        |     60.29 |     32.02 | -46.88% |
| Median TTFT (s)                      |     84.74 |     31.71 | -62.58% |
| p95 TTFT (s)                         |    120.68 |     61.36 | -49.16% |
| Mean TPOT (ms)                       |    151.79 |    205.57 | +35.43% |
| Mean request E2E (s)                 |    137.85 |    137.07 |  -0.57% |
| Peak active KV (%)                   |    100.00 |     43.07 | -56.93% |
| Peak simultaneous two-card HBM (MiB) | 80,208.00 | 80,512.00 |  +0.38% |
| Exact lifecycle preemptions          |         5 |         0 |       — |

## Comparison with the previous C64 pair

| Configured pool | Physical cards               | OFF tokens/s | ON tokens/s | ON throughput change |
| --------------- | ---------------------------- | -----------: | ----------: | -------------------: |
| 373,248 tokens  | 1/6 (previous)               |       189.38 |      225.24 |              +18.94% |
| 228,352 tokens  | 2/3 (current, user-approved) |       143.41 |      224.01 |              +56.20% |

The
[previous report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8afbf04a57963dcefd5c9f1bb54556a023173740/reports/pyramidkv-pressure-20261103)
is immutable. Its original analysis is included for reference, with SHA256
`3a2eae1a384e7575fb27847d99dbd2bf929c323aea64259f07f042080c47b70a`. The two capacity settings were
not rerun together on the current physical cards. A single pair does not quantify run-to-run
variance.

## Preserved settings and input identity

Qwen2.5-14B-Instruct, TP2/PP1, BF16, 64 maximum sequences and simultaneous long requests, 7168 input
tokens and 512 forced output tokens, context 8192, batch-token budget 4096, utilization 0.8, seed 0,
APC, async scheduling, chunked prefill, Eager, no MTP, and `generation-config=vllm` remain as in the
prior C64 pair. Each fresh lifecycle first receives 48 short 1024/256 requests at rate 1, then 64
long requests at rate infinity; one separate first-prompt readiness request precedes each cohort.
There are 224 measured requests and 4 readiness requests in total. One OFF/ON pair was run, without
repeated rounds.

The exact prior C64 request file, algorithm config, client and candidate wheel are reused. Request
SHA256: `625cf0084c879fae3e78a06c483e22bb102e0f7d9c89ea48e3c24e6a6ea5fe1b`. The original C64 client
is retained byte for byte; its 100-connection ceiling does not bind 64 requests plus telemetry.
`pressure-coverage.json` verifies 64 server-side running-plus-waiting requests before any completion
and reports all telemetry errors and maximum sampling gaps.

Candidate wheel version 0.4.0rc2.dev0, source `c8742d038a1a5680c6de21dbd126b647a5fd3944`; core
`d0f22d2bda562156e4dbf433ce645e1769b4f804`, Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`.
Runtime remains CANN 9.1, Torch 2.10.0+cpu, torch-npu 2.10.0.post4, transformers 5.14.1. Installed
vLLM metadata 0.23.0+empty and Ascend 0.23.0.post1 are distinct from the prepared ABI selector
`VLLM_VERSION=0.25.1`. All package versions, model size/mtime pins and clean native source commits
were rechecked before freezing. CPU quota remains 8 cores and host memory limit 64 GiB. Startup-only
checkpoint file-cache eviction is recorded; it does not run during inference.

Algorithm settings remain capacity 512/beta 20/window 8/kernel 7/maxpool, KV-head/mean GQA, no
merge, and prompt admission above 4096 tokens. For 7168-token prompts, dense compressed allocation
remains 1024 physical tokens (8 pages); layer-specific selected lengths are separate from this
physical allocation. Both TP ranks acknowledge every admitted commit.

Contract SHA256: `67db31451c2026f353d97e638b403692579c1016bdf210cde451b6234fd8f95e`. The parameter
override changes native allocation and admission capacity; it does not alter the algorithm, batch
budget or request texts.

## Pool memory and metric definitions

Configured token capacity is 1784×128. With 48 attention layers, 4 local KV heads per TP rank, head
dimension 128, separate K/V and BF16, each native block occupies 12 MiB per rank. Thus the
configured KV tensors occupy20.90625 GiB per rank, versus 34.171875 GiB at 2916 blocks. These tensor
bytes are derived from pinned model geometry; total device HBM is measured separately and includes
weights, workspace and runtime allocations. The worker's `Available KV cache memory` log is its
profiled budget, not the overridden allocation.

KV usage is active block occupancy inside this configured pool. HBM peaks are sampled on physical
cards 2/3; their sum is the peak simultaneous sampled sum. Metrics are sampled about every 0.2s and
HBM about every 2s, so peaks can miss events between samples. Exact preemption totals span the whole
fresh lifecycle, including readiness and the short prelude; cohort sampled preemption lower bounds
are separately labeled in analysis.json.

TTFT ends at the first choice event, including an empty choice. E2E ends at the last choice,
excluding usage/DONE. TPOT divides the interval after first choice by output tokens minus one.
Cohort throughput uses output tokens divided by cohort wall time. Latency means, median and p95 plus
per-request data are retained. Synchronous artifact writes and telemetry add common overhead. This
synthetic run provides no task-quality evidence.

## Originals and offline verification

The capsule retains frozen inputs, launchers, wheel, source/model identities, both complete server
logs, SSE streams, timestamped events, request results, telemetry and cleanup receipts. Every
measured/readiness request is verified for exact payload and token usage, raw SSE/event
correspondence and recomputed timing. Both startup and final pool configurations must match the
requested target. Final active/waiting counts must be zero and both devices free. Known native
output-handler teardown errors are distinguished from serving errors.

`SHA256SUMS` verifies public files; `members.json.gz` identifies every archived member. Use Python
3.12 or newer, CPU only, without installing model/runtime packages:

```sh
python verify.py
```

The verifier safely extracts originals, checks all hashes, recomputes request/aggregate metrics,
TP-rank commits, pool identity and actual concurrency, then byte-compares the published analysis and
coverage. Public pinned-commit files are read back and verified after upload.
