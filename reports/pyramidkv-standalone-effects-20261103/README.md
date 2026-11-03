# Standalone PyramidKV 0.4.0rc1: quality, service cost and capacity

This report records the current-release validation requested by
[PyramidKV #31](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/31), under
[#1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1). All 12 preregistered
lifecycles completed: 1,834 requests and 682 compression commits, each matched to acknowledgements
from both TP workers. The measurement services have stopped and both assigned devices are free.

The fixed LongBench development and holdout gates pass. At an equal 8 GiB KV reservation per worker,
pooled throughput is lower with compression in every tested service cell. At 2 GiB, the
preregistered 24K/C4 pressure cell improves by 7.93%, with preemptions falling from 33 in each OFF
process to zero in each ON process. Other 2 GiB cells are slower. C1/C2/C4 capacity probes complete
in both arms and do not establish greater maximum resident capacity. Sampled physical HBM peaks are
higher with compression; lower KV-pool occupancy is not an HBM saving.

## Measured configuration

- Public plugin: `vllm_ascend_pyramidkv_hust-0.4.0rc1-py3-none-any.whl`, source
  `bdc20e248c9202560bf44c086bcdede2952b3274`, SHA256
  `46a0a9b2dd02b7523fe61d5032a78abd7dceb653b660d0304f2dc52d9ae5ae81`. Both arms load this installed
  wheel from the same release overlay.
- Evaluation runner: `06ab70a5dc58b6a2d4f1995c9d68ac61216d5b6a`, the tested source tree merged
  through PyramidKV PR #34. This runner revision does not replace the measured plugin wheel.
- Host: vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`, vLLM-Ascend
  `03766ac696fde5ab1980d80ca0b8543d3580c989`, CANN 9.1, torch 2.10.0+cpu and torch-npu 2.10.0.post4;
  eight pinned native-operator files.
- Model: Qwen3.5-35B-A3B, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`; tokenizer, template
  and all 28 model files are pinned. Initial full-file hashes are retained; the final check repeats
  model size/mtime checks, not a second weight-content hash pass.
- Hardware: two actual 910B2 devices, physical IDs 1 and 6, visible IDs 0 and 1; aarch64 container
  limited to 8 CPU and 64 GiB RAM.
- BF16, TP2, 32K, APC, MTP2, async scheduling, chunked prefill, FULL_AND_PIECEWISE, capture sizes
  `[3,6,12]`, max-num-seqs 4 and max-num-batched-tokens 4096. Both arms set batched metadata to 1
  and aligned1368/beta2; only compression activation changes within each reservation budget.
- Initial contract SHA256 `94b1f3b26f5d2ba3d5ad7ae75ed685035a0fa34a976ddb6324e755c694e7a4ad`;
  amended contract SHA256 `787559a9cfc357664749c18fc656d81a42358822f430116f8a744da7b7c0046a`. The
  first three lifecycles used V1, the remaining nine V2. Both versions, preregistration links and
  actual process receipts are retained.

These are experimental BF16 results for this pinned profile, separate from the organization's public
v0.18.0/FP16 official leaderboard baseline. Historical shared-adapter results keep their original
identity.

## Quality

Each arm runs the first 100 official examples from narrativeqa, qasper and 2wikimqa. Development
indices 0–49 and holdout indices 50–99 are judged separately. Every subset requires mean F1 loss ≤3
points and every task loss ≤5 points. Positive loss below means OFF minus ON.

| Subset      | Task                       |  OFF F1 |   ON F1 | Loss, points |
| ----------- | -------------------------- | ------: | ------: | -----------: |
| development | narrativeqa                | 27.0150 | 27.2387 |      -0.2237 |
| development | qasper                     | 51.7397 | 51.7560 |      -0.0163 |
| development | 2wikimqa                   | 65.0333 | 63.0333 |      +2.0000 |
| development | **Mean loss; gate passed** |         |         |   **0.5867** |
| holdout     | narrativeqa                | 30.7408 | 29.9475 |      +0.7933 |
| holdout     | qasper                     | 49.3472 | 48.8741 |      +0.4731 |
| holdout     | 2wikimqa                   | 64.2667 | 64.7667 |      -0.5000 |
| holdout     | **Mean loss; gate passed** |         |         |   **0.2555** |

This is a 300-example regression selection per arm, not the full LongBench leaderboard. The 96
additional service-smoke QA requests are retained separately and do not replace this gate. Natural
APC reuse is retained in the quality arms; the strict zero-cache condition applies to the 8 GiB
synthetic service requests.

## Service results

Each reservation uses fresh OFF/ON/ON/OFF processes. Every process has 18 formal cohorts: 1K/8K/24K
× C1/C4 × three repeats, with three requests per C1 cohort and eight per C4 cohort. Its 99 formal
requests are pooled as output tokens divided by the sum of cohort wall times. The 33 warmups, five
repeatability probes and 12 QA smoke requests are excluded from these throughput figures. All eight
service processes use byte-identical corresponding request payloads and 149 unique request salts.

| KV per worker | Prompt / concurrency | OFF tok/s | ON tok/s | ON/OFF change | Pair 1 / pair 2 change |
| ------------- | -------------------- | --------: | -------: | ------------: | ---------------------: |
| 8gib          | 1024-c1              |    79.795 |   76.187 |        -4.52% |        -4.45% / -4.60% |
| 8gib          | 1024-c4              |   188.592 |  186.102 |        -1.32% |        +1.62% / -4.19% |
| 8gib          | 8192-c1              |    52.408 |   48.330 |        -7.78% |        -8.31% / -7.25% |
| 8gib          | 8192-c4              |    65.106 |   63.767 |        -2.06% |        -3.33% / -0.76% |
| 8gib          | 24576-c1             |    31.331 |   30.393 |        -2.99% |        -3.60% / -2.40% |
| 8gib          | 24576-c4             |    24.947 |   24.681 |        -1.06% |        -2.22% / +0.09% |
| 2gib          | 1024-c1              |    79.108 |   76.655 |        -3.10% |        -3.84% / -2.35% |
| 2gib          | 1024-c4              |   199.570 |  193.308 |        -3.14% |        -8.34% / +2.29% |
| 2gib          | 8192-c1              |    52.585 |   49.227 |        -6.39% |        -5.95% / -6.83% |
| 2gib          | 8192-c4              |    65.425 |   63.707 |        -2.63% |        -1.27% / -3.96% |
| 2gib          | 24576-c1             |    31.597 |   30.054 |        -4.88% |        -4.86% / -4.90% |
| 2gib          | 24576-c4             |    22.634 |   24.428 |        +7.93% |        +8.45% / +7.42% |

The primary pressure cell was fixed as 2 GiB / 24576 / C4 before measurement. Its latency
distributions are:

| Client metric, seconds |  OFF mean |   ON mean |   OFF p50 |    ON p50 |   OFF p95 |    ON p95 |
| ---------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| ttft_seconds           | 10.969499 |  8.940469 | 10.829427 |  8.167274 | 18.622911 | 17.914412 |
| e2e_seconds            | 21.646718 | 20.565426 | 22.337862 | 20.573353 | 34.912068 | 31.995073 |
| tpot_seconds           |  0.084070 |  0.091532 |  0.083624 |  0.103242 |  0.127951 |  0.131061 |

Pair 1 compares ON-1 with OFF-1; pair 2 compares ON-2 with OFF-2. There are only two process
replicates per arm/budget. These are descriptive matched observations, not a statistical
significance or universal speedup claim. The full [analysis](analysis.json) preserves every formal
cohort, cell, pair and process, including negative results.

## Preemption, cache, MTP and memory

The following counters and peaks span each complete service client phase, including warmups, probes
and QA smoke. HBM values are sampled per physical device in MiB; they exclude startup. Phase-level
observations remain in `analysis.json` and the raw telemetry. MTP shows accepted draft tokens /
total draft tokens.

| Process        | Preemptions | Cached prompt tokens | MTP accepted / drafted | Peak KV use | Peak HBM device 1 / 6, MiB |
| -------------- | ----------: | -------------------: | ---------------------: | ----------: | -------------------------: |
| service8-off-1 |           0 |                    0 |          11498 / 12440 |      29.12% |              47687 / 47679 |
| service8-on-1  |           0 |                    0 |          11383 / 12676 |      21.70% |              47956 / 47948 |
| service8-on-2  |           0 |                    0 |          11410 / 12622 |      21.70% |              47956 / 47948 |
| service8-off-2 |           0 |                    0 |          11480 / 12472 |      29.12% |              47686 / 47679 |
| service2-off-1 |          33 |                32768 |          11485 / 12424 |      96.67% |              41566 / 41558 |
| service2-on-1  |           0 |                    0 |          11416 / 12622 |      87.78% |              41836 / 41828 |
| service2-on-2  |           0 |                    0 |          11414 / 12624 |      87.78% |              41836 / 41828 |
| service2-off-2 |          33 |                34816 |          11497 / 12424 |      96.67% |              41566 / 41558 |

The pressure OFF processes have four and five cached cases respectively, all in the 24K/C4 warmup or
formal cohorts with increasing sampled preemption counters. The ON pressure processes have no cached
cases or preemptions. [PRESSURE.md](PRESSURE.md) documents the pinned source path for same-request
prefix reuse after initial-prefill preemption. This attribution uses unique salts, complete
HTTP/engine counts and phase counters; individual request preemption events were not captured at
INFO level. The 2 GiB result must not be called all-uncached or an isolated kernel speedup.

At 8 GiB the zero-cache synthetic requirement remains strict. The pressure-policy correction was
made before any 2 GiB requests and changed validation metadata only. Both contract versions, the
original policy, unchanged payload hashes and amendment timing evidence are retained.

## Capacity operating points

Every probe uses 32,704 input tokens and a 64-token output budget. C1/C2/C4 each complete in both
arms, with zero failures, zero cached tokens and no preemptions in the observed capacity phases.
These points are within max-num-seqs=4; no larger-buffer or higher-concurrency profile is qualified
here.

| KV per worker | Concurrency | OFF completed | ON completed | OFF mean E2E, s | ON mean E2E, s | OFF p95, s | ON p95, s |
| ------------- | ----------- | ------------: | -----------: | --------------: | -------------: | ---------: | --------: |
| 8gib          | c1          |             1 |            1 |          3.8891 |         3.9346 |     3.8891 |    3.9346 |
| 8gib          | c2          |             2 |            2 |         10.8465 |        10.3809 |    11.0479 |   10.6358 |
| 8gib          | c4          |             4 |            4 |         20.5818 |        20.5356 |    24.2694 |   23.5879 |
| 2gib          | c1          |             1 |            1 |          3.7396 |         4.0505 |     3.7396 |    4.0505 |
| 2gib          | c2          |             2 |            2 |         10.3244 |        10.5485 |    10.5346 |   10.7808 |
| 2gib          | c4          |             4 |            4 |         18.4391 |        21.0261 |    23.7531 |   23.7811 |

The sample count at each capacity point is small (one, two or four requests). A p95 from one request
equals that observation. The results establish admission/completion at these configured operating
points; they do not demonstrate an increased maximum resident capacity or a production latency SLO.

## Evidence, verification and limitations

- [REPRODUCE.md](REPRODUCE.md) gives public and dataset-backed offline commands and the exact input
  reconstruction contract. Public verification covers all 1,138 synthetic request payloads and 1,834
  SSE/result pairs. The 696 QA payloads, three complete case plans and three official data files
  remain private, with exact byte/hash inventories. Dataset-backed verification reconstructs all 696
  QA payloads and recomputes all 696 F1 scores; public-only verification explicitly reports zero for
  those two private checks.
- `analysis.json` is deterministically recomputed from the originals. `public-verification.json`
  records per-process verification and explicit coverage. `private-reconstruction.json` records
  byte-identical fresh tokenizer preparation. Raw archive parts and the compressed member inventory
  retain 5,100 public members; the capsule contains 702 private-file inventory records and no
  private payload contents.
- All 682 commits match both TP workers. A few logs interleave two explicit TP prefixes with two
  byte-identical ACK bodies. The verifier accepts only that exact recoverable pattern and records
  every recovery; originals are unchanged, and missing or differing bodies fail.
- First quality-OFF has the frozen launcher, unchanged environment source and parsed startup
  profile, but no captured `/proc` argv/environment snapshot. The other 11 lifecycles have actual
  snapshots. This gap is disclosed and was not hidden by rerunning the first arm.
- Client TTFT/E2E/TPOT are original observations. SSE bodies cannot independently reconstruct token
  receive times. The 600-second setting is a socket timeout, with separate preregistered client and
  eight-hour series limits; it is not a strict whole-stream deadline.
- One five-request repeatability group (`service8-on-1`) produced two distinct output hashes; the
  other seven groups each produced one. Fixed greedy settings and seed do not establish bitwise
  output determinism on this host. Every probe is retained and excluded from performance pooling.
- Startup compiler warnings, cache-advice receipts, final return codes and all logs are retained.
  Startup-only cache advice ended before readiness. Local tests and data audits ran during startup
  or after the series, never during measured requests.
- Model contents were hashed before the series; the driver checks their size/mtime before every
  lifecycle, and the final receipt repeats that check. Host Git identities, eight native-operator
  hashes and external launch-file hashes were checked unchanged at the end. No second complete
  weight rehash is claimed.

The conclusion is specific: this independent candidate passes the fixed quality regression and shows
a conditional service gain under the selected 2 GiB pressure workload. It does not show a general
service speedup, lower sampled HBM, or larger maximum capacity. The five-dataset program under #8
has separate execution and judge requirements.
