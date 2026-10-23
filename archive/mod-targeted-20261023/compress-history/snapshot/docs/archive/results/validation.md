# Historical 2026-09-20 Working-Tree Validation Record

English | [简体中文](validation.zh.md)

> **Historical archive.** This snapshot predates the 0.8 Frontier protocol and
> is not a current release acceptance result. See [the active protocol](frontier-benchmarking-0.9.md).

Date: 2026-09-20. Verdict: **partial engineering PASS, not formal V4.6
acceptance**. The complete A2 commissioning matrix and the three public quality
workloads passed. The frozen A3 stability matrix completed but failed its
minimum-completion and six-window throughput-CV gates. The negative result is
retained; it was not replaced by extra rounds. The supplementary Qwen3.5
matrix also completed: two of eight A2 cells passed and its A3 stability result
failed, while all model outputs and compression transactions remained correct.

| Scope | Result | Publication boundary |
| --- | --- | --- |
| Unit/contract suite | PASS | Repository correctness only |
| Ascend numerical smoke and kernel microbenchmark | PASS | Synthetic cache shape, not serving throughput |
| Qwen2.5 A2, eight cells, three cold lifecycles per arm | PASS | Commissioning fixture and same-host control |
| Qwen2.5 A3, three 30-minute lifecycles per arm | **FAIL** | Completion count and throughput CV failed |
| Qwen2.5 public LongBench-v2/LongBench | PASS | One run per arm; quality and integrity are primary |
| Qwen3.5-35B-A3B adaptation and public workloads | PASS, supplementary | TP=2/eager engineering evidence only |
| Qwen3.5 transferred A2/A3 matrix | **FAIL**, supplementary | Six A2 cells missed throughput; A3 stability failed |
| Formal V4.6 release | **NOT CLAIMED** | Official B0 and authorized LONG-PUBLIC evidence are absent |

## Implementation under test

The working tree adds the TriAttention V3 position policy: configurable hard
protection for prefix and recent tokens, with proportional eviction quotas
across equal middle-context segments. It also adds Qwen3.5 hybrid support:
only the ten full-attention layers are scored and compacted, while
Gated-DeltaNet recurrent state remains under the host lifecycle. Qwen3.5's
192 non-rotary key dimensions receive a calibrated direct Q·K content term;
TP ranks all-reduce layer scores before selecting an identical token set.

Both model families use schema-3 calibration artifacts:

| Model | Calibration SHA-256 |
| --- | --- |
| Qwen2.5-14B-Instruct | `cf3be9376043bb9c47fb8beb461d3951c3d82caa90d33bffcfc4dff5d96a6f26` |
| Qwen3.5-35B-A3B | `c09e9ec41c34d876802865a5ef489c1d4514245a90b71d1574663ddcfa2c637b` |

Neither `vllm-hust` nor `vllm-ascend-hust` was modified. All integration is in
the separately packaged plugin. Qwen3.5's opt-in legacy `int[]` GDN ABI bridge
is schema-gated to the validated host binary.

## Frozen environment

| Component | Validated value |
| --- | --- |
| Plugin | 0.7.0 release candidate based on `de038e7` |
| vLLM-HUST | `6cdc0304a8ba`, `0.28.1.post1.dev260` |
| vLLM-Ascend-HUST | `5901bedbb718`, `0.25.1rc2.dev232+hust.20260903.4.g5901bedbb` |
| Python / torch / torch-npu | 3.11.16 / 2.10 / 2.10.post2 |
| Standard model | `/data/shared_models/Qwen--Qwen2.5-14B-Instruct`, FP16, TP=1 |
| Supplementary model | `/workspace/models/Qwen--Qwen3.5-35B-A3B`, BF16, TP=2, eager |
| Device | Ascend 910B2; devices 0–5 only for this run |

Services used `--generation-config vllm`. Clients fixed temperature 0, top-p
1, top-k -1, min-p 0, neutral penalties including repetition penalty 1, n=1,
beam search off, empty stop, seed 0, usage-bearing streams, and
`add_special_tokens=true`.

## Automated and NPU checks

- `pytest`: 114 passed and 1 environment skip.
- Ruff lint and format checks: pass.
- Ascend numerical kernel smoke: pass.
- Fresh 2,176-token, eight-KV-head BF16 microbenchmark: paged K/V copy 1.64×,
  direct scoring 1.85×, and fused aggregation 1.35× versus their references.

The kernel values are isolated microbenchmarks, not model-serving claims. The
cross-version normalized view is the
[HTML benchmark leaderboard](../../results/benchmark-leaderboard.html).

## Qwen2.5 standard commissioning matrix

The deterministic fixture hash is
`6911a586376fbb9472be9017d68660cd3fb38dda2a054dd76cd2b11c408e208e`.
B0 and B1 each used three independent cold service lifecycles. A2 measured 16
requests per cell and per lifecycle at concurrency 4.

| Profile | RPS | B0 → B1 median total tok/s | Delta | Quality | Minimum physical KV reduction | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8K + 512 | 0.05 | 439.13 → 438.90 | −0.05% | 100% → 100% | n/a: no threshold crossing | PASS |
| 8K + 512 | 0.10 | 829.81 → 828.68 | −0.14% | 100% → 100% | n/a: no threshold crossing | PASS |
| 8K + 512 | 0.20 | 1294.64 → 1290.71 | −0.30% | 100% → 100% | n/a: no threshold crossing | PASS |
| 8K + 512 | 0.40 | 1370.00 → 1368.79 | −0.09% | 100% → 100% | n/a: no threshold crossing | PASS |
| 16K + 1,024 | 0.05 | 820.79 → 826.60 | +0.71% | 100% → 100% | 50.00% | PASS |
| 16K + 1,024 | 0.10 | 1085.70 → 1232.85 | +13.55% | 100% → 100% | 50.00% | PASS |
| 16K + 1,024 | 0.20 | 1128.82 → 1300.52 | +15.21% | 100% → 100% | 50.00% | PASS |
| 16K + 1,024 | 0.40 | 1147.92 → 1335.78 | +16.36% | 100% → 100% | 50.00% | PASS |

Across the measured A2 cells, all 768 arm-requests completed with no failures,
short forced outputs, or silent truncations. The six warm-up requests were also
clean. The 16K B1 cells produced 192 scheduler commits and 192 worker
acknowledgements. Peak device HBM was 87%; vLLM reserves its KV pool at startup,
so no smaller reserved-pool claim is made. The 8K cells stayed below the 9,216
semantic-token threshold and correctly report M3 as not applicable.

## Qwen2.5 A3 negative stability result

A3 used exact 30,720-token inputs plus 2,048 forced output tokens, concurrency
1, a five-minute warm-up, and a 30-minute measurement divided into six
five-minute windows. Each arm used three independent cold lifecycles.

| Median of three runs | B0 | B1 | Change |
| --- | ---: | ---: | ---: |
| Total token throughput | 407.91 tok/s | 461.62 tok/s | +13.17% |
| Mean TTFT | 6304.90 ms | 6318.90 ms | +0.22% |
| Mean TPOT | 36.16 ms | 31.58 ms | −12.68% |
| Mean E2E | 80.32 s | 70.96 s | −11.66% |

All 147 measured arm-requests were correct; failures, OOMs, short outputs, and
silent truncations were zero. B1 recorded 156 commits and 156 acknowledgements,
with at least 73.33% physical KV reduction. All TTFT/TPOT median and p99 drift
limits passed. A3 nevertheless fails both frozen stability conditions:

- B0 completed 23 requests in every lifecycle, below the required 24; B1
  completed 26 in every lifecycle.
- Six-window throughput CV was 12.86% for every B0 lifecycle and 8.94% for
  every B1 lifecycle, above the 5% limit.

The result is a test-gate failure, not a crash or corruption failure. No extra
round was used to replace it.

## Public long-context quality and performance

Qwen2.5 used all eligible non-truncated cases: LongBench-v2 116, passage
retrieval 200, and Qasper pressure subset 93. All 818 arm-requests completed.

| Dataset | B0 → B1 quality | Request throughput | Mean E2E | Physical blocks | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| LongBench-v2 | 40.52 → 39.66 accuracy (−0.86 pp) | +7.78% | −6.99% | −61.96% | PASS |
| Passage retrieval | 98.75 → 98.75 (0.00 pp) | +2.41% | −2.33% | bypassed | PASS |
| Qasper | 43.60 → 43.36 F1 (−0.24 pp) | +1.35% | −1.41% | −38.73% on compressed cases | PASS |

All quality changes are inside the frozen one-percentage-point gate. B1
produced 127 commits and 127 acknowledgements; compressed cases changed 20,666
source blocks to 8,128 destination blocks (−60.67%). These are one-run-per-arm
engineering measurements while other NPUs carried concurrent work, so
performance deltas are exploratory. See
[Public long-context benchmarks](public-long-context-benchmarks.md).

## Qwen3.5 supplementary adaptation

The downloaded model snapshot is revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`; its 14 weight shards contain
71,903,655,008 tensor bytes. TP=2 16K smoke passed with 50% physical KV
reduction. A non-thinking public run completed 806 arm-requests with no errors
or silent truncations; all three quality gates passed. LongBench-v2 accuracy
was unchanged, passage retrieval stayed at 100% while bypassing compression,
and Qasper changed by −0.43 pp. TP=2 transaction evidence was exact: 121
scheduler commits and 242 per-rank acknowledgements.

Performance did not beat B0 on those public workloads. This host also requires
eager execution and the explicit legacy GDN ABI bridge. Qwen3.5 results are
therefore supplementary hybrid-compatibility evidence, never a replacement for
the standard Qwen2.5 matrix.

The transferred A2 matrix completed all 768 measured arm-requests correctly,
with 192 commits and 384 TP-rank acknowledgements. Only 8K@0.05 and 16K@0.40
passed; the other six cells exceeded the frozen 1% total-throughput regression
budget. The transferred A3 matrix completed four correct requests in every B0
and B1 lifecycle and improved median total throughput 1.23%, but failed the
24-completion gate, reported 100% six-window throughput CV, and could not
compute latency drift across empty windows. B1 still achieved at least 73.33%
physical KV reduction with 24 commits and 48 acknowledgements. See
[Qwen3.5 adaptation](../../design/qwen3.5-35b-a3b-adaptation.md).

## Release boundary

This working tree cannot claim formal V4.6 PASS or production readiness:

1. B0 is a current-host no-compression engineering control, not the prescribed
   official vLLM 0.18 plus matching official Ascend stack.
2. A2/A3 use a repository commissioning fixture, not an independently approved
   and signed LONG-PUBLIC corpus.
3. Qwen2.5 A3 and the transferred Qwen3.5 A3 both failed the
   minimum-completion and six-window throughput-CV gates; six Qwen3.5 A2 cells
   also failed the throughput-regression gate.
4. Public performance has one run per arm and concurrent host load.
5. Reserved device HBM did not fall, and system cost per token was not measured
   independently.
6. Qwen3.5 uses a non-standard TP=2/eager topology and a host-specific ABI
   bridge; strict compressed multi-position NIAH remains a deployment gate.

Machine-readable records:

- [Qwen2.5 A2/A3 summary](../../results/evidence/kvcompress-working-tree-20260920-qwen25-standard-summary.json)
- [Qwen2.5 public summary](../../results/evidence/kvcompress-working-tree-20260920-qwen25-public-summary.json)
- [Qwen3.5 public summary](../../results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)
- [Qwen3.5 transferred A2/A3 summary](../../results/evidence/kvcompress-working-tree-20260920-qwen35-standard-summary.json)
- [Leaderboard history](../../results/evidence/leaderboard-history.json)
