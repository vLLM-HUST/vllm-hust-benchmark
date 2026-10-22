# Qwen3.5 paired Ascend evaluation — 2026-10-09 UTC

This receipt addresses the remaining task-quality, capacity, throughput,
latency, and HBM evidence in [issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1).
Results apply to the pinned source installation and explicit 1368/beta-2
configuration below. Release promotion and Workshop publication require
review; this receipt does not certify other configurations.

## What changed after the failed run

The original paired evaluation returned HTTP 200 for every request and matched
all expected compression transactions, but task quality failed badly. Some
answers degenerated into repeated exclamation marks after their first token.
FULL graph capture had kept the shared native cache-write slot mapping;
subsequent replay updated different per-layer buffers. The shared fix binds
the preallocated per-layer buffers during synthetic capture, includes the MTP
auxiliary cache, and invalidates padded slots before replay. See
[KVCompress PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19).
The provider algorithm and model weights are unchanged.

The initial mean F1 loss was 15.13 points. With only the graph fix, the original
512/beta-20 profile produced no repeated-`!!!!` answers in 150 development
cases, but qasper still lost 6.19 points. These diagnostics are not the final
paired comparison: they reused a development server after five diagnostic
requests. Both failed stages remain in the archives below.

Before seeing holdout outputs, the next candidate was fixed to
`max_capacity_prompt=1368`, `beta=2`, window 8, max pooling, kernel 7,
`kv_head` granularity, mean GQA aggregation, no merge, and admission above
4096 prompt tokens. It retains more keys in deeper full-attention layers while
staying within the same 2048-token aligned physical allocation bound. This is
an explicit evaluated configuration, not a change to the package's historical
512/beta-20 default. Use [method-config.json](method-config.json).

## Exact scope and sources

Two Ascend 910B2 devices (physical 6/7, logical 0/1, 64 GiB each), CANN 9.1,
BF16, TP2, APC, MTP2, async scheduling, chunked prefill,
`FULL_AND_PIECEWISE`, and `mamba_cache_mode=align`. Both arms use max context
32768, max four sequences, 4096 batched tokens, and an 8 GiB per-worker KV
pool. The text-only service uses direct shared-adapter activation, disabled
for baseline and enabled for PyramidKV. The two saved launch commands match.

| Component | Exact revision |
| --- | --- |
| vLLM | `d0f22d2bda562156e4dbf433ce645e1769b4f804` |
| vLLM Ascend | `03766ac696fde5ab1980d80ca0b8543d3580c989` |
| Shared adapter, evaluated validation branch | `19f322130e1b3841da953b1b421bcf3259e9b8dc` |
| Shared fix PR head against newer master | `2a0dc155d5a6aced292808f852892d4383878f7a` |
| PyramidKV / evaluation launch source | `01379d29a3f3ce938be4271760cb7ac0e61ef5d4` |
| Unchanged PyramidKV provider source | `b0cd3fd73635b97b7c7372219e393e366df7c8d0` |
| Model `Qwen/Qwen3.5-35B-A3B` | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| LongBench data | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` |
| LongBench templates and reference metric | `2e00731f8d0bff23dc4325161044d0ed8af94c1e` |

[environment.json](environment.json) records Python/packages, worker and Manager
revisions, input hashes, and method configuration. The validation branch
applies the fix to the older host-compatible shared source and includes the
existing Manifest 0.3 migration. The package version `0.9.0` alone does not
identify this implementation. All 14 model shards were verified in the
[preceding model receipt](../2026-10-09-qwen35-serving/model-hashes.json).

## Final paired results

Both arms completed **409/409 requests**, including 300 quality cases, with
zero request failures. Baseline had zero compression commits; PyramidKV had
**325 matched commits**, each acknowledged with matching semantic/physical
lengths by both TP ranks. All committed physical prompt lengths were 2048.
The 1024/4096-token boundary requests remained uncompressed; 4097 compressed.

### Task quality

QA-F1 is on a 0–100 scale. Positive loss means PyramidKV scored lower.
The gate is mean loss <= 3 points and every task loss <= 5 points, enforced
separately for development and holdout. **Overall gate: PASS**.

| Subset | Task | Baseline F1 | PyramidKV F1 | Loss (points) |
| --- | --- | ---: | ---: | ---: |
| development | narrativeqa | 27.01 | 27.24 | -0.22 |
| development | qasper | 51.74 | 51.76 | -0.02 |
| development | 2wikimqa | 65.03 | 63.03 | +2.00 |
| development | **Mean (PASS)** | 47.93 | 47.34 | **+0.59** |
| holdout | narrativeqa | 30.74 | 29.95 | +0.79 |
| holdout | qasper | 49.35 | 48.87 | +0.47 |
| holdout | 2wikimqa | 64.27 | 64.77 | -0.50 |
| holdout | **Mean (PASS)** | 48.12 | 47.86 | **+0.26** |

Every subset/task cell contains 50 examples. Coverage below reports all 100
examples per task, including short examples that are ineligible for compression.

| Task | Examples above 4096 tokens | Middle-truncated examples | Final prompt token range |
| --- | ---: | ---: | --- |
| narrativeqa | 100/100 | 69 | 8735–16212 |
| qasper | 64/100 | 3 | 2090–16212 |
| 2wikimqa | 87/100 | 1 | 1042–16211 |

### Throughput and latency

Three cohorts per row, 128 forced output tokens per request. Concurrency 1
has 9 requests total per row; concurrency 4 has 24. Output throughput includes
prefill and is pooled over cohort wall time. Positive throughput change is
higher; positive latency change is slower. All cohorts are retained.

| Input tokens | Concurrency | Baseline output tok/s | PyramidKV output tok/s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 66.53 | 77.71 | +16.8% |
| 1024 | 4 | 182.88 | 181.73 | -0.6% |
| 8192 | 1 | 52.66 | 47.22 | -10.3% |
| 8192 | 4 | 66.27 | 63.63 | -4.0% |
| 24576 | 1 | 32.41 | 29.54 | -8.8% |
| 24576 | 4 | 25.68 | 25.01 | -2.6% |

Each latency cell is **baseline → PyramidKV**, using request means. The full
analysis also retains p50/p95 and each cohort throughput, so dispersion is
visible rather than replaced with a best-run number.

| Input tokens | Concurrency | TTFT (s) | TPOT (ms/token) | E2E (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 0.60 → 0.39 | 10.24 → 9.68 | 1.90 → 1.62 |
| 1024 | 4 | 0.87 → 0.83 | 14.78 → 15.22 | 2.74 → 2.76 |
| 8192 | 1 | 1.20 → 1.18 | 9.47 → 11.83 | 2.41 → 2.69 |
| 8192 | 4 | 3.44 → 3.05 | 32.86 → 38.23 | 7.62 → 7.91 |
| 24576 | 1 | 2.75 → 2.78 | 9.23 → 11.96 | 3.92 → 4.30 |
| 24576 | 4 | 8.44 → 8.80 | 88.27 → 88.90 | 19.65 → 20.09 |

### Capacity, resources, and speculative decoding

Both arms completed 32704 input + 64 output tokens at concurrency 1 (one
request) and 4 (four requests), without preemption. PyramidKV committed
32704 semantic prompt tokens to a 2048-token physical full-attention allocation
for each capacity request. Actual layer retention remains nonuniform.

| Measurement | Baseline | PyramidKV |
| --- | ---: | ---: |
| Sampled peak device 6 HBM (MiB) | 47683 | 48094 |
| Sampled peak device 7 HBM (MiB) | 47682 | 48092 |
| Sampled peak KV pool utilization | 29.12% | 21.70% |
| Preemptions | 0 | 0 |
| MTP drafted tokens | 11973 | 12474 |
| MTP accepted tokens | 10751 | 10446 |
| MTP accepted / drafted | 89.79% | 83.74% |
| Telemetry sampling errors | 0 | 0 |

HBM includes the reserved KV pool and graph/workspace allocations. The
compression transaction proves reduced request KV ownership, not an equal
drop in whole-device HBM. This run does not justify a general HBM-saving
or across-the-board speedup claim. Per-phase cache/HBM peaks and MTP counters
are retained in the full analysis.

In this run, the compressed 8K/24K workloads lost 2.6%–10.3% output
throughput relative to baseline. The 1024-token case is below the admission
threshold, so its positive variation cannot be presented as a compression
speedup. These are measured tradeoffs, not positive performance-promotion
results.

## Startup failure retained

The first final-candidate startup failed before health or any test request:
TP1's Ascend driver rejected a 4096-byte pinned host allocation with error
207001. The container has a 64 GiB cgroup limit despite the host reporting
about 2 TiB RAM; `memory.current` was near the limit with about 61.3 GB of
file cache. No inference request from this attempt is counted as successful.

After stopping the failed attempt's processes, `POSIX_FADV_DONTNEED` on only
the local model shards reduced container usage from 62.55 GB to 1.44 GB.
The same candidate command/configuration was retried with a startup-only
model page-cache monitor (48 GiB threshold). It stops before evaluation.
No global cache drop, memory-limit change, model edit, or serving-argument
change was made. The baseline needed no retry. Startup latency is excluded
from the performance tables; this asymmetry and the one-lifecycle measurement
limit general performance claims. The archive retains the failed server and
driver logs, cgroup readings, retry metadata and cache-reclaim events.

## Reproduction and raw records

Follow the [fixed protocol and commands](../../../docs/issue-1-evaluation-protocol.md)
and [prepared-host runbook](../../../docs/qwen35-serving-smoke.md), selecting
`scripts/qwen35/pyramidkv-aligned.json`. `PYRAMIDKV_ENABLED=0` and `1` select
the two arms through the same launcher. Use the fixed dataset revision in
[dataset-source.json](dataset-source.json), prepare with `--samples-per-task 100`,
and verify the [plan manifest](plan-manifest.json). Dataset contexts, reference
answers, model weights, and credentials are not redistributed here.
[case-index.jsonl](case-index.jsonl) retains all case IDs, truncation metadata,
and prompt hashes. Recreate the saved token inputs from the pinned source
model/tokenizer and dataset to recompute QA scores independently.

Verify downloaded records with `sha256sum -c SHA256SUMS`. Extract
`paired-results.tar.gz` into an empty directory, then recompute the analysis:

```bash
python scripts/qwen35/analyze_evaluation.py \
  --root /path/to/extracted/aligned --output /tmp/comparison.json
```

| Artifact | Contents |
| --- | --- |
| [analysis.json](analysis.json) | Full paired analysis, per-request transaction verification and source-file hashes |
| [paired-results.tar.gz](paired-results.tar.gz) | Both final arms: raw per-request SSE, predictions/scores, usage/latency, cohort results, metrics, NPU telemetry, server/client logs, exact launch records and idle-device checks |
| [initial-failure-summary.json](initial-failure-summary.json) / [initial-failure.tar.gz](initial-failure.tar.gz) | Original paired failure, with original adapter/configuration and raw records |
| [graph-fix-development-scores.json](graph-fix-development-scores.json) / [graph-fix-development.tar.gz](graph-fix-development.tar.gz) | Five diagnostic requests and 150-case development rerun with fixed slots and original budget |
| [environment.sh.txt](environment.sh.txt) / [serve-evaluated.sh.txt](serve-evaluated.sh.txt) | Evaluated host environment and launcher snapshots |
| [run-pair-original.py.txt](run-pair-original.py.txt) / [retry-candidate.py.txt](retry-candidate.py.txt) | Original server-specific orchestration and startup retry; absolute paths intentionally retained for provenance |
| [shared-base-existing-failures.txt](shared-base-existing-failures.txt) | Two shared-master tests failing identically on the pristine base with this older host |

## Validation and limits

PyramidKV [local tests](pyramidkv-tests.txt): 68 passed, one opt-in NPU oracle
skipped; Ruff and [sdist/wheel build](package-build.txt) passed. The extracted
archive reproduced the identical full analysis, and all 600 paired QA
predictions were rescored against the pinned reference answers. Portable launcher argument/environment checks matched
the evaluated command for both activation values. Shared integration tests:
45 passed. Shared full suite: 203 passed, one skipped, two failures due to the
old host lacking `vllm_ascend.ops.fused_moe.router`; both failures reproduced
on pristine shared master. Those two failures are not described as passed.

This is a fixed first-100 subset for three LongBench tasks, split into separate
50-case development and holdout sets per task, not the full leaderboard.
Middle truncation caps pre-template instructions at 16200 tokens. Synthetic
performance inputs are not task-quality evidence. Greedy generation uses seed
17 and disabled thinking; outputs and MTP acceptance can still differ between
arms.

There is one fresh server lifecycle per arm, baseline first, with three
performance cohorts per configuration. Repeated cohorts are not independent
process-level trials. Both arms receive the same two excluded warmup requests;
later shape/JIT/cache effects remain in the retained measurements. Reported
throughput is output tokens divided by cohort wall time, including prefill.
TTFT/TPOT/E2E are client-observed; speculative streaming does not expose the
latency of every individual token. Sampled HBM and cache peaks are lower bounds
on instantaneous peaks. Prefix caching may help quality cases naturally;
performance cohorts use distinct deterministic prompts.

The 32K capacity point is a configured-context check, not maximum hardware
capacity, a 256K result, or a larger supported context than baseline. No
low-memory/OOM frontier was measured. Full-attention KV retention is not
whole-model memory consumption; hybrid recurrent state, weights, workspaces,
graph pools and reserved KV storage remain resident.

The pinned Manager joint launch still conflicts on `vllm.environment`;
this source installation uses direct activation. Published-wheel installation
acceptance, shared-fix merge/review, and Workshop/public-surface promotion are
separate from this device receipt. Unsupported model/dtype/TP/CANN/scheduling
combinations remain fail-closed as documented in the support matrix.
