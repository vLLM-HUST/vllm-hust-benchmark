# Issue #136 fixed-1-RPS matrix independent audit

Date: 2026-10-07 UTC

## Verdict

**PASS, with no blocking findings.** The archived fixed-load matrix is internally
complete and reproducible from its raw results. It is suitable as a fixed 1 RPS
offered-load latency checkpoint. It is **not** capacity or scaling evidence.

## Audited material

- Archive commit: `1141df1` (`evidence: archive issue 136 fixed-load matrix`)
- Archive path at that commit:
  `reports/issue-136-current-main-fixed-1-rps-20261007`
- Completed source matrix:
  `/root/issue-214-136-formal-20261007/formal/issue136-matrix-fixed-1-rps-20261007T034200Z-4034702`
- Audit method: CPU-only and read-only. No NPU, service port, process, lock, or
  active benchmark worktree was touched.

## Integrity checks

- Archive `SHA256SUMS`: 418/418 entries passed. The archive contains 419 files;
  the only intentionally uncovered file is `SHA256SUMS` itself.
- `SOURCE_SHA256SUMS`: 905/905 entries passed against the completed source
  matrix.
- `POSTPROCESS_SHA256SUMS`: 907/907 entries were independently rehashed against
  the completed source matrix.
- All 36 submission-local `checksums.sha256` manifests passed.
- 413 files common to the archive and original source manifest had identical
  hashes; 414 files common to the archive and postprocess manifest had identical
  hashes. There were no mismatches.
- The archive inventory declares the excluded cache, runtime-cwd, and duplicate
  runtime-state trees. The retained `submissions` trees contain the publishable
  raw measurements, resolved specs, environment manifests, server logs, and
  local checksums.

## Coverage and completion

- Exact coverage: four workloads x TP1/TP2/TP4 = 12 cells.
- Workloads: `random-online`, `sharegpt-online`,
  `prefix-repetition-online`, and `agent-research-online`.
- Exact repetition count: three successful submissions per cell = 36
  submissions.
- All 12 cell `EXIT_CODE` files are `0`; all 36 submission `STATUS` files are
  `OK`; every campaign reports 3 requested, 3 attempted, and 3 successful runs.
- Raw totals: 5,688 completed requests and 0 failed requests. Each
  random/sharegpt/prefix repeat completed 200/200; each agent repeat completed
  32/32.
- All 26 top-level and per-batch before/after NPU snapshots report four healthy
  910B2 devices (physical 0,1,2,7) and no running NPU processes at their
  respective boundaries.

## Frozen identity and configuration

All 36 submissions consistently record:

- Benchmark commit: `cc6e98b659a2a4f7c389108e7f13f5c4c4f8923f`
- Core commit, declared and observed:
  `c696cc916ef30c7eb57c3e90a9f278e82e6e0bc7`
- Plugin commit, declared and observed:
  `7c8ec865a12f7b6a14d53b61e7f61b5e811dd483`
- Image: `sha256:cd3c36c1832d8b9dfd193405d52a73b22ca2699fbe95271015bc7932bef3e2ee`
- Model: `Qwen/Qwen2.5-14B-Instruct`, FP16; frozen model revision
  `4cebf06566bc02922cce8ec2e44368204024e56a6813c5139192b01b7ee956d3`
- CANN declared/detected: 9.1.0; torch-npu declared/detected: 2.10.0.post4
- Topology: single node, four Ascend 910B2 devices, full HCCS, physical
  0,1,2,7; resolved chip count and tensor parallel size agree for TP1/TP2/TP4.
- Graph configuration: `FULL_AND_PIECEWISE` in all 36 resolved specs and all 36
  server logs.
- Client contract: temperature 0 and request rate 1 in all 36 resolved specs;
  raw results also report request rate 1.

Server logs demonstrate that the imported core and plugin came from the frozen
source worktrees, including plugin path `vllm-ascend-main-7c8ec865`. This agrees
with the declared/observed Git commits.

## Independent statistics reproduction

For every cell and each of the ten reported metrics, the three raw values were
reloaded from `raw_benchmark_result.json`. Median, q1, q3, and IQR were then
recomputed independently using the three-value linear percentile convention:
median = middle value, q1 = midpoint(min, median), q3 = midpoint(median, max),
and IQR = q3 - q1.

All 12 x 10 aggregates matched `analysis-summary.json` within `1e-10`, including
the raw value ordering and raw-file SHA256 bindings. No repair selection or
invalid original attempt is present.

Median output throughput by workload and TP:

| Workload | TP1 | TP2 | TP4 |
| --- | ---: | ---: | ---: |
| random-online | 241.0541 | 236.8167 | 235.3977 |
| sharegpt-online | 196.1837 | 189.1981 | 186.3105 |
| prefix-repetition-online | 180.1713 | 154.9127 | 148.6059 |
| agent-research-online | 186.0071 | 171.7635 | 166.2348 |

## Claim boundary

The analysis correctly states: `fixed 1 RPS is an offered-load latency
checkpoint and does not support capacity-scaling claims`.

The prefix workload is visibly saturated even at this fixed offered rate:

| TP | Median request throughput | Median duration | Median p99 TTFT |
| --- | ---: | ---: | ---: |
| 1 | 0.7038 req/s | 284.17 s | 63.66 s |
| 2 | 0.6051 req/s | 330.51 s | 106.13 s |
| 4 | 0.5805 req/s | 344.54 s | 118.71 s |

Its achieved request rate is well below the offered 1 RPS and its tail queueing
latency is tens to more than one hundred seconds. The decreasing fixed-load
throughput points must therefore not be presented as a TP scaling curve,
scaling efficiency result, or capacity comparison. Capacity claims require the
separate rate-search/scaled matrix.

## Non-blocking observations

- 21 of 36 server logs contain an `AsyncLLM output_handler failed` /
  `EngineDeadError` traceback only after the runner triggers post-measurement
  shutdown. Those logs also show completed request processing and FastAPI
  shutdown completion. With raw failures = 0, submission status `OK`, and cell
  exit 0, these are teardown artifacts rather than measurement failures.
- The environment package inventory reports the installed distribution metadata
  `vllm-ascend=0.23.0.post1`, while the actual imported source path, resolved
  target label, and declared/observed plugin commit identify the tested current
  plugin checkout at `7c8ec865` (derived from the 0.25.1rc1 line). Publication
  should use the Git commit/import-path provenance, not the stale installed
  distribution metadata, and should retain this distinction for auditability.
- `matrix-plan.json` retains its immutable pre-execution status
  `planned-not-executed`. Completion is established by top-level `STATUS=ok`,
  `matrix-summary.json status=ok`, `analysis-summary.json status=validated`, and
  the per-cell/per-submission evidence. Consumers should not treat the plan's
  creation-time status as the final run status.

## Conclusion

The fixed 1 RPS archive at commit `1141df1` passes integrity, coverage,
completion, frozen-environment, and independent statistical checks. It may be
published only with its fixed-load checkpoint boundary. The prefix points are
saturated observations and must remain excluded from scaling claims.
