# Selected-page materialization diagnostic

This operator-level optimization for [issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1)
reduces the work needed to materialize PyramidKV selections. Scoring still sees the same full K
tensor, with the same arithmetic and top-k order. The method resolves each request's page addresses
once, reads only selected V rows, and reuses the first layer's selected addresses for MTP K/V.
Destination addressing is shared across layers and K/V copies. Per-head selections, chronological
tail, private destinations, and nonuniform layer lengths are preserved. No model/hardware
compatibility gate or compression budget changes.

The initial test is an **isolated operator diagnostic**, not a service throughput result. It ran on
a verified 910B2 device with BF16, local 8 query heads / 1 KV head, head dimension 256, ten
full-attention layers and one MTP cache. Both implementations receive identical queries and
scrambled source pages, alternating order across 30 repetitions after five warmups. The baseline is
`47580977a2dca108054470b3132811bd690020a6`.

| Input tokens | Original median (ms) | Candidate median (ms) | Reduction |
| ------------ | -------------------: | --------------------: | --------: |
| 8,192        |              17.8141 |               11.4263 |    35.86% |
| 24,576       |              17.9188 |               11.4653 |    36.02% |
| 32,704       |              18.1324 |               11.6188 |    35.92% |

Whole K/V caches match bit for bit at all three lengths, including MTP and untouched padding; source
caches remain unchanged. CPU regression tests additionally cover distinct per-head selections,
scattered destinations and partial pages. The full local suite passed 78 tests with one opt-in NPU
test skipped; the separate diagnostic above actually ran on NPU.

[Raw timing samples, logs, source hashes and checksums](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-diagnostic)
are retained. Temporary allocations and individual operator events do not establish a service HBM
reduction. The subsequent serving pair completed 409 requests per implementation, with zero failures
and identical outputs/F1 on all 300 quality samples. There were 33 synthetic output differences (32
at C4, one at C1); their cause is still under investigation. CPU-heavy checks overlapped the
original arm, so its timing cannot establish a serving speedup.

A separate live-cache shadow diagnostic completed 20 requests and 40 TP worker checks. All 880 K/V
destination comparisons and all compression results matched bit for bit, including 32K/C4, MTP and
an APC repeat with 20,480 cached prompt tokens. The same query/source cache was used by both
methods, restoring private destination pages between them. This narrows the materialization
investigation; it does not prove cross-process generation determinism.

[Serving and real-cache diagnostic evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-selected-page-serving)
includes raw SSE, failures, worker hashes and the diagnostic overlay. The dedicated fresh-process
ABBA series is now complete: 548 requests succeeded, 372 compression transactions paired with both
TP workers, and all SSE/usage/input hashes were verified. The canonical
[benchmark report and raw archive](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a2684c3701237c0b5a8cb6e73d25ce305d6f0e71/reports/pyramidkv-selected-page-abba-20261009)
preserve all four processes, six measured cells and resource observations.

**No serving acceleration was observed.** Pooled throughput changes for 8K/C1, 8K/C4, 24K/C1 and
24K/C4 are -0.24%, -0.45%, -0.45% and -0.25%; HBM peaks are essentially unchanged. Only two processes
per implementation were measured, so these small changes do not establish a general regression
or benefit. The approximately 36% isolated operator reduction cannot be promoted as end-to-end
acceleration. These observations do **not** supersede the earlier compression-off/on regressions.

Same-implementation fresh-process pairs have 35 and 34 synthetic output differences; cross-method
pairs have 41 and 28. All 20 disputed-C1 repeatability controls have one identical output hash.
Cross-process text differences alone therefore do not identify a selected-page copy defect, while
the underlying generation variability is not proven. No mismatches are discarded.

The report discloses a Harbor constructor check that overlapped the first already-excluded
selected-2 repeatability control. Formal measured telemetry begins 109.564 seconds after that
check ended. All controls/warmups were excluded in the frozen plan, and no known competing
test/lint/audit jobs were sampled during the arms. The scanner is not an exhaustive host-activity
audit. PR #13 is accepted specifically as a local operator optimization, as confirmed by the
project owner on 2026-10-10 (Asia/Shanghai). This acceptance does not establish serving acceleration,
HBM reduction, improved maximum capacity or public release readiness.

Reproduce in the qualified isolated runtime, while it is not processing requests:

```bash
PYTHONPATH=/path/to/candidate/src python scripts/qwen35/benchmark_materialization.py \
  --baseline-source /path/to/4758097/src/vllm_ascend_pyramidkv/method.py \
  --config scripts/qwen35/pyramidkv-aligned.json \
  --hardware Ascend910B2 \
  --output /path/to/new-diagnostic.json
```

Use the baseline's immutable checkout with the unchanged provider. The benchmark script records
hashes of both methods, the provider, configuration and itself. Revert the method commit or point
the runtime back to the baseline checkout to restore old materialization; no cache-state migration
is needed between fresh processes.
