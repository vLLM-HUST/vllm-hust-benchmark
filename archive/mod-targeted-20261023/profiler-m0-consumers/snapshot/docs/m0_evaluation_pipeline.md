# M0 Evaluation Pipeline (development-level, 2026-08-11)

> Implements issue #1's first-stage evaluation machinery: normalize request
> traces into a phase DAG, diff baseline vs anomaly on the critical path, rank
> top-k mechanism candidates with confidence and unexplained residual, and
> score against ground truth. This is the **development-level** evaluator; the
> Team-A custody/reveal blind protocol is a separate team-frozen activity.

## Pipeline stages

1. **Normalize** — per-request phase vectors:
   - General phases: `tokenization, queueing, prefill, decode, streaming,
     cleanup` (from `runtime_internal_spans.csv` / runtime-hook traces).
   - KV-recovery phases: `kv_recovery` = `preempt -> first_prefill_or_decode`
     span (from `trace.rlp-kv-recovery.*.jsonl`) plus client end-to-end timing.
2. **Aggregate** — per-phase median over requests (robust).
3. **Critical-path diff** — per-phase delta `anomaly_median - baseline_median`.
4. **Rank** — mechanisms by positive delta share, sorted descending; top-1/top-3
   share = fraction of total positive delta explained.
5. **Residual** — total positive delta minus top-1 delta (unexplained).
6. **Abstain rule** — abstain (`no_single_root_cause` / `insufficient_evidence`)
   if total delta is below a floor or the top-1 share is below a dominance
   threshold.
7. **Score** (when ground truth is known) — top-1/top-3 match, MRR.

## Development validation (NOT blind-scored)

Module: `src/vllm_request_lifecycle_profiler/m0_evaluation.py`
Driver: `.benchmarks/run_m0_development_validation.py`
Results: `.benchmarks/results/m0_development_validation_20260811/validation.json`

| case | baseline | anomaly | ground truth | top-1 | share | residual | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A restore tail | tiering_disabled (no restore) | tiering_enabled (restore) | restore/wakeup | restore/wakeup | 1.000 | 0.000 | correct |
| B concurrency-2 prefill tail | concurrency 1 | concurrency 2 | prefill | prefill | 0.912 | 0.087 | correct |
| C no single root cause | synthetic | synthetic spread | (abstain) | — | — | 1.000 | abstained |
| D low evidence | concurrency 1 | concurrency 3 | (abstain) | — | — | 1.000 | abstained |

Unit tests: `tests/test_m0_evaluation.py` (5 passed).

## Boundary

- Development-level only: ground truth is known and not hidden. It validates
  that the pipeline can localize the two known anomalies and abstain correctly.
- It does **not** establish blind accuracy/MRR/time-to-localize; that requires
  the separately frozen Team-A custody/reveal protocol with fresh opaque cases.
- `time-to-localize` is currently represented by ground-truth rank (rank 1 ⇒
  fastest localization); wall-clock human localization time is out of scope.
- Project remains `NOT_M0_PROVEN`.
