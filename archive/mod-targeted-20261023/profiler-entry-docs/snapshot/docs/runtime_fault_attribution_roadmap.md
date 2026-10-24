# Runtime Fault Attribution Roadmap

> **Current route (2026-08-03):** the historical evidence below remains valid
> development context. PR #4 is now composed with the frozen parent checkpoint;
> the immediate work is the remaining KV-recovery profile/runtime ladder in
> `experiment_plan.md`, not another direct concurrency-2 rerun.

The repository has passed the trace-capture readiness gate: NPU6 hook-disabled
and hook-enabled source runs are paired, the hook-enabled runtime writes the
full lifecycle schema, and all observed chains contain the required stages.
The next submission-critical question is live attribution accuracy under known
faults.

## Current Runtime Evidence

| Evidence | Result directory | Current meaning |
| --- | --- | --- |
| Hook-disabled baseline | `.benchmarks/results/npu6_runtime_hooks_disabled_low_overhead_baseline/` | Matched NPU6 source row with 12/12 measured success, TTFT p95 80.94 ms, latency p95 179.48 ms |
| Hook-enabled runtime trace | `.benchmarks/results/npu6_runtime_hooks_low_overhead_enabled_smoke/` | 117 internal JSONL events over 13 chains; all chains include received, tokenized, queued, scheduled, prefill_done, first_token, decode_done, stream_done, cleanup_done |
| Pair-plan summary | `.benchmarks/results/npu6_runtime_hook_pair_plan/` | Derived paired gate with complete runtime chains and small-smoke deltas: -2.14 ms TTFT p95 and +0.93 ms latency p95 |
| Slow-stream controlled fault | `.benchmarks/results/npu6_runtime_hooks_slow_stream_fault_smoke/` | Hook-enabled NPU6 run with client slow-read fault: 4/4 measured success, latency p95 5334.74 ms, client diagnosis `streaming` for 4/4 measured requests, and 5/5 complete internal runtime chains |
| Decode-heavy controlled fault | `.benchmarks/results/npu6_runtime_hooks_structured_decode_fault_smoke/` | Hook-enabled NPU6 run with structured-agent decode: 4/4 measured success, latency p95 2135.82 ms, client diagnosis `decode` for 4/4 measured requests, decode-span p95 1986.54 ms, and 5/5 complete internal runtime chains |
| Concurrency anomaly analysis | `.benchmarks/results/npu6_runtime_hooks_concurrency_sweep/concurrency_anomaly_analysis/` | Chain-level derived artifact over checked-in runtime hooks: prompt tokens 2335, generation tokens 8, cached-token ratio p50 0.987, and the only two >1 s runtime-prefill outliers both occur in concurrency 2 |
| Invalid long-context candidates | `.benchmarks/results/npu6_runtime_hooks_long_prefill_fault_smoke/`, `.benchmarks/results/npu6_runtime_hooks_decode_heavy_fault_smoke/` | Both candidate workloads failed before usable attribution under the current 4096-token service profile; keep as boundary evidence only |

This evidence proves full lifecycle capture on the smoke path and coverage
during two known live faults: client-side slow-stream and structured
decode-heavy output. The concurrency anomaly analysis narrows one live anomaly
to coarse runtime prefill under a fixed token/cache shape. It does not yet prove
internal-only fault attribution accuracy, memory overhead, TPOT overhead, or
broad workload generality.

## Next Fault Classes

1. KV recovery: integrate the full preempt/restore/wakeup/requeue/admission/
   first-compute chain and verify one controlled pressure episode.
2. Fixed-8-GiB tiering/HBM-only: run only after the profile, compatibility,
   CPU, and preflight gates pass; split copy from scheduler/admission waiting.
3. Capacity surface: expand to benchmark #134's 8/16/24/32-GiB by three
   workloads at steady 1 RPS.
4. Historical prefill and cleanup: revisit the concurrency-2 anomaly and a
   cleanup stall only after the KV-recovery priority is complete.

## Evidence Rules

- Use NPU6 and `vllm-request-lifecycle-profiler-exp`.
- Treat the old `third_party/vllm-hust`
  `feature/request-lifecycle-profiler-runtime-hooks-faculty` pointer as a
  historical reproducer, not the new integration base. Use the approved exact
  runtime/device pair and record any newly approved compatibility SHA.
- Run one fault class per result directory and record the injected ground truth
  in `run_metadata.json`.
- Preserve both client-observed anchors and internal runtime JSONL.
- Compare causal attribution with raw logs, simple timers, and causal rules
  disabled before claiming decision impact.
- Rerun `make top-tier-readiness` and `make paper-assets` after each valid
  evidence batch.
