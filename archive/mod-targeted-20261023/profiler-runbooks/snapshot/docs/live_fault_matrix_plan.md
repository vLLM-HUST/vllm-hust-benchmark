# Live Fault Matrix Plan

> **Priority correction (2026-08-03):** faculty comments on profiler issue #1
> supersede the old “rerun concurrency-2 next” instruction. Preserve that
> artifact as a development trigger. PR #4 reconciliation is now complete on
> `feature/kv-recovery-integration`; next complete the versioned
> KV-recovery/communication profile, controlled runtime-trace validation, and
> the remaining G0-G3 gates in `experiment_plan.md`. The next matched candidate
> is benchmark #134's fixed-8-GiB tiering/HBM-only comparison.

The current artifact proves that lifecycle traces can be collected and that
client-visible diagnosis changes under known slow-stream and decode-heavy
faults. That is not yet enough for a systems-paper diagnosis claim. The next
submission-critical milestone is a live fault matrix with ground-truth fault
classes, raw-timer baselines, and explicit overhead measurements.

## Current Supported Evidence

- Synthetic lifecycle attribution: 7/7 deterministic faults are classified
  correctly in the no-NPU harness.
- Client-observed NPU6 proxy traces: warmup-controlled shared workload runs
  produce complete proxy lifecycle events and derived stage hypotheses.
- Runtime hook coverage: the hook-enabled vLLM-HUST submodule records all nine
  lifecycle stages on the NPU6 smoke workload with bounded smoke overhead.
- Controlled slow-stream fault: the known client slow-read fault switches
  client-visible diagnosis to `streaming`, while runtime hooks still record
  complete chains.
- Structured decode-heavy fault: the known decode-heavy workload switches
  client-visible diagnosis to `decode`, while runtime hooks again record
  complete chains.
- Prompt-heavy low-output fault: the same runtime-entering workload with only
  4 requested output tokens switches client-visible diagnosis to `prefill`,
  with 8/8 measured success and complete runtime hook chains.
- Concurrency-sensitive structured-agent sweep: the hook-enabled NPU6 service
  succeeds for 27/27 measured requests across measured concurrency 1/2/3. It
  exposes a non-monotonic prefill/TTFT boundary: concurrency 2 has 8/9
  `prefill` diagnoses and TTFT p95 12299.08 ms, while concurrency 3 has 9/9
  `prefill` diagnoses but TTFT p95 131.61 ms. Runtime hooks record 30/30
  complete chains with no missing stages. Post-hoc internal span analysis
  confirms that the concurrency-2 tail is inside the runtime prefill span:
  runtime prefill p95 is 12261.98 ms at concurrency 2, versus 150.31 ms at
  concurrency 1 and 106.94 ms at concurrency 3. The chain-level anomaly
  artifact
  `.benchmarks/results/npu6_runtime_hooks_concurrency_sweep/concurrency_anomaly_analysis/`
  holds prompt tokens (2335), generation tokens (8), and cached-token ratio p50
  (0.987) constant, and finds the only two >1 s runtime-prefill outliers in
  concurrency 2.

These results support trace feasibility and coverage during known faults. They
do not yet prove internal-only causal diagnosis accuracy.

## Missing Submission Gates

### Gate 1: Valid Prefill or KV-Pressure Fault

The two long-context candidates in
`.benchmarks/results/npu6_runtime_hooks_long_prefill_fault_smoke/` and
`.benchmarks/results/npu6_runtime_hooks_decode_heavy_fault_smoke/` are invalid
boundary evidence because requests failed before usable runtime attribution.
The prompt-heavy low-output run now provides a valid prefill-dominant case that
enters the runtime and produces successful measured requests. The concurrency
sweep adds a concrete anomaly to explain: the concurrency-2 run produces a
12.3 s TTFT/prefill tail that does not appear at concurrency 3, and internal
runtime spans confirm the tail is in prefill under a fixed prompt/output/cache
shape. The remaining gate is to split that prefill span into
scheduler-to-prefill transition, KV allocation/cache pressure, graph capture or
batch transition, and prefill kernel execution, rather than treating a coarse
prefill span as KV proof.

Required evidence:

- Keep `.benchmarks/results/npu6_runtime_hooks_concurrency_sweep/` as the
  candidate boundary evidence.
- Keep `.benchmarks/results/npu6_runtime_hooks_concurrency_sweep/concurrency_anomaly_analysis/`
  as the chain-level audit table for fixed token/cache shape and c=2-only
  outliers.
- Add finer internal span or counter analysis for the concurrency-2 tail:
  scheduler-to-prefill transition, prefill execution, KV allocation/cache
  pressure, graph capture or batch transition, and request-level token counts.
- Preserve client proxy trace and internal `runtime_trace.jsonl`.
- Report whether internal evidence confirms a KV/prefill mechanism or
  overturns the proxy hypothesis.
- Use `FAILED.txt` only if a follow-up candidate is invalid, with the exact
  boundary.

### Gate 2: Raw-Timer Baseline Comparison

The paper must show why lifecycle attribution changes the optimization decision
relative to ordinary timers. The next derived artifact should compare, for each
controlled fault, what a timer-only observer can conclude versus what the
lifecycle spans conclude.

Required rows:

- normal shared workload;
- slow-stream fault;
- decode-heavy fault;
- valid prefill/KV fault once found.

Required columns:

- TTFT p95 and latency p95;
- timer-only likely diagnosis;
- lifecycle dominant span;
- next optimization action under timer-only reasoning;
- next optimization action under lifecycle reasoning;
- whether the two actions differ.

### Gate 3: Runtime Hook Overhead Beyond Smoke

The current hook-disabled/enabled pair is a small smoke. Before broad overhead
claims, add at least one longer decode-heavy workload and one prompt-heavy
workload.

Required metrics:

- TTFT p50/p95 delta;
- latency p50/p95 delta;
- TPOT or per-token decode latency delta when token counts are available;
- trace bytes per request;
- process RSS/HBM snapshot before and after the run;
- complete-chain coverage.

### Gate 4: Claim Discipline

Use these terms until all gates close:

- "runtime hook coverage during known faults";
- "client-visible diagnosis";
- "derived optimization decision aid";
- "candidate prefill/KV fault boundary".

Do not use these terms yet:

- "internal-only causal diagnosis accuracy";
- "root cause is proven by runtime hooks";
- "broadly negligible overhead";
- "time-to-root-cause reduction" unless a raw-log/manual baseline is measured.

## Suggested Next NPU6 Run

Do not launch another coarse-prefill run yet. After G0-G2 admit the environment,
run one minimal controlled pressure episode and split its chain into D2H/H2D
copy, restore-to-wakeup, wakeup-to-admission, admission-to-first-compute,
requeue/wait, and total recovery while preserving explicit request/sequence/
block and transfer handoffs. Then execute the fixed-8-GiB matched modes defined
in `experiment_plan.md`.

The old concurrency-2 shape may be reused later as a pressure trigger or
secondary control after the recovery profile is live, but it cannot substitute
for #134's Qwen2.5-14B, 32K-context target. If a run has no real recovery
episode, mark negative coverage; if admission or trace validation fails, keep
the exact BLOCKED/FAILED artifact rather than silently deleting it.
