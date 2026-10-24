# Experiment Plan

> **Current research direction (2026-08-26):** The G1--G6 ladder below is
> retained as historical development context. The corrected Issue #19 paired
> M0 is an evidence-valid scoped mechanism `NO_GO` (resource pathology 1/10,
> matched latency pathology 0/10, fixed gate 8/10), so do not rerun that M0 or
> implement reconciliation. The active protocol is
> [`blind_attribution_contract.md`](blind_attribution_contract.md): consume
> TraceLoom's structured execution evidence, compare five equal-budget
> diagnosis arms on two fresh opaque positives and one valid negative, and
> validate every non-abstained top-1 with a rank-one counterfactual. A longest
> span or complete trace is not a causal-root-cause result.

> **G1 CPU development checkpoint (2026-08-08, revised after audit):** the
> controlled-trace cases succeeded for the current working trees (profiler
> `feature/kv-recovery-config-cleanup`, runtime
> `feature/rlp-kv-recovery-g1-default-off`). Artifacts in
> `.benchmarks/results/g1_cpu_controlled_trace_20260808/`:
> `recovery_episode/` (real connector flow, six milestones, requeue=0),
> `recovery_episode_requeue/` (full eight-milestone chain with two requeues
> and real monotonic-ns artifact-level decomposition), `no_pressure/` (stores
> only), all `drained` with zero loss; `g1_verification.md/json`,
> `run_metadata.json`, and three test-batch logs (plugin 124 passed in the
> vLLM venv; 122 passed in the prescribed conda env with the 2 cross-repo
> integration tests requiring torch/vllm; runtime KV suites 82 passed).
> Left to later gates: multi-process roster receipt (`[api_server,
> engine_core]`) and the conda-env cross-repo runs (environment dependency).

> **G2 admission (2026-08-08, revised after audit):** read-only
> version-aware whole-trace preflight passed for the current working trees
> (device-owner, model, schema/profile vs sealed G1 shards, expected-process
> receipt via G1 summaries, trace-export, environment). Marker discipline
> verified including the neither-marker and both-markers (fail-closed
> archive) cases. Artifacts:
> `.benchmarks/results/g2_readonly_admission_20260808/` (`READY.txt`,
> `run_metadata.json`). Deferred to G3: live `/v1/models` endpoint check and
> the real multi-process roster receipt, because G2 must not start a service.

> **G3 minimum online trace smoke development capture (2026-08-09):** the
> current working trees produced a successful smoke capture. The Ascend
> `NPUModelRunner.execute_model` override
> (`vllm-ascend vllm_ascend/worker/model_runner_v1.py`) did not call
> `observe_kv_recovery_first_compute`, so the `first_prefill_or_decode` child
> observation never fired on the live V1 Ascend runner; the call was added
> immediately before `_model_forward` (mirroring the base V1 runner). A
> controlled service (Qwen2.5-Coder-14B-Instruct on NPU6,
> `gpu_memory_utilization=0.6`, `max_model_len=32768`, OffloadingConnector /
> TieringOffloadingSpec, `kv_recovery_profile_enabled`) induced two real
> preemption + H2D-restore episodes and produced two complete seven-stage
> chains (`preempt -> restore_start -> restore_done -> scheduler_wakeup ->
> admission -> first_prefill_or_decode`) with zero trace loss, live endpoint
> verified, per-process committed receipts `[api_server, engine_core]` both
> `drained`. Artifacts:
> `.benchmarks/results/g3_minimum_online_trace_smoke_20260809/`
> (`g3_verification.md/json`, `run_metadata.json`, raw shards/logs), labeled
> `real-online` + `smoke-only` (capture evidence, not a matched performance
> result and not benchmark #95 accepted evidence).

> **Current status (2026-08-12):** G1-G6 artifacts are retained as development
> mechanism/capture results. They are not benchmark #95 accepted performance
> evidence. The immediate work is ordinary integration review and CI for the
> profiler, runtime, and Ascend PRs. M0 remains `NOT_M0_PROVEN` until fresh
> opaque cases, custody/reveal blind evaluation, and a flat-summary comparison
> are complete.

## 2026-08-03 KV-Recovery Plan Correction

Faculty comments `5157885904` and `5157931714` assign the next concrete work:
connect PR #4's `preempt`, `restore_start`, `restore_done`,
`scheduler_wakeup`, `requeue`, `admission`, and
`first_prefill_or_decode` fields to a controlled runtime trace while retaining
the complete request ID and its stage association. Only after that trace is
valid should tiering/HBM-only matched runs begin.

This changes the immediate experiment priority, not the M0 research goal or
the existing base-schema semantics. `restore_start`, `restore_done`, and
`scheduler_wakeup` are not existing v1alpha1 event names; PR #4 also uses float
milliseconds and array-valued block IDs while P0 uses monotonic uint64
nanoseconds and scalar bounded metadata. The runtime integration therefore
requires a reviewed, versioned optional adapter/profile. It must define exact
stage boundaries, request/sequence/block-ID bounds, requeue reasons, clock
conversion, and fail-closed handling before adding call sites. An actual
tiering/offload trace also needs a supported versioned communication or
specialty profile; it cannot be mislabeled as
`communication_mode=none`.

Execute the corrected ladder in order:

1. **G0 — offline composition and compatibility, no hardware.** Keep the
   optional profile default-off, review the resolved runtime/device call sites,
   run the profiler and runtime CPU suites, and build the package. Before
   choosing a tiering mode, check scheduler API compatibility: runtime
   `f229ba7...` passes `throttle_prefills` to `schedule()`, whereas device
   `cafad89...`'s `RecomputeScheduler.schedule()` accepts no such argument.
   The selected `OffloadingConnector` path must avoid that incompatible
   scheduler, or the interface must be fixed and tested on CPU.
2. **G1 — CPU controlled trace.** Produce at least one complete synthetic/fake
   runtime recovery episode and one valid trace with no pressure episode.
   Verify an untruncated request-ID association to canonical `trace_id`, engine
   request/sequence identity, recovery stages, and a stable scoped logical
   block association; seven-stage order; repeated requeue reason/count; exact latency
   decomposition; monotonic-ns conversion; zero unaccounted loss; and one
   committed receipt for every expected process.
3. **G2 — read-only NPU6 admission.** Use a version-aware whole-trace preflight.
   `READY.txt` must exist only when `BLOCKED.txt` is absent and every endpoint,
   model, device-owner, schema/profile, expected-process, and trace-export check
   passes. If either marker state is ambiguous or any check fails, stop without
   starting a service or collecting performance data.
4. **G3 — minimum online trace smoke.** Start a controlled service and induce
   one recovery episode. Confirm client request ID through
   engine sequence and every recovery stage, plus complete copy/wait/requeue
   fields. Label it `real-online` with `smoke-only` validity; it proves capture,
   not speedup or representative performance.
5. **G4 — fixed-8-GiB matched modes.** Define nonoverlapping resolved meanings
   for `tiering_disabled`, `tiering_enabled`, and `HBM-only`. Use the same
   request set/seed and all fixed #134 settings below. Run each mode in at least
   three independent service lifecycles in rotating/alternating order and
   publish every repetition plus median/IQR. If supported, test copy
   optimization on/off as a separate one-variable pair. Measure tracing
   disabled/enabled separately to quantify observer overhead.
   Do not run a `RecomputeScheduler`-based row until the G0 signature mismatch
   has been resolved and the exact tested pair is recorded.
   For the local #134 path, select exactly one tiering implementation; do not
   mix the device plugin's NPU tiering spec with the runtime's separate spec.
   HBM-only removes `--kv-transfer-config` rather than using zero CPU capacity.
   If `tiering_disabled` and HBM-only resolve to the same configuration, stop
   and obtain the intended mode definitions instead of publishing duplicate
   controls under different names. An FS tier must use a run-owned root,
   `PYTHONHASHSEED=0`, and a fixed cold/warm-cache policy so state cannot leak
   across independent service lifecycles.
6. **G5 — full capacity surface.** Run the #134 8/16/24/32-GiB capacity and
   workload matrix after the fixed-8-GiB mechanism comparison is sound.
   **Development capture recorded 2026-08-10** — see
   `docs/g5_completion_record.md` and
   `.benchmarks/results/g5_capacity_surface_20260810/`.
7. **G6 — counterfactual.** If causal ranking selects copy, restore/wakeup,
   admission/requeue, or another mechanism, change only the rank-one predicted
   mechanism and rerun a matched pair. The public #134 experiment is mechanism
   evidence; it counts toward blind M0 scoring only in a genuine opaque-case
   custody/reveal evaluation.
   **Development counterfactual recorded 2026-08-11** — see
   `docs/g6_counterfactual_spec.md` / `docs/g6_completion_record.md` and
   `.benchmarks/results/g6_counterfactual_20260811/`. Blind M0 scoring remains
   pending fresh opaque cases and custody/reveal evaluation.

Stop and fail the run closed on stage loss/duplication/inversion, request or
sequence ID drift, restore block drift, unreasoned requeue, trace loss, missing
committed process receipt, correctness failure, or unmatched resolved
configuration. If no real preempt/recovery episode occurs, report negative
coverage rather than a tiering or attribution result.

### Benchmark #134 fixed matrix and measurements

- Model/hardware: `Qwen2.5-14B-Instruct`, Ascend 910B2 x1, FP16,
  `max_model_len=32768`, identical request set, and fully resolved parameters.
- Capacity surface: device KV 8/16/24/32 GiB; `random-online`,
  `sharegpt-online`, and `prefix-repetition-online`; steady 1 RPS; at least
  three independent service processes per point in alternating order.
- Fixed-8-GiB modes: tiering disabled, tiering enabled, HBM-only, and, only if
  supported, copy optimization on/off.
- Service metrics: throughput; mean/P50/P95/P99 TTFT, TPOT, and ITL; error rate;
  running/waiting; batch size; per-step prefill/decode tokens; KV usage, prefix
  hit, preemption, and eviction; episode start/end and affected-request count.
- Per recovery: stable request/sequence/logical-block association, with raw
  physical block IDs explicitly process/rank/allocator scoped; block count,
  bytes, and direction; H2D/D2H copy; inflight migration and budget wait; full stage
  timestamps; copy time; restore-to-wakeup, wakeup-to-admission,
  restore-to-admission, admission-to-first-compute, and total recovery; requeue
  count/reason; CPU-tier-hit-but-waiting duration; copy/decode overlap.
- Statistics/evidence: every repetition, median/IQR, exact parent/core/plugin
  SHAs, model revision, CANN/driver/torch-npu versions, raw artifacts,
  environment and resolved-config manifests, trace schema/profile version,
  request-ID association artifact, expected-process receipts, observer
  overhead, and #89 status `canonical-admitted`, `negative-result`, or
  `blocked` as applicable.

The existing Qwen2.5-7B NPU6 smoke and concurrency-2 prefill-tail artifacts are
useful development triggers only. They do not substitute for #134's 14B,
32K-context, fixed-capacity matched target.

## Phase 0: Trace Schema Validation

- Unit-test lifecycle event ordering, missing stages, and bottleneck
  attribution rules.
- Generate synthetic traces for known tokenizer delay, queueing, prefill,
  decode, KV pressure, streaming backpressure, and cleanup bottlenecks.
- Report attribution accuracy, false positives, false negatives, complete span
  coverage, and missing-event rate. Trace overhead is out of scope for this
  phase because no serving runtime is launched.

Evidence label: `simulation/model`.

Latest valid no-NPU result:
`.benchmarks/results/synthetic_fault_injection_env/`, generated in
`vllm-request-lifecycle-profiler-exp`, reports 7/7 correct synthetic
attributions, zero false positives, zero false negatives, and zero missing-event
rate across all deterministic fault cases. This supports schema/harness
coverage only; it does not support live NPU6 diagnosis or speedup claims.

## Phase 1: Existing-Server Probe on NPU6

- Follow the local NPU6 trace-probe runbook (kept as a local working doc) before
  probing any server.
- Instrument baseline serving without changing runtime behavior.
- Run shared workloads and emit per-request timelines.
- Compare profiler reports against raw logs and simple stage timers.

Evidence label: `existing-server-probe`.

Latest attempt:
`.benchmarks/results/npu6_existing_server_probe_blocked/` records that NPU6 was
visible with no `npu-smi` process owner, but no project-defined existing-server
endpoint, model identity, request client, or trace export hook was available.
This is a blocked/invalid probe directory and must not support paper claims.

Readiness preflight and first trace probe:
`.benchmarks/preflight_npu6_trace_probe.py` now defines the repo-local
read-only preflight. It verifies endpoint authentication, `/v1/models`, model
path, Ascend runtime root, NPU6 process ownership, trace export schema, and the
pinned workload submodule commit. If a runtime trace hook is missing, the
preflight records the missing hook as `trace_export_invalid:*` rather than
claiming an online measurement.

Do not conflate two different readiness snapshots. The historical checked-in
`.benchmarks/results/npu6_trace_probe_preflight/` below was READY for its older
client-proxy run. PR #4's newer read-only preflight was BLOCKED by a failed
models endpoint, missing trace export path, and no process on NPU6; it launched
no service and produced no latency or throughput evidence. That BLOCKED state
does not invalidate the old historical run, but the old READY marker also does
not prove that the current environment is admitted.

Current live trace evidence:

- `.benchmarks/results/npu6_existing_server_trace_probe_smoke/` is an
  `existing-server-probe` client-observed lifecycle proxy trace. It sent 4
  streaming requests to the managed NPU6 endpoint, all succeeded, and emitted
  36 JSONL lifecycle events to
  `/tmp/codex-vllm-request-lifecycle-profiler-npu6-trace.jsonl`.
- `.benchmarks/results/npu6_trace_probe_preflight/` now contains `READY.txt`;
  the trace export schema check passed with 36 records and no missing required
  fields.
- `.benchmarks/results/npu6_existing_server_trace_probe_repeated_smoke/`
  upgrades the probe to a warmup-controlled repeated suite. It sent 1 warmup
  request plus 12 measured streaming requests to the managed NPU6 endpoint,
  all measured requests succeeded, and the measured TTFT summary was p50 83.43
  ms, p95 85.76 ms, and p99 86.65 ms. This removes the first-request warmup
  outlier from the measured summary and gives a stable baseline for later
  enabled/disabled overhead comparisons.
- `.benchmarks/results/npu6_existing_server_trace_overhead_smoke/` is the
  matched no-trace/trace client-probe overhead suite. Each mode sent 12 measured
  streaming requests with 12/12 success. Trace mode emitted 117 client-observed
  lifecycle events. Relative to no-trace, trace mode changed TTFT p95 by
  +4.07 ms and latency p95 by +4.03 ms. This supports a narrow proxy-overhead
  claim for client-side event construction only.
- `.benchmarks/results/npu6_trace_diagnosis/` is a `derived-artifact`
  diagnosis pass over the repeated NPU6 trace probe. It converts 117
  client-observed events into per-request spans for 13 requests with 0.0
  missing-event rate. Among the 12 measured requests, the dominant
  client-visible span is `decode` for 12/12 requests. Decode-proxy p95 is
  98.75 ms and prefill-proxy p95 is 85.50 ms. This is useful because it turns
  raw TTFT/latency into a per-request stage hypothesis, but it remains a
  client-visible proxy diagnosis rather than an internal runtime root cause.
- Boundary: these are client-observed events (`received`, local tokenization,
  HTTP send, first streamed chunk, stream done, cleanup). They are not yet
  internal vLLM scheduler/KV/preemption events and the matched overhead suite
  must not be used as internal runtime-hook or memory-overhead evidence.
- `.benchmarks/results/npu6_existing_server_slow_stream_trace_smoke/` adds a
  controlled client-visible slow-stream shape. It uses the same managed NPU6
  endpoint, `proxy_stage_mode=streaming-proxy`, `per_chunk_read_delay_ms=80`,
  two repeated measured requests over two shared-workload rows, and a
  64-token output cap. All 4 measured requests succeeded. TTFT p95 stayed
  81.14 ms while latency p95 rose to 5208.20 ms. The derived artifact
  `.benchmarks/results/npu6_slow_stream_trace_diagnosis/` attributes 4/4
  measured requests to the `streaming` span with streaming p95 5127.09 ms and
  missing-event-rate p95 0.0. This is client-visible backpressure evidence,
  not internal runtime streaming or scheduler evidence.

Runtime hook integration readiness:
the pinned `third_party/vllm-hust` submodule is on
`feature/request-lifecycle-profiler-runtime-hooks-faculty` at commit `0daab7a30`. It
adds an optional dependency-free shim that imports
`vllm_request_lifecycle_profiler.runtime_hooks` only when
`VLLM_RLP_TRACE_EXPORT_PATH` is set, then emits internal events for request
receipt, tokenization, queue admission, scheduling, prefill completion, first
token, decode completion, stream handoff, and cleanup.

Current paired runtime-hook evidence:
`.benchmarks/results/npu6_runtime_hooks_disabled_low_overhead_baseline/` and
`.benchmarks/results/npu6_runtime_hooks_low_overhead_enabled_smoke/` run the same
warmup-controlled shared-workload shape from the pinned submodule. Both source
runs are `existing-server-probe` evidence with clean `dirty_excluding_output_dir`
metadata and 12/12 successful measured requests. The enabled run writes
`.benchmarks/results/npu6_runtime_hooks_low_overhead_enabled_smoke/runtime_trace.jsonl`
with 117 internal runtime events over 13 external request chains. The derived
pair-plan `.benchmarks/results/npu6_runtime_hook_pair_plan/` reports 13/13
complete chains with all nine stages present and no missing stages. Current
overhead on this small smoke is -3.66 ms TTFT p50, -2.14 ms TTFT p95,
-4.28 ms latency p50, and +0.93 ms latency p95 for hook-enabled versus
hook-disabled, after replacing per-event open/write/close with a persistent
append fd sink.

Next step:
the first hook-enabled controlled fault is now checked in at
`.benchmarks/results/npu6_runtime_hooks_slow_stream_fault_smoke/`. It reruns
the slow-stream client fault while preserving both the client proxy trace and
the internal runtime JSONL. The client diagnosis
`.benchmarks/results/npu6_runtime_hooks_slow_stream_fault_diagnosis/`
attributes 4/4 measured requests to `streaming`; the runtime trace summary
records 45 internal events over 5/5 complete chains with all nine stages
present. This closes the "hook coverage during a known fault" gap for
slow-stream, but it is not yet internal-only diagnosis accuracy. The next
hook-enabled decode-heavy fault is also checked in at
`.benchmarks/results/npu6_runtime_hooks_structured_decode_fault_smoke/`, with
derived diagnosis in
`.benchmarks/results/npu6_runtime_hooks_structured_decode_fault_diagnosis/`.
It uses `shared_scenario_structured_agent_decode` with 128 requested output
tokens, records 4/4 measured success, and attributes 4/4 measured requests to
`decode` with decode-span p95 1986.54 ms while runtime hooks again record 5/5
complete chains.

A valid prompt-heavy low-output fault is now checked in at
`.benchmarks/results/npu6_runtime_hooks_prompt_heavy_low_output_smoke/`, with
derived diagnosis in
`.benchmarks/results/npu6_runtime_hooks_prompt_heavy_low_output_diagnosis/`.
It keeps the same runtime-entering shared workload but requests only 4 output
tokens, records 8/8 measured success, and attributes 8/8 measured requests to
`prefill` with prefill-span p95 133.66 ms, decode-span p95 36.70 ms, and
runtime hooks over 9/9 complete chains.

A concurrency-sensitive sweep is now checked in at
`.benchmarks/results/npu6_runtime_hooks_concurrency_sweep/`. It uses the same
structured-agent workload, 8 requested output tokens, and measured concurrency
1/2/3. All 27 measured requests succeed. Concurrency 1 is mostly `decode`
dominant (7 decode, 2 prefill) with TTFT p95 163.30 ms. Concurrency 2 exposes
a severe prefill/TTFT tail: 8/9 measured requests are `prefill` dominant,
TTFT p95 is 12299.08 ms, latency p95 is 12404.06 ms, and prefill-span p95 is
12298.88 ms. Concurrency 3 is 9/9 `prefill` dominant but has TTFT p95
131.61 ms. Runtime hooks record 270 events across 30/30 complete chains with
no missing-stage counts. Post-hoc internal span analysis confirms that the
concurrency-2 tail is inside runtime prefill: internal prefill p95 is
12261.98 ms at concurrency 2, versus 150.31 ms at concurrency 1 and 106.94 ms
at concurrency 3. The chain-level anomaly analysis in
`.benchmarks/results/npu6_runtime_hooks_concurrency_sweep/concurrency_anomaly_analysis/`
holds prompt tokens at 2335, generation tokens at 8, and cached-token ratio p50
at 0.987; the only two >1 s runtime-prefill outliers occur in concurrency 2.
Treat this as an internally confirmed prefill-tail boundary under fixed
token/cache shape, not as KV allocator proof yet.

The benchmark-owned controlled-fault audit is checked in at
`.benchmarks/results/npu6_controlled_fault_matrix/`. It aggregates the available
NPU6 fault artifacts into seven required rows: four available
existing-server-probe/derived cases (slow streaming backpressure, decode-heavy,
prompt-heavy prefill, and the concurrency-2 prefill tail) and three explicit
missing gates (queue pressure, KV pressure, cleanup). The current matrix reports
4/4 available-row attribution agreement and 0.0 p95 missing-event rate, but its
`real_online_matrix_status` is `incomplete`; use it as a readiness audit, not as
live internal diagnosis accuracy. Raw-log and manual time-to-root-cause
baselines are also recorded as missing/not measured.

Two attempted long-context candidates are intentionally marked invalid:
`.benchmarks/results/npu6_runtime_hooks_long_prefill_fault_smoke/` and
`.benchmarks/results/npu6_runtime_hooks_decode_heavy_fault_smoke/` both failed
before usable runtime-hook attribution. Treat them as service-boundary evidence
for workload design, not as attribution failures. The next submission-critical
gap is finer internal decomposition of the concurrency-2 tail:
scheduler-to-prefill transition, prefill kernel execution, KV allocation/cache
pressure, graph capture or batch transition, and scheduler-output/batch-shape
counters under the same hook-enabled pattern.

The stricter live-fault matrix gate is documented in
the local live-fault matrix plan (kept as a local working doc). It separates
current coverage claims from the
missing ASPLOS-level diagnosis claims: KV-pressure ground truth, timer-only
baseline comparison, TPOT/HBM overhead beyond the smoke workload, and a larger
workload matrix.

## Phase 2: Controlled Fault Injection

- Inject tokenizer slow path, queue surge, long-prompt prefill, decode-heavy
  output, KV pressure boundary, slow streaming client, and cleanup stall
  conditions.
- Measure whether attribution matches injected ground truth.
- For the next NPU6 pass, follow G0-G4 above: validate one complete KV-recovery
  pressure episode before a fixed-8-GiB tiering/HBM-only matched run. Do not
  begin by rerunning the older coarse concurrency-2 experiment.
- Split the recovery interval into copy, restore-to-wakeup,
  wakeup-to-admission, admission-to-first-compute, requeue/wait, and total
  recovery. Keep scheduler-output/batch shape, KV pressure, graph transition,
  and prefill/decode timing as supporting mechanism fields.

Evidence label: `real-online` only when the runtime actually serves requests on
NPU6; otherwise use `replay` or `simulation/model`.

## Required Artifact Fields

Every run directory must include `run_metadata.json` with parent commit,
environment, NPU id, model, runtime, workload source, injected fault, command,
dirty state, and evidence label.

For KV-recovery work it must additionally record the PR #4 merge commit, exact
runtime-core and device-plugin SHAs, model revision, CANN/driver/torch-npu
versions, all resolved server/client parameters, target/spec hash, trace
schema and optional-profile version, request-ID association artifact,
expected-process roster and committed receipts, preflight marker/admission
status, correctness result, trace-loss result, and a separate validity status.
Readiness (`READY` or `BLOCKED`) is orthogonal to the six source labels and is
never itself a performance measurement.


Historical blocked-server and client trace-overhead directories now refer to
[benchmark archive members](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/afd47a8e9964b71844dec75a8e908f279ae98202/archive/mod-history-20261023/profiler/manifest.json), not tracked local files.
See EVIDENCE.md for retrieval. Other entries above retain their stated scope.
