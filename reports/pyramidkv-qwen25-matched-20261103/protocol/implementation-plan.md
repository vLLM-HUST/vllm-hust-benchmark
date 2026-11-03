# Qwen2.5-14B reproduction implementation plan

> Execute with superpowers:executing-plans. Steps use checkboxes for tracking.

**Goal:** Measure current independent PyramidKV OFF/ON at the user's historical
Qwen2.5-14B, 7168/512, concurrency32 workload on the approved current runtime.

**Architecture:** Add one exact dense eager compatibility profile to the
existing adapter; reuse selection, private-page transactions and per-layer
metadata. Freeze a reproducible evaluation package and preserve every run.

**Tech Stack:** Python, pytest, Torch/Torch-NPU, pinned vLLM/Ascend, one 910B2.

**Spec:** `docs/superpowers/specs/2026-11-03-qwen25-retest-design.md`.

## Global constraints

- Current runtime approved by user; both arms identical except activation.
- Historical 512/beta20, single-card eager, memory utilization0.8,
  max lengths8192/2048, seq32, APC/async/chunked, no speculation.
- Six alternating fresh lifecycles, same frozen requests; preserve failures.
- Label restored support as a new candidate; retain strict Qwen3.5 validation.
- No local development work during formal requests. Keep workspace clean.

## Review focus

- A qwen2 geometry mismatch or unqualified graph/TP/speculation must fail.
- Dense GQA must conserve layer-specific selected K/V and semantic positions.
- Disabled control must receive identical host compatibility and launch values.
- Generated requests must match frozen legacy semantics and exact payloads.
- C32 request reordering/preemption must not retain stale query or page state.

## Task 1: Narrow Qwen2.5 eager profile

Files: `src/vllm_ascend_pyramidkv/method.py`,
`tests/test_external_method.py`, `docs/support-matrix.md`, `scripts/qwen25/`.

- [ ] Record clean baseline tests and original unsupported-profile behavior.
- [ ] Add failing tests for exact Qwen2.5 acceptance and changed geometry,
  TP, graph, speculation and scheduler rejection; verify failures.
- [ ] Implement exact profile alongside current Qwen3.5 checks and add a
  launcher/config preserving the historical parameters.
- [ ] Run targeted and full tests; repair real host seams only with regression
  coverage if live preflight exposes them. Freeze candidate source/package.
- [ ] Commit code/tests/docs after checks pass.

## Task 2: Correctness and frozen paired measurement

Files: evaluation scripts under `scripts/qwen25/`; request/telemetry originals
under `/root/.cache/pyramidkv-qwen25-20261103`.

- [ ] Recover historical generator/launch identity and verify shared model pins.
- [ ] Add tests for request generation/counts, stream metrics and paired
  aggregation using hand-calculated and changed-payload negative fixtures.
- [ ] Run real NPU oracle and serving preflight at short/admission/long/C32
  boundaries; verify transactions and cleanup before timing.
- [ ] Freeze requests, package, launch, six-run order and explicit stop rules.
- [ ] Run OFF/ON repeated three times, one process at a time; retain all outputs,
  timers, counters and failures, then verify each request/transaction offline.

## Task 3: Review, publication and user-facing comparison

- [ ] Produce OFF/ON throughput, TTFT, TPOT, E2E, KV and HBM comparison plus
  historical columns; preserve the approved runtime difference and limits.
- [ ] Run full checks and one fresh whole-branch review; fix substantive issues.
- [ ] Publish reviewed code and canonical benchmark originals/recomputation;
  verify public readback and write results to existing issue #1.
- [ ] Stop owned services, preserve necessary originals, remove temporary copies.
