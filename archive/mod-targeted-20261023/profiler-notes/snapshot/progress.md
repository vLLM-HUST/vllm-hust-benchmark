# Progress Log

## 2026-07-18

- Loaded required workflow, planning, seven-step, systems-paper, and
  real-experiment readiness skills.
- Read optimization repository, experiment provenance, runbook, and claim
  discipline references.
- Checked repository status, remotes, submodules, AGENTS, README, project
  brief, and top-tier plan.
- Created persistent planning files for this Potential 04 push.
- Audited current NPU6 probe, diagnosis, runtime hook, overhead, and
  concurrency anomaly artifacts. Found that the strongest current non-obvious
  behavior is a concurrency-2 runtime-prefill tail, but root cause remains
  unresolved below the coarse prefill span.
- Generated `.benchmarks/results/npu6_controlled_fault_matrix/` with
  `make npu6-controlled-fault-matrix PYTHON=python3`.
  - Available checked-in cases: 4/7.
  - Missing real-online gates: queue pressure, KV pressure, cleanup.
  - Available-row causal-rule agreement: 4/4; missing-event-rate p95 max: 0.0.
  - Evidence label: `derived-artifact`; not a repo-launched real-online matrix.
- Updated claim ledger, experiment plan, and paper scaffold to include the
  controlled-fault audit while preserving the
  simulation/existing-server/derived/real-online boundary.
- Verification passed:
  - `make shared-workloads-smoke PYTHON=python3`
  - `PYTHONPATH=src pytest -q` (32 passed)
  - `make synthetic-fault-injection PYTHON=python3`
  - `make top-tier-readiness PYTHON=python3`

## 2026-07-26

- Re-read the full repository instructions, planning files, and required
  systems-paper, experiment-provenance, optimization-workflow, figure, and PDF
  skills/references.
- Audited the dirty worktree without reset, clean, or hardware access.
- Confirmed that the in-flight concurrency analyzer turns the 12.2 s tail into
  four falsifiable sub-prefill instrumentation requirements while retaining
  `root_cause_status=unresolved`.
- Reframed the paper with the seven-step research logic: aggregate-timer
  ambiguity as the problem; ordered events plus completeness/causal rules as
  the mechanism; and only the supported scenario correspondence and coarse
  tail localization as results.
- Replaced eight main-text micro-tables with two readable generated displays.
- Added normal related-work positioning, moderate hyphenation penalties, an
  explicit generative-AI disclosure, and a research-draft/not-candidate
  conclusion.
- No hardware experiment was launched: the exact repository-controlled
  graph-mode protocol and missing queue/KV/cleanup ground truth are scientific
  blockers that must be frozen before device admission.
- Added `docs/submission_readiness_audit.md` with the evidence boundary,
  verification record, verdict, and exact next protocol.
- Final no-hardware verification passed:
  - `make lint`
  - `make test` (32 passed)
  - `make shared-workloads-smoke`
  - `make top-tier-readiness` (32 passed plus all derived gates)
- Built the paper twice after adding page-by-page pypdf canonicalization.
  Both complete builds have SHA-256
  `b4da67d59a466b107bd5e8a4d3e3236b39ed1b397415d7099ad5dcba6543cf71`.
- Rendered and visually inspected all three final pages at 144 dpi. Fonts are
  embedded; no overfull boxes, undefined references, clipping, collisions,
  internal readiness strings, or unreadable tables remain.
- Updated the central workspace PDF at
  `/home/shuhao/llm-optimizations/docs/papers/08-vllm-request-lifecycle-profiler-plugin__request_lifecycle_causal_profiler.pdf`.

## 2026-07-27

- Added a CPU-only intervention-linked lifecycle fixture and evaluator.
- Added tests proving that matched single-stage intervention support is
  distinct from dominant-span localization and that unintended second-stage
  changes invalidate support.
- Focused CPU checks passed (13 tests). No NPU, service, port, Docker, network
  experiment, runtime code, or submodule pointer was touched.
- Evidence status: `simulation/model`; controlled live causal M0 is
  `NOT_M0_PROVEN`.
