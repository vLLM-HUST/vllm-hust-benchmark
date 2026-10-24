# Potential 04 Request Lifecycle Profiler Push Plan

## Goal

Move the request lifecycle profiler from probe/readiness evidence toward
benchmark-owned real-online controlled-fault evidence, without overstating
synthetic or existing-server-probe results.

## Constraints

- Protect the dirty worktree; do not reset, clean, or revert unrelated changes.
- Parent repository is the source of truth for benchmark scripts, artifacts,
  paper scaffolding, and submodule pointers.
- Runtime changes must stay in the correct submodule feature branch.
- Formal results must use graph mode; eager diagnostics must not be mixed into
  paper evidence.
- Every result directory needs a provenance label and manifest.
- Do not kill unrelated processes or silently switch hardware.

## Phases

| Phase | Status | Notes |
| --- | --- | --- |
| 1. Repository and evidence audit | complete | Read AGENTS/README/status/submodules/results/planning. |
| 2. Identify reusable current artifacts | complete | Existing positive cases: slow-stream, decode, prefill, concurrency anomaly; missing queue/KV/cleanup. |
| 3. Add controlled-fault matrix/baseline tooling | complete | Added derived controlled-fault matrix over checked-in NPU6 artifacts; missing queue/KV/cleanup gates remain explicit. |
| 4. Add 12.2s tail root-cause analysis/ablation scaffold | complete | Tail remains localized only to coarse prefill; four candidate substages and required fields are explicit and tested. |
| 5. Run safe tests/probes | complete | Lint, 32 tests, shared-workload smoke, readiness gates, two stable PDF builds, and three-page visual audit passed; no hardware used. |
| 6. Update claims/paper/artifacts/handoff | complete | Paper and ledger are evidence-scoped; repo audit and central PDF are complete. Verdict: research draft, not a submission candidate. |
| 7. Offline causal-contract readiness | complete | Matched intervention fixture/evaluator separates longest-span localization from intervention support; controlled live gate remains `NOT_M0_PROVEN` |

## Errors Encountered

| Error | Attempt | Resolution |
| --- | --- | --- |
