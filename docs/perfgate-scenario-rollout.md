# Perfgate Scenario Rollout Plan

This document tracks the requirement and development plan for extending the vLLM-HUST performance
gate from the current smoke scenario to more benchmark scenarios.

It is intended to be the durable handoff context for GitHub issues, PRs, and future development
sessions.

Detailed registry/resolver design: `docs/perfgate-spec-registry-rollout-plan.md`

Tracking issue: https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/38

## Background

The current performance gate flow mainly validates the `random-online` scenario. This is useful as a
fast smoke gate, but it does not provide enough coverage for real benchmark workloads.

The target direction is to let each important benchmark scenario have:

- a same-spec benchmark definition
- a perfgate baseline definition
- a workflow path that can run the scenario consistently
- a comparison result that can be used for PR gating or reporting
- published benchmark data that can be aggregated by the website

The rollout should be incremental. Each scenario should be added, observed, and stabilized before
the next scenario is enabled.

## Goals

- Extend performance gate coverage beyond `random-online`.
- Add scenarios in small PRs so data can appear continuously.
- Keep PR preview gates fast enough for normal development.
- Keep formal main benchmark runs comprehensive enough for release and website reporting.
- Use consistent spec selection across `vllm-hust` and `vllm-ascend-hust`.
- Avoid hard-coding a single perfgate spec file when multiple scenarios and hardware chips are
  supported.

## Non-Goals

- Do not enable every scenario in one PR.
- Do not mix emergency fixes, such as B2/B3 spec defaults, with broad scenario rollout work.
- Do not make PR preview gates publish formal benchmark results.
- Do not require every scenario to be blocking from day one. New scenarios can start in report mode
  while data quality is verified.

## Current State

### Implemented

- `random-online` is the current primary perfgate scenario.
- The GitHub self-hosted Ascend runner is expected to be 910B2.
- The B2 random-online perfgate spec exists in `vllm-hust-benchmark`:
  `docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json`.
- Multiple official baseline specs already exist under `docs/official-baselines/` for formal main
  benchmark coverage. These specs are useful inputs for scenario rollout, but they do not by
  themselves mean the scenario is already wired into PR perfgate workflows.

### Recently Fixed Separately

The workflow default spec naming had a B3/B2 mismatch. That issue should stay separate from scenario
rollout work:

- `vllm-hust`: defaults were aligned with the B2 runner.
- `vllm-ascend-hust`: defaults should also align with the B2 runner.

### Known Gap

Only `random-online` has the complete gate path today. Other benchmark scenarios do not yet have
complete perfgate coverage.

Existing official baseline candidates include:

- `sharegpt-online`
- `sharegpt-throughput`
- `random-latency`
- `prefix-repetition-online`
- `sonnet-throughput`
- `instructcoder-online`
- `agent-research-online`
- `visionarena-online`

For each candidate, the remaining work is to confirm or add the perfgate spec, wire scenario-aware
workflow selection, run in report mode, and only then decide whether it should become a blocking
gate.

## Target Mechanism

The desired mechanism is:

1. A workflow receives or derives the benchmark scenario.
1. The workflow resolves the hardware chip model, for example `910B2`.
1. A resolver maps `(scenario, hardware_chip_model)` to a spec file.
1. The workflow uses the resolved spec file for same-spec benchmark and perfgate comparison.
1. PR preview results stay separate from formal main benchmark results.
1. Aggregation and website display use only eligible formal benchmark data.

In other words, workflow code should move from a single default:

```text
PERFGATE_SPEC_FILE=docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json
```

to scenario-aware selection:

```text
scenario=random-online, chip=910B2 -> docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json
scenario=sharegpt-online, chip=910B2 -> docs/official-baselines/perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json
```

Exact file names can change, but the naming must remain stable and should include scenario, model
family, model size, and hardware chip.

## Spec Types

There are two related but different spec categories:

- Official baseline specs describe formal benchmark workloads and baseline targets for merged-code
  evaluation and website-visible results.
- Perfgate specs describe the workload used by PR gate or report-mode comparison jobs.

An official baseline spec can be reused as a perfgate spec only if its runtime cost, dataset
requirements, and stability are acceptable for CI. Otherwise, a scenario may need a smaller perfgate
spec that preserves the same workload shape while keeping PR feedback practical.

For example, the current `random-online` perfgate spec uses a smaller 3B smoke configuration, while
formal official baseline specs may use larger 14B workloads.

## Repository Scope

### vllm-hust-benchmark

Responsibilities:

- Own official same-spec and perfgate spec files.
- Add one spec per scenario and hardware target.
- Store or expose baseline data used by comparison jobs.
- Keep spec naming consistent and reviewable.

Expected changes per new scenario:

- add same-spec definition
- add perfgate definition
- add constraints file if the scenario needs special tolerances
- update tests or validation scripts for spec discoverability
- add the scenario to the shared perfgate spec registry after the registry exists

### vllm-hust

Responsibilities:

- Run performance gate for the vLLM-HUST engine.
- Select the correct spec for the requested scenario.
- Keep PR preview behavior separate from formal main benchmark behavior.
- Publish only eligible formal data to downstream aggregation.

Expected changes per new scenario:

- workflow scenario selection or enablement policy update
- static tests for workflow wiring
- optional report-mode rollout before blocking mode

After the shared registry/resolver is wired once, new scenarios should not require adding
per-scenario spec mappings in this repository.

### vllm-ascend-hust

Responsibilities:

- Align the Ascend plugin benchmark gate with `vllm-hust`.
- Use the same scenario and chip selection model.
- Keep B2/B3 hardware naming consistent with the actual runner.

Expected changes per new scenario:

- workflow scenario selection or enablement policy update
- static tests for workflow wiring
- runner-specific defaults if required

After the shared registry/resolver is wired once, new scenarios should not require adding
per-scenario spec mappings in this repository.

### vllm-hust-website

Responsibilities:

- Aggregate and display formal benchmark data.
- Avoid mixing PR preview data into formal leaderboard results.
- Display multiple scenarios clearly.

Expected changes only if needed:

- aggregation filters
- scenario labels
- leaderboard/table display updates

## Rollout Strategy

Roll out one scenario at a time.

Cross-repository dependency order:

1. Add or confirm the scenario spec in `vllm-hust-benchmark`.
1. Wire scenario-aware selection in `vllm-hust`.
1. Wire the same mechanism in `vllm-ascend-hust`.
1. Verify website aggregation and display.
1. Decide whether the scenario stays report-only or becomes blocking.

Do not merge workflow PRs that depend on spec files unavailable on `vllm-hust-benchmark@main`.

### Phase 0: Stabilize random-online

Status: in progress / mostly done.

Tasks:

- Ensure both `vllm-hust` and `vllm-ascend-hust` default to B2 specs for the current 910B2
  self-hosted runner.
- Confirm `random-online` reruns successfully after B2 default fixes.
- Confirm PR preview results do not pollute formal main benchmark aggregation.

Exit criteria:

- PR gate can run with the B2 spec.
- Main benchmark can store baseline data.
- Website aggregation does not include new PR preview data as formal data.

### Phase 1: Add sharegpt-online specs

Status: planned.

Tasks:

- Add `sharegpt-online` specs in `vllm-hust-benchmark`.
- Confirm dataset path availability on the self-hosted runner.
- Confirm model, precision, chip, concurrency, request rate, and constraints.
- Add validation so the spec is discoverable and schema-compatible.

Exit criteria:

- The `sharegpt-online` perfgate spec is available on `main`.
- The spec can be used by downstream workflow PRs without cross-branch dependency.

### Phase 2: Wire sharegpt-online in vllm-hust

Status: planned.

Tasks:

- Consume the shared `vllm-hust-benchmark` perfgate spec resolver.
- Add workflow selection or report-mode enablement for `sharegpt-online`.
- Run `sharegpt-online` in report mode first.
- Observe runtime, stability, variance, and failure modes.

Exit criteria:

- `vllm-hust` can run `sharegpt-online` using scenario-aware spec selection.
- Report-mode results are available for review.
- No blocking gate is enabled until the baseline is considered stable.

### Phase 3: Wire sharegpt-online in vllm-ascend-hust

Status: planned.

Tasks:

- Align with the `vllm-hust` resolver mechanism.
- Consume the shared `vllm-hust-benchmark` perfgate spec resolver.
- Add workflow selection or report-mode enablement for `sharegpt-online`.
- Confirm B2 hardware defaults and spec selection stay consistent.
- Run `sharegpt-online` in report mode first.
- Observe runtime, stability, variance, and failure modes.

Exit criteria:

- `vllm-ascend-hust` can run `sharegpt-online` using scenario-aware spec selection.
- Report-mode results are available for review.
- No blocking gate is enabled until the baseline is considered stable.

### Phase 4: Verify website aggregation and display

Status: planned.

Tasks:

- Confirm PR preview artifacts do not enter formal aggregation.
- Confirm main benchmark artifacts aggregate by scenario.
- Confirm the website display distinguishes `random-online` and `sharegpt-online`.
- Add website aggregation or UI fixes only if the display is unclear.

Exit criteria:

- Formal data and PR preview data remain separated.
- Scenario labels are visible and unambiguous in the website output.

### Phase 5: Repeat for additional scenarios

Status: planned.

Candidate scenarios should be selected based on product value, runtime cost, and stability. Each
scenario should follow the same pattern as `sharegpt-online`.

Definition of done for a scenario:

- The scenario has a reviewed spec on `vllm-hust-benchmark@main`.
- `vllm-hust` can select and run the scenario without hard-coded spec paths.
- `vllm-ascend-hust` can select and run the same scenario consistently.
- PR preview data and formal main benchmark data remain separated.
- Report-mode results have been reviewed for runtime and variance.
- Blocking mode is enabled only after the team accepts the stability tradeoff.

## Spec Selection Requirements

The resolver should support:

- scenario name
- hardware chip model
- optional model family or model size
- repository-specific defaults
- explicit override through GitHub repository variables or workflow inputs

The resolver should fail clearly when no spec exists:

```text
No perfgate spec registered for scenario=<scenario>, hardware_chip_model=<chip>.
```

The resolver should not silently fall back to a different scenario. Fallback from B3 to B2 or from
one scenario to another can hide real data mismatches.

## Data Separation Requirements

PR preview and formal main benchmark data have different meanings.

PR preview data:

- validates a candidate change
- can be used for comments and gate decisions
- should not be treated as formal leaderboard data

Main benchmark data:

- represents merged code
- can update baselines or website-visible benchmark history
- can be aggregated for public display

Aggregation should filter by data source, event type, branch/ref, and publish intent so PR preview
data does not appear as formal benchmark data.

## Validation Plan

For each scenario PR, include:

- static workflow tests
- shell syntax checks for workflow scripts
- spec file validation
- one dry-run or report-mode workflow run when hardware is available
- benchmark result inspection for scenario, model, hardware, source, and metric fields

Suggested commands vary by repository, but common checks include:

```bash
python3 -m py_compile <changed-python-files>
bash -n <changed-shell-scripts>
git diff --check
```

When pytest cannot run locally because hardware or heavyweight dependencies are missing, the PR must
state that explicitly and rely on CI or workflow reruns.

## Risk Register

| Risk                                                        | Impact                                       | Mitigation                                                                 |
| ----------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------- |
| Spec file missing on `vllm-hust-benchmark@main`             | Gate fails before benchmark starts           | Add benchmark spec PR first, verify file exists on main before workflow PR |
| Hardware chip mismatch, for example B2 runner using B3 spec | Invalid comparison or file-not-found failure | Resolve spec by chip and assert defaults in tests                          |
| PR preview data enters website aggregation                  | Misleading leaderboard results               | Keep publish flags and aggregation filters strict                          |
| New scenario is too slow for PR gating                      | Developer feedback becomes slow              | Start in report mode and only gate selected scenarios                      |
| Baseline variance is high                                   | Flaky gates                                  | Observe first, tune thresholds, and gate only after stable                 |
| Cross-repo PRs merge in wrong order                         | CI failures                                  | Use issue checklist and link dependent PRs                                 |

## Tracking Issue

Progress is tracked in:

- https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/38

Each scenario rollout PR should link back to that issue.

## Implementation Checklist

### Phase 0: Stabilize current random-online gate

- [ ] Merge this rollout plan document.
- [ ] Confirm the `vllm-hust` B2 default fix is merged.
- [ ] Confirm the `vllm-ascend-hust` B2 default fix is merged.
- [ ] Confirm `vllm-hust-benchmark` provides the B2 random-online perfgate spec.
- [ ] Rerun the `vllm-hust` random-online gate and confirm it no longer fails because of a missing
  B3 spec.
- [ ] Rerun the `vllm-ascend-hust` random-online gate and confirm it no longer fails because of a
  missing B3 spec.
- [ ] Confirm PR preview benchmark data does not enter formal website aggregation.

### Phase 1: Add sharegpt-online specs in vllm-hust-benchmark

- [ ] Review the existing `sharegpt-online` official baseline spec.
- [ ] Confirm the ShareGPT dataset path and CI runner accessibility.
- [ ] Confirm model, precision, chip, concurrency, request rate, and related benchmark parameters.
- [ ] Add the `sharegpt-online` perfgate spec.
- [ ] Add or reuse a constraints file if the scenario needs special tolerances.
- [ ] Add spec discoverability or schema validation tests.
- [ ] Submit and merge the `vllm-hust-benchmark` spec PR.

### Phase 2: Wire sharegpt-online in vllm-hust

- [ ] Add or extend the `scenario + hardware chip -> spec` resolver.
- [ ] Wire `sharegpt-online` perfgate spec selection.
- [ ] Ensure manual dispatch, issue comment, and PR trigger paths can select the scenario.
- [ ] Keep PR preview and main benchmark publish paths separate.
- [ ] Add workflow static tests.
- [ ] Run `sharegpt-online` in report mode first.
- [ ] Observe result stability and runtime cost.
- [ ] Decide whether to promote `sharegpt-online` to a blocking gate.

### Phase 3: Wire sharegpt-online in vllm-ascend-hust

- [ ] Align with the `vllm-hust` resolver mechanism.
- [ ] Wire `sharegpt-online` perfgate spec selection.
- [ ] Confirm B2 default hardware and spec selection stay consistent.
- [ ] Ensure manual dispatch, issue comment, and PR trigger paths can select the scenario.
- [ ] Keep PR preview and main benchmark publish paths separate.
- [ ] Add workflow static tests.
- [ ] Run `sharegpt-online` in report mode first.
- [ ] Observe result stability and runtime cost.
- [ ] Decide whether to promote `sharegpt-online` to a blocking gate.

### Phase 4: Website and aggregation verification

- [ ] Confirm PR preview benchmark artifacts do not enter formal aggregation.
- [ ] Confirm main benchmark artifacts can aggregate by scenario.
- [ ] Confirm website display can distinguish `random-online` and `sharegpt-online`.
- [ ] If the display is unclear, add a website aggregation or UI follow-up PR.

### Phase 5: Repeat for next scenarios

Candidate scenarios:

- [ ] `sharegpt-throughput`
- [ ] `random-latency`
- [ ] `prefix-repetition-online`
- [ ] `sonnet-throughput`
- [ ] `instructcoder-online`
- [ ] `agent-research-online`
- [ ] `visionarena-online`

Selection criteria:

- Existing official baseline/spec availability
- Dataset availability on the CI runner
- Runtime cost suitable for PR or report-mode gate
- Metric stability
- Product relevance
- Hardware complexity, especially single-chip versus multi-chip

For each selected scenario:

- [ ] Add or confirm the matching specs.
- [ ] Wire the `vllm-hust` workflow.
- [ ] Wire the `vllm-ascend-hust` workflow.
- [ ] Run in report mode first.
- [ ] Decide whether to make it blocking based on stability.
- [ ] Verify website aggregation and display.

## New Conversation Handoff

When continuing this work in a new conversation, start with:

```text
继续多场景性能门禁接入。
进度 issue: https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/38
开发文档:
- docs/perfgate-scenario-rollout.md
- docs/perfgate-spec-registry-rollout-plan.md
请先读取 issue 和两个文档，然后继续下一步。
```
