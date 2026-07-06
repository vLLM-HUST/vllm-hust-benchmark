# Performance Gate Spec Registry Rollout Plan

This document defines the plan for moving performance gate spec selection from
hard-coded workflow defaults to a shared registry and resolver owned by
`vllm-hust-benchmark`.

The goal is to make future scenario onboarding mostly data-driven. After the
initial mechanism is in place, adding a new performance gate scenario should
usually require adding a spec file and one registry entry in this repository,
instead of repeatedly editing workflows in `vllm-hust` and `vllm-ascend-hust`.

Tracking issue:
https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/38

## Background

The current performance gate flow still relies on a single default perfgate spec
path in downstream workflows:

```text
docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json
```

This works for the existing `random-online` smoke gate, but it does not scale
well when multiple scenarios need to be enabled. Each new case would otherwise
require repeated edits in:

- `vllm-hust-benchmark`, to add the spec.
- `vllm-hust`, to choose the spec in the benchmark workflow.
- `vllm-ascend-hust`, to mirror the same workflow wiring.
- optionally `vllm-hust-website`, if aggregation or display needs adjustment.

That repeated workflow editing increases the chance of B2/B3 mismatches,
missing files, cross-branch dependencies, and inconsistent behavior between
`vllm-hust` and `vllm-ascend-hust`.

## Goals

- Centralize the mapping from `(scenario, hardware_chip_model)` to perfgate spec
  files in `vllm-hust-benchmark`.
- Keep perfgate spec selection consistent across `vllm-hust` and
  `vllm-ascend-hust`.
- Preserve explicit workflow override support for emergency fixes and manual
  experiments.
- Fail early with a clear error when a scenario/chip pair is unsupported or the
  selected spec file is missing.
- Keep policy decisions, such as report-only versus blocking mode, in the
  downstream workflow repositories.
- Make future scenario rollout incremental and reviewable.

## Non-Goals

- Do not enable every benchmark scenario as a blocking gate in one change.
- Do not move workflow policy into `vllm-hust-benchmark`.
- Do not make PR preview data eligible for formal website aggregation.
- Do not replace official baseline specs. This registry only selects perfgate
  specs used by PR gate or report-mode comparison jobs.
- Do not remove manual `PERFGATE_SPEC_FILE` overrides in the first migration.

## Proposed Mechanism

Add a source-of-truth registry file to `vllm-hust-benchmark`, for example:

```text
src/vllm_hust_benchmark/data/perfgate_spec_registry.json
```

The registry maps a scenario and hardware chip model to a repo-relative spec
file path:

```json
{
  "entries": [
    {
      "scenario": "random-online",
      "hardware_chip_model": "910B2",
      "spec_file": "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json"
    },
    {
      "scenario": "sharegpt-online",
      "hardware_chip_model": "910B2",
      "spec_file": "docs/official-baselines/perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json"
    }
  ]
}
```

Add a resolver module, for example:

```text
src/vllm_hust_benchmark/perfgate_specs.py
```

The resolver should expose both a Python API and a CLI:

```bash
PYTHONPATH="$BENCHMARK_REPO/src" python -m vllm_hust_benchmark.perfgate_specs resolve \
  --scenario "$BENCH_SCENARIO" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --repo-root "$BENCHMARK_REPO"
```

The CLI should print the resolved spec path. By default it should print an
absolute path when `--repo-root` is provided, because workflows ultimately need a
file path that can be passed to same-spec runners.

## Selection Precedence

Downstream workflows should resolve the spec with this precedence:

1. If `PERFGATE_SPEC_FILE` or the existing repo variable override is explicitly
   set, use it.
2. Otherwise resolve by `(BENCH_SCENARIO, HARDWARE_CHIP_MODEL)` through the
   registry.
3. If no scenario is configured, default to `random-online`.
4. If no hardware chip model is configured, default to `910B2` for the current
   self-hosted runner.

The resolver itself should not silently fall back from an unsupported scenario
to `random-online`. Unsupported pairs should fail early with a readable message.
Fallbacks belong in workflow configuration, not inside the registry lookup.

## Resolver Output Contract

The first version should keep the workflow interface simple:

```bash
python -m vllm_hust_benchmark.perfgate_specs resolve ...
```

Standard output:

- one path only
- no extra log lines on success
- absolute path when `--repo-root` is provided
- repo-relative path only if no `--repo-root` is provided

Standard error:

- validation and resolution errors
- supported scenario/chip pairs when lookup fails

Exit codes:

- `0`: resolved successfully
- `2`: invalid input, invalid registry, unsupported pair, or missing spec file

This keeps shell workflow usage straightforward:

```bash
resolved_spec_file="$(
  PYTHONPATH="$BENCHMARK_REPO/src" python -m vllm_hust_benchmark.perfgate_specs resolve \
    --scenario "$BENCH_SCENARIO" \
    --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
    --repo-root "$BENCHMARK_REPO"
)"
```

A future `--format json` option can be added later if workflows need richer
outputs such as spec id, model, or benchmark type.

## Registry Contract

Each registry entry should include:

- `scenario`: the exact scenario name used by `official_scenarios.json`.
- `hardware_chip_model`: a normalized chip model such as `910B2`.
- `spec_file`: a repo-relative path to a perfgate spec JSON file.

The resolver should validate:

- the registry file is valid JSON
- every entry has required fields
- duplicate `(scenario, hardware_chip_model)` pairs are rejected
- `scenario` exists in the official scenario registry
- `spec_file` exists under the provided repo root
- the target spec JSON has the same `scenario`
- the target spec JSON has the same `hardware_chip_model`
- the target spec id is non-empty

Recommended future optional fields:

- `description`
- `owner`
- `status`: `experimental`, `report-only`, or `candidate`
- `notes`

These optional fields are metadata only. Workflow enforcement should not depend
on them until there is an explicit policy decision.

## Repository Responsibilities

### vllm-hust-benchmark

Owns the shared contract:

- perfgate spec files
- perfgate spec registry
- resolver CLI/API
- tests for registry consistency and spec existence
- same-spec validation compatibility

First implementation should include:

- existing `random-online` B2 registry entry
- new `sharegpt-online` B2 registry entry
- tests for successful resolution
- tests for unsupported scenario/chip pairs
- tests for duplicate entries, if the loader is structured enough

### vllm-hust

Consumes the contract:

- checkout `vllm-hust-benchmark`
- install or load the benchmark package as it already does today
- resolve `PERFGATE_SPEC_FILE` after checkout
- pass the resolved spec to same-spec benchmark and perfgate comparison
- keep explicit spec override behavior
- keep scenario selection policy in workflow logic

Initial behavior should keep `random-online` as the default. `sharegpt-online`
can be enabled through manual dispatch or label/report-mode first.

### vllm-ascend-hust

Mirrors `vllm-hust` behavior:

- use the same resolver interface
- keep B2 as the current default runner chip
- preserve manual override support
- align tests with `vllm-hust` so B2/B3 defaults do not drift again

This should be a separate PR after the benchmark registry lands.

### vllm-hust-website

No immediate change is required for the resolver itself.

Website changes are only needed if new formal benchmark data changes
aggregation or display behavior. PR preview perfgate data must remain excluded
from formal leaderboard aggregation.

## Rollout Phases

### Phase 1: Add Registry and Resolver in vllm-hust-benchmark

Scope:

- Add the registry JSON.
- Add the resolver module.
- Register `random-online` and `sharegpt-online` for 910B2.
- Add tests.
- Keep this as a benchmark-only PR.

Exit criteria:

- `random-online` resolves to the existing B2 spec.
- `sharegpt-online` resolves to the new B2 spec.
- resolver fails clearly for unsupported pairs.
- same-spec parsing still works for both specs.

Suggested PR content:

- `docs/official-baselines/perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json`
- `src/vllm_hust_benchmark/data/perfgate_spec_registry.json`
- `src/vllm_hust_benchmark/perfgate_specs.py`
- `tests/test_perfgate_specs.py`
- focused updates to existing official baseline tests

This PR should not change downstream workflow behavior.

### Phase 2: Wire vllm-hust to Use the Resolver

Scope:

- Replace hard-coded default perfgate spec selection with resolver output.
- Keep `VLLM_HUST_PERFGATE_SPEC_FILE` override support.
- Keep default scenario as `random-online`.
- Add static workflow tests.
- Enable `sharegpt-online` only through report-mode or explicit manual trigger
  first.

Exit criteria:

- existing random-online PR gate behavior is unchanged by default.
- manual or label-selected `sharegpt-online` can resolve the correct spec.
- missing registry entries fail before the benchmark starts.

Implementation sketch:

1. checkout `vllm-hust-benchmark`
2. determine `BENCH_SCENARIO`
3. determine `HARDWARE_CHIP_MODEL`
4. if an explicit perfgate spec override is set, use it
5. otherwise call the resolver
6. export the resolved value as `SAME_SPEC_SPEC_FILE`
7. reuse the existing same-spec benchmark and compare steps

This phase should not change main benchmark publication rules.

### Phase 3: Wire vllm-ascend-hust to Use the Resolver

Scope:

- Mirror the `vllm-hust` resolver integration.
- Keep `VLLM_ASCEND_HUST_PERFGATE_SPEC_FILE` override support.
- Keep default scenario as `random-online`.
- Add or update static workflow tests.

Exit criteria:

- `vllm-ascend-hust` no longer hard-codes the B3-era perfgate path.
- B2 default and scenario-aware resolution are both covered by tests.
- `sharegpt-online` can be selected without changing workflow YAML again.

This PR should intentionally mirror the `vllm-hust` implementation instead of
inventing a second resolver or a second mapping table.

### Phase 4: Add More Scenarios One by One

For each new scenario:

1. Add a perfgate spec in `vllm-hust-benchmark`.
2. Add one registry entry.
3. Add or extend tests.
4. Run in report-only mode first.
5. Review runtime, variance, dataset availability, and failure modes.
6. Decide whether to keep report-only or promote to blocking.

Candidate order:

1. `sharegpt-online`
2. `prefix-repetition-online`
3. `random-latency`
4. `sharegpt-throughput`
5. `instructcoder-online`
6. `agent-research-online`
7. `visionarena-online`

The exact order can change based on data availability, runtime cost, and
stakeholder priority.

## PR Breakdown

Recommended PR sequence:

1. `vllm-hust-benchmark`: add registry, resolver, `random-online` entry, and
   `sharegpt-online` entry.
2. `vllm-hust`: consume resolver while keeping default behavior as
   `random-online`.
3. `vllm-ascend-hust`: consume resolver with the same default behavior.
4. `vllm-hust`: enable `sharegpt-online` as report-only through manual dispatch
   or PR label.
5. `vllm-ascend-hust`: enable `sharegpt-online` as report-only through the same
   selection model.
6. `vllm-hust-benchmark`: add the next scenario spec and registry entry.

Do not combine steps 1, 2, and 3 into one PR. The benchmark registry must be
available on `vllm-hust-benchmark@main` before downstream workflows rely on it.

## Validation Plan

Benchmark repository:

```bash
PYTHONPATH=src pytest tests/test_official_baselines.py tests/test_same_spec.py tests/test_registry.py
PYTHONPATH=src pytest tests/test_perfgate_specs.py
git diff --check
```

Resolver smoke checks:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.perfgate_specs resolve \
  --scenario random-online \
  --hardware-chip-model 910B2 \
  --repo-root "$PWD"

PYTHONPATH=src python -m vllm_hust_benchmark.perfgate_specs resolve \
  --scenario sharegpt-online \
  --hardware-chip-model 910B2 \
  --repo-root "$PWD"
```

Downstream repositories:

- static workflow tests for resolved spec wiring
- one `random-online` rerun to prove behavior did not regress
- one `sharegpt-online` report-mode run before enforcing

Hardware validation:

- confirm the runner chip model is still 910B2
- confirm ShareGPT dataset availability or downloader behavior
- confirm benchmark runtime is acceptable for PR preview
- confirm results are not published as formal main benchmark data unless the
  run is explicitly a formal main benchmark

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Downstream workflow references registry before it is on benchmark `main` | CI fails before benchmark | merge benchmark registry PR first |
| Wrong chip default reintroduces B2/B3 mismatch | wrong spec or missing file | resolver tests and downstream static tests assert B2 default |
| Manual override behavior is removed too early | harder emergency rollback | keep explicit `PERFGATE_SPEC_FILE` overrides |
| Registry silently falls back to random-online | false confidence | resolver should fail unsupported pairs |
| New scenario is too slow or unstable | noisy PR gate | start report-only, then promote later |
| PR preview data enters formal aggregation | misleading website data | keep publication/aggregation policy outside resolver |
| Dataset path is unavailable on runner | benchmark fails after startup | validate dataset availability during report-mode rollout |

## Open Questions

- Should registry metadata include `status`, or should status live only in the
  tracking issue and downstream workflows?
- Should `hardware_chip_model` matching be strict, or should aliases such as
  `ascend910b2` normalize to `910B2`?
- Should the resolver print only the path, or support `--format json` for
  richer workflow outputs?
- Should `sharegpt-online` start as manual dispatch only, PR label report-only,
  or both?

## Self Review

This plan is feasible because it fits the current repository boundaries:

- `vllm-hust-benchmark` already owns specs and Python utilities.
- downstream workflows already checkout `vllm-hust-benchmark`.
- the resolver can be introduced without changing benchmark execution logic.
- manual spec overrides provide a rollback path.

The main sequencing constraint is cross-repository ordering. The benchmark
registry must land before downstream workflows depend on it. For that reason,
the first PR should be limited to `vllm-hust-benchmark`.

The plan intentionally does not put enforcement policy in the registry. This is
important because the same scenario may be report-only in one repository while
still experimental or disabled in another. Keeping policy in workflows avoids
turning the registry into a hidden gate controller.

The largest remaining risk is operational rather than architectural:
`sharegpt-online` depends on dataset availability and may have higher runtime or
variance than `random-online`. It should therefore be observed in report-only
mode before becoming a blocking gate.
