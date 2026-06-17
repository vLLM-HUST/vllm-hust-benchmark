# Changelog

All notable changes to vllm-hust-benchmark will be documented in this file.

The format is based on Keep a Changelog, and this project aims to keep
Unreleased entries concise and workflow-focused.

## [Unreleased]

### Added
- Added quantized Ascend same-spec benchmark support so workflow-dispatch runs
  can override the runtime model, precision, quantization scheme, dtype, and
  hardware chip metadata without mutating the official FP16 spec files.
- Registered the W8A8 benchmark model identity for `aly16/Qwen2.5-14B-W8A8`
  so exported leaderboard artifacts preserve canonical quantized-model metadata.
- Added a shared same-spec resolver in `src/vllm_hust_benchmark/same_spec.py` so
  official baseline runs and current runs now materialize server/client
  parameters from the same benchmark spec instead of duplicating shell-side
  resolution logic.
- Added `scripts/run-current-ascend-same-spec.sh` to run the current
  `vllm-hust` plus `vllm-ascend-hust` stack against the official Ascend Jan 2026
  benchmark spec while preferring the `vllm-hust-dev` environment Python and the
  local workspace source trees.

### Changed
- Updated current same-spec export to carry workflow-provided model identity,
  model parameters, precision, quantization, chip model, runtime provenance, and
  resolved dtype into `run_leaderboard.json` and same-spec metadata.
- Hardened Ascend same-spec startup by disabling Bash nounset only while
  sourcing Ascend toolkit environment scripts, avoiding `CMAKE_PREFIX_PATH`
  unbound-variable failures without weakening the rest of the runner.
- Corrected leaderboard aggregation validation for hard-constraint snapshots:
  `hard_constraints.scopes` is valid as a standalone section and does not
  require compare groups, goal-progress pairs, or matching baseline rows in the
  current merged snapshot.

- Switched trusted benchmark publication back to a direct bot-authenticated
  commit to `main` that carries both `submissions/<run-id>/` and refreshed
  `leaderboard-data/snapshots/**`, leaving `push-to-hf.yml` as the only
  downstream publish step and making the old snapshot PR / auto-merge workflows
  obsolete.

- Updated `scripts/run-official-ascend-goal-baseline.sh` to resolve runtime
  parameters through the shared same-spec payload before launching the official
  `v0.11.0` baseline.
- Hardened the official and current same-spec runners with managed server
  lifecycle state, startup lock files, and stale-server reclamation so reruns
  can stop their own leftover benchmark services before restarting, while still
  refusing to kill unrelated listeners.
- Extended leaderboard export to persist same-spec metadata
  (`spec_id` / `resolved_spec_hash` / resolved runtime parameters) together with
  runtime provenance for both the engine repo and the Ascend plugin repo, so
  exported artifacts remain auditable after publication.
- Added regression coverage for same-spec resolution, artifact export metadata,
  and runtime provenance propagation.