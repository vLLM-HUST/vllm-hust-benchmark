# Changelog

All notable changes to vllm-hust-benchmark will be documented in this file.

The format is based on Keep a Changelog, and this project aims to keep
Unreleased entries concise and workflow-focused.

## [Unreleased]

### Added
- Added a shared same-spec resolver in `src/vllm_hust_benchmark/same_spec.py` so
  official baseline runs and current runs now materialize server/client
  parameters from the same benchmark spec instead of duplicating shell-side
  resolution logic.
- Added `scripts/run-current-ascend-same-spec.sh` to run the current
  `vllm-hust` plus `vllm-ascend-hust` stack against the official Ascend Jan 2026
  benchmark spec while preferring the `vllm-hust-dev` environment Python and the
  local workspace source trees.

### Changed
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