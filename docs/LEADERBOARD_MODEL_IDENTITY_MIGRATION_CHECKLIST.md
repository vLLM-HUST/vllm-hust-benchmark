# Leaderboard Model Identity Migration Checklist

This checklist tracked the implementation wave after Phase 0 decisions were frozen in `docs/LEADERBOARD_MODEL_IDENTITY_NORMALIZATION.md`.

Current status: complete. The transition window is closed.

## Close-Out Summary

- benchmark exporters and validators emit and enforce normalized model identity fields
- checked-in historical `submissions/**/run_leaderboard.json` artifacts are backfilled to the final contract
- website schema and examples require normalized model identity fields
- website aggregation and frontend filters consume normalized identity directly and no longer infer repo ids from short aliases

## Scope Of The Initial Seed

The initial registry seed intentionally covers only models already present in the checked-in leaderboard snapshots.

Current raw values observed in website snapshots:

- `Qwen/Qwen2.5-14B-Instruct` with 13 rows
- `Qwen2.5-14B-Instruct` with 6 rows
- `Qwen2.5-7B-Instruct` with 3 rows

Initial canonical model set:

- `hf:Qwen/Qwen2.5-14B-Instruct`
- `hf:Qwen/Qwen2.5-7B-Instruct`

Seed file:

- `src/vllm_hust_benchmark/data/model_identity_registry.json`

## Seed Mapping

### Canonical record: `hf:Qwen/Qwen2.5-14B-Instruct`

- `repo_id`: `Qwen/Qwen2.5-14B-Instruct`
- `short_name`: `Qwen2.5-14B-Instruct`
- `display_name`: `Qwen2.5-14B-Instruct`
- legacy inputs to normalize:
  - `Qwen/Qwen2.5-14B-Instruct`
  - `Qwen2.5-14B-Instruct`

### Canonical record: `hf:Qwen/Qwen2.5-7B-Instruct`

- `repo_id`: `Qwen/Qwen2.5-7B-Instruct`
- `short_name`: `Qwen2.5-7B-Instruct`
- `display_name`: `Qwen2.5-7B-Instruct`
- legacy inputs to normalize:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen2.5-7B-Instruct`

Note: the checked-in benchmark snapshots currently expose only the short-name form for 7B, but the canonical repo id is already used elsewhere in the workspace model catalog and bootstrap scripts. That is sufficient for the initial seed.

## Implementation Checklist

- [x] Add a loader helper in the benchmark package that reads `model_identity_registry.json` through `importlib.resources`
- [x] Extend leaderboard schema and field spec docs to require `model.canonical_id`, `model.repo_id`, `model.short_name`, and `model.display_name`
- [x] Update `export-leaderboard-artifact` so new writes emit the normalized field set
- [x] Freeze `model.name` writes to `model.repo_id` for all new exporters
- [x] Backfill checked-in historical submissions and website snapshots from the normalized pipeline
- [x] Update website filters and table rendering to use `canonical_id` as value and `display_name` as label
- [x] Add tests that prove mixed raw inputs collapse to one canonical model during normalization and that legacy short-alias-only writes are rejected at publish/import time
- [x] Remove website import-boundary heuristic fallback after historical submissions were normalized

## Touchpoints

Expected first implementation files:

- `src/vllm_hust_benchmark/leaderboard_export.py`
- `src/vllm_hust_benchmark/integration.py`
- `src/vllm_hust_benchmark/data/model_identity_registry.json`
- `vllm-hust-website/data/schemas/leaderboard_v1.schema.json`
- `vllm-hust-website/data/FIELD_SPECIFICATION.md`
- `vllm-hust-website/assets/leaderboard.js`
- benchmark and website snapshot tests that currently assert raw `model.name`

## Exit Condition For This Seed

The initial seed is complete when:

- [x] new exports always write normalized model identity fields
- [x] legacy short-name and repo-id rows collapse into one logical model in filters and compare scopes
- [x] checked-in 7B rows publish the same normalized identity structure
- [x] no checked-in snapshot introduces a bare short alias as the only machine-readable model identifier
