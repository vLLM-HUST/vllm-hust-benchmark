# Leaderboard Model Identity Migration Checklist

This checklist tracks the first implementation wave after Phase 0 decisions were frozen in `docs/LEADERBOARD_MODEL_IDENTITY_NORMALIZATION.md`.

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
- `display_name`: `Qwen 2.5 14B Instruct`
- legacy inputs to normalize:
  - `Qwen/Qwen2.5-14B-Instruct`
  - `Qwen2.5-14B-Instruct`

### Canonical record: `hf:Qwen/Qwen2.5-7B-Instruct`

- `repo_id`: `Qwen/Qwen2.5-7B-Instruct`
- `short_name`: `Qwen2.5-7B-Instruct`
- `display_name`: `Qwen 2.5 7B Instruct`
- legacy inputs to normalize:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen2.5-7B-Instruct`

Note: the checked-in benchmark snapshots currently expose only the short-name form for 7B, but the canonical repo id is already used elsewhere in the workspace model catalog and bootstrap scripts. That is sufficient for the initial seed.

## Implementation Checklist

- [ ] Add a loader helper in the benchmark package that reads `model_identity_registry.json` through `importlib.resources`
- [ ] Extend leaderboard schema and field spec docs to permit `model.canonical_id`, `model.repo_id`, `model.short_name`, and `model.display_name`
- [ ] Update `export-leaderboard-artifact` so new writes emit the normalized field set
- [ ] Freeze `model.name` writes to `model.repo_id` for all new exporters
- [ ] Update aggregation logic to resolve legacy `model.name` values through the registry before grouping or deduplicating
- [ ] Keep fallback reads from legacy snapshots during the transition window
- [ ] Backfill `leaderboard_single.json`, `leaderboard_multi.json`, and `leaderboard_compare.json` from the normalized pipeline
- [ ] Update website filters and table rendering to use `canonical_id` as value and `display_name` as label
- [ ] Add tests that prove `Qwen/Qwen2.5-14B-Instruct` and `Qwen2.5-14B-Instruct` collapse to one canonical model
- [ ] Add tests that reject unknown or ambiguous short aliases at publish time

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

- new exports always write normalized model identity fields
- legacy 14B short-name and repo-id rows collapse into one logical model in filters and compare scopes
- 7B rows also publish the same normalized identity structure
- no new checked-in snapshot introduces a bare short alias as the only machine-readable model identifier