# LLM Engine Leaderboard MVP Field Specification

This document defines the MVP leaderboard data contract used by the website.

## Source of Truth

- Schema: `data/schemas/leaderboard_v1.schema.json`
- Display data: `data/leaderboard_single.json`, `data/leaderboard_multi.json`
- Derived compare snapshot: `data/leaderboard_compare.json`
- Samples: `data/examples/single_node_example.json`, `data/examples/multi_node_example.json`

## Top-level format

Schema accepts both:

1. **single entry object** (for example files)
1. **entry array** (for website data files)

## Required fields per entry

| Field            | Type         | Notes                                       |
| ---------------- | ------------ | ------------------------------------------- |
| `entry_id`       | string(UUID) | unique record id                            |
| `engine`         | string       | inference engine id (e.g. vllm, sglang)     |
| `engine_version` | string       | engine version string                       |
| `config_type`    | string       | `single_gpu` / `multi_gpu` / `multi_node`   |
| `hardware`       | object       | hardware metadata                           |
| `model`          | object       | model metadata                              |
| `workload`       | object       | workload shape                              |
| `metrics`        | object       | benchmark metrics                           |
| `constraints`    | object       | mandatory hard-constraint context + metrics |
| `versions`       | object       | component versions                          |
| `environment`    | object       | runtime env                                 |
| `metadata`       | object       | provenance and reproducibility              |

## Key nested fields used by website MVP

### `hardware`

- Required: `vendor`, `chip_model`, `chip_count`
- Must provide one of: `interconnect` or `intra_node_interconnect`
- `vendor` supports `Intel/AMD/NVIDIA/Huawei/Unknown/Other`

### `metrics`

- Required: `ttft_ms`, `throughput_tps`, `peak_mem_mb`, `error_rate`
- Optional: `prefix_hit_rate`, `tbt_ms`, `tpot_ms`, KV/evict metrics
- Numeric values are non-negative (`error_rate` in `[0,1]`)

### `model`

- Required: `canonical_id`, `repo_id`, `short_name`, `display_name`, `name`, `parameters`,
  `precision`
- `canonical_id`: machine identifier in `<registry>:<repo_id>` format, for example
  `hf:Qwen/Qwen2.5-14B-Instruct`
- `repo_id`: upstream repository coordinate, for example `Qwen/Qwen2.5-14B-Instruct`
- `short_name`: namespace-free alias, for example `Qwen2.5-14B-Instruct`
- `display_name`: final UI label, for example `Qwen2.5-14B-Instruct`
- Compatibility rule: `model.name` remains in schema v1 and must mirror `model.repo_id`

`display_name` is presentation-only. In the current contract it mirrors the industry-standard public
release string carried by `short_name`, for example `Qwen2.5-14B-Instruct`. Writers may override it
only through an explicit normalized registry entry, not by frontend heuristics.

### `constraints` (Hard constraints)

- Required `scenario_source = "vllm-benchmark"`
- Required `accountable_scope`:
  - `domestic_chip_class`
  - `representative_model_band` (fixed to `7B-13B`)
  - `representative_business_scenario`
  - `baseline_engine`
- Required `constraints.metrics`:
  - `single_chip_effective_utilization_pct`
  - `typical_throughput_ratio_vs_baseline`
  - `typical_ttft_reduction_pct_vs_baseline`
  - `typical_tpot_reduction_pct_vs_baseline`
  - `long_context_length`
  - `long_context_throughput_stable`
  - `long_context_ttft_p95_ms`, `long_context_ttft_p99_ms`
  - `long_context_tpot_p95_ms`, `long_context_tpot_p99_ms`
  - `long_context_ttft_p95_stable`, `long_context_ttft_p99_stable`
  - `long_context_tpot_p95_stable`, `long_context_tpot_p99_stable`
  - `unit_token_cost_reduction_pct`
  - `multi_tenant_high_utilization`

### `cluster`

- `null` for single-node configs
- object for multi-node, with required:
  - `node_count >= 2`
  - `comm_backend`
  - `topology_type`

### `versions`

- Required: `protocol`, `backend`, `core`
- Optional: `control_plane`, `gateway`, `kv_cache`, `comm`, `compression`, `benchmark`
- Version format accepts semver-like string or `N/A`

## MVP consistency rule

Website render fields must be present in schema and examples:

- version: `engine_version`
- config filters: `hardware.chip_model`, `model.canonical_id`, `model.display_name`,
  `model.precision`, `workload.name`
- trend metrics: `metrics.ttft_ms`, `metrics.throughput_tps`, `metrics.peak_mem_mb`,
  `metrics.error_rate`, `metrics.prefix_hit_rate`

Normalized exporters must emit `model.canonical_id`, `model.repo_id`, `model.short_name`, and
`model.display_name`. Website filters and grouping logic treat `canonical_id` as the machine
identity and `display_name` as presentation-only.

## `metadata.runtime_provenance`

Optional but strongly recommended for any `vllm-hust` engine entry. The website trend chart uses the
contained `(engine, plugin)` commit pair to compute the x-axis position; a missing block is
tolerated for legacy entries but loses trend-axis precision.

Expected shape:

```json
"runtime_provenance": {
  "engine": {
    "ref": "a46abb7ae6",  // pragma: allowlist secret
    "commit": "a46abb7ae68acc13a4fc5870db98619b3f97c6e0"  // pragma: allowlist secret
  },
  "plugin": {
    "engine": "vllm-ascend-hust",
    "repository": "vllm-hust/vllm-ascend-hust",
    "ref": "f430530ada",  // pragma: allowlist secret
    "commit": "f430530ada2c0c2ec2f925606494bc95a474d9c8"  // pragma: allowlist secret
  }
}
```

### Plugin commit alignment rule

For any `metadata.git_commit` already present in
`leaderboard-data/snapshots/leaderboard_single.json`, a new submission's
`runtime_provenance.plugin.commit` MUST equal the canonical plugin commit, defined as **the plugin
commit of the earliest-submitted existing entry of that commit group**. This prevents the same
vllm-hust binary from rendering as two separate x-axis positions on the trend chart (the `a46abb7ae`
split incident; see `vllm-hust-benchmark/docs/HISTORICAL_PR_BACKFILL.md`).

The rule is enforced at three layers:

1. **Backfill-time guard** — `vllm-hust-benchmark/scripts/backfill_single_gpu.py` raises
   `PluginCommitMismatch` during `run` and warns during `plan`. `--force-mismatched-plugin-commit`
   overrides and writes an audit entry into `.benchmarks/backfill-single-gpu/state.json`.
1. **Publish-time filter** — this repository's
   `scripts/aggregate_results.py::plugin_commit_mismatch_rejection_reason` drops any entry whose
   plugin commit disagrees with the canonical derived from the merged entry set. The rejection
   message includes the entry_id of the canonical-side row for traceability.
1. **CI sentinel** — `scripts/check_engine_version_consistency.py` and `.github/workflows/ci.yml`
   surface a runtime WARNING when a release `engine_version` (e.g. `0.18.0.post1`) co-exists with a
   dev-build `engine_version` (e.g. `v0.17.2rc0-2810-ga46abb7ae`) for the same
   `metadata.git_commit`. The sentinel fails CI only on hard dev-build sha / `git_commit`
   disagreement, not on the warn-only co-existence case.

## Derived compare snapshot

`leaderboard_compare.json` is generated from validated leaderboard entries after deduplication. It
is not covered by `leaderboard_v1.schema.json` because it is a website-facing derived view, not a
primary benchmark result contract.

Expected top-level fields:

- `schema_version = leaderboard-compare-snapshot/v1`
- `generated_at`
- `group_count`
- `preferred_pair_count`
- `groups[]`

Each compare group carries:

- `scope_key`: exact compare scope key used by the website
  (`model|hardware|precision|workload|config_type|chip_count|node_count|setting_signature`)
- `scope`: human-readable decomposition of that same compare scope
- `scope.setting_signature`: stable setting fingerprint, usually sourced from
  `same_spec.resolved_spec_hash`
- `scope.setting_summary`: readable setting summary such as IO length, TP/PP, dtype, and request
  rate
- `engines[]`: one preferred row per engine after deduplication
- `engines[].constraints`: compare-facing hard-constraint summary copied from the source leaderboard
  artifact, including utilization, long-context stability, and token-cost fields when available
- `preferred_pair`: the head-to-head pair the website should render first, selected by score
  ordering (throughput first, then latency)
