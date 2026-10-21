# Leaderboard MVP Validation Rules

## Validation command

```bash
python data/validate_schema.py \
  data/leaderboard_single.json \
  data/leaderboard_multi.json \
  data/examples/single_node_example.json \
  data/examples/multi_node_example.json
```

## Rule set (MVP)

1. **Payload shape**
   - accepts entry object or entry array
1. **Entry required keys**
   - `entry_id`, `engine`, `engine_version`, `config_type`, `hardware`, `model`, `workload`,
     `metrics`, `constraints`, `versions`, `environment`, `metadata`
1. **Model identity normalization fields**
   - schema v1 continues to require `model.name`, `model.parameters`, and `model.precision`
   - normalized artifacts should carry `model.canonical_id`, `model.repo_id`, `model.short_name`,
     and `model.display_name`, and all four fields must be strings
   - newly exported normalized artifacts should set `model.name = model.repo_id`
   - `model.display_name` is a presentation label only: in the current contract it should mirror the
     normalized `model.short_name`, not be synthesized from legacy raw values in the frontend
1. **Hardware minimum**
   - `vendor`, `chip_model`, `chip_count`
   - at least one of `interconnect` / `intra_node_interconnect`
1. **Metrics minimum**
   - `ttft_ms`, `throughput_tps`, `peak_mem_mb`, `error_rate`
   - all non-negative, and `error_rate` in `[0,1]`
1. **Cluster rules**
   - single-node: `cluster = null`
   - multi-node: object with `node_count>=2`, `comm_backend`, `topology_type`
1. **Hard constraints rules**
   - `constraints.scenario_source` must be `vllm-benchmark`
   - `constraints.accountable_scope` must include domestic chip class, 7B-13B model band,
     representative business scenario, baseline engine
   - all hard-constraint metric fields are mandatory and must pass type/range checks
   - baseline-relative reduction fields may be negative when the current run regresses the baseline
1. **Version rules**
   - required `protocol`, `backend`, `core`
   - semver-like `X.Y.Z(.W)` or `N/A`

## Fail-fast policy

- Schema mismatch = invalid data, do not publish to website data files.
- Missing required metrics = reject record.
- Multi-node record without valid `cluster` = reject record.
