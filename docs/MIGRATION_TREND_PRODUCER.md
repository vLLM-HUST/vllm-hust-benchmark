# Trend Producer Migration Guide

## Overview

The trend producer (`trend_producer.py`) is a new entry point for generating benchmark entry
artifacts that carry all required trend-coverage fields defined in T06/T08 and pass the T09
admission validator.

## New API

### `produce_trend_entry()` — recommended for new entries

This function wraps the legacy `export_leaderboard_artifacts` and extends the output with explicit
trend-coverage fields. It accepts the same base parameters plus the trend parameters listed below.

**Key difference from the old API:** Every trend parameter is an **explicit keyword argument**. The
producer never infers coverage class, campaign, comparison, repeat metadata, or aggregate from
filenames, environment variables, or frontend context.

```python
from vllm_hust_benchmark.trend_producer import produce_trend_entry

artifact_path = produce_trend_entry(
    # Original export_leaderboard_artifacts parameters (unchanged)
    scenario=scenario,
    metrics_file=Path("results/metrics.json"),
    benchmark_result_file=Path("results/benchmark.json"),
    constraints_file=Path("results/constraints.json"),
    same_spec_file=Path("results/same_spec.json"),
    output_dir=Path("./artifacts"),
    artifact_name="run_leaderboard.json",
    run_id="run-001",
    engine="vllm-hust",
    engine_version="v0.23.1-rc0",
    model_name="Qwen/Qwen2.5-14B-Instruct",
    model_parameters="14B",
    model_precision="BF16",
    hardware_chip_model="910B2",
    chip_count=2,
    submitter="vllm-hust-team",
    # … other base parameters …

    # NEW: explicit trend coverage parameters
    coverage_class="full-matrix",
    campaign_id="full-stack-jul-2026/v1",
    point_role="checkpoint",
    repeat_group="full-stack-jul-2026/v1::qwen25-14b::910B2::BF16::random-online::2chip::multi_gpu::vllm-hust",
    repeat_index=0,
    canonical_aggregate={
        "method": "mean",
        "count": 3,
        "metrics": {"ttft_ms": {"value": 42.0}},
        "outlier_handling": "none",
    },
    trend_status="default",
    # validate=True by default — validates against T09 admission
)
```

### `add_trend_fields_to_existing_entry()` — migration helper

For **existing** (legacy) entries that were produced without trend fields:

```python
from vllm_hust_benchmark.trend_producer import add_trend_fields_to_existing_entry

add_trend_fields_to_existing_entry(
    Path("submissions/legacy-submission/run_leaderboard.json"),
    coverage_class="full-matrix",
    campaign_id="migration/v1",
    point_role="checkpoint",
    repeat_group="mig::group",
    repeat_index=0,
    canonical_aggregate={
        "method": "mean", "count": 3,
        "metrics": {"ttft_ms": {"value": 42.0}},
        "outlier_handling": "none",
    },
    trend_status="default",
    validate=True,
)
```

## Trend Parameters Reference

| Parameter             | Required For                         | Rules                                                                 |
| --------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| `coverage_class`      | All trend entries                    | `"full-matrix"`, `"targeted-pair"`, or `"experimental"`               |
| `campaign_id`         | `full-matrix`, `targeted-pair`       | Unique campaign identifier (e.g. `"full-stack-jul-2026/v1"`)          |
| `comparison_id`       | `targeted-pair`                      | Shared ID linking baseline ↔ head                                     |
| `point_role`          | `full-matrix`, `targeted-pair`       | `"checkpoint"` for matrix, `"baseline"`/`"head"` for pair             |
| `repeat_group`        | When `repeat_index` is set           | Composite key grouping repetitions of the same series                 |
| `repeat_index`        | When `repeat_group` is set           | 0-based index within the repeat group                                 |
| `canonical_aggregate` | Non-experimental with `repeat_group` | Aggregate of the repeat group (from T10)                              |
| `trend_status`        | All trend entries                    | `"default"`, `"experimental"`, `"blocked"`, `"invalid"`, `"excluded"` |
| `trend_reason`        | `blocked`, `invalid`, `excluded`     | Human-readable explanation                                            |

## Validation Behavior

- **`validate=True`** (default): After building the entry, the producer runs the T09 admission
  validator on it. If validation fails, `ValueError` is raised and the entry is **not written**.
- **`validate=False`**: The entry is written even if it fails T09 checks. This is useful for
  migration of known-blocked entries where you want to preserve the provenance.

If any parameter combination violates the schema rules (e.g., `full-matrix` without `campaign_id`),
validation fails **before** the base entry is built, avoiding wasted work.

## Migration Path

### For new benchmark runs

Replace calls to `export_leaderboard_artifacts` with `produce_trend_entry` and supply the required
trend parameters.

### For existing (legacy) entries

1. Read the legacy entry with `add_trend_fields_to_existing_entry`.
1. If the entry lacks some trend data (e.g., no `repeat_group`), set it to `"experimental"` status
   with an appropriate `trend_reason`.
1. Re-run validation to confirm the migrated entry passes.

### For automated scripts (CI/CD)

Update the CLI command to pass `--coverage-class`, `--campaign-id`, etc. to the export command. The
CLI will be extended in T12.

## File Layout

```
src/vllm_hust_benchmark/
  trend_producer.py       # NEW — trend-coverage producer
  leaderboard_export.py   # unchanged — base entry builder
  trend_validator.py      # from T09 — admission validator
  workload_config_contract.py  # from earlier work — effective config checks

tests/
  test_trend_producer.py  # NEW — 31 test cases

schemas/
  leaderboard_trend_v1.schema.json  # from T08 — JSON Schema
```

## Dependencies

- T09: `trend_validator.py` — admission validation
- T10: `aggregate_results.py` — repeat-run aggregation → `canonical_aggregate`
