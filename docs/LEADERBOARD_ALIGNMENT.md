# Benchmark Scenario Alignment With Website Leaderboard

This document defines the contract between benchmark scenarios in this repository and the website leaderboard ingestion pipeline.

## Goal

Keep benchmark outputs merge-safe and directly consumable by `vllm-hust-website/scripts/aggregate_results.py` without ad-hoc conversion code.

## Required Website Inputs

Website aggregation scans benchmark exports and expects:

- `leaderboard_manifest.json`
- one or more referenced `*_leaderboard.json` artifacts

Each artifact must satisfy `vllm-hust-website/data/schemas/leaderboard_v1.schema.json`.

## Scenario Taxonomy

Scenario source of truth is `src/vllm_hust_benchmark/data/official_scenarios.json`.
Each scenario includes a `leaderboard` block for schema alignment:

- `workload_name`
- `representative_business_scenario`
- `default_config_type`

Current mapping:

- `sharegpt-online` -> workload `sharegpt-online`, business `online-chat`
- `random-online` -> workload `random-online`, business `online-chat`
- `prefix-repetition-online` -> workload `prefix-repetition-online`, business `long-context-chat`
- `instructcoder-online` -> workload `instructcoder-online`, business `code-assistant`
- `visionarena-online` -> workload `visionarena-online`, business `multimodal-chat`
- `sharegpt-throughput` -> workload `sharegpt-throughput`, business `offline-batch-serving`
- `sonnet-throughput` -> workload `sonnet-throughput`, business `offline-batch-serving`
- `random-latency` -> workload `random-latency`, business `latency-slo`

## Field Mapping

`export-leaderboard-artifact` applies this mapping:

- scenario `leaderboard.workload_name` -> artifact `workload.name`
- scenario `leaderboard.representative_business_scenario` -> `constraints.accountable_scope.representative_business_scenario`
- scenario source -> `constraints.scenario_source = vllm-benchmark`
- benchmark metrics payload `metrics` -> artifact `metrics`
- benchmark metrics payload `constraints_metrics` -> `constraints.metrics`
- run metadata -> `metadata.*`, including deterministic `metadata.idempotency_key`

## Metrics Payload Contract

Exporter input is a JSON file with both blocks:

```json
{
  "metrics": {
    "ttft_ms": 42.0,
    "throughput_tps": 321.0,
    "peak_mem_mb": 10240,
    "error_rate": 0.0
  },
  "constraints_metrics": {
    "single_chip_effective_utilization_pct": 92.0,
    "typical_throughput_ratio_vs_baseline": 2.2,
    "typical_ttft_reduction_pct_vs_baseline": 23.0,
    "typical_tpot_reduction_pct_vs_baseline": 25.0,
    "long_context_length": 32768,
    "long_context_throughput_stable": true,
    "long_context_ttft_p95_ms": 80.0,
    "long_context_ttft_p99_ms": 95.0,
    "long_context_tpot_p95_ms": 9.0,
    "long_context_tpot_p99_ms": 10.0,
    "long_context_ttft_p95_stable": true,
    "long_context_ttft_p99_stable": true,
    "long_context_tpot_p95_stable": true,
    "long_context_tpot_p99_stable": true,
    "unit_token_cost_reduction_pct": 35.0,
    "multi_tenant_high_utilization": true
  }
}
```

## HF Delivery Convention

Recommended HF dataset layout:

- `benchmarks/<engine>/<run_id>/leaderboard_manifest.json`
- `benchmarks/<engine>/<run_id>/run_leaderboard.json`

Website CI flow:

1. Download HF snapshot directory.
2. Run `aggregate_results.py --source-dir <snapshot_root> --output-dir vllm-hust-website/data`.
3. Commit updated website data snapshots.
