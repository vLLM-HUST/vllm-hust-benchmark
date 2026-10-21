# Dataset Matrix Results

The canonical publication lives in
`vLLM-HUST/vllm-hust-benchmark/leaderboard-data/dataset-validation`. This website contains a
generated read-only mirror. It does not start evaluation jobs, apply merge gates, publish to Hugging
Face, or own benchmark facts.

`dataset_validation_index_v1.json` is the model/scenario catalog used by the page. Each entry points
to one model-specific `dataset-validation-v1` artifact. Model selection is preserved in the
`?model=` query parameter. Qwen3.5-35B therefore has a separate measured artifact and index entry;
its cells are never appended to the Qwen2.5 artifact.

## Contract

B1 means **currently best measured vLLM-HUST optimization solution** （当前已实测最优的vLLM-HUST优化方案）.
Eligible candidates may use any compatible MOD or combination. Engine optimization settings are
candidate variables; model, precision, hardware resource budget, input, offered workload and quality
constraints remain matched. Each cell keeps its selected configuration and evidence; optima from
different configurations are not one jointly measured deployment. Unconfirmed candidates remain
candidates, not selected B1 values.

The first frontend contract is `dataset-validation-v1`. A result document contains:

- `source`: producer service, source repository, commit, run id, and an optional immutable artifact
  URL;
- `baseline`: the B0 reference identity used for comparison;
- `scenario`: model, hardware, precision, and scenario identity;
- `datasets` and `metrics`: the declared matrix dimensions;
- `results`: zero or more cells keyed by `dataset_id` and `metric_id`.

Each result cell may provide `status` (`not_tested`, `baseline_only`, `queued`, `running`, `passed`,
`failed`, or `not_applicable`), `value`, `baseline_value`, `delta_pct`, `unit`, `updated_at`, and a
`provenance` object. Missing cells are rendered as `not_tested`; this makes an empty run explicit
instead of presenting zeros as measurements.

`baseline_only` is used when an evidenced B0 value exists but no matching B1 candidate has been
selected. B0 is fixed for the scenario. B1 is selected independently for each dataset/metric cell
from the best compliant measured result; a single B1 engine version is not required. Every populated
B1 cell must retain its exact engine versions and evidence in `provenance`, and must match the B0
model, hardware, precision, workload, and measurement definition.

The production `dataset_validation_v1.b0.json` imports the B0 screenshots supplied in the Home
archive. Its archive SHA-256 and screenshot path are retained per cell. Preparation instructions are
not measurements: datasets without a benchmark-result screenshot remain `not_tested`.

`candidate_search` records each completed B1 evidence audit. A near match is not a B1 value: changes
to model, hardware, precision, dataset materialization, endpoint, arrival policy, execution mode, or
metric definition make the result ineligible for the corresponding cell. The 2026-10-03 audit covers
the benchmark snapshots, result-bearing remote branches, A1-A4 handoff, and evidence already
published by the Plugin page. The vSpec raw artifacts provide 12 eligible B1 cells across GSM8K and
ARC-Easy: request, output-token and total-token throughput, mean TTFT, mean TPOT, and request
success rate. Each cell retains the vSpec repository revision, engine revision, raw artifact, model,
hardware, precision, endpoint, request count, and arrival policy.

Output-token throughput is a separate metric from total-token throughput. The Home workbook supplies
the B0 output-token values for all 28 evidenced B0 datasets. GSM8K has a vSpec B1 measurement but no
measured B0 value in the Home archive or workbook, so its B0 side remains empty. The available
ShareGPT and Sonnet results still use different online/offline or arrival contracts, while the
available InstructCoder result also uses the Coder model. Those near matches remain in their
original leaderboard contexts and are not copied into this matrix.

The Qwen3.5 selector currently exposes one separately configured BidKV measurement: EvoScientist
first 12 requests at TP4, C12, request rate 4, and 512 MiB KV cache per NPU. B0 and B1 are geometric
means across the two order-reversed paired repeats. The artifact preserves both repeat values,
policy calls and selections, exact source revisions, input hash, and archive hash. It is one
measured cell, not a claim of coverage for the rest of the Qwen3.5 matrix, and it is not pooled with
the TP2 SWE Prefix Reuse series.

The checked-in `data/dataset_validation_v1.empty.json` file is a schema-shaped empty fixture for
local UI development. It is not a benchmark result and must be replaced by a signed or otherwise
authenticated service artifact before production ingestion is enabled. For visual smoke testing
only, opening the page with `?demo=1` overlays the fixture's `_demo_results` and shows B0/B1
comparison colors; this private fixture field is not part of the production contract.

## Integration boundary

Add model/configuration artifacts in the benchmark repository; each central `data_file` points to
one `dataset-validation-v1` document. Run `scripts/sync_dataset_validation_snapshots.py` to project
those filenames into website `data_url` values. The adapter rejects unknown statuses, dimensions,
duplicate cells, scenario identity mismatches, and invalid index defaults. Keep it independent from
`leaderboard_v1` and retain the empty, loading, error, and stale-source states.
