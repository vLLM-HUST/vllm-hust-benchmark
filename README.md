# vllm-hust-benchmark

Independent benchmark repository for vllm-hust.

The goal is not to fork random legacy scripts from upstream vLLM. The goal is to mirror the current upstream benchmark boundary, keep the command surface compatible with `vllm bench`, and add a clean scenario registry so new AGI4S and domestic-hardware workloads can be added without turning the repo into a pile of ad-hoc shell scripts.

## What This Repo Mirrors From Upstream vLLM

Upstream vLLM has already moved the main benchmark entrypoints into CLI subcommands:

- `vllm bench serve`
- `vllm bench throughput`
- `vllm bench latency`
- `vllm bench sweep ...`

The real implementation boundary now lives in:

- `vllm/entrypoints/cli/benchmark/`
- `vllm/benchmarks/`

Within that boundary, the most important upstream ideas are:

1. Benchmark entrypoints are CLI-first, not standalone scripts.
2. Dataset handling is centralized around a shared registry and sampling layer.
3. Result production is standardized around throughput, TTFT, TPOT, ITL, E2EL, and related metrics.
4. Online and offline benchmarks share common request and dataset abstractions.
5. CI and regression runs are driven by explicit scenario definitions rather than hand-written command fragments.

## What This Repo Adds

This independent repo adds a scenario registry on top of upstream parity:

1. Official vLLM benchmark scenarios are recorded as data, not scattered across docs and CI JSON files.
2. Command construction is separated from scenario definition.
3. New scenarios can be added by extending a JSON manifest instead of rewriting the CLI.
4. The command builder can print or execute the exact upstream `vllm bench ...` command.

## Repository Layout

- `src/vllm_hust_benchmark/cli.py`: CLI entrypoint
- `src/vllm_hust_benchmark/models.py`: scenario data model and command rendering
- `src/vllm_hust_benchmark/registry.py`: official scenario loading and lookup
- `src/vllm_hust_benchmark/leaderboard_export.py`: website-compatible artifact and manifest exporter
- `src/vllm_hust_benchmark/data/official_scenarios.json`: initial official scenario mirror
- `docs/UPSTREAM_ANALYSIS.md`: analysis of upstream benchmark architecture and why this repo is structured this way
- `docs/LEADERBOARD_ALIGNMENT.md`: scenario taxonomy and exact mapping to website leaderboard schema
- `tests/`: focused tests for registry loading and command generation

## Quick Start

```bash
# list the mirrored upstream scenarios
python -m vllm_hust_benchmark.cli list-scenarios

# build the upstream-equivalent command for an official serving scenario
python -m vllm_hust_benchmark.cli build-command sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct

# execute the constructed command directly
python -m vllm_hust_benchmark.cli run sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct --execute

# inspect how scenarios map to leaderboard fields
python -m vllm_hust_benchmark.cli list-leaderboard-map

# export website-compatible artifact + manifest after a run
python -m vllm_hust_benchmark.cli export-leaderboard-artifact \
	sharegpt-online \
	--metrics-file docs/examples/metrics_payload.sample.json \
	--output-dir .benchmarks/exports/run-001 \
	--run-id run-001 \
	--engine vllm-hust \
	--engine-version 0.7.3 \
	--model-name meta-llama/Llama-3.1-8B-Instruct \
	--hardware-chip-model Ascend-910B \
	--submitter ci
```

## Extending With New Scenarios

The intended extension path is:

1. Preserve upstream benchmark semantics where possible.
2. Add new scenarios as new registry entries with explicit tags, defaults, and dataset metadata.
3. Keep hardware-specific overrides isolated in scenario config rather than baking them into the CLI.
4. Export results in a stable way so website and regression pipelines can consume them later.

## Leaderboard And HF Integration

The intended production chain is:

1. Run benchmark scenarios from this repository.
2. Convert measured metrics to a standard leaderboard artifact with `export-leaderboard-artifact`.
3. Upload the export directory to Hugging Face dataset storage.
4. In website CI, run `vllm-hust-website/scripts/aggregate_results.py` against the downloaded HF snapshot.
5. Publish `leaderboard_single.json`, `leaderboard_multi.json`, and `leaderboard_compare.json` to website data.

`export-leaderboard-artifact` writes two files:

- `run_leaderboard.json` (configurable): one schema-compatible leaderboard entry.
- `leaderboard_manifest.json`: manifest that points to artifact files and idempotency keys.

This is the exact input pattern consumed by website aggregation.

For the upstream benchmark analysis behind this design, see `docs/UPSTREAM_ANALYSIS.md`.