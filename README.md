# vllm-hust-benchmark

Wrapper benchmark repository for vllm-hust.

The goal is to keep `vllm-hust-benchmark` as the stable entrypoint for running experiments, while the real benchmark implementations stay in `vllm-hust`. This repo resolves the sibling `vllm-hust` and `vllm-hust-website` repositories, invokes the benchmark entrypoints from there, and keeps result export and website publication flows in one place.

For runtime comparisons, the wrapper now understands two execution targets:

- `vllm-hust` (default): the sibling workspace checkout at `../vllm-hust`
- `vllm`: the baseline checkout at `../reference-repos/vllm` (override with `VLLM_BASELINE_VLLM_REPO`)

Use the already configured conda environment for this workspace. Do not rely on the system Python installation for benchmark execution or validation.

## What This Repo Wraps

The actual benchmark implementations live in the sibling `vllm-hust` repository:

- `vllm bench serve`
- `vllm bench throughput`
- `vllm bench latency`
- `vllm bench sweep ...`
- `benchmarks/*.py`

This repository adds a stable orchestration layer around them:

- upstream performance test discovery from `vllm-hust/.buildkite/performance-benchmarks/tests/*.json`
- upstream suite delegation via `vllm-hust/.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`
- scenario registry and command construction
- sibling-repo path resolution
- standard leaderboard artifact export
- website snapshot aggregation via `vllm-hust-website/scripts/aggregate_results.py`

Within that boundary, the important design point is:

1. `vllm-hust-benchmark` should stay thin and should not re-implement benchmark runtime logic that already belongs in `vllm-hust`.
2. Benchmark result shaping and website publishing can still live here because they are cross-repo orchestration concerns.

## Repository Layout

- `src/vllm_hust_benchmark/cli.py`: CLI entrypoint
- `src/vllm_hust_benchmark/integration.py`: sibling repo resolution and wrapper execution helpers
- `src/vllm_hust_benchmark/models.py`: scenario data model and command rendering
- `src/vllm_hust_benchmark/registry.py`: official scenario loading and lookup
- `src/vllm_hust_benchmark/leaderboard_export.py`: website-compatible artifact and manifest exporter
- `src/vllm_hust_benchmark/data/official_scenarios.json`: initial official scenario mirror
- `docs/UPSTREAM_ANALYSIS.md`: analysis of upstream benchmark architecture and why this repo is structured this way
- `docs/LEADERBOARD_ALIGNMENT.md`: scenario taxonomy and exact mapping to website leaderboard schema
- `tests/`: focused tests for registry loading and command generation

## Quick Start

```bash
# inspect the resolved sibling repositories
python -m vllm_hust_benchmark.cli show-repos --validate

# list upstream official tests directly from sibling vllm-hust
python -m vllm_hust_benchmark.cli list-tests --benchmark-type serve

# inspect one upstream test and the wrapped commands it would use
python -m vllm_hust_benchmark.cli show-test serving_llama8B_tp1_sharegpt

# delegate one official test run to vllm-hust's own performance suite
python -m vllm_hust_benchmark.cli run-test serving_llama8B_tp1_sharegpt --execute

# delegate the whole upstream suite from this repository entrypoint
python -m vllm_hust_benchmark.cli run-suite --execute

# list the mirrored upstream scenarios
python -m vllm_hust_benchmark.cli list-scenarios

# build the upstream-equivalent command for an official serving scenario
python -m vllm_hust_benchmark.cli build-command sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct

# execute the constructed command through the sibling vllm-hust repo
python -m vllm_hust_benchmark.cli run sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct --execute

# execute the same scenario against the pinned baseline reference-repos/vllm checkout
python -m vllm_hust_benchmark.cli run sharegpt-online --runtime vllm --model meta-llama/Llama-3.1-8B-Instruct --execute

# run a raw vllm bench command from the sibling vllm-hust repo
python -m vllm_hust_benchmark.cli bench -- serve --model meta-llama/Llama-3.1-8B-Instruct --dataset-name sharegpt

# run a raw vllm bench command from the baseline reference-repos/vllm checkout
python -m vllm_hust_benchmark.cli bench --runtime vllm -- serve --model meta-llama/Llama-3.1-8B-Instruct --dataset-name sharegpt

# run a specific script from vllm-hust/benchmarks
python -m vllm_hust_benchmark.cli run-script benchmark_serving.py -- --help

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
	--hardware-chip-model <hardware-chip-model> \
	--submitter ci

# export directly from a raw vllm bench result json plus constraints metrics
python -m vllm_hust_benchmark.cli export-leaderboard-artifact \
	sharegpt-online \
	--benchmark-result-file ../vllm-hust/benchmarks/results/serving_llama8B_tp1_sharegpt_qps_inf_concurrency_200.json \
	--constraints-file docs/examples/constraints_metrics.sample.json \
	--peak-mem-mb 10240 \
	--output-dir .benchmarks/exports/run-002 \
	--run-id run-002 \
	--engine vllm-hust \
	--engine-version 0.7.3 \
	--model-name meta-llama/Llama-3.1-8B-Instruct \
	--hardware-chip-model <hardware-chip-model> \
	--submitter ci \
	--publish-website \
	--execute

# aggregate exported artifacts into the sibling vllm-hust-website repo
python -m vllm_hust_benchmark.cli publish-website \
	--source-dir .benchmarks/exports/run-001 \
	--execute
```

## Extending With New Scenarios

The intended extension path is:

1. Preserve upstream benchmark semantics where possible.
2. Add new scenarios as new registry entries with explicit tags, defaults, and dataset metadata.
3. Keep hardware-specific overrides isolated in scenario config rather than baking them into the CLI.
4. Export results in a stable way so website and regression pipelines can consume them later.

## Leaderboard And HF Integration

The intended production chain is:

1. Run `vllm-hust` benchmark entrypoints or the upstream performance suite from this repository.
2. Convert measured metrics to a standard leaderboard artifact with `export-leaderboard-artifact`.
3. Optionally run `publish-website`, or pass `--publish-website --execute` to `export-leaderboard-artifact`, to aggregate exports directly into the sibling `vllm-hust-website` checkout for local validation.
4. Upload the aggregated snapshot or raw export directory to the production data channel.
5. Website consumes the generated `leaderboard_single.json`, `leaderboard_multi.json`, and `leaderboard_compare.json` snapshots.

`export-leaderboard-artifact` writes two files:

- `run_leaderboard.json` (configurable): one schema-compatible leaderboard entry.
- `leaderboard_manifest.json`: manifest that points to artifact files and idempotency keys.

When `export-leaderboard-artifact` or `submit` runs inside GitHub Actions, the exporter now also captures GitHub provenance into `metadata`, including the triggering user, commit SHA, commit URL, repository, ref, and optional PR metadata. You can override any of those fields explicitly with `--git-commit`, `--github-user`, `--github-commit-url`, `--github-repository`, `--github-ref`, `--github-event-name`, `--github-pr-number`, and `--github-pr-url`.

For cross-repository CI, `sync-submission-to-hf` is the preferred publish entrypoint after a run has already been exported into one submission directory. It downloads historical raw submissions from a Hugging Face dataset prefix, merges in the new submission, regenerates `leaderboard_single.json`, `leaderboard_multi.json`, `leaderboard_compare.json`, and `last_updated.json`, and uploads both the refreshed snapshots and the new raw submission in one commit.

This is the exact input pattern consumed by website aggregation.

If you already have a raw `vllm bench` result JSON, you do not need to hand-author the full metrics payload anymore. The wrapper can derive the main website metrics from the raw result and only requires a separate constraints metrics JSON.

For the upstream benchmark analysis behind this design, see `docs/UPSTREAM_ANALYSIS.md`.