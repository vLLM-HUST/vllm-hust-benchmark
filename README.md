# vllm-hust-benchmark

Wrapper benchmark repository for vllm-hust.

The goal is to keep `vllm-hust-benchmark` as the stable entrypoint for running experiments, while the real benchmark implementations stay in `vllm-hust`. This repo resolves the sibling `vllm-hust` and `vllm-hust-website` repositories, invokes the benchmark entrypoints from there, and keeps result export and website publication flows in one place.

For runtime comparisons, the wrapper now understands two execution targets:

- `vllm-hust` (default): the sibling workspace checkout at `../vllm-hust`
- `vllm`: the baseline checkout at `../reference-repos/vllm` (override with `VLLM_BASELINE_VLLM_REPO`)

Use the already configured conda environment for this workspace. Do not rely on the system Python installation for benchmark execution or validation.

## Local Validation Dependencies

Install the benchmark test toolchain from this repository's optional `test` extra before running local validation:

```bash
python -m pip install -e .[test]
```

This extra intentionally includes `jsonschema`, because benchmark publication and some regression paths invoke the sibling website aggregation script, which validates leaderboard snapshots with `jsonschema`.

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
- `docs/LEADERBOARD_HANDOFF.md`: handoff guide for the leaderboard publication chain and official baseline operations
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

# inspect the wrapped Ascend CI benchmark command without executing it
python -m vllm_hust_benchmark.cli run-ascend-ci \
	--scenario random-online \
	--model Qwen/Qwen2.5-14B-Instruct \
	--num-prompts 200

# execute the same CI wrapper for a ShareGPT-backed run
python -m vllm_hust_benchmark.cli run-ascend-ci \
	--execute \
	--scenario sharegpt-online \
	--model Qwen/Qwen2.5-14B-Instruct \
	--dataset-path /data/sharegpt/sharegpt.json \
	--constraints-file docs/examples/constraints_metrics.sample.json \
	--num-prompts 200 \
	--max-concurrency 4

# publish a CI run to HF after export aggregation
python -m vllm_hust_benchmark.cli run-ascend-ci \
	--execute \
	--scenario random-online \
	--model Qwen/Qwen2.5-14B-Instruct \
	--publish-to-hf \
	--allow-random-hf-publish \
	--hf-repo-id vLLM-HUST/leaderboard-preview

# run the Ascend W8A8 quantized model path; dtype=auto is required for modelslim
# quantized models and should be propagated through the same-spec resolver.
python -m vllm_hust_benchmark.cli run-ascend-ci \
	--execute \
	--scenario random-online \
	--model aly16/Qwen2.5-14B-W8A8 \
	--model-precision INT8 \
	--env MODEL_QUANTIZATION=W8A8 \
	--dtype auto \
	--hardware-chip-model 910B2 \
	--num-prompts 200 \
	--request-rate 8 \
	--max-concurrency 4

# list the mirrored upstream scenarios
python -m vllm_hust_benchmark.cli list-scenarios

# build the upstream-equivalent command for an official serving scenario
python -m vllm_hust_benchmark.cli build-command sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct

# execute the constructed command through the sibling vllm-hust repo
python -m vllm_hust_benchmark.cli run sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct --execute

# execute the same scenario against the pinned baseline reference-repos/vllm checkout
python -m vllm_hust_benchmark.cli run sharegpt-online --runtime vllm --model meta-llama/Llama-3.1-8B-Instruct --execute

# execute both runtimes sequentially from one entrypoint
python -m vllm_hust_benchmark.cli run-both sharegpt-online --model meta-llama/Llama-3.1-8B-Instruct --execute

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

For run-ascend-ci:

1. The command stays a thin wrapper around vllm-hust/.github/workflows/scripts/run_ascend_benchmark_ci.sh, so CI-side defaults still live in that script.
2. random-online can run with script defaults, but sharegpt-online requires both --dataset-path and --constraints-file.
3. --publish-to-hf requires --hf-repo-id, and random-online also requires --allow-random-hf-publish as a deliberate safeguard.
4. Use repeated --env KEY=VALUE when the local Ascend runtime still needs extra environment injection such as toolkit or cache overrides.

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

For cross-repository CI, `sync-submission-to-hf` is the preferred publish entrypoint after a run has already been exported into one submission directory. It downloads historical raw submissions from the production Hugging Face dataset `intellistream/vllm-hust-benchmark-results`, merges in the new submission, regenerates `leaderboard_single.json`, `leaderboard_multi.json`, `leaderboard_compare.json`, and `last_updated.json`, and uploads both the refreshed snapshots and the new raw submission in one commit.

Because the production website currently prioritizes `github -> hf -> local`, a successful HF upload alone is not sufficient to refresh the live site. The production chain therefore has two stages:

1. `vllm-hust` benchmark CI writes the exported `submissions/<run-id>/` payload and refreshed `leaderboard-data/snapshots/` files directly into `vllm-hust-benchmark@main` in one bot-authenticated commit.
2. The `push-to-hf.yml` workflow in this repository reacts to that `submissions/**` change, syncs the merged raw submission plus refreshed snapshots to the HF dataset, and keeps the GitHub repository itself as the website's first freshness source.

The older snapshot-PR and auto-merge workflows are obsolete under this model. The benchmark repository submission directories and generated snapshots are the authoritative source for public leaderboard data; the HF dataset is a synchronized distribution mirror for consumers that cannot read the GitHub snapshots directly.

If you already have a raw `vllm bench` result JSON, you do not need to hand-author the full metrics payload anymore. The wrapper can derive the main website metrics from the raw result and only requires a separate constraints metrics JSON.

For the upstream benchmark analysis behind this design, see `docs/UPSTREAM_ANALYSIS.md`.

## Official Goal Baseline

The canonical home for the official Ascend goal-baseline runner is this repository, not `reference-repos/*`.
`reference-repos` stays read-only for upstream comparison, while `vllm-hust-benchmark` owns cross-repo orchestration and website artifact export.

Current public baseline target:

- official `vllm v0.18.0`
- official `vllm-ascend v0.18.0`
- canonical spec set under `docs/official-baselines/*.json`
- current hardware target `Huawei 910B2`

The official baseline set is no longer a single `random-online` spec. The canonical spec files under `docs/official-baselines/` are the source of truth for all official scenarios that need a baseline under the pinned January 2026 `v0.18.0` runtime pair.

Retired `vllm v0.11.0` / `vllm-ascend v0.11.0` runs must not be republished to
`leaderboard-data/snapshots`, the website mirror, or the HF snapshot root. Run
`python scripts/validate_public_leaderboard_snapshots.py` before publishing
curated leaderboard data.

Files:

- `scripts/prepare-official-ascend-baseline-env.sh`
- `scripts/run-official-ascend-goal-baseline.sh`
- `docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json`
- `docs/official-baselines/official-ascend-constraints.stub.json`

Example:

```bash
export ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0180"
bash scripts/prepare-official-ascend-baseline-env.sh

export GOAL_BASELINE_ENV_PREFIX="$ENV_PREFIX"
bash scripts/run-official-ascend-goal-baseline.sh \
	docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json
```

Batch trigger for all official specs:

```bash
export GOAL_BASELINE_ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0180"
bash scripts/run-official-ascend-goal-baseline-matrix.sh
```

Batch trigger with repeated candidate selection for missing canonical specs:

```bash
export GOAL_BASELINE_ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0180"
REPEAT_COUNT=3 bash scripts/run-official-ascend-goal-baseline-matrix.sh
```

Notes:

- `prepare-official-ascend-baseline-env.sh` creates or repairs a dedicated conda env for the fixed official baseline only.
- `prepare-official-ascend-baseline-env.sh` now starts with a health check. If the existing env already matches the pinned official baseline, it skips the heavy uninstall/reinstall path and reuses the env as-is.
- If `ENV_PREFIX` is unset, the prepare script now defaults it to `$(conda info --base)/envs/vllm-ascend-official-v0180` instead of assuming `/root/miniconda3/...`.
- `prepare-official-ascend-baseline-env.sh` also owns the benchmark admission preflight: it proactively cleans residual `api_server` / `bench serve` / `EngineCore_DP0` processes and clears the benchmark port before a new run is allowed to start.
- The baseline runtime is pinned to `reference-repos/vllm@v0.11.0` and `reference-repos/vllm-ascend@v0.11.0` worktrees.
- The prepare script intentionally does not install `vllm-hust` or `vllm-ascend-hust` into the official env, to avoid plugin-entry-point contamination.
- The prepare script now defaults `PYTORCH_CPU_INDEX_URL` to `https://download.pytorch.org/whl/cpu` in addition to the Ascend mirror, so `torch==...+cpu` dependencies from `torch-npu` metadata can resolve.
- The torch family package pins are configurable with `OFFICIAL_TORCH_VERSION`, `OFFICIAL_TORCH_NPU_VERSION`, `OFFICIAL_TORCHVISION_VERSION`, and `OFFICIAL_TORCHAUDIO_VERSION`.
- For the default official version set on `Python 3.11`, the prepare script now auto-selects built-in archived wheel URLs for both `x86_64` and `aarch64`, based on the local machine architecture.
- You can bypass drifting package indexes with explicit wheel URLs: `OFFICIAL_TORCH_WHEEL_URL`, `OFFICIAL_TORCH_NPU_WHEEL_URL`, `OFFICIAL_TORCHVISION_WHEEL_URL`, and `OFFICIAL_TORCHAUDIO_WHEEL_URL`.
- If you override the default torch-family versions or use a different Python ABI, the script falls back to index-based resolution unless you also provide explicit wheel URLs.
- The runner executes against the pinned worktrees through `PYTHONPATH` and defaults `VLLM_CACHE_ROOT` to `.cache/official-ascend-goal-baseline/` in this repository.
- The runner calls the same prepare script in admission-only mode immediately before benchmark startup, so residual-process cleanup is a hard precondition rather than a manual step.
- The runner prefers a locally cached Hugging Face snapshot for the target model when one already exists. You can also force a specific local model directory with `OFFICIAL_MODEL_PATH=/abs/model/path`.
- `run-official-ascend-goal-baseline-matrix.sh` is the batch trigger for official baseline establishment. It skips specs that already have canonical submissions under `submissions/<spec-id>/` and prints a hint instead of re-running them.
- When a spec has no canonical submission yet, `run-official-ascend-goal-baseline-matrix.sh` targets `REPEAT_COUNT` successful runs (default `3`), allows bounded transient repeat failures, chooses the run whose primary metric is closest to the median successful candidate set, and promotes only that run into `submissions/<spec-id>/`.
- Set `FORCE_RUN_EXISTING=1` only when you intentionally want a review-only rerun. The matrix runner preserves the current canonical submission and leaves the new result under `.benchmarks/` for manual comparison instead of auto-replacing it.
- By default the matrix runner accepts a degraded but still usable sample set when at least `2` successful repeats exist (`1` when `REPEAT_COUNT=1`). Override with `MIN_SUCCESSFUL_REPEATS=<n>` if you need a stricter bar.
- The matrix runner uses `MAX_REPEAT_ATTEMPTS` (default `REPEAT_COUNT + 1`) to absorb transient engine or runtime crashes without throwing away earlier successful repeats.
- For official baseline workflow dispatch in GitHub Actions, use `.github/workflows/run-official-ascend-baselines.yml`. It runs the same matrix trigger on the self-hosted Ascend runner, promotes missing canonical `submissions/official-ascend-*`, and now publishes those canonical directories plus refreshed `leaderboard-data/snapshots/` to `vllm-hust-benchmark@main` by default so the website can see the result.

This produces:

- a raw benchmark result under `.benchmarks/official-ascend-goal-baseline/`
- a website-compatible leaderboard artifact under `.benchmarks/official-ascend-goal-baseline/submission/`

The exported artifact can then be aggregated into `vllm-hust-website` with `publish-website` or `vllm-hust-website/scripts/aggregate_results.py`.

### Triggering Locally

Recommended first establishment run for all missing official specs:

```bash
cd /path/to/vllm-hust-benchmark
export GOAL_BASELINE_ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0180"
REPEAT_COUNT=3 bash scripts/run-official-ascend-goal-baseline-matrix.sh
```

Run only one spec file:

```bash
cd /path/to/vllm-hust-benchmark
export GOAL_BASELINE_ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0180"
REPEAT_COUNT=3 bash scripts/run-official-ascend-goal-baseline-matrix.sh \
	docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json
```

Run a review-only rerun for a spec that already has canonical data:

```bash
cd /path/to/vllm-hust-benchmark
export GOAL_BASELINE_ENV_PREFIX="$(conda info --base)/envs/vllm-ascend-official-v0180"
FORCE_RUN_EXISTING=1 REPEAT_COUNT=3 bash scripts/run-official-ascend-goal-baseline-matrix.sh \
	docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json
```

Useful local switches:

- `PREPARE_OFFICIAL_ENV=1` (default): create or repair the pinned official env before the batch.
- `FORCE_REPAIR_OFFICIAL_ENV=1`: bypass the prepare script health check and force a full reinstall/repair of the official env.
- `REPEAT_COUNT=3`: recommended default for establishing a missing canonical spec.
- `FORCE_RUN_EXISTING=1`: rerun even if canonical data already exists, but do not overwrite canonical automatically.
- `MIN_SUCCESSFUL_REPEATS=<n>`: minimum successful repeats required before canonical selection is allowed. Default is `2` when `REPEAT_COUNT >= 2`, otherwise `1`.
- `MAX_REPEAT_ATTEMPTS=<n>`: hard cap on total attempts per missing canonical spec, including retries after transient failures. Default is `REPEAT_COUNT + 1`.
- `PUBLISH_WEBSITE=1`: rebuild the checked-out local `vllm-hust-website/data` from the full `submissions/` tree after the batch. This is a local workspace refresh, not the live publication path.
- `PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu`: default CPU wheel index used by the prepare script for torch-family dependency resolution.
- `OFFICIAL_TORCH_NPU_WHEEL_URL=<exact-wheel-url>`: preferred escape hatch when the historical `torch-npu` build has disappeared from the live mirror.

Operational behavior notes:

- Each matrix run now persists a preferred single-card Ascend device in `.benchmarks/<matrix-run-id>/preferred-ascend-device`, so later repeats and later specs in the same batch reuse the same idle NPU when it remains available.
- `FORCE_RUN_EXISTING=1` is a manual-review path for an existing canonical spec. It reruns that spec once, suppresses canonical promotion, and collapses the repeat-selection knobs to a single attempt even if `REPEAT_COUNT` is larger.

Outputs:

- Per-run artifacts go to `.benchmarks/<matrix-run-id>/...`
- Promoted canonical submissions go to `submissions/<spec-id>/`
- The batch summary goes to `.benchmarks/<matrix-run-id>/summary.md` unless `MATRIX_SUMMARY_FILE` is overridden.

### Triggering The Workflow

The manual workflow entrypoint is `.github/workflows/run-official-ascend-baselines.yml` and is intended for a self-hosted Ascend runner.

Recommended `workflow_dispatch` inputs for the first full establishment run:

- `spec_paths`: leave blank to cover `docs/official-baselines`
- `repeat_count`: `3`
- `min_successful_repeats`: `0` (workflow default; resolves to `2` when `repeat_count >= 2`)
- `max_repeat_attempts`: `0` (workflow default; resolves to `repeat_count + 1`)
- `force_run_existing`: `false`
- `prepare_official_env`: `true`
- `force_repair_official_env`: `false`
- `publish_results`: `true`
- `publish_website`: `false`
- `publication_target_branch`: `main`
- `website_ref`: `main`
- `goal_baseline_env_prefix`: leave blank unless the runner must use a non-default conda base; blank resolves to `$(conda info --base)/envs/vllm-ascend-official-v0180` on the runner

Workflow visibility note:

- GitHub can only dispatch workflows that already exist on the remote repository. Before using `gh workflow run`, commit and push `.github/workflows/run-official-ascend-baselines.yml` to the branch you want to dispatch.
- If you are validating from a feature branch before merge, pass `--ref <branch>` so GitHub dispatches the workflow definition from that remote branch.

Trigger from GitHub UI:

1. Open the Actions page for this repository.
2. Select `Run Official Ascend Baselines`.
3. Click `Run workflow`.
4. Fill `spec_paths` only when you want a subset. Leave it blank for the full official set.

Trigger from `gh` CLI:

```bash
cd /path/to/vllm-hust-benchmark
gh workflow run run-official-ascend-baselines.yml \
	--ref ws/official-baseline-v2 \
	-f repeat_count=3 \
	-f min_successful_repeats=0 \
	-f max_repeat_attempts=0 \
	-f force_run_existing=false \
	-f prepare_official_env=true \
	-f force_repair_official_env=false \
	-f publish_results=true \
	-f publish_website=false \
	-f publication_target_branch=main \
	-f website_ref=main
```

Trigger a subset from `gh` CLI by passing newline-separated `spec_paths`:

```bash
cd /path/to/vllm-hust-benchmark
gh workflow run run-official-ascend-baselines.yml \
	--ref ws/official-baseline-v2 \
	-f spec_paths='docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json
docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json' \
	-f repeat_count=3 \
	-f force_repair_official_env=false \
	-f force_run_existing=false
```

Workflow artifacts:

- `official-baseline-summary-<run_id>-<attempt>`: the batch summary markdown.
- `official-baseline-results-<run_id>-<attempt>`: `.benchmarks/...` results and any promoted `submissions/official-ascend-*` directories.

`publish_results=true` is the formal publication path. It now checks `vllm-hust-benchmark@main` for already-published canonical submissions before launching a missing spec, publishes each newly promoted spec immediately, regenerates `leaderboard-data/snapshots/`, and still runs a final safety sync at the end. This means a later spec failure no longer forces you to rerun already-published official baselines.

`publish_website=true` remains a local-only convenience switch. It refreshes the checked-out `vllm-hust-website/data/` workspace copy after the batch, but by itself it does not update the live benchmark data source.

## Ascend msprof Profiling

For a same-spec run that needs Ascend `msprof` collection, use the profiling wrapper. With no argument, it uses the current default Ascend same-spec baseline, so confirm the target model and hardware before starting a long profiling run:

```bash
bash scripts/run-current-ascend-same-spec-msprof.sh \
  docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json
```

It keeps the existing current same-spec runner as the workload and writes this layout:

```text
.benchmarks/current-ascend-msprof/<run-id>/
  msprof_raw/
  benchmark/
  msprof.log
  run_meta.env
```

Default `msprof` settings live in `scripts/run-current-ascend-same-spec-msprof.env`.

Useful overrides:

- `CONFIG_FILE`: load another shell config file instead of `scripts/run-current-ascend-same-spec-msprof.env`.
- `MSPROF_EXECUTABLE`: use a specific `msprof` binary path.
- `PROFILE_RUN_ID` or `PROFILE_RUN_DIR`: choose the output location.
- `PROFILE_RUN_OVERWRITE=1`: allow reusing an existing output location.
- `MSPROF_FLAGS`: replace the default `--ascendcl=on --runtime-api=on --task-time=l1 --hccl=on --type=text`. This is split on whitespace.
- `MSPROF_ARGS`: in a sourced config file, define a bash array for arguments that need quoting, for example `MSPROF_ARGS=(--foo "a b")`.

The raw profile directory can be analyzed later by a separate analyzer. For example, with TraceLoom:

```bash
traceloom analysis .benchmarks/current-ascend-msprof/<run-id>/msprof_raw \
  --out-dir .benchmarks/current-ascend-msprof/<run-id>/analysis
```

## Batch Same-Spec Matrices

When you want to benchmark one machine group under multiple runtime settings and keep every setting visible on the website leaderboard, use the matrix wrapper instead of invoking the single-spec runner repeatedly by hand.

Files:

- `scripts/run-current-ascend-same-spec.sh`: execute one resolved same-spec benchmark and export one submission
- `scripts/run-current-ascend-same-spec-msprof.sh`: wrap one same-spec benchmark with `msprof`
- `scripts/run-current-ascend-same-spec-msprof.env`: default `msprof` wrapper configuration
- `scripts/run-current-ascend-same-spec-matrix.sh`: iterate a directory or list of spec files, assign one result directory per spec, and optionally regenerate website data after the whole batch

Examples:

```bash
# run every JSON spec under one directory
bash scripts/run-current-ascend-same-spec-matrix.sh docs/spec-matrix/

# run a hand-picked setting list and refresh the local website snapshot afterwards
PUBLISH_WEBSITE=1 bash scripts/run-current-ascend-same-spec-matrix.sh \
	docs/spec-matrix/random-online-tp1.json \
	docs/spec-matrix/random-online-tp2.json \
	docs/spec-matrix/random-online-tp4.json
```

Notes:

- The matrix wrapper does not replace the single-spec runner; it only orchestrates repeated calls to it.
- Every spec gets an isolated `RESULT_DIR` under `.benchmarks/<matrix-run-id>/`, so raw results, resolved same-spec payloads, and exported submissions do not overwrite each other.
- Set `PUBLISH_WEBSITE=1` only after you are ready to regenerate `vllm-hust-website/data` from the full batch output.
