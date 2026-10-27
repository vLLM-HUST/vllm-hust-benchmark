# vllm-hust-benchmark

只安装独立数据组件、避免检出整个历史归档：见 [固定提交 wheel 安装](docs/workload-data/INSTALLATION.zh-CN.md)。

可选的原始数据目录、读取适配与准备入口见 [Workload Data 使用说明](workload-preparation/README.md)；逐来源实际验证进度见
[数据集清单](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)。目录登记不代表任务或性能已验证，不改变现有测试规范与结果。
数据检索与维护入口（Agent）：[数据索引与读写指南](DATA_INDEX.md) ·
[机器可读集合索引](docs/data-index.json)。覆盖原始运行、报告、历史归档、网站来源和契约，明确读取、 追加、订正与验证规则；不移动或覆盖历史证据。

SWE-bench Pro
V1：[从全新 CPU 工作区部署并运行一个真实 Agent](docs/SWE_BENCH_PRO.zh-CN.md#%E6%96%B0%E7%94%A8%E6%88%B7%E6%8E%A8%E8%8D%90%E5%85%A5%E5%8F%A3)。
安装入口自动准备固定源码、独立客户端依赖、V1 数据和 ARM 任务镜像；只连接已有模型 API，无需 Controller。 平台管理员先提供独立 Docker
执行模板和固定镜像仓库，不能把模型所在的 Ray 工作区当作任务执行环境。

> The legacy Ascend benchmark, baseline, HF publication, and website-dispatch workflows have been
> removed. The execution scripts and historical design notes below are retained for reference only;
> new PR validation is performed by the external fixed-machine dataset service.

Wrapper benchmark repository for vllm-hust.

The goal is to keep `vllm-hust-benchmark` as the stable entrypoint for running experiments, while
the real benchmark implementations stay in `vllm-hust`. This repo resolves the sibling `vllm-hust`
and `vllm-hust-website` repositories, invokes the benchmark entrypoints from there, and keeps result
export and website publication flows in one place.

## Current Delivery Contract

Historical Frontier raw measurements and offline recomputation:
[56-run raw archive](reports/frontier-raw-recovery-20261020/README.md) (BetterScale/Native and six
additional MOD series; includes complete concurrency curves and SHA-256).

The current Project 4 midterm authority is **Project Milestone Test Outline V1.5**, sections 2.4 and
3.4.1–3.4.4. See the
[verified extract and current report](reports/vllm-hust-midterm-20261019/README.md). Prior
standalone V5.5/V5.27 contracts and their validators are retained for historical records, not as
additional current midterm acceptance requirements. Existing measurements remain immutable; reuse or
recalculate applicable records before proposing missing measurements.

Validate the historical V5.5 contract with:

```bash
PYTHONPATH=src python scripts/validate_acceptance_v5_5.py
```

For runtime comparisons, the wrapper now understands two execution targets:

- `vllm-hust` (default): the sibling workspace checkout at `../vllm-hust`
- `vllm`: the baseline checkout at `../reference-repos/vllm` (override with
  `VLLM_BASELINE_VLLM_REPO`)

Use the already configured conda environment for this workspace. Do not rely on the system Python
installation for benchmark execution or validation.

## Local Validation Dependencies

Install the benchmark test toolchain from this repository's optional `test` extra before running
local validation:

```bash
python -m pip install -e .[test]
```

This extra intentionally includes `jsonschema`, because benchmark publication and some regression
paths invoke the sibling website aggregation script, which validates leaderboard snapshots with
`jsonschema`.

## What This Repo Wraps

The actual benchmark implementations live in the sibling `vllm-hust` repository:

- `vllm bench serve`
- `vllm bench throughput`
- `vllm bench latency`
- `vllm bench sweep ...`
- `benchmarks/*.py`

This repository adds a stable orchestration layer around them:

- upstream performance test discovery from
  `vllm-hust/.buildkite/performance-benchmarks/tests/*.json`
- upstream suite delegation via
  `vllm-hust/.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`
- scenario registry and command construction
- sibling-repo path resolution
- standard leaderboard artifact export
- website snapshot aggregation via `vllm-hust-website/scripts/aggregate_results.py`

Within that boundary, the important design point is:

1. `vllm-hust-benchmark` should stay thin and should not re-implement benchmark runtime logic that
   already belongs in `vllm-hust`.
1. Benchmark result shaping and website publishing can still live here because they are cross-repo
   orchestration concerns.

## Repository Layout

- `src/vllm_hust_benchmark/cli.py`: CLI entrypoint
- `src/vllm_hust_benchmark/integration.py`: sibling repo resolution and wrapper execution helpers
- `src/vllm_hust_benchmark/models.py`: scenario data model and command rendering
- `src/vllm_hust_benchmark/registry.py`: official scenario loading and lookup
- `src/vllm_hust_benchmark/leaderboard_export.py`: website-compatible artifact and manifest exporter
- `src/vllm_hust_benchmark/data/official_scenarios.json`: initial official scenario mirror
- `leaderboard-data/official-targets.json`: versioned machine-readable official target registry
- `docs/OFFICIAL_TARGETS.zh-CN.md`: generated Chinese official target matrix
- `docs/OFFICIAL_TARGETS.md`: generated English official target matrix
- `docs/UPSTREAM_ANALYSIS.md`: analysis of upstream benchmark architecture and why this repo is
  structured this way
- `docs/LEADERBOARD_ALIGNMENT.md`: scenario taxonomy and exact mapping to website leaderboard schema
- `docs/TREND_COVERAGE_SCHEMA.md`: definitions and maintenance rules for the trend coverage schema
  extension
- `docs/LEADERBOARD_HANDOFF.md`: handoff guide for the leaderboard publication chain and official
  baseline operations
- `docs/perfgate-scenario-rollout.md`: requirement and rollout plan for multi-scenario performance
  gates
- `tests/`: focused tests for registry loading and command generation

## Measured experimental reports

- [BetterScale + solver mixed graph pool (2026-10-20)](reports/betterscale-solver-mixed-pool-20261020/README.md):
  actual HTTP serving, C8/C16/C32 ×4K/8K, MMLU-Pro200 and HLE Gold text200, payload-free repeat
  receipts and card-rental costs. Solver-enabled measurements, not predictions, native-OFF
  comparisons, task accuracy or official fixed-target leaderboard submissions.

## Quick Start

```bash
# show the active public target matrix; 3B perfgate profiles are excluded
official-targets

# inspect the complete machine-readable registry
official-targets --json

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

1. The command stays a thin wrapper around
   vllm-hust/.github/workflows/scripts/run_ascend_benchmark_ci.sh, so CI-side defaults still live in
   that script.
1. random-online can run with script defaults, but sharegpt-online requires both --dataset-path and
   --constraints-file.
1. --publish-to-hf requires --hf-repo-id, and random-online also requires --allow-random-hf-publish
   as a deliberate safeguard.
1. Use repeated --env KEY=VALUE when the local Ascend runtime still needs extra environment
   injection such as toolkit or cache overrides.

## Extending With New Scenarios

The intended extension path is:

1. Preserve upstream benchmark semantics where possible.
1. Add new scenarios as new registry entries with explicit tags, defaults, and dataset metadata.
1. Keep hardware-specific overrides isolated in scenario config rather than baking them into the
   CLI.
1. Export results in a stable way so website and regression pipelines can consume them later.

## Leaderboard And HF Integration

The intended production chain is:

1. Run `vllm-hust` benchmark entrypoints or the upstream performance suite from this repository.
1. Convert measured metrics to a standard leaderboard artifact with `export-leaderboard-artifact`.
1. Optionally run `publish-website`, or pass `--publish-website --execute` to
   `export-leaderboard-artifact`, to aggregate exports directly into the sibling `vllm-hust-website`
   checkout for local validation.
1. Upload the aggregated snapshot or raw export directory to the production data channel.
1. Website consumes the generated `leaderboard_single.json`, `leaderboard_multi.json`, and
   `leaderboard_compare.json` snapshots.

`export-leaderboard-artifact` writes two files:

- `run_leaderboard.json` (configurable): one schema-compatible leaderboard entry.
- `leaderboard_manifest.json`: manifest that points to artifact files and idempotency keys.

When `export-leaderboard-artifact` or `submit` runs inside GitHub Actions, the exporter now also
captures GitHub provenance into `metadata`, including the triggering user, commit SHA, commit URL,
repository, ref, and optional PR metadata. You can override any of those fields explicitly with
`--git-commit`, `--github-user`, `--github-commit-url`, `--github-repository`, `--github-ref`,
`--github-event-name`, `--github-pr-number`, and `--github-pr-url`.

For cross-repository CI, `sync-submission-to-hf` is the preferred publish entrypoint after a run has
already been exported into one submission directory. It downloads historical raw submissions from
the production Hugging Face dataset `intellistream/vllm-hust-benchmark-results`, merges in the new
submission, regenerates `leaderboard_single.json`, `leaderboard_multi.json`,
`leaderboard_compare.json`, and `last_updated.json`, and uploads both the refreshed snapshots and
the new raw submission in one commit.

Because the production website currently prioritizes `github -> hf -> local`, a successful HF upload
alone is not sufficient to refresh the live site. The production chain therefore has two stages:

1. `vllm-hust` benchmark CI writes the exported `submissions/<run-id>/` payload and refreshed
   `leaderboard-data/snapshots/` files directly into `vllm-hust-benchmark@main` in one
   bot-authenticated commit.
1. The legacy `push-to-hf.yml` publication path is retired. Results are now supplied by the external
   fixed-machine dataset validator and will be consumed by the website result path after its input
   schema and synchronization contract are finalized.

The older snapshot-PR and auto-merge workflows are obsolete under this model. The benchmark
repository submission directories and generated snapshots are the authoritative source for public
leaderboard data; the HF dataset is a synchronized distribution mirror for consumers that cannot
read the GitHub snapshots directly.

If you already have a raw `vllm bench` result JSON, you do not need to hand-author the full metrics
payload anymore. The wrapper can derive the main website metrics from the raw result and only
requires a separate constraints metrics JSON.

For the upstream benchmark analysis behind this design, see `docs/UPSTREAM_ANALYSIS.md`.

## Official Goal Baseline

The canonical home for the official Ascend goal-baseline runner is this repository, not
`reference-repos/*`. `reference-repos` stays read-only for upstream comparison, while
`vllm-hust-benchmark` owns cross-repo orchestration and website artifact export.

Current public baseline target:

- official `vllm v0.18.0`
- official `vllm-ascend v0.18.0`
- canonical spec set under `docs/official-baselines/*.json`
- current hardware target `Huawei 910B2`

The official baseline set is no longer a single `random-online` spec. The canonical spec files under
`docs/official-baselines/` are the source of truth for all official scenarios that need a baseline
under the pinned January 2026 `v0.18.0` runtime pair.

Retired `vllm v0.11.0` / `vllm-ascend v0.11.0` runs must not be republished to
`leaderboard-data/snapshots`, the website mirror, or the HF snapshot root. Run
`python scripts/validate_public_leaderboard_snapshots.py` before publishing curated leaderboard
data.

Files:

- `scripts/run-official-v0180-baselines.sh`
- `scripts/prepare-official-ascend-baseline-env.sh`
- `scripts/run-official-ascend-goal-baseline.sh`
- `docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json`
- `docs/official-baselines/official-ascend-constraints.stub.json`

One-command local entrypoint:

```bash
bash scripts/run-official-v0180-baselines.sh
```

This wrapper sets the local 910B2 defaults used by the public baseline run: `/data` result and HF
cache paths, `HF_ENDPOINT=https://hf-mirror.com`, the pinned `vllm-ascend-official-v0180` conda env,
local Qwen2.5-14B model path when present, single-threaded CPU libraries, and
`SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD=1` so official v0.18 uses the sampler fallback instead of
the fragile custom-op registration path.

Useful one-command variants:

```bash
# Run all missing official baselines with three candidate repeats.
bash scripts/run-official-v0180-baselines.sh --repeat-count 3

# Run one spec on one card.
bash scripts/run-official-v0180-baselines.sh --devices 0 \
	docs/official-baselines/official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json

# Rerun existing canonical specs for review without replacing canonical data.
bash scripts/run-official-v0180-baselines.sh --review-existing --devices 0 \
	docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json
```

Lower-level example:

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

- `prepare-official-ascend-baseline-env.sh` creates or repairs a dedicated conda env for the fixed
  official baseline only.
- `prepare-official-ascend-baseline-env.sh` now starts with a health check. If the existing env
  already matches the pinned official baseline, it skips the heavy uninstall/reinstall path and
  reuses the env as-is.
- If `ENV_PREFIX` is unset, the prepare script now defaults it to
  `$(conda info --base)/envs/vllm-ascend-official-v0180` instead of assuming `/root/miniconda3/...`.
- `prepare-official-ascend-baseline-env.sh` also owns the benchmark admission preflight: it
  proactively cleans residual `api_server` / `bench serve` / `EngineCore_DP0` processes and clears
  the benchmark port before a new run is allowed to start.
- The baseline runtime is pinned to `reference-repos/vllm@v0.18.0` and
  `reference-repos/vllm-ascend@v0.18.0` worktrees.
- The prepare script intentionally does not install `vllm-hust` or `vllm-ascend-hust` into the
  official env, to avoid plugin-entry-point contamination.
- The prepare script now defaults `PYTORCH_CPU_INDEX_URL` to `https://download.pytorch.org/whl/cpu`
  in addition to the Ascend mirror, so `torch==...+cpu` dependencies from `torch-npu` metadata can
  resolve.
- The torch family package pins are configurable with `OFFICIAL_TORCH_VERSION`,
  `OFFICIAL_TORCH_NPU_VERSION`, `OFFICIAL_TORCHVISION_VERSION`, and `OFFICIAL_TORCHAUDIO_VERSION`.
- For the default official version set on `Python 3.11`, the prepare script now auto-selects
  built-in archived wheel URLs for both `x86_64` and `aarch64`, based on the local machine
  architecture.
- You can bypass drifting package indexes with explicit wheel URLs: `OFFICIAL_TORCH_WHEEL_URL`,
  `OFFICIAL_TORCH_NPU_WHEEL_URL`, `OFFICIAL_TORCHVISION_WHEEL_URL`, and
  `OFFICIAL_TORCHAUDIO_WHEEL_URL`.
- If you override the default torch-family versions or use a different Python ABI, the script falls
  back to index-based resolution unless you also provide explicit wheel URLs.
- The runner executes against the pinned worktrees through `PYTHONPATH` and defaults
  `VLLM_CACHE_ROOT` to `.cache/official-ascend-goal-baseline/` in this repository.
- Before model loading, the runner now proves that imported `vllm` and `vllm_ascend` modules,
  package versions, generated build commit IDs, and discovered compiled extensions correspond to
  those pinned worktrees. It repeats the capture after the benchmark and fails closed if the runtime
  changes. The resulting `official_runtime_provenance` is stored identically in
  `run_leaderboard.json`, `leaderboard_manifest.json`, and `env-manifest.json`; publication rejects
  a missing or mismatched copy before snapshots are generated. Historical artifacts without this new
  field remain readable, but newly captured official evidence cannot omit it.
- The runner calls the same prepare script in admission-only mode immediately before benchmark
  startup, so residual-process cleanup is a hard precondition rather than a manual step.
- The runner prefers a locally cached Hugging Face snapshot for the target model when one already
  exists. You can also force a specific local model directory with
  `OFFICIAL_MODEL_PATH=/abs/model/path`.
- `run-official-ascend-goal-baseline-matrix.sh` is the batch trigger for official baseline
  establishment. It skips specs that already have canonical submissions under
  `submissions/<spec-id>/` and prints a hint instead of re-running them.
- When a spec has no canonical submission yet, `run-official-ascend-goal-baseline-matrix.sh` targets
  `REPEAT_COUNT` successful runs (default `3`), allows bounded transient repeat failures, chooses
  the run whose primary metric is closest to the median successful candidate set, and promotes only
  that run into `submissions/<spec-id>/`.
- Set `FORCE_RUN_EXISTING=1` only when you intentionally want a review-only rerun. The matrix runner
  preserves the current canonical submission and leaves the new result under `.benchmarks/` for
  manual comparison instead of auto-replacing it.
- By default the matrix runner accepts a degraded but still usable sample set when at least `2`
  successful repeats exist (`1` when `REPEAT_COUNT=1`). Override with `MIN_SUCCESSFUL_REPEATS=<n>`
  if you need a stricter bar.
- The matrix runner uses `MAX_REPEAT_ATTEMPTS` (default `REPEAT_COUNT + 1`) to absorb transient
  engine or runtime crashes without throwing away earlier successful repeats.
- The former `.github/workflows/run-official-ascend-baselines.yml` entrypoint is removed. Use the
  local matrix scripts for historical engineering analysis; publication requires a separately
  reviewed external path.

This produces:

- a raw benchmark result under `.benchmarks/official-ascend-goal-baseline/`
- a website-compatible leaderboard artifact under
  `.benchmarks/official-ascend-goal-baseline/submission/`

The exported artifact can then be aggregated into `vllm-hust-website` with `publish-website` or
`vllm-hust-website/scripts/aggregate_results.py`.

### Triggering Locally

Recommended first establishment run for all missing official specs:

```bash
cd /path/to/vllm-hust-benchmark
bash scripts/run-official-v0180-baselines.sh --repeat-count 3
```

Run only one spec file:

```bash
cd /path/to/vllm-hust-benchmark
bash scripts/run-official-v0180-baselines.sh --repeat-count 3 \
	docs/official-baselines/official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json
```

Run a review-only rerun for a spec that already has canonical data:

```bash
cd /path/to/vllm-hust-benchmark
bash scripts/run-official-v0180-baselines.sh --review-existing --repeat-count 3 \
	docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json
```

Useful local switches:

- `PREPARE_OFFICIAL_ENV=1` (default): create or repair the pinned official env before the batch.
- `FORCE_REPAIR_OFFICIAL_ENV=1`: bypass the prepare script health check and force a full
  reinstall/repair of the official env.
- `REPEAT_COUNT=3`: recommended default for establishing a missing canonical spec.
- `FORCE_RUN_EXISTING=1`: rerun even if canonical data already exists, but do not overwrite
  canonical automatically.
- `MIN_SUCCESSFUL_REPEATS=<n>`: minimum successful repeats required before canonical selection is
  allowed. Default is `2` when `REPEAT_COUNT >= 2`, otherwise `1`.
- `MAX_REPEAT_ATTEMPTS=<n>`: hard cap on total attempts per missing canonical spec, including
  retries after transient failures. Default is `REPEAT_COUNT + 1`.
- `PUBLISH_WEBSITE=1`: rebuild the checked-out local `vllm-hust-website/data` from the full
  `submissions/` tree after the batch. This is a local workspace refresh, not the live publication
  path.
- `PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu`: default CPU wheel index used by the
  prepare script for torch-family dependency resolution.
- `OFFICIAL_TORCH_NPU_WHEEL_URL=<exact-wheel-url>`: preferred escape hatch when the historical
  `torch-npu` build has disappeared from the live mirror.

Operational behavior notes:

- Each matrix run now persists a preferred single-card Ascend device in
  `.benchmarks/<matrix-run-id>/preferred-ascend-device`, so later repeats and later specs in the
  same batch reuse the same idle NPU when it remains available.
- `FORCE_RUN_EXISTING=1` is a manual-review path for an existing canonical spec. It reruns that spec
  once, suppresses canonical promotion, and collapses the repeat-selection knobs to a single attempt
  even if `REPEAT_COUNT` is larger.

Outputs:

- Per-run artifacts go to `.benchmarks/<matrix-run-id>/...`
- Promoted canonical submissions go to `submissions/<spec-id>/`
- The batch summary goes to `.benchmarks/<matrix-run-id>/summary.md` unless `MATRIX_SUMMARY_FILE` is
  overridden.

### Retired GitHub Workflow

The former `run-official-ascend-baselines.yml` workflow has been removed. Historical
`gh workflow run` examples are no longer valid. Run the local matrix scripts above for engineering
analysis; automated publication must use a separately reviewed external service and contract.

## Ascend msprof Profiling

For a same-spec run that needs Ascend `msprof` collection, use the profiling wrapper. With no
argument, it uses the current default Ascend same-spec baseline, so confirm the target model and
hardware before starting a long profiling run:

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

- `CONFIG_FILE`: load another shell config file instead of
  `scripts/run-current-ascend-same-spec-msprof.env`.
- `MSPROF_EXECUTABLE`: use a specific `msprof` binary path.
- `PROFILE_RUN_ID` or `PROFILE_RUN_DIR`: choose the output location.
- `PROFILE_RUN_OVERWRITE=1`: allow reusing an existing output location.
- `MSPROF_FLAGS`: replace the default
  `--ascendcl=on --runtime-api=on --task-time=l1 --hccl=on --type=text`. This is split on
  whitespace.
- `MSPROF_ARGS`: in a sourced config file, define a bash array for arguments that need quoting, for
  example `MSPROF_ARGS=(--foo "a b")`.

The raw profile directory can be analyzed later by a separate analyzer. For example, with TraceLoom:

```bash
traceloom analysis .benchmarks/current-ascend-msprof/<run-id>/msprof_raw \
  --out-dir .benchmarks/current-ascend-msprof/<run-id>/analysis
```

## Batch Same-Spec Matrices

When you want to benchmark one machine group under multiple runtime settings and keep every setting
visible on the website leaderboard, use the matrix wrapper instead of invoking the single-spec
runner repeatedly by hand.

Files:

- `scripts/run-current-ascend-same-spec.sh`: execute one resolved same-spec benchmark and export one
  submission
- `scripts/run-current-ascend-same-spec-msprof.sh`: wrap one same-spec benchmark with `msprof`
- `scripts/run-current-ascend-same-spec-msprof.env`: default `msprof` wrapper configuration
- `scripts/run-current-ascend-same-spec-matrix.sh`: iterate a directory or list of spec files,
  assign one result directory per spec, and optionally regenerate website data after the whole batch

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

- The matrix wrapper does not replace the single-spec runner; it only orchestrates repeated calls to
  it.
- Every spec gets an isolated `RESULT_DIR` under `.benchmarks/<matrix-run-id>/`, so raw results,
  resolved same-spec payloads, and exported submissions do not overwrite each other.
- Set `PUBLISH_WEBSITE=1` only after you are ready to regenerate `vllm-hust-website/data` from the
  full batch output.

For formal current-main multi-card campaigns, follow `docs/issue-136-ascend-evidence-execution.md`.
Its strict launch mode requires frozen source, image, model, runtime, and topology identity; uses a
fresh service for every repetition; and writes a machine-readable campaign summary without treating
readiness evidence as a performance result.

## Contributing

The benchmark repo uses the same pre-commit toolchain as `vllm-hust-website` so commits land on
`main` with consistent formatting. Once-per-clone setup:

```bash
./quickstart.sh        # installs dev deps + editable package + git hooks
```

After that, every `git commit` runs the local fallback hook (`hooks/pre-commit`) automatically:
trailing whitespace, conflict markers, large files, ruff format + check, API keys, debug statements.
When `pre-commit` framework + Python 3.11 are both available locally, the same hook transparently
switches to running the full `.pre-commit-config.yaml` hooks set.

For CI-parity local validation before opening a PR:

```bash
./scripts/validate-local.sh
```

This mirrors what `.github/workflows/ci.yml` runs: `pre-commit run --all-files` followed by
`python -m pytest -q tests`. Pass `--skip-pre-commit` or `--skip-tests` to skip a phase.

The `no-commit-to-branch` hook blocks direct commits to `main`; create a feature branch first. The
pre-push hook additionally re-checks whitespace / conflict markers on main-bound pushes only.

<!-- workload-catalog:start -->

## 按 workload 查找原生 benchmark 与数据资产

本仓库统一提供原始数据获取、读取适配、准备和必要的原生运行入口，方便实验直接复用；不包含研究侧的观测、特征挖掘与分析。**本表为 2026-10-27 的分支交付快照，不代表已合入 main。**

当前范围为 **40 类 workload、113 项交付来源及 5 项仅分类来源**，其中包含 8 份企业私有请求资产。**61/113 项已交付，52 项待完成**；首批 PR
[#432](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/432) 固定 60 项，第二批已增加
ScienceWorld。不要将跨类别重复引用的数据集重复计数。逐项状态以[唯一交付清单](docs/workload-data/dataset-validation-status.json)的
`delivery_inventory` 和[打勾清单](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)为准。

### 怎么使用、怎么理解状态

1. 从下表选择 workload，再点击来源 ID，查看官方来源、仓库入口和证据。
1. 按[安装说明](docs/workload-data/INSTALLATION.zh-CN.md)准备数据组件，按对应脚本帮助和[来源使用文档](workload-preparation/README.md)准备独立原生环境。**纯数据
   wheel 不包含仓库运行脚本**；执行时需检出包含该入口的固定代码版本。
1. 原始数据和模型下载到仓库外，不随 Git 提交。保留原题、完整工具/环境流程及作者评分；不能只读取一条数据就视为 benchmark 跑通，也不能因失败换题或缩减任务。
1. 小规模套件尽量全量验证；大规模套件按固定规则抽样。**[x] 已交付只覆盖证据中明确的原生配置/子集，或明确的数据资产契约，不等于全量模型评测、高分或性能结论。**
   模型、依赖、抽样和未通过项见对应交付回执；某个子集跑通不代表同类所有能力都验证过。
1. **[ ] 未闭环/阻塞/待接入均不可视为可完整运行**。下表列出的脚本可能只完成部分准备，不能从文件存在推断通过；当前缺口见打勾清单。
1. MMLU-Pro、HLE-Verified、SWE-bench-Pro、FrontierScience、TerminalBench 2.1
   **本批只补分类、复用已有独立实现**，不重复接入，也不据此判定其已有实现不可用。参见[已有能力盘点](docs/workload-data/WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)。

查目录无需下载数据或启动模型；下列命令只查询分类与入口，**不是 benchmark 执行命令**：

```bash
python -m vllm_hust_benchmark.workload_data classes --scheme delivery-plan
python -m vllm_hust_benchmark.workload_data class-plan W11 --scheme delivery-plan
python -m vllm_hust_benchmark.workload_data source-plan SCIENCEWORLD
python -m vllm_hust_benchmark.workload_data delivery-assets
```

### 40 类 workload 与来源

分类描述原生任务的语义；数据集可关联多类，但须选择对应原始子任务。长输入、长输出、共享前缀、负载强度等是请求/运行维度，不额外冒充语义任务类别。 W17 的 Spider 仅关联只读 SQL
查询，不代表数据库更新；W37/W38 的原数据或对话语料也不自动证明流式／语音生成原生执行能力。

| 类别 | Workload                   | 对应数据集／原生任务来源                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W01  | 通用问答与开放生成         | [WILDCHAT](#workload-source-wildchat)、[LMSYS_CHAT](#workload-source-lmsys-chat)、[DOLLY](#workload-source-dolly)、[NATURALQUESTIONS](#workload-source-naturalquestions)、[DUREADER](#workload-source-dureader)、[NATURAL_INSTRUCTIONS](#workload-source-natural-instructions)、[OASST1](#workload-source-oasst1)、[TULU3](#workload-source-tulu3)、[ULTRACHAT](#workload-source-ultrachat)、[LONGWRITER](#workload-source-longwriter)            |
| W02  | 对话续接                   | [WILDCHAT](#workload-source-wildchat)、[LMSYS_CHAT](#workload-source-lmsys-chat)、[OASST1](#workload-source-oasst1)、[ULTRACHAT](#workload-source-ultrachat)、[SPOKENWOZ](#workload-source-spokenwoz)、[DAILYTALK](#workload-source-dailytalk)                                                                                                                                                                                                    |
| W03  | 给定检索材料的问答与跟进   | [QRECC](#workload-source-qrecc)、[QASPER](#workload-source-qasper)、[NARRATIVEQA](#workload-source-narrativeqa)、[DUREADER](#workload-source-dureader)、[NATURALQUESTIONS](#workload-source-naturalquestions)、[LONGBENCH](#workload-source-longbench)                                                                                                                                                                                            |
| W04  | 文档分析与理解             | [LONGBENCH](#workload-source-longbench)、[LONGBENCH_V2](#workload-source-longbench-v2)、[INFINITEBENCH](#workload-source-infinitebench)、[QUALITY](#workload-source-quality)、[QASPER](#workload-source-qasper)、[NARRATIVEQA](#workload-source-narrativeqa)、[LONGALPACA](#workload-source-longalpaca)                                                                                                                                           |
| W05  | 独立知识与推理求解         | [GSM8K](#workload-source-gsm8k)、[MATH500](#workload-source-math500)、[BBH](#workload-source-bbh)、[GPQA](#workload-source-gpqa)、[MMLU_PRO](#workload-source-mmlu-pro)、[AI2_ARC](#workload-source-ai2-arc)、[OPENMATH2](#workload-source-openmath2)、[OPENTHOUGHTS3](#workload-source-openthoughts3)、[LABBENCH](#workload-source-labbench)、[HLE_VERIFIED](#workload-source-hle-verified)、[FRONTIERSCIENCE](#workload-source-frontierscience) |
| W06  | 结构化生成与格式转换       | [JSONSCHEMABENCH](#workload-source-jsonschemabench)、[STRUCTEVAL](#workload-source-structeval)、[SPIDER](#workload-source-spider)、[SCHEMASTORE](#workload-source-schemastore)、[NATURAL_INSTRUCTIONS](#workload-source-natural-instructions)                                                                                                                                                                                                     |
| W07  | 摘要                       | [XSUM](#workload-source-xsum)、[MULTINEWS](#workload-source-multinews)、[BILLSUM](#workload-source-billsum)、[XLSUM](#workload-source-xlsum)、[DOLLY](#workload-source-dolly)、[NATURAL_INSTRUCTIONS](#workload-source-natural-instructions)                                                                                                                                                                                                      |
| W08  | 翻译                       | [FLORES200](#workload-source-flores200)、[NATURAL_INSTRUCTIONS](#workload-source-natural-instructions)                                                                                                                                                                                                                                                                                                                                            |
| W09  | 信息抽取与分类             | [DOLLY](#workload-source-dolly)、[NATURAL_INSTRUCTIONS](#workload-source-natural-instructions)、[TULU3](#workload-source-tulu3)、[PAWS](#workload-source-paws)                                                                                                                                                                                                                                                                                    |
| W10  | 改写与编辑                 | [NATURAL_INSTRUCTIONS](#workload-source-natural-instructions)、[TULU3](#workload-source-tulu3)、[DOLLY](#workload-source-dolly)                                                                                                                                                                                                                                                                                                                   |
| W11  | 单题代码生成与测试         | [HUMANEVAL](#workload-source-humaneval)、[MBPP](#workload-source-mbpp)、[BIGCODEBENCH](#workload-source-bigcodebench)、[DS1000](#workload-source-ds1000)、[LIVECODEBENCH](#workload-source-livecodebench)、[OPENCODE](#workload-source-opencode)、[OPENTHOUGHTS3](#workload-source-openthoughts3)                                                                                                                                                 |
| W12  | 代码评判                   | [CODEJUDGE_EVAL](#workload-source-codejudge-eval)                                                                                                                                                                                                                                                                                                                                                                                                 |
| W13  | 仓库上下文代码补全         | [REPOBENCH](#workload-source-repobench)、[CROSSCODEEVAL](#workload-source-crosscodeeval)                                                                                                                                                                                                                                                                                                                                                          |
| W14  | 交互式代码编辑助手         | [TRACELAB](#workload-source-tracelab)                                                                                                                                                                                                                                                                                                                                                                                                             |
| W15  | 长程仓库软件工程任务       | [SWE_BENCH](#workload-source-swe-bench)、[SWE_VERIFIED](#workload-source-swe-verified)、[SWESMITH](#workload-source-swesmith)、[SWE_BENCH_PRO](#workload-source-swe-bench-pro)                                                                                                                                                                                                                                                                    |
| W16  | 状态化环境操作             | [AGENTBENCH](#workload-source-agentbench)、[SCIENCEWORLD](#workload-source-scienceworld)                                                                                                                                                                                                                                                                                                                                                          |
| W17  | 数据库查询与更新执行       | [AGENTBENCH](#workload-source-agentbench)、[SPIDER](#workload-source-spider)                                                                                                                                                                                                                                                                                                                                                                      |
| W18  | 知识图谱查询与遍历         | [AGENTBENCH](#workload-source-agentbench)                                                                                                                                                                                                                                                                                                                                                                                                         |
| W19  | 网页导航与交互             | [AGENTBENCH](#workload-source-agentbench)、[MIND2WEB](#workload-source-mind2web)、[WEBARENA](#workload-source-webarena)、[VISUALWEBARENA](#workload-source-visualwebarena)、[WORKARENA](#workload-source-workarena)                                                                                                                                                                                                                               |
| W20  | 操作系统与终端任务         | [AGENTBENCH](#workload-source-agentbench)、[TERMINAL_BENCH_2_1](#workload-source-terminal-bench-2-1)                                                                                                                                                                                                                                                                                                                                              |
| W21  | 桌面GUI操作                | [OSWORLD](#workload-source-osworld)                                                                                                                                                                                                                                                                                                                                                                                                               |
| W22  | 应用API多步工作流          | [APPWORLD](#workload-source-appworld)、[BFCL](#workload-source-bfcl)、[TOOLBENCH](#workload-source-toolbench)、[DEEPPLANNING](#workload-source-deepplanning)                                                                                                                                                                                                                                                                                      |
| W23  | 规则约束业务办理与排障     | [TAUBENCH](#workload-source-taubench)                                                                                                                                                                                                                                                                                                                                                                                                             |
| W24  | 工具选择与参数生成         | [BFCL](#workload-source-bfcl)、[XLAM_TOOL](#workload-source-xlam-tool)、[TOOLACE](#workload-source-toolace)                                                                                                                                                                                                                                                                                                                                       |
| W25  | 主动证据检索与综合         | [GAIA](#workload-source-gaia)、[BFCL](#workload-source-bfcl)                                                                                                                                                                                                                                                                                                                                                                                      |
| W26  | 约束规划与重规划           | [DEEPPLANNING](#workload-source-deepplanning)                                                                                                                                                                                                                                                                                                                                                                                                     |
| W27  | 科学实验规划               | [DISCOVERYBENCH](#workload-source-discoverybench)、[SCIENCEWORLD](#workload-source-scienceworld)、[LABBENCH](#workload-source-labbench)、[MLAGENTBENCH](#workload-source-mlagentbench)、[SCIENCEAGENTBENCH](#workload-source-scienceagentbench)                                                                                                                                                                                                   |
| W28  | 仿真、分析与验证流水线     | [SCIENCEWORLD](#workload-source-scienceworld)、[DISCOVERYBENCH](#workload-source-discoverybench)、[BIXBENCH](#workload-source-bixbench)、[NATUREBENCH](#workload-source-naturebench)、[SCIENCEAGENTBENCH](#workload-source-scienceagentbench)                                                                                                                                                                                                     |
| W29  | 科学代码工作流             | [SCIENCEAGENTBENCH](#workload-source-scienceagentbench)、[NATUREBENCH](#workload-source-naturebench)、[MLEBENCH](#workload-source-mlebench)、[MLAGENTBENCH](#workload-source-mlagentbench)、[BIXBENCH](#workload-source-bixbench)、[PAPERBENCH](#workload-source-paperbench)、[DS1000](#workload-source-ds1000)                                                                                                                                   |
| W30  | 演化语料检索               | [DRAGON_PUBLIC](#workload-source-dragon-public)、[TEMPORALWIKI](#workload-source-temporalwiki)                                                                                                                                                                                                                                                                                                                                                    |
| W31  | 跨会话对话记忆             | [LONGMEMEVAL](#workload-source-longmemeval)、[LOCOMO](#workload-source-locomo)                                                                                                                                                                                                                                                                                                                                                                    |
| W32  | 任务经历与环境轨迹记忆     | [LONGMEMEVAL_V2](#workload-source-longmemeval-v2)                                                                                                                                                                                                                                                                                                                                                                                                 |
| W33  | 一般图像理解               | [VISIONARENA](#workload-source-visionarena)、[MMMU](#workload-source-mmmu)                                                                                                                                                                                                                                                                                                                                                                        |
| W34  | 视觉文档与图表理解         | [CHARTQA](#workload-source-chartqa)、[MMMU](#workload-source-mmmu)、[DOCVQA](#workload-source-docvqa)                                                                                                                                                                                                                                                                                                                                             |
| W35  | 视频理解                   | [VIDEOMME](#workload-source-videomme)、[VIDEOMME_V2](#workload-source-videomme-v2)、[MVBENCH](#workload-source-mvbench)、[LONGVIDEOBENCH](#workload-source-longvideobench)、[STREAMINGBENCH](#workload-source-streamingbench)、[OVOBENCH](#workload-source-ovobench)                                                                                                                                                                              |
| W36  | 离线音频识别与理解         | [LIBRISPEECH](#workload-source-librispeech)、[FLEURS](#workload-source-fleurs)、[WAVCAPS](#workload-source-wavcaps)                                                                                                                                                                                                                                                                                                                               |
| W37  | 音频流交互                 | [FULL_DUPLEX_BENCH](#workload-source-full-duplex-bench)、[SPOKENWOZ](#workload-source-spokenwoz)、[DAILYTALK](#workload-source-dailytalk)                                                                                                                                                                                                                                                                                                         |
| W38  | 文本到语音生成             | [LIBRITTS](#workload-source-libritts)、[LJSPEECH](#workload-source-ljspeech)                                                                                                                                                                                                                                                                                                                                                                      |
| W39  | 向量表示与相似度／检索任务 | [BEIR_SCIFACT](#workload-source-beir-scifact)、[BEIR_NFCORPUS](#workload-source-beir-nfcorpus)、[MSMARCO_PASSAGE](#workload-source-msmarco-passage)、[STSB](#workload-source-stsb)、[PAWS](#workload-source-paws)                                                                                                                                                                                                                                 |
| W40  | 相关性重排                 | [BEIR_SCIFACT](#workload-source-beir-scifact)、[BEIR_NFCORPUS](#workload-source-beir-nfcorpus)、[MSMARCO_PASSAGE](#workload-source-msmarco-passage)                                                                                                                                                                                                                                                                                               |

### 服务轨迹、受控探针与企业请求资产

这些资产与上述语义任务分开登记，不虚构第 41 类，也不靠文件名推断实际 KV 命中。服务元数据缺少
prompt/媒体时不能伪造原始请求；有官方回放工具的按原工具契约使用，无原始回放能力的按数据资产使用。

| 资产类型             | 来源                                                                      | 使用边界                                                                       |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 服务轨迹／会话元数据 | [AZURE_LLM](#workload-source-azure-llm)                                   | 服务元数据和时序；非原始请求正文。                                             |
| 服务轨迹／会话元数据 | [AZURE_LMM_2025](#workload-source-azure-lmm-2025)                         | 多模态服务元数据；不补造缺失图像、视频或请求。                                 |
| 服务轨迹／会话元数据 | [BURSTGPT](#workload-source-burstgpt)                                     | 原流量轨迹；官方模拟器使用的外部 prompt 不等于原用户请求。                     |
| 服务轨迹／会话元数据 | [MOONCAKE](#workload-source-mooncake)                                     | 原服务／存储轨迹及官方回放契约；不生成 prompt 冒充原始请求。                   |
| 服务轨迹／会话元数据 | [COPILOT_AGENT_2026](#workload-source-copilot-agent-2026)                 | 原编码 Agent 会话元数据；保留 session/turn/tool 关系，不含原始代码与正文。     |
| 多 Agent 协作关系    | [MULTIAGENTBENCH](#workload-source-multiagentbench)                       | 按原场景保留多 Agent 协作／竞争关系、环境和评分。                              |
| 官方受控长上下文探针 | [RULER](#workload-source-ruler)                                           | 官方生成的受控探针，不冒称自然用户请求。                                       |
| 企业私有原始请求     | [PAIO-CODE-EVAL-1000](#workload-source-paio-code-eval-1000)               | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-JSON-500](#workload-source-paio-json-500)                           | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-CHAT-1000](#workload-source-paio-chat-1000)                         | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-LONG-PREFILL-5000](#workload-source-paio-long-prefill-5000)         | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-LONGTEXT-4000](#workload-source-paio-longtext-4000)                 | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-PREFIX-SHARED-5000](#workload-source-paio-prefix-shared-5000)       | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-REUSE-CONV-5000](#workload-source-paio-reuse-conv-5000)             | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |
| 企业私有原始请求     | [PAIO-SEMANTIC-SIMILAR-5000](#workload-source-paio-semantic-similar-5000) | 仅外部授权数据；保留原始请求与关系，不公开正文，不从文件名推断任务或实际复用。 |

### 逐来源：官方出处、仓库入口与验证范围

“准备／执行入口”仅列已登记的真实代码路径；读取适配器不冒充执行入口。没有原生入口索引的来源明确标记待补。原始训练语料、schema 资源、轨迹资产并不天然是完整
benchmark；缺少原生评测流程的须继续补齐，不包装成作者未提供的官方评分。授权、gated 访问及商用限制遵从各发布方条款；公开链接不等于可再分发或可商用。

| 来源 ID                                                                           | 官方来源／数据所有方                                                                                                                  | 准备／原生执行入口                                                                                                                                                                                                                                                         | 本批状态与证据／缺口                                                                                                  |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| <a id="workload-source-wildchat"></a>WILDCHAT                                     | [官方来源](https://huggingface.co/datasets/allenai/WildChat-1M)                                                                       | 原生：[run_historical_dialogue_original.py](scripts/run_historical_dialogue_original.py)                                                                                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/wildchat-original-continuation-20261026.json)              |
| <a id="workload-source-lmsys-chat"></a>LMSYS_CHAT                                 | [官方来源](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)                                                                       | 原生：[run_historical_dialogue_original.py](scripts/run_historical_dialogue_original.py)                                                                                                                                                                                   | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-dolly"></a>DOLLY                                           | [官方来源](https://huggingface.co/datasets/databricks/databricks-dolly-15k)                                                           | 原生：[run_dolly_original_categories.py](scripts/run_dolly_original_categories.py)                                                                                                                                                                                         | [x] 已交付；[交付范围与环境](docs/workload-data/validation/dolly-request-corpus-delivery-20261024.json)               |
| <a id="workload-source-naturalquestions"></a>NATURALQUESTIONS                     | [官方来源](https://github.com/google-research-datasets/natural-questions)                                                             | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)<br>原生：[run_nq_open_original.py](scripts/run_nq_open_original.py)                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/nq-open-native-delivery-20261026.json)                     |
| <a id="workload-source-dureader"></a>DUREADER                                     | [官方来源](https://github.com/baidu/DuReader)                                                                                         | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)<br>原生：[run_dureader_original.py](scripts/run_dureader_original.py)<br>隔离评分：[score_dureader_original.py](scripts/score_dureader_original.py)                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/dureader-native-delivery-20261026.json)                    |
| <a id="workload-source-natural-instructions"></a>NATURAL_INSTRUCTIONS             | [官方来源](https://github.com/allenai/natural-instructions)                                                                           | 原生：[run_natural_instructions_original.py](scripts/run_natural_instructions_original.py)<br>原生：[run_ni_editing_extraction_original.py](scripts/run_ni_editing_extraction_original.py)                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/ni-native-delivery-20261024.json)                          |
| <a id="workload-source-oasst1"></a>OASST1                                         | [官方来源](https://huggingface.co/datasets/OpenAssistant/oasst1)                                                                      | 原生：[run_oasst_original_continuation.py](scripts/run_oasst_original_continuation.py)                                                                                                                                                                                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/oasst-corpus-delivery-20261024.json)                       |
| <a id="workload-source-tulu3"></a>TULU3                                           | [官方来源](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)                                                                | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)<br>原生：[run_historical_dialogue_original.py](scripts/run_historical_dialogue_original.py)                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/tulu3-native-delivery-20261026.json)                       |
| <a id="workload-source-ultrachat"></a>ULTRACHAT                                   | [官方来源](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                                                              | 原生：[run_historical_dialogue_original.py](scripts/run_historical_dialogue_original.py)                                                                                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/ultrachat-original-continuation-20261026.json)             |
| <a id="workload-source-longwriter"></a>LONGWRITER                                 | [官方来源](https://huggingface.co/datasets/zai-org/LongWriter-6k)                                                                     | 原生：[run_longwriter_original_samples.py](scripts/run_longwriter_original_samples.py)                                                                                                                                                                                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/longwriter-native-delivery-20261024.json)                  |
| <a id="workload-source-spokenwoz"></a>SPOKENWOZ                                   | [官方来源](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/spokenwoz)                                                        | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-dailytalk"></a>DAILYTALK                                   | [官方来源](https://github.com/keonlee9420/DailyTalk)                                                                                  | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-qrecc"></a>QRECC                                           | [官方来源](https://github.com/apple-aiml-research/ml-qrecc)                                                                           | 原生：[score_qrecc_original_qa.py](scripts/score_qrecc_original_qa.py)                                                                                                                                                                                                     | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-qasper"></a>QASPER                                         | [官方来源](https://huggingface.co/datasets/allenai/qasper)                                                                            | 准备：[prepare_qasper_original_assets.py](scripts/prepare_qasper_original_assets.py)<br>原生：[run_qasper_original.py](scripts/run_qasper_original.py)                                                                                                                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/qasper-native-delivery-20261026.json)                      |
| <a id="workload-source-narrativeqa"></a>NARRATIVEQA                               | [官方来源](https://github.com/google-deepmind/narrativeqa)                                                                            | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)<br>原生：[run_narrativeqa_original_summary.py](scripts/run_narrativeqa_original_summary.py)                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/narrativeqa-native-delivery-20261027.json)                 |
| <a id="workload-source-longbench"></a>LONGBENCH                                   | [官方来源](https://huggingface.co/datasets/THUDM/LongBench)                                                                           | 原生：[run_longbench_v1_qasper_original.py](scripts/run_longbench_v1_qasper_original.py)                                                                                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/longbench-qasper-native-delivery-20261024.json)            |
| <a id="workload-source-longbench-v2"></a>LONGBENCH_V2                             | [官方来源](https://huggingface.co/datasets/THUDM/LongBench-v2)                                                                        | 原生：[run_longbench_v2_original_samples.py](scripts/run_longbench_v2_original_samples.py)                                                                                                                                                                                 | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-infinitebench"></a>INFINITEBENCH                           | [官方来源](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)                                                            | 原生：[run_infinitebench_original.py](scripts/run_infinitebench_original.py)                                                                                                                                                                                               | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-quality"></a>QUALITY                                       | [官方来源](https://github.com/nyu-mll/quality)                                                                                        | 原生：[run_quality_original_mc.py](scripts/run_quality_original_mc.py)                                                                                                                                                                                                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/quality-native-delivery-20261024.json)                     |
| <a id="workload-source-longalpaca"></a>LONGALPACA                                 | [官方来源](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)                                                                     | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)<br>原生：[run_longalpaca_original.py](scripts/run_longalpaca_original.py)                                                                                                             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/longalpaca-native-delivery-20261026.json)                  |
| <a id="workload-source-gsm8k"></a>GSM8K                                           | [官方来源](https://huggingface.co/datasets/openai/gsm8k)                                                                              | 原生：[run_gsm8k_original_samples.py](scripts/run_gsm8k_original_samples.py)                                                                                                                                                                                               | [x] 已交付；[交付范围与环境](docs/workload-data/validation/gsm8k-native-delivery-20261024.json)                       |
| <a id="workload-source-math500"></a>MATH500                                       | [官方来源](https://github.com/openai/prm800k)                                                                                         | 原生：[run_math500_native.py](scripts/run_math500_native.py)                                                                                                                                                                                                               | [x] 已交付；[交付范围与环境](docs/workload-data/validation/math500-native-delivery-20261024.json)                     |
| <a id="workload-source-bbh"></a>BBH                                               | [官方来源](https://github.com/suzgunmirac/BIG-Bench-Hard)                                                                             | 原生：[run_bbh_original_tasks.py](scripts/run_bbh_original_tasks.py)                                                                                                                                                                                                       | [x] 已交付；[交付范围与环境](docs/workload-data/validation/bbh-native-delivery-20261024.json)                         |
| <a id="workload-source-gpqa"></a>GPQA                                             | [官方来源](https://github.com/idavidrein/gpqa)                                                                                        | 原生：[run_gpqa_original.py](scripts/run_gpqa_original.py)                                                                                                                                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/gpqa-native-delivery-20261025.json)                        |
| <a id="workload-source-mmlu-pro"></a>MMLU_PRO                                     | [官方来源](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)                                                                        | 复用已有实现；[独立入口盘点](docs/workload-data/WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)                                                                                                                                                                                  | — 本批仅分类；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                               |
| <a id="workload-source-ai2-arc"></a>AI2_ARC                                       | [官方来源](https://huggingface.co/datasets/allenai/ai2_arc)                                                                           | 原生：[run_arc_original_samples.py](scripts/run_arc_original_samples.py)                                                                                                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/arc-native-delivery-20261024.json)                         |
| <a id="workload-source-openmath2"></a>OPENMATH2                                   | [官方来源](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)                                                                 | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-openthoughts3"></a>OPENTHOUGHTS3                           | [官方来源](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)                                                          | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)<br>原生：[run_historical_dialogue_original.py](scripts/run_historical_dialogue_original.py)                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/openthoughts3-native-delivery-20261027.json)               |
| <a id="workload-source-labbench"></a>LABBENCH                                     | [官方来源](https://github.com/Future-House/LAB-Bench)                                                                                 | 准备：[prepare_labbench_original_assets.py](scripts/prepare_labbench_original_assets.py)<br>原生：[run_labbench_original.py](scripts/run_labbench_original.py)                                                                                                             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/labbench-native-delivery-20261027.json)                    |
| <a id="workload-source-hle-verified"></a>HLE_VERIFIED                             | [官方来源](https://huggingface.co/datasets/skylenage-ai/HLE-Verified)                                                                 | 复用已有实现；[独立入口盘点](docs/workload-data/WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)                                                                                                                                                                                  | — 本批仅分类；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                               |
| <a id="workload-source-frontierscience"></a>FRONTIERSCIENCE                       | [官方来源](https://huggingface.co/datasets/openai/frontierscience)                                                                    | 复用已有实现；[独立入口盘点](docs/workload-data/WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)                                                                                                                                                                                  | — 本批仅分类；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                               |
| <a id="workload-source-jsonschemabench"></a>JSONSCHEMABENCH                       | [官方来源](https://github.com/guidance-ai/jsonschemabench)                                                                            | 准备：[prepare_jsonschemabench_original_assets.py](scripts/prepare_jsonschemabench_original_assets.py)<br>原生：[run_jsonschemabench_official.py](scripts/run_jsonschemabench_official.py)                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/jsonschemabench-official-workflow-readiness-20261024.json) |
| <a id="workload-source-structeval"></a>STRUCTEVAL                                 | [官方来源](https://huggingface.co/datasets/TIGER-Lab/StructEval)                                                                      | 原生：[run_structeval_original_json.py](scripts/run_structeval_original_json.py)                                                                                                                                                                                           | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-spider"></a>SPIDER                                         | [官方来源](https://github.com/taoyds/spider)                                                                                          | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-schemastore"></a>SCHEMASTORE                               | [官方来源](https://github.com/SchemaStore/schemastore)                                                                                | 准备：[prepare_schemastore_original_assets.py](scripts/prepare_schemastore_original_assets.py)<br>原生：[run_schemastore_original_validation.py](scripts/run_schemastore_original_validation.py)                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/schemastore-native-delivery-20261027.json)                 |
| <a id="workload-source-xsum"></a>XSUM                                             | [官方来源](https://huggingface.co/datasets/EdinburghNLP/xsum)                                                                         | 准备：[prepare_xsum_original_assets.py](scripts/prepare_xsum_original_assets.py)<br>原生：[run_xsum_original.py](scripts/run_xsum_original.py)                                                                                                                             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/xsum-native-delivery-20261025.json)                        |
| <a id="workload-source-multinews"></a>MULTINEWS                                   | [官方来源](https://huggingface.co/datasets/alexfabbri/multi_news)                                                                     | 准备：[prepare_multinews_original_assets.py](scripts/prepare_multinews_original_assets.py)<br>原生：[run_multinews_original.py](scripts/run_multinews_original.py)                                                                                                         | [x] 已交付；[交付范围与环境](docs/workload-data/validation/multinews-native-delivery-20261025.json)                   |
| <a id="workload-source-billsum"></a>BILLSUM                                       | [官方来源](https://huggingface.co/datasets/FiscalNote/billsum)                                                                        | 准备：[prepare_billsum_original_assets.py](scripts/prepare_billsum_original_assets.py)<br>原生：[run_billsum_original.py](scripts/run_billsum_original.py)                                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/billsum-native-delivery-20261025.json)                     |
| <a id="workload-source-xlsum"></a>XLSUM                                           | [官方来源](https://huggingface.co/datasets/csebuetnlp/xlsum)                                                                          | 原生：[run_xlsum_original.py](scripts/run_xlsum_original.py)                                                                                                                                                                                                               | [x] 已交付；[交付范围与环境](docs/workload-data/validation/xlsum-native-delivery-20261025.json)                       |
| <a id="workload-source-flores200"></a>FLORES200                                   | [官方来源](https://github.com/facebookresearch/flores/tree/a6c830c6e1051fb4ac1a44b32358f00463f332bd/flores200)                        | 准备：[prepare_flores200_original_assets.py](scripts/prepare_flores200_original_assets.py)<br>原生：[run_flores200_original.py](scripts/run_flores200_original.py)<br>原生：[score_flores200_original_results.py](scripts/score_flores200_original_results.py)             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/flores200-native-delivery-20261026.json)                   |
| <a id="workload-source-paws"></a>PAWS                                             | [官方来源](https://huggingface.co/datasets/google-research-datasets/paws)                                                             | 准备：[prepare_paws_original_assets.py](scripts/prepare_paws_original_assets.py)<br>原生：[run_paws_original.py](scripts/run_paws_original.py)                                                                                                                             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/paws-native-delivery-20261027.json)                        |
| <a id="workload-source-humaneval"></a>HUMANEVAL                                   | [官方来源](https://github.com/openai/human-eval)                                                                                      | 生成：[run_humaneval_original_generation.py](scripts/run_humaneval_original_generation.py)<br>隔离评分：[run_humaneval_isolated_scoring.py](scripts/run_humaneval_isolated_scoring.py)                                                                                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/humaneval-native-delivery-20261025.json)                   |
| <a id="workload-source-mbpp"></a>MBPP                                             | [官方来源](https://huggingface.co/datasets/google-research-datasets/mbpp)                                                             | 准备：[prepare_mbpp_original_assets.py](scripts/prepare_mbpp_original_assets.py)<br>生成：[run_mbpp_original_generation.py](scripts/run_mbpp_original_generation.py)<br>隔离评分：[run_mbpp_isolated_scoring.py](scripts/run_mbpp_isolated_scoring.py)                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/mbpp-original-native-sample-20261026.json)                 |
| <a id="workload-source-bigcodebench"></a>BIGCODEBENCH                             | [官方来源](https://huggingface.co/datasets/bigcode/bigcodebench)                                                                      | 生成：[run_bigcodebench_original_generation.py](scripts/run_bigcodebench_original_generation.py)<br>隔离评分：[run_bigcodebench_container_scoring.py](scripts/run_bigcodebench_container_scoring.py)                                                                       | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-ds1000"></a>DS1000                                         | [官方来源](https://huggingface.co/datasets/xlangai/DS-1000)                                                                           | 生成：[run_ds1000_original_generation.py](scripts/run_ds1000_original_generation.py)<br>隔离评分：[run_ds1000_isolated_scoring.py](scripts/run_ds1000_isolated_scoring.py)                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/ds1000-native-delivery-20261025.json)                      |
| <a id="workload-source-livecodebench"></a>LIVECODEBENCH                           | [官方来源](https://github.com/LiveCodeBench/LiveCodeBench)                                                                            | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-opencode"></a>OPENCODE                                     | [官方来源](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)                                                                   | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-codejudge-eval"></a>CODEJUDGE_EVAL                         | [官方来源](https://aclanthology.org/2025.coling-main.7/)                                                                              | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-repobench"></a>REPOBENCH                                   | [官方来源](https://github.com/Leolty/repobench)                                                                                       | 准备：[prepare_reused_research_assets.py](scripts/prepare_reused_research_assets.py)                                                                                                                                                                                       | [x] 已交付；[交付范围与环境](docs/workload-data/validation/repobench-native-delivery-20261027.json)                   |
| <a id="workload-source-crosscodeeval"></a>CROSSCODEEVAL                           | [官方来源](https://github.com/amazon-science/cceval)                                                                                  | 准备：[prepare_crosscodeeval_original_assets.py](scripts/prepare_crosscodeeval_original_assets.py)<br>原生：[run_crosscodeeval_original.py](scripts/run_crosscodeeval_original.py)<br>隔离评分：[score_crosscodeeval_original.py](scripts/score_crosscodeeval_original.py) | [x] 已交付；[交付范围与环境](docs/workload-data/validation/crosscodeeval-original-acquisition-20261026.json)          |
| <a id="workload-source-tracelab"></a>TRACELAB                                     | [官方来源](https://github.com/uw-syfi/TraceLab)                                                                                       | 准备：[build_tracelab_original.py](scripts/build_tracelab_original.py)<br>原生：[run_tracelab_original.py](scripts/run_tracelab_original.py)                                                                                                                               | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-swe-bench"></a>SWE_BENCH                                   | [官方来源](https://huggingface.co/datasets/princeton-nlp/SWE-bench)                                                                   | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-swe-verified"></a>SWE_VERIFIED                             | [官方来源](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)                                                          | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-swesmith"></a>SWESMITH                                     | [官方来源](https://github.com/SWE-bench/SWE-smith)                                                                                    | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-swe-bench-pro"></a>SWE_BENCH_PRO                           | [官方来源](https://github.com/scaleapi/SWE-bench_Pro-os)                                                                              | 复用已有实现；[独立入口盘点](docs/workload-data/WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)                                                                                                                                                                                  | — 本批仅分类；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                               |
| <a id="workload-source-agentbench"></a>AGENTBENCH                                 | [官方来源](https://github.com/THUDM/AgentBench)                                                                                       | 原生：[run_agentbench_alfworld_original.py](scripts/run_agentbench_alfworld_original.py)                                                                                                                                                                                   | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-scienceworld"></a>SCIENCEWORLD                             | [官方来源](https://github.com/allenai/ScienceWorld)                                                                                   | 准备：[prepare_scienceworld_original_assets.py](scripts/prepare_scienceworld_original_assets.py)<br>原生：[run_scienceworld_original.py](scripts/run_scienceworld_original.py)                                                                                             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/scienceworld-original-native-delivery-20261027.json)       |
| <a id="workload-source-mind2web"></a>MIND2WEB                                     | [官方来源](https://huggingface.co/datasets/osunlp/Mind2Web)                                                                           | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-webarena"></a>WEBARENA                                     | [官方来源](https://github.com/web-arena-x/webarena)                                                                                   | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-visualwebarena"></a>VISUALWEBARENA                         | [官方来源](https://github.com/web-arena-x/visualwebarena)                                                                             | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-workarena"></a>WORKARENA                                   | [官方来源](https://github.com/ServiceNow/WorkArena)                                                                                   | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-terminal-bench-2-1"></a>TERMINAL_BENCH_2_1                 | [官方来源](https://github.com/harbor-framework/terminal-bench-2-1)                                                                    | 复用已有实现；[独立入口盘点](docs/workload-data/WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)                                                                                                                                                                                  | — 本批仅分类；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                               |
| <a id="workload-source-osworld"></a>OSWORLD                                       | [官方来源](https://github.com/xlang-ai/OSWorld)                                                                                       | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-appworld"></a>APPWORLD                                     | [官方来源](https://github.com/StonyBrookNLP/appworld)                                                                                 | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-bfcl"></a>BFCL                                             | [官方来源](https://github.com/ShishirPatil/gorilla)                                                                                   | 原生：[run_bfcl_v4_simple_python_original.py](scripts/run_bfcl_v4_simple_python_original.py)                                                                                                                                                                               | [x] 已交付；[交付范围与环境](docs/workload-data/validation/bfcl-native-delivery-20261024.json)                        |
| <a id="workload-source-toolbench"></a>TOOLBENCH                                   | [官方来源](https://github.com/OpenBMB/ToolBench)                                                                                      | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-deepplanning"></a>DEEPPLANNING                             | [官方来源](https://qwenlm.github.io/Qwen-Agent/en/benchmarks/deepplanning/)                                                           | 准备：[prepare_deepplanning_original_assets.py](scripts/prepare_deepplanning_original_assets.py)；原生 Agent 执行／评分待补                                                                                                                                                | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-taubench"></a>TAUBENCH                                     | [官方来源](https://github.com/sierra-research/tau2-bench)                                                                             | 原生：[run_tau2_original_telecom.py](scripts/run_tau2_original_telecom.py)                                                                                                                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/tau2-native-delivery-20261024.json)                        |
| <a id="workload-source-xlam-tool"></a>XLAM_TOOL                                   | [官方来源](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)                                                      | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-toolace"></a>TOOLACE                                       | [官方来源](https://huggingface.co/datasets/Team-ACE/ToolACE)                                                                          | 准备：[audit_toolace_original_corpus.py](scripts/audit_toolace_original_corpus.py)<br>原生：[run_toolace_original_continuation.py](scripts/run_toolace_original_continuation.py)                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/toolace-original-continuation-20261026.json)               |
| <a id="workload-source-gaia"></a>GAIA                                             | [官方来源](https://huggingface.co/datasets/gaia-benchmark/GAIA)                                                                       | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-discoverybench"></a>DISCOVERYBENCH                         | [官方来源](https://github.com/allenai/discoverybench)                                                                                 | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-mlagentbench"></a>MLAGENTBENCH                             | [官方来源](https://github.com/snap-stanford/MLAgentBench)                                                                             | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-scienceagentbench"></a>SCIENCEAGENTBENCH                   | [官方来源](https://huggingface.co/datasets/osunlp/ScienceAgentBench)                                                                  | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-bixbench"></a>BIXBENCH                                     | [官方来源](https://github.com/Future-House/BixBench)                                                                                  | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-naturebench"></a>NATUREBENCH                               | [官方来源](https://github.com/FrontisAI/NatureBench)                                                                                  | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-mlebench"></a>MLEBENCH                                     | [官方来源](https://github.com/openai/mle-bench)                                                                                       | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-paperbench"></a>PAPERBENCH                                 | [官方来源](https://github.com/openai/frontier-evals/tree/main/project/paperbench)                                                     | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-dragon-public"></a>DRAGON_PUBLIC                           | [官方来源](https://github.com/RussianNLP/DRAGON)                                                                                      | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-temporalwiki"></a>TEMPORALWIKI                             | [官方来源](https://github.com/joeljang/temporalwiki)                                                                                  | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-longmemeval"></a>LONGMEMEVAL                               | [官方来源](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)                                                            | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-locomo"></a>LOCOMO                                         | [官方来源](https://github.com/snap-research/locomo)                                                                                   | 原生：[run_locomo_original_qa.py](scripts/run_locomo_original_qa.py)                                                                                                                                                                                                       | [x] 已交付；[交付范围与环境](docs/workload-data/validation/locomo-native-delivery-20261025.json)                      |
| <a id="workload-source-longmemeval-v2"></a>LONGMEMEVAL_V2                         | [官方来源](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)                                                                 | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 未闭环；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-visionarena"></a>VISIONARENA                               | [官方来源](https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat)                                                               | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-mmmu"></a>MMMU                                             | [官方来源](https://huggingface.co/datasets/MMMU/MMMU)                                                                                 | 准备：[prepare_mmmu_original_assets.py](scripts/prepare_mmmu_original_assets.py)<br>原生：[run_mmmu_original.py](scripts/run_mmmu_original.py)                                                                                                                             | [x] 已交付；[交付范围与环境](docs/workload-data/validation/mmmu-native-delivery-20261027.json)                        |
| <a id="workload-source-chartqa"></a>CHARTQA                                       | [官方来源](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)                                                                     | 准备：[prepare_chartqa_native_toolkit.py](scripts/prepare_chartqa_native_toolkit.py)<br>原生：[run_chartqa_original.py](scripts/run_chartqa_original.py)                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/chartqa-native-delivery-20261026.json)                     |
| <a id="workload-source-docvqa"></a>DOCVQA                                         | [官方来源](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA)                                                                   | 准备：[prepare_reused_source_records.py](scripts/prepare_reused_source_records.py)<br>原生：[run_docvqa_original.py](scripts/run_docvqa_original.py)                                                                                                                       | [x] 已交付；[交付范围与环境](docs/workload-data/validation/docvqa-original-native-delivery-20261027.json)             |
| <a id="workload-source-videomme"></a>VIDEOMME                                     | [官方来源](https://huggingface.co/datasets/lmms-lab/Video-MME)                                                                        | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-videomme-v2"></a>VIDEOMME_V2                               | [官方来源](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2)                                                               | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-mvbench"></a>MVBENCH                                       | [官方来源](https://huggingface.co/datasets/OpenGVLab/MVBench)                                                                         | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-longvideobench"></a>LONGVIDEOBENCH                         | [官方来源](https://huggingface.co/datasets/longvideobench/LongVideoBench)                                                             | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-streamingbench"></a>STREAMINGBENCH                         | [官方来源](https://streamingbench.github.io/)                                                                                         | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-ovobench"></a>OVOBENCH                                     | [官方来源](https://github.com/JoeLeelyf/OVO-Bench)                                                                                    | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-librispeech"></a>LIBRISPEECH                               | [官方来源](https://www.openslr.org/12/)                                                                                               | 原生：[run_librispeech_original.py](scripts/run_librispeech_original.py)                                                                                                                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/librispeech-native-delivery-20261025.json)                 |
| <a id="workload-source-fleurs"></a>FLEURS                                         | [官方来源](https://huggingface.co/datasets/google/fleurs)                                                                             | 准备：[prepare_fleurs_original_assets.py](scripts/prepare_fleurs_original_assets.py)<br>原生：[run_fleurs_original.py](scripts/run_fleurs_original.py)<br>隔离评分：[score_fleurs_original.py](scripts/score_fleurs_original.py)                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/fleurs-native-delivery-20261026.json)                      |
| <a id="workload-source-wavcaps"></a>WAVCAPS                                       | [官方来源](https://huggingface.co/datasets/cvssp/WavCaps)                                                                             | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-full-duplex-bench"></a>FULL_DUPLEX_BENCH                   | [官方来源](https://full-duplex-bench.github.io/)                                                                                      | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-libritts"></a>LIBRITTS                                     | [官方来源](https://openslr.org/60)                                                                                                    | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-ljspeech"></a>LJSPEECH                                     | [官方来源](https://keithito.com/LJ-Speech-Dataset/)                                                                                   | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-beir-scifact"></a>BEIR_SCIFACT                             | [官方来源](https://huggingface.co/datasets/BeIR/scifact)                                                                              | 原生：[run_beir_original.py](scripts/run_beir_original.py)                                                                                                                                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/beir-native-delivery-20261025.json)                        |
| <a id="workload-source-beir-nfcorpus"></a>BEIR_NFCORPUS                           | [官方来源](https://huggingface.co/datasets/BeIR/nfcorpus)                                                                             | 原生：[run_beir_original.py](scripts/run_beir_original.py)                                                                                                                                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/beir-native-delivery-20261025.json)                        |
| <a id="workload-source-msmarco-passage"></a>MSMARCO_PASSAGE                       | [官方来源](https://microsoft.github.io/msmarco/)                                                                                      | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 阻塞；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                   |
| <a id="workload-source-stsb"></a>STSB                                             | [官方来源](https://sbert.net/datasets/stsbenchmark.tsv.gz)                                                                            | 原生：[run_stsb_original.py](scripts/run_stsb_original.py)                                                                                                                                                                                                                 | [x] 已交付；[交付范围与环境](docs/workload-data/validation/stsb-native-delivery-20261025.json)                        |
| <a id="workload-source-azure-llm"></a>AZURE_LLM                                   | [官方来源](https://github.com/Azure/AzurePublicDataset)                                                                               | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [x] 已交付；[交付范围与环境](docs/workload-data/validation/azure-serving-trace-delivery-20261024.json)                |
| <a id="workload-source-azure-lmm-2025"></a>AZURE_LMM_2025                         | [官方来源](https://github.com/Azure/AzurePublicDataset/blob/215becdacba1ce682c7368642ce97d5a332de7a6/AzureLMMInferenceDataset2025.md) | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [x] 已交付；[交付范围与环境](docs/workload-data/validation/azure-lmm-2025-asset-delivery-20261027.json)               |
| <a id="workload-source-burstgpt"></a>BURSTGPT                                     | [官方来源](https://github.com/HPMLL/BurstGPT)                                                                                         | 原生：[run_burstgpt_original_example.py](scripts/run_burstgpt_original_example.py)                                                                                                                                                                                         | [x] 已交付；[交付范围与环境](docs/workload-data/validation/burstgpt-native-delivery-20261025.json)                    |
| <a id="workload-source-mooncake"></a>MOONCAKE                                     | [官方来源](https://github.com/kvcache-ai/Mooncake)                                                                                    | 准备：[prepare_mooncake_storage_plan.py](scripts/prepare_mooncake_storage_plan.py)<br>原生：[run_mooncake_storage_original.py](scripts/run_mooncake_storage_original.py)                                                                                                   | [x] 已交付；[交付范围与环境](docs/workload-data/validation/mooncake-official-storage-delivery-20261027.json)          |
| <a id="workload-source-copilot-agent-2026"></a>COPILOT_AGENT_2026                 | [官方来源](https://github.com/Azure/AzurePublicDataset/releases/tag/ghcp-coding-agent-2026)                                           | 准备：[prepare_copilot_original_assets.py](scripts/prepare_copilot_original_assets.py)                                                                                                                                                                                     | [x] 已交付；[交付范围与环境](docs/workload-data/validation/copilot-2026-metadata-delivery-20261027.json)              |
| <a id="workload-source-multiagentbench"></a>MULTIAGENTBENCH                       | [官方来源](https://github.com/ulab-uiuc/MARBLE)                                                                                       | 原生入口索引待补；[目录／适配说明](workload-preparation/README.md)                                                                                                                                                                                                         | [ ] 待接入；[当前缺口](docs/workload-data/DATASET_CHECKLIST.zh-CN.md)                                                 |
| <a id="workload-source-ruler"></a>RULER                                           | [官方来源](https://github.com/NVIDIA/RULER)                                                                                           | 原生：[run_ruler_original_profile.py](scripts/run_ruler_original_profile.py)                                                                                                                                                                                               | [x] 已交付；[交付范围与环境](docs/workload-data/validation/ruler-native-delivery-20261025.json)                       |
| <a id="workload-source-paio-code-eval-1000"></a>PAIO-CODE-EVAL-1000               | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-json-500"></a>PAIO-JSON-500                           | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-chat-1000"></a>PAIO-CHAT-1000                         | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-long-prefill-5000"></a>PAIO-LONG-PREFILL-5000         | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-longtext-4000"></a>PAIO-LONGTEXT-4000                 | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-prefix-shared-5000"></a>PAIO-PREFIX-SHARED-5000       | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-reuse-conv-5000"></a>PAIO-REUSE-CONV-5000             | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |
| <a id="workload-source-paio-semantic-similar-5000"></a>PAIO-SEMANTIC-SIMILAR-5000 | 派欧云提供，原始数据私有、不提供公开下载                                                                                              | 请求回放：[run_source_request_samples.py](scripts/run_source_request_samples.py)                                                                                                                                                                                           | [x] 已交付；[交付范围与环境](docs/workload-data/validation/enterprise-fixed-ci-handoff-20261024.json)                 |

<!-- workload-catalog:end -->
