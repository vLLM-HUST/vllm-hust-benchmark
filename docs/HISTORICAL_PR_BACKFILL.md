# Historical PR Benchmark Backfill

Use `scripts/backfill_historical_pr_benchmarks.py` for real-online historical PR
backfills. It is intentionally a higher-level driver around
`scripts/run-current-ascend-same-spec.sh`; the benchmark implementation and
leaderboard export path stay in the existing runner.

## Invariants

- Default workload coverage is the official `v0.18.0`, `910B2`, single-chip spec
  set under `docs/official-baselines/`.
- Historical PR backfills must launch the serving system through
  `/home/shuhao/vllm-hust-dev-hub/manage.sh` with `--managed-dev-hub`. Do not
  bypass dev-hub by manually running `vllm.entrypoints.openai.api_server`.
- Managed runs must use a real systemd unit name ending in `.service`, for
  example `vllm-hust-bf-pr66-random.service`. A bare unit filename can be
  written by `manage.sh` but then `systemctl --user restart <name>.service`
  cannot find it.
- The runner uses detached git worktrees for historical refs. It must not mutate
  the active `/home/shuhao/vllm-hust` or `/home/shuhao/vllm-ascend-hust`
  checkouts.
- Each completed result is a `real-online-historical-pr-backfill` submission.
- Do not republish archived `v0.11.0`, `910B3`, BF16, or missing-same-spec
  records as substitutes for a real run.
- Keep `docs/official-baselines/` curated to public comparable specs only:
  `vllm 0.18.0`, `vllm-ascend 0.18.0`, `910B2`, `FP16`, and the actual
  workload model. Do not place perfgate, tuning, `910B3`, BF16, or 3B
  experiment specs in this directory.
- Do not let previous optimization experiments leak into backfills. The managed
  backfill defaults explicitly use an isolated container/unit, a dedicated port,
  `--enforce-eager`, no prefix caching, no chunked prefill, no custom kernels,
  and disabled Ascend fusion passes unless a run plan explicitly opts in.
- Hardware metadata is detected from the actual managed NPU devices before a
  run. If the real chip model does not match the official same-spec baseline
  chip model, the run must fail instead of exporting data.
- Precision metadata comes from the official same-spec file. `FP16` workloads
  must launch the server with `float16`; `BF16` must launch with `bfloat16`.
  Never relabel a result to make two incomparable rows look comparable.
- A failed startup, failed health probe, or failed client benchmark must not be
  published, mirrored to website data, or committed.
- Benchmark submissions and `leaderboard-data/snapshots/` in this repository are
  the real data source. The website repository must only mirror those snapshots;
  do not make website-local data authoritative and do not add a local-first
  loader mode that can hide fresher GitHub or HF results from other users.
- For top-level compare cards, never display an arithmetic average across
  unrelated rows. Choose one representative matched same-spec sample: same
  workload, precision, model, hardware chip model/count, node count, and baseline
  target. If no matched sample exists, show the scope as blocked for comparison
  instead of synthesizing a value.
- In managed-server mode, the host-side benchmark client is only an HTTP load
  generator. It must not import the historical `vllm-ascend-hust` source tree;
  server-side Ascend plugin code is loaded only inside the dev-hub container
  launched by `manage.sh`.
- Managed historical source worktrees must be visible inside the dev-hub
  container. The container maps `/home/shuhao` to `/workspace`, so use
  `--worktree-root` under `/home/shuhao` for historical source refs. Keep large
  result roots, model caches, and dataset caches under `/data`, but do not put
  managed source worktrees only under `/data`; otherwise the server can fall
  back to the default `/workspace/vllm-hust` checkout while metadata claims a
  historical commit.
- The managed dev-hub service requires a real API key. The backfill runner maps
  `VLLM_HUST_API_KEY` to the client's `OPENAI_API_KEY` environment variable; do
  not pass bearer tokens through command-line `--header` arguments.
- Keep model identity and request routing separate. Leaderboard metadata should
  keep the official model ID, while the online client request should use the
  server's `served_model_name` when dev-hub starts the model under a basename.
  The client tokenizer should still point at the local model snapshot path.
- Online backfill requests follow the same-spec client parameters by default.
  Only pass `--current-client-temperature 0` when the matching official
  baseline/spec explicitly uses that override. The managed server is still
  launched with `--generation-config vllm`; do not let a model's bundled
  generation config silently enable top-k/top-p sampling paths.
- Dataset and model downloads must use `HF_ENDPOINT=https://hf-mirror.com` and
  cache under `/data/shared_datasets/vllm-hust-benchmark/huggingface` or another
  explicit `/data` path. Do not let backfills populate `$HOME` with HF datasets,
  model caches, or benchmark artifacts.
- When `--publish-each` is enabled, every completed submission is immediately
  merged into the HF dataset and the local benchmark snapshots are regenerated.
- When `--sync-website-each` is enabled, the website data mirror is updated from
  the regenerated benchmark snapshots after each result.
- When `--commit-push-each` is enabled, the benchmark and website repos are
  committed and pushed after each successful result.

## Dry Run

Preview the default current-ref matrix:

```bash
cd /home/shuhao/vllm-hust-benchmark
python scripts/backfill_historical_pr_benchmarks.py
```

Preview important historical refs discovered from commit subjects:

```bash
python scripts/backfill_historical_pr_benchmarks.py \
  --discover-from-log \
  --max-discovered-refs 12
```

Preview a curated plan:

```bash
python scripts/backfill_historical_pr_benchmarks.py \
  --plan-file docs/historical-pr-backfill-plan.sample.json
```

## Real Run

Run a curated plan through the dev-hub managed service, upload every completed
result to HF, refresh the website mirror, and push both repositories:

```bash
PYTHONPATH=src python scripts/backfill_historical_pr_benchmarks.py \
  --plan-file docs/historical-pr-backfill-plan.json \
  --managed-dev-hub \
  --dev-hub-dir /home/shuhao/vllm-hust-dev-hub \
  --managed-container vllm-hust-backfill \
  --managed-systemd-unit vllm-hust-backfill.service \
  --managed-npu-devices 0 \
  --server-port 8001 \
  --runtime-python /home/shuhao/miniconda3/envs/vllm-hust-dev/bin/python \
  --current-env-prefix /home/shuhao/miniconda3/envs/vllm-hust-dev \
  --result-root /data/shared_datasets/vllm-hust-benchmark/historical-pr-backfill \
  --worktree-root /home/shuhao/.cache/vllm-hust-benchmark-worktrees/historical-pr-backfill \
  --execute \
  --publish-each \
  --sync-website-each \
  --commit-push-each
```

To run only one workload while debugging:

```bash
PYTHONPATH=src python scripts/backfill_historical_pr_benchmarks.py \
  --plan-file docs/historical-pr-backfill-plan.json \
  --workload sharegpt-online \
  --managed-dev-hub \
  --managed-container vllm-hust-backfill \
  --managed-systemd-unit vllm-hust-backfill.service \
  --managed-npu-devices 0 \
  --server-port 8001 \
  --runtime-python /home/shuhao/miniconda3/envs/vllm-hust-dev/bin/python \
  --current-env-prefix /home/shuhao/miniconda3/envs/vllm-hust-dev \
  --result-root /data/shared_datasets/vllm-hust-benchmark/historical-pr-backfill \
  --worktree-root /home/shuhao/.cache/vllm-hust-benchmark-worktrees/historical-pr-backfill \
  --execute \
  --publish-each \
  --sync-website-each
```

The state file defaults to:

```text
.benchmarks/historical-pr-backfill/state.json
```

Completed matrix cells are skipped on later invocations unless
`--rerun-completed` is passed.

## Default Single-Chip Workloads

The default official `910B2` single-chip workload set currently resolves to:

- `agent-research-online`
- `instructcoder-online`
- `prefix-repetition-online`
- `random-latency`
- `random-online`
- `sharegpt-online`
- `sharegpt-throughput`
- `sonnet-throughput`
- `visionarena-online`

Multi-chip sonnet specs are excluded by default. Use `--include-multi-chip` only
when the corresponding leaderboard view and hardware reservation are intended.

## Incident Notes

Recent local optimization experiments left service-manager state that could have
corrupted historical results if reused blindly:

- dev-hub `.env` had graph-mode compilation, prefix caching, chunked prefill,
  custom kernel, and optimization-plugin settings intended for a separate
  experiment;
- host-side client benchmarks failed when the CANN `set_env.sh` path was not
  sourced, even though the containerized server was healthy;
- stale assumptions around `910B2` versus `910B3`, and `FP16` versus `BF16`,
  made rows look comparable when they were not.

The backfill harness now treats these as hard failures or explicit opt-ins. If a
future experiment needs graph mode, prefix caching, chunked prefill, BF16, B3, or
another non-default setting, create a separate plan and label the data source so
it cannot enter the `v0.18.0`, `910B2`, single-chip public leaderboard by
accident.

The old `perfgate-ascend-qwen25-3b-910b3.json` spec was removed from
`docs/official-baselines/` because it described a BF16 3B run on 910B3 and was
not comparable with the public v0.18.0 / 910B2 / FP16 leaderboard. Keep any
future non-public performance gates outside the official-baseline directory.
