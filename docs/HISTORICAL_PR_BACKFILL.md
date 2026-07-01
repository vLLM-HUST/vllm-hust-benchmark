# Historical PR Benchmark Backfill

Use `scripts/backfill_historical_pr_benchmarks.py` for real-online historical PR
backfills. It is intentionally a higher-level driver around
`scripts/run-current-ascend-same-spec.sh`; the benchmark implementation and
leaderboard export path stay in the existing runner.

## Invariants

- Default workload coverage is the official `v0.18.0`, `910B2`, single-chip spec
  set under `docs/official-baselines/`.
- The runner uses detached git worktrees for historical refs. It must not mutate
  the active `/home/shuhao/vllm-hust` or `/home/shuhao/vllm-ascend-hust`
  checkouts.
- Each completed result is a `real-online-historical-pr-backfill` submission.
- Do not republish archived `v0.11.0`, `910B3`, BF16, or missing-same-spec
  records as substitutes for a real run.
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

Run a curated plan, upload every completed result to HF, refresh the website
mirror, and push both repositories:

```bash
PYTHONPATH=src python scripts/backfill_historical_pr_benchmarks.py \
  --plan-file docs/historical-pr-backfill-plan.json \
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
