# Leaderboard Data (Repo-Hosted)

This folder is the Git-repo-hosted source for leaderboard publishing.

## Layout

- `snapshots/leaderboard_single.json`
- `snapshots/leaderboard_multi.json`
- `snapshots/leaderboard_compare.json`
- `snapshots/last_updated.json`
- `../submissions/` (raw run exports, each run has `run_leaderboard.json` and `leaderboard_manifest.json`)

## Refresh snapshots from submissions

Run from repository root:

```bash
/home/shuhao/miniconda3/envs/vllm-hust-dev/bin/python -m vllm_hust_benchmark.cli publish-website \
  --source-dir submissions \
  --output-dir leaderboard-data/snapshots \
  --execute
```

## Git LFS guidance

Current payload is very small (JSON metadata only). Do **not** use Git LFS unless large binary artifacts are added (for example traces, logs, model binaries, or files > 50MB).
