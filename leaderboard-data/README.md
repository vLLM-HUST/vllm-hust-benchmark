# Leaderboard Data (Repo-Hosted)

This folder is the Git-repo-hosted source for public leaderboard publishing.

`snapshots/` is a curated publish set, not a blind aggregate of every historical
file under `submissions/`. The raw submissions directory may contain old CI
probes, failed tuning attempts, and historical experiments that are useful for
audit but should not automatically appear on the website.

The public baseline is `vllm 0.18.0` plus the matching `vllm-ascend 0.18.0`
official Ascend baseline. Older `vllm 0.11.0` / `v0110` baseline runs are
retired and must not appear in `snapshots/`, even if they still exist in a
local archive or external dataset history.

## Layout

- `snapshots/leaderboard_single.json`
- `snapshots/leaderboard_multi.json`
- `snapshots/leaderboard_compare.json`
- `snapshots/last_updated.json`
- `../submissions/` (raw run exports, each run has `run_leaderboard.json` and `leaderboard_manifest.json`)
- `../archive/` (archived submissions not included in aggregation, see `archive/pre-v0.18.0/README.md`)

## Refresh snapshots from submissions

For private/audit rebuilds, run from repository root:

```bash
/home/shuhao/miniconda3/envs/vllm-hust-dev/bin/python -m vllm_hust_benchmark.cli publish-website \
  --source-dir submissions \
  --output-dir leaderboard-data/snapshots \
  --execute
```

Before committing refreshed snapshots, curate the publish set and verify that
the website mirror remains identical:

```bash
python scripts/validate_public_leaderboard_snapshots.py \
  --snapshot-dir leaderboard-data/snapshots

python ../vllm-hust-website/scripts/sync_leaderboard_snapshots.py \
  --source-dir leaderboard-data/snapshots \
  --target-dir ../vllm-hust-website/data \
  --check
```

## Git LFS guidance

Current payload is very small (JSON metadata only). Do **not** use Git LFS unless large binary artifacts are added (for example traces, logs, model binaries, or files > 50MB).
