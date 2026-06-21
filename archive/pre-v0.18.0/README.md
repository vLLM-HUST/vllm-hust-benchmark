# Archive: Pre-v0.18.0 Benchmark Submissions

Archived on: 2026-06-21
Cutoff commit: `2d6f5de` (2026-06-04 04:02:29 UTC) — "feat: refresh official v0.18.0 ascend baseline snapshots"

## Contents

47 submission directories archived, including:

- **8 × `official-ascend-jan-2026-v0.11.0-*`**: Old official baselines (vllm v0.11.0 + vllm-ascend v0.11.0), superseded by v0.18.0
- **30 × `ci-*`**: CI benchmark runs from May 12 – June 3, 2026 (pre-v0.18.0 era)
- **9 × `qwen25-*`**: Early manual benchmark runs (April–May 2026)

## Reason for Archival

These submissions were archived because:

1. They contain outdated engine versions (v0.11.0, early dev builds) that are no longer the official baseline
2. The `publish-website` aggregation (`_publish_to_website_snapshots()`) scans ALL directories under `submissions/`, which caused stale v0.11.0 data to be re-introduced into leaderboard snapshots on every CI run
3. Moving them to `archive/` removes them from the aggregation path while preserving historical data

## Impact

- `publish-website` will no longer include these 47 submissions in leaderboard snapshots
- The data remains available in this archive for historical reference
- To restore: `mv archive/pre-v0.18.0/* submissions/`

## Directory Structure After Archival

```
vllm-hust-benchmark/
├── submissions/          # 17 active submissions (v0.18.0+)
│   ├── ci-2702*~ci-2767* # CI runs after v0.18.0 baseline (8 dirs)
│   └── official-ascend-jan-2026-v0.18.0-*  # Current official baselines (9 dirs)
├── archive/
│   └── pre-v0.18.0/      # 47 archived submissions (this directory)
│       ├── ci-257*~ci-268*
│       ├── official-ascend-jan-2026-v0.11.0-*
│       └── qwen25-*
└── leaderboard-data/snapshots/
```
