# Archived 7a63 main random-only records

Archived on 2026-07-03 while auditing the single-card trend chart by x-axis column coverage.

These three `main` / `7a63f81e86bd` records only covered `random-online`, so they created a partial x-axis column in the default all-workloads chart. Two records are duplicate FP16 same-spec `random-online` runs, and one record uses `aly16/Qwen2.5-14B-W8A8` with `dtype=auto` / INT8, which should not be mixed into the FP16 Qwen2.5-14B-Instruct trend line.

The same core revision is already represented by the complete clean gapfill column `main-7a63-d40e-gapfill`, which includes all default online workloads. Keep these raw records for auditability only; do not aggregate them into active public leaderboard snapshots.
