# Suspect Historical Targets

The following historical target combinations are intentionally excluded from active
leaderboard trend data and future backfill runs.

## 7fa0e3ed4b / c56ccf1e

- Core: `7fa0e3ed4b8166e14f42c97951e33ca9f06e2d2b`
- Plugin: `c56ccf1e778b59bb7f29bba2152c453ab162811f`
- Archived submissions:
  - `qwen25-14b-910b2-prefix-repetition-online-1npu-20260506-corrected`
  - `qwen25-14b-910b2-random-latency-1npu-20260506-corrected`
  - `qwen25-14b-910b2-sonnet-throughput-1npu-20260506-corrected`

Reason: the `prefix-repetition-online` record has inconsistent workload metadata
versus its same-spec client parameters, and a clean graph-mode rerun on 2026-07-03
failed before serving due to `ImportError: cannot import name ops from
vllm_ascend`. Keeping this target in active trends creates misleading gaps and
outlier points, so the data is retained only under `archive/suspect/`.

## PR77 duplicate sharegpt-throughput low run

- Archived submission: `historical-pr-pr-77-ceec19abb0-sharegpt-throughput-ceec19abb0-51e577b17b`
- Core: `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad`
- Plugin: `51e577b17b46`
- Workload: `sharegpt-throughput`

Reason: this entry is a duplicate PR77 same-spec run with the same core/plugin pair as
`historical-pr-pr77-perfgate-l2-scenario-registry-sharegpt-throughput-ceec19abb0-51e577b17b`,
but reports `1067.21 tok/s` while neighboring same-spec PR77/PR70/PR66 points are
about `1568-1662 tok/s`. Since there is an aligned PR77 same-spec point for the
same commit/plugin already present, the lower duplicate is treated as a suspect
bad run and retained only under `archive/suspect/`.
