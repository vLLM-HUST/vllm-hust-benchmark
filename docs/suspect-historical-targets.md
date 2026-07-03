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
