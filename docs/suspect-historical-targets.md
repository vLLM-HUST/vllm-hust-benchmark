# Suspect Historical Targets

The following historical target combinations are intentionally excluded from active leaderboard
trend data and future backfill runs.

## 7fa0e3ed4b / c56ccf1e

- Core: `7fa0e3ed4b8166e14f42c97951e33ca9f06e2d2b`
- Plugin: `c56ccf1e778b59bb7f29bba2152c453ab162811f`
- Archived submissions:
  - `qwen25-14b-910b2-prefix-repetition-online-1npu-20260506-corrected`
  - `qwen25-14b-910b2-random-latency-1npu-20260506-corrected`
  - `qwen25-14b-910b2-sonnet-throughput-1npu-20260506-corrected`

Reason: the `prefix-repetition-online` record has inconsistent workload metadata versus its
same-spec client parameters, and a clean graph-mode rerun on 2026-07-03 failed before serving due to
`ImportError: cannot import name ops from vllm_ascend`. Keeping this target in active trends creates
misleading gaps and outlier points, so the data is retained only under `archive/suspect/`.

## PR77 duplicate sharegpt-throughput low run

- Archived submission: `historical-pr-pr-77-ceec19abb0-sharegpt-throughput-ceec19abb0-51e577b17b`
- Core: `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad`
- Plugin: `51e577b17b46`
- Workload: `sharegpt-throughput`

Reason: this entry is a duplicate PR77 same-spec run with the same core/plugin pair as
`historical-pr-pr77-perfgate-l2-scenario-registry-sharegpt-throughput-ceec19abb0-51e577b17b`, but
reports `1067.21 tok/s` while neighboring same-spec PR77/PR70/PR66 points are about
`1568-1662 tok/s`. Since there is an aligned PR77 same-spec point for the same commit/plugin already
present, the lower duplicate is treated as a suspect bad run and retained only under
`archive/suspect/`.

## Closed PR #7 single-card 910B3 random-online record

- Entry ID: `6a6b8250-e8de-4eb6-86b5-a611b3211bc0`
- Archived: `archive/suspect/pr7-910b3-singlecard-random-online-unverified/`
- Core: `cee0aff18d987ba6fdd86d5e5fec80a20cfc97eb`
- Plugin: `f16e2c1a419430e4f018ad41a5cdd5e6b48d0702`
- Hardware: `910B3`, 1 chip
- Workload: `random-online` (Qwen/Qwen2.5-14B-Instruct, FP16)

Reason: the record only exists in a closed, never-merged PR (#7). `metadata.verified` is null, its
same-spec points to the retired `v0.11.0` target while the recorded engine is `0.17.2rc1...`, the
attached server log proves the run was served with `enforce_eager=True`, and no env-manifest,
pip-packages, or original checksums were produced. It is quarantined as non-public audit data and
must not be admitted to `submissions/`, snapshots, or the website mirror, and must not be compared
against `910B2` or the retired `v0.11.0` target.

## Rerun disposition (2026-08-14, issue #179)

The original record was re-run on 910B3 with the current clean runtime as an isolated 910B3 hardware
series. The rerun is admitted under `submissions/` but must never be compared against 910B2 or the
retired v0.11.0 target.

- New entry ID: `919147ca-5719-4283-98e3-36fcf27287c0`
- Submission: `submissions/single-gpu-backfill-random-online-4861aab3a-20260814/`
- Spec: `docs/official-baselines/specialty-ascend-current-random-online-qwen25-14b-910b3.json`
- Core: `4861aab3af39e721c1b5a8b27b72c4f6bebda888`
- Plugin: `03adecc4de8c3752df3ce687558a8774fb7b84cd`
- Served: graph mode (`enforce_eager=False`, `PIECEWISE` cudagraph), `temperature=0.0`
  (`aclnnApplyTopKTopPCustom` workaround required on 910B3)
- Result: throughput `243.57 tok/s`, TTFT `254.14 ms`, TPOT `43.13 ms`, error_rate `0.0`
- Hardware identity: verified via `npu-smi` on host (Huawei Ascend 910B3), `metadata.verified=true`
