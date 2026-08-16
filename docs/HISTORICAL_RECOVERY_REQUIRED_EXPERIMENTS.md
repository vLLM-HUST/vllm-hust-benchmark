# Historical recovery: experiments that still require reruns

The recovery pass evaluated 347 raw historical entries. It directly admitted 276 entries after
filling every field that could be determined from existing specs and revision evidence. Missing
`verified`, peak-memory, checksum, or environment-manifest fields do **not** require reruns for the
historical trend dataset.

The six rows below are the complete rerun queue. They all use the same registered target:
`official-ascend-jan-2026-v0.18.0-random-latency-qwen25-14b-910b2` (Qwen2.5-14B-Instruct,
FP16, one 910B2). Their recorded latency values are usable, but `error_rate` is absent and cannot be
derived from any retained raw result.

| Priority | Historical label | vllm-hust commit | vllm-ascend-hust commit | Missing measurement |
| --- | --- | --- | --- | --- |
| P0 | Official v0.18.0 baseline | `bcf2be96120005e9aea171927f85055a6a5c0cf6` | `e18643f8a4d5bd9990727654318ad069ea0b56e2` | `error_rate` |
| P1 | current-main-single-npu-offline-graph | `1aa7cd10b7b16e82fdb29fcc47d3a3cd93bd01dc` | `03ae1d03db8049cd2a5c3f824039814459542e25` | `error_rate` |
| P1 | ascend-pr66-simllm-kv-manager | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `e0686f12d1af74e6df97dfdaf7d314b4b3de10f7` | `error_rate` |
| P1 | ascend-pr70-simllm-baseline-validation | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `312ca80a90cbd28438bce3b59e3fbaad749451f3` | `error_rate` |
| P1 | pr69-perfgate-l2-targeted-verification | `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7` | `51e577b17b46babba210858686d577161296a420` | `error_rate` |
| P1 | pr77-perfgate-l2-scenario-registry | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `51e577b17b46babba210858686d577161296a420` | `error_rate` |

For each row, rerun only `random-latency` with the registered spec and retain the raw benchmark
result. A successful rerun must explicitly export a finite `error_rate` in `[0, 1]`. No other
historical workload currently needs a new experiment.

The machine-readable source of truth is
`leaderboard-data/snapshots/historical_recovery_report.json`; its `required_experiments` array is
intended for Codex-agent task generation.
