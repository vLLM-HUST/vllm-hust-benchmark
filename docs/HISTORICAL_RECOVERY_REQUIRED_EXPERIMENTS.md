# Historical recovery: required experiment status

The historical recovery queue is complete. The recovery pass evaluated 349 raw historical entries,
selected 282 auditable trend entries, and reports zero required experiments. Missing `verified`,
peak-memory, checksum, or environment-manifest fields do not require reruns for the historical trend
dataset.

All six formerly queued rows use the registered target
`official-ascend-jan-2026-v0.18.0-random-latency-qwen25-14b-910b2` (Qwen2.5-14B-Instruct, FP16, one
910B2). They were resolved as follows:

| Historical label                       | vllm-hust commit                           | vllm-ascend-hust commit                    | Evidence                                                       |          TTFT (ms) | `error_rate` |
| -------------------------------------- | ------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------- | -----------------: | -----------: |
| Official v0.18.0 baseline              | `bcf2be96120005e9aea171927f85055a6a5c0cf6` | `e18643f8a4d5bd9990727654318ad069ea0b56e2` | Existing strict three-repeat suite; selected repeat 03         |   84910.6170389879 |            0 |
| current-main-single-npu-offline-graph  | `1aa7cd10b7b16e82fdb29fcc47d3a3cd93bd01dc` | `03ae1d03db8049cd2a5c3f824039814459542e25` | Fresh same-spec NPU rerun, 10 warmups + 30 measured iterations |   4887.08614447775 |            0 |
| ascend-pr66-simllm-kv-manager          | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `e0686f12d1af74e6df97dfdaf7d314b4b3de10f7` | Derived successful atomic offline latency artifact             | 7025.4118066901965 |            0 |
| ascend-pr70-simllm-baseline-validation | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `312ca80a90cbd28438bce3b59e3fbaad749451f3` | Derived successful atomic offline latency artifact             |  6874.877021748884 |            0 |
| pr69-perfgate-l2-targeted-verification | `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7` | `51e577b17b46babba210858686d577161296a420` | Derived successful atomic offline latency artifact             |  6903.954959474504 |            0 |
| pr77-perfgate-l2-scenario-registry     | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `51e577b17b46babba210858686d577161296a420` | Derived successful atomic offline latency artifact             |   7014.38210135481 |            0 |

The derivation uses rule `atomic-offline-latency-success/v1`. `vllm bench latency` runs in-process
and emits its aggregate artifact only after every warmup and measured `LLM.generate` call returns; a
request failure raises and exits non-zero instead. A retained successful artifact with 10 warmup and
30 measured iterations therefore proves `error_rate=0`. This rule is limited to the successful
`real-online-historical-pr-backfill` `random-latency` artifacts and is not applied to online
workloads.

No fusion setting was changed and no incompatible diagnostic run was published. The recovery
projection preserves each original entry ID, idempotency key, and SHA-256. The machine-readable
source of truth is `leaderboard-data/snapshots/historical_recovery_report.json`; its
`satisfied_experiments` array distinguishes `existing-strict-repeat-suite`, `fresh-rerun`, and
`derived-success`, while `required_experiments` is empty.
