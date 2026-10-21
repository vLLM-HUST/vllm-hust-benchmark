# Historical R5 matched real-NPU result

This directory preserves compact source evidence for the accepted StateAxis issue
#76 R5 campaign. The test used Qwen2.5-14B BF16, TP1, an Ascend 910B2 selected as
physical NPU3, and three independent fresh processes per arm.

| Metric | Serialized control | Async candidate | Descriptive delta |
|---|---:|---:|---:|
| Host-tier-hit wall p50 | 592.597 ms | 333.716 ms | -43.686% |
| Host-tier-hit wall p95 | 611.814 ms | 343.321 ms | -43.884% |
| Concurrent-pair makespan p50 | 607.648 ms | 355.473 ms | -41.500% |
| Pair operations/s p50 | 3.291 | 5.626 | +70.941% |
| Peak HBM observations | 44,175--44,177 MiB | 44,174--44,177 MiB | no resolved increase |
| Device-overlap-proven runs | 0/3 | 3/3 | candidate-only |

All six runs passed output exactness, lifecycle exactness and state-identity
preservation. Every run applied one host-tier prefetch; authority and prefetch
fallback counts were zero. Candidate measured 1.578/1.730/1.730 ms of device
overlap at transfer completion.

The accepted preregistration used only three repetitions per arm and did not define
a significance claim. These numbers therefore remain directional component evidence,
not a general end-to-end serving speedup. The fixture covers one host-tier restore
concurrent with unrelated compute; it does not cover 2K/8K/16K contexts, logical
capacity scaling, cross-instance transfer or arbitrary request mixes.

The active MOD implementation and ECPA activation contract were extracted later.
Their matched-process 3+3 campaign proves current identity, effect counters,
exactness, lifecycle and fail-closed wiring with a deterministic protocol worker,
but performs no accelerator kernels. Consequently the historical performance result
does not qualify MOD version 0.2.0; a fresh ECPA-launched real-NPU campaign is still
required.
