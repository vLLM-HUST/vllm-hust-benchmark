# Issue #76 asynchronous KV transfer lane closure (2026-08-15)

> **Document role: point-in-time report, not current repository status or a task
> assignment.** Current classification and priorities live only in
> [`NATIVE_ENGINE_STATUS.md`](../../NATIVE_ENGINE_STATUS.md).

## Decision

Issue #76 is implemented as a default-off, fail-closed production candidate and
has passed its frozen correctness and real-device overlap gate. The accepted r5
campaign is a matched fresh-process 3+3 on physical NPU 3. It is a bounded,
directionally positive result; it is not a claim of statistical significance,
general serving speedup, larger logical capacity, or superiority over vLLM.

The feature is enabled only with
`STATE_ENGINE_PROTOCOL_ASYNC_TRANSFER_LANE=1`. The default path retains the
existing serialized control behavior.

## Frozen identity

- official local image:
  `quay.io/ascend/vllm-ascend:v0.23.0rc1-openeuler`, image id
  `sha256:f4c89c293e076453e9eef9edb5fb9669740dccbd3c48619a9f976d775fc29b81`;
  no pull was performed;
- worker SHA-256:
  `cdda03cf1491f4e7e18296f966b89232bae64abb78bbd578cc5bfffd08524744`;
- artifact SHA-256:
  `eda976712755e07825549bd9c3346536878425c4235b422c60b27043ab45da51`;
- execution-config SHA-256:
  `408ad3b68c38df1cde4f335264c5554a89d476f3e9c179c71063ffd1ca0caea2`;
- aggregate frozen identity:
  `b75925e6bba3cd02957fb1d6dda7049c36fcee32f2ebeb7b07b98e84c2dfc608`.

The runtime binding and independent artifact audits report no mismatch. The
complete identity is in
`benchmarks/issue76_async_transfer_lane_20260815_r5_identity.json`.

## Correctness and lifecycle boundary

The implementation adds a dedicated ACL transfer stream and timeline events,
correlation-safe out-of-order native-frame completion, serialized protocol
stdout, bounded request queues, and transactional restore
(`BeginRestore`/`CommitRestore`/`AbortRestore`). Duplicate restores for the same
state generation are coalesced. Residency is published only after DMA
completion, full-snapshot digest equality, and state/generation/location checks;
failure aborts without publishing partial state.

All six formal runs passed token exactness, lifecycle exactness, and state
identity preservation. Every arm applied one host-tier prefetch per run;
authority and prefetch fallback counts were zero. The relevant offline suite
also passed 233 Rust library tests, 11 Python tests, and allocator regression
tests.

## Matched NPU result

| Metric | serialized control | async candidate |
|---|---:|---:|
| fresh-process repetitions | 3 | 3 |
| host-tier-hit wall p50 | 592.597 ms | 333.716 ms |
| host-tier-hit wall p95 | 611.814 ms | 343.321 ms |
| host-tier-hit p50 delta | -- | -43.686% |
| concurrent-pair makespan p50 | 607.648 ms | 355.473 ms |
| pair operations/s p50 | 3.291 | 5.626 |
| peak HBM observations | 44,175--44,177 MiB | 44,174--44,177 MiB |
| device-overlap-proven runs | 0/3 | 3/3 |
| measured device overlap | 0 | 1.578/1.730/1.730 ms |

Candidate host-hit wall coefficient of variation was 1.925%; control was
2.319%. The overlap value comes from paired ACL compute and transfer events at
transfer completion, not from HTTP wall-clock intersection. The candidate also
overlapped full-snapshot host integrity verification with H2D submission while
retaining commit-after-verification semantics.

Only three repetitions per arm were preregistered. Therefore the observed
43.686% p50 reduction is reported as directional evidence, not a statistically
significant speedup. The fixture exercises one host-tier restore concurrent
with unrelated compute and does not establish 2K/8K/16K behavior or 2x logical
capacity; descriptor/device-slot decoupling remains Issue #77.

## Preserved evidence and cleanup

- accepted formal evidence:
  `results/issue76-async-transfer-lane-20260815-r5/`;
- r3 diagnostic: integrity verification preceded H2D, so device overlap was not
  established;
- r4 diagnostic: host overlap was present, but request-phase telemetry sampled
  the wrong completion boundary; it is retained and unscored;
- accepted preregistration and runner:
  `benchmarks/issue76_async_transfer_lane_20260815_r5_preregistration.json` and
  `scripts/run_issue76_async_transfer_lane_20260815_r5_gate.sh`.

The experiment container was removed. The post-run device snapshot reports no
process on NPU 3; unrelated business containers were not stopped or modified.
