# Issue #95 bounded automatic prefix-cache closure

> **Point-in-time report — not current repository status or a task
> assignment.** Current classification and defaults live in
> [`NATIVE_ENGINE_STATUS.md`](../../NATIVE_ENGINE_STATUS.md).

Issue #95 closes the unbounded-residency gap left by the positive, default-off
automatic prefix-cache result in Issue #55. The production native runtime now
binds an explicit positive `automatic_prefix_cache_max_entries` into config and
deployment identity, maintains LRU order, skips every leased generation, evicts
the selected worker state between batches, and reports entries, evictions and
lease-blocked insertions. The feature remains default-off.

## Correctness repair found by the real gate

The first candidate run exposed a production ownership error rather than a
benchmark mismatch. APC had indexed the workload's explicitly retained seed
state. Between request cohorts that caller-owned state had no active request
lease, so unique-prefix churn could evict it and later exact requests correctly
failed with HTTP 409 `StaleState`.

The narrow fix prevents a cache miss for `retain_state=true` from transferring
ownership to APC. Explicit retained handles remain caller-owned; such requests
may still query an existing APC entry. A regression now executes retained seed,
cache-owned churn at capacity one, and exact continuation of the seed. It passes
without weakening state generation, lease, stale-handle or digest checks.

## Frozen experiment

- version: `issue95-bounded-apc-20260815-r3`
- identity: `a9b0b823b394062ebcce2add88d80a6a46066bc9bb663e573086130c5aaffde3`
- Rust bundle: `a3e0f99ac03d795c227eb29f72e2849aac6e12a5e65715952b3a6587f04c5017`
- official local image: `quay.io/ascend/vllm-ascend:v0.23.0rc1-openeuler`, ID
  `sha256:f4c89c293e076453e9eef9edb5fb9669740dccbd3c48619a9f976d775fc29b81`
- workload: Qwen2.5-14B BF16, 2,177-token prompt, 32 output tokens,
  32 requests, concurrency four, 24 exact-hot plus eight unique cold prefixes
- capacity: 17 states, 256 physical KV blocks, scheduler/append batch one,
  APC capacity two
- control: APC off; candidate: APC on; every other input and timing boundary
  matched
- order: control/candidate, candidate/control, control/candidate; fresh
  server and worker per arm

Two independent r2 cache-miss worker gates generated the same 161-line digest
cache byte-for-byte, SHA-256
`3cfa499de77b60f739645074f4d6650cedfea80093416a6413f93ec2e9b8725a`.
R3 froze that verified cache for matched cache-hit startup in every arm.

## Result

The 1+1 gate and all six formal arms were token exact and lifecycle-clean.
Every control run emitted zero APC counters and retained no APC state. Every
candidate run emitted exactly eight lookups, zero hits, eight inserts, six
evictions and two final APC residents. Lease-blocked insertions, hash
collisions and stale entries were zero. Peak HBM was 53,550 MiB in both arms.

| metric | control median | candidate median |
|---|---:|---:|
| request/s | 1.021090 | 1.020829 |
| output token/s | 32.674894 | 32.666517 |
| wall p50 | 2,897.428 ms | 2,897.714 ms |
| wall p95 | 3,920.504 ms | 3,921.170 ms |
| cold TTFT p50 | 587.562 ms | 586.703 ms |
| cold TTFT p95 | 595.431 ms | 591.352 ms |
| cold TPOT p50 | 107.288 ms | 107.338 ms |
| cold TPOT p95 | 107.586 ms | 108.229 ms |
| request/s CV | 0.326% | 0.268% |

Candidate throughput overhead is 0.0256%, within the preregistered 2% guard.
This accepts the bounded safety fix for the opt-in APC path; it is not a speedup
result and does not support a vLLM comparison.

## Rejected attempts and evidence

R1 failed before worker launch because the host attempted to hash a
container-only execution-config path. R2 corrected that binding, completed an
exact control, and then exposed the caller-owned seed eviction in candidate.
Neither failed run was retried under the same identity or removed. Their raw
logs, status and cleanup evidence remain under `results/issue95-bounded-apc-*`.
The accepted 1+1 and 3+3 evidence is under the r3 result roots.
