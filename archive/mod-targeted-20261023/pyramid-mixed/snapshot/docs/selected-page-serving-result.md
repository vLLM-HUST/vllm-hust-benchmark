# Selected-page serving result and acceptance boundary

[PR #13](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/13) is an operator optimization that avoids
full V gathers and reuses page mappings. Its approximately 36% isolated operator latency reduction
does not translate into a demonstrated service throughput improvement.

The [canonical four-arm report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a2684c3701237c0b5a8cb6e73d25ce305d6f0e71/reports/pyramidkv-selected-page-abba-20261009)
compares original–selected–selected–original with a fresh process each time, identical BF16 TP2
910B2 graph/MTP2 settings, 1368/beta-2 compression and unique salted prompts. Both methods enable
compression; this is not a new compression-off/on B0/B1 result.

All 548 requests succeed, including 396 measured requests and 152 predeclared excluded controls
and warmups. All SSE/usage/input hashes and 372 compression transactions on both TP workers were
verified. At 8K/C1, 8K/C4, 24K/C1 and 24K/C4, pooled throughput changes are -0.24%, -0.45%, -0.45%
and -0.25%. HBM peaks are essentially unchanged. Only two processes per implementation were
measured; the small differences do not establish a general regression, equivalence or benefit.

Cross-process synthetic output differences also appear with the same implementation (35 and 34
of 137 outputs); all 20 disputed-C1 repeat controls match. Existing separate evidence includes
300 identical sampled quality outputs and 880 bitwise-equal real-cache tensor comparisons. These
checks narrow the investigation but do not establish the numerical cause of generation variability.

The report preserves all positive/negative observations and raw archives. It discloses the
selected-2 startup constructor overlap with an already excluded control; formal measured telemetry
starts 109.564 seconds after the check ends. The final server's later Harbor API diagnostic is
outside the performance plan. Historical compression-off/on throughput regressions remain
unchanged, and there is no new official leaderboard or HBM-reduction claim.

On 2026-10-10 (Asia/Shanghai), the project owner explicitly accepted PR #13 on the basis of its
local operator improvement, with no claim of serving acceleration. The acceptance covers the
selected-page copying and shared slot mappings supported by the isolated operator measurements
and correctness checks. It does not promote service throughput, HBM usage, maximum capacity or
release readiness. The original negative off/on B0/B1 results remain unchanged. A corrected public
shared-adapter artifact and its release-specific acceptance remain outstanding.
