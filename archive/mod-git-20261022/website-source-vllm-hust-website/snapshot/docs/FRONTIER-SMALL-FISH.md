# BetterScale draft communication / copy increment

These are new **15-minute SWE Prefix Reuse smoke observations**, using the same prepared35B workload
and exact-token continuation as the earlier capacity16 series. They are not AgentX scores or SWE
answer-quality measurements.

The experimental prototype uses BF16 Qwen3.5-35B-A3B, TP2, natural MTP2, APC, 16 request slots,4096
query budget and262144 configured context. Its explicit KV budget is24.25GiB/chip; the older
BetterScale curve used20.25GiB. Keep the new fixed-configuration concurrency line separate. Client
revisions59ea20a and 29136f1 differ only in repository knowledge documentation, not measurement
code.

The change reuses native distributed greedy for the draft only, retaining target sampling. It also
removes gate copies under the owned stride-aware consumer and shares the identical mixed-QKV pack.
Request-bounded sampling predates this increment. Qualified two-rank TraceLoom profiles show decode
sampling payload 7,946,240→320 local tensor bytes across two draft rounds, target gate slices60→0,
and mixed QKV packs60→30. These are structural/operator observations, not physical wire traffic or
an end-to-end speedup percentage.

Clean control/candidate serving passes24 targeted retrievals through262080 prompt tokens and16
concurrent requests. An earlier deployment with diagnostic collective graphs allocated during
startup failed32K retrieval; the clean startup passed. The low-level cause remains unresolved. These
bounded checks are not general model-quality certification.

## 35B concurrency observations

All five900s windows completed with zero request failures:

| HTTP concurrency | Output tokens/s/chip | P90 request decode tokens/s | Requests started |
| ---------------- | -------------------: | --------------------------: | ---------------: |
| 1                |               58.922 |                     133.343 |              178 |
| 2                |              102.734 |                     125.854 |              289 |
| 4                |              160.835 |                     102.356 |              477 |
| 8                |              242.462 |                      74.058 |              702 |
| 16               |              368.928 |                      55.379 |             1071 |

At C4, the historical BetterScale point was160.008 output tokens/s/chip and 103.552 P90 decode
tokens/s: the new point has slightly higher throughput and slightly lower P90 decode speed. At C16,
the historical values were349.676 and 54.069; the new throughput is about5.5% higher. These are
single shared-host observations with **different explicit KV budgets**, not a controlled ablation of
this increment. Request-bounded sampling had already enabled the larger KV budget. Do not advertise
removed profile work as a universal throughput gain. Finite-window request mixes, natural acceptance
and host colocation remain part of the evidence.

## 27B extension and correctness boundary

The separate Qwen3.8-27B option uses the same source sessions,900s policy, TP2/BF16/natural MTP2,16
slots,4096 query budget,262144 configured context and 24.25GiB KV/chip. It uses the actual27B
tokenizer, so its prepared token IDs, output budgets and workload fingerprint differ from35B. The
local checkpoint name is user-confirmed; no immutable upstream model revision is attested.

BetterScale passed the new dense-geometry independent GDN/FIA gates and all24
cold/warm/long/concurrent retrieval checks. Its exact-member TraceLoom ledger shows zero draft
vocabulary gathers,320 local greedy-stat tensor bytes across two rounds in the complete decode
captures, zero target gate slices and48 QKV packs per target. This is a candidate observation, not
a27B before/after ablation; the matched35B profiles establish the structural-removal comparison.

The native27B deployment passed all8 serial cold/warm retrievals through262080 prompt tokens, but5
of16 concurrent requests returned `cobalt-seven-` followed by EOS instead of `cobalt-seven-42`.
Retain this failed correctness boundary; neither its cause nor harmless numerical variation has been
established. Fletcher authorized displaying all five native points as red, failed-correctness
throughput references. The failed check was C16 (request IDs 2, 5, 8, 11, 13); C1/2/4/8 were not
separately correctness-qualified. The shared deployment failure is attached to every reference
point, not claimed as five independently observed failures. Red references remain outside the Pareto
envelope, including when BetterScale is filtered out. Neither arm's protocol-valid performance is a
SWE answer-quality score.

All five27B BetterScale900s windows completed with zero request failures:

| HTTP concurrency | Output tokens/s/chip | P90 request decode tokens/s | Requests started |
| ---------------- | -------------------: | --------------------------: | ---------------: |
| 1                |               33.007 |                      77.784 |              118 |
| 2                |               57.138 |                      68.271 |              175 |
| 4                |               96.477 |                      59.967 |              281 |
| 8                |              145.924 |                      46.172 |              439 |
| 16               |              119.482 |                      31.868 |              396 |

The27B C16 observation is deliberately retained even though it is slower than C8: 119.482
versus145.924 output tokens/s/chip, with TTFT P95 17.788 versus1.358 seconds. Server counter deltas
across the sending window **plus drain** record205 preemptions at C16 versus0 at C8, and
prefix-cache hit fractions11.47% versus93.88%. This is an observed capacity/reuse degradation under
the fixed KV budget, not a slower repeat of an identical concurrency setting. Performance protocol
validity does not establish semantic correctness across those preemptions. The five35B windows
recorded zero preemptions. These counters are retained in downloaded metrics with the explicit
`measurement_and_drain` suffix; they do not credit drain tokens to plotted throughput.

Only complete valid windows with observed prefix reuse, continuous device-owner guards, clean server
exit and resource release are imported. Full metric extracts remain in
`data/leaderboard_frontier_swe_evidence.json`; point downloads retain source/configuration/protocol
identities. Same-configuration repeated observations follow
[whole-run best-of selection](FRONTIER-REPEAT-SELECTION.md), with all inferior raw evidence
retained. Different source/configuration curves are not silently pooled into repeats.

## Native27B throughput references (correctness failed)

| Concurrency | Native tokens/s/chip | BetterScale tokens/s/chip |
| ----------- | -------------------: | ------------------------: |
| 1           |               27.904 |                    33.007 |
| 2           |               48.544 |                    57.138 |
| 4           |               80.434 |                    96.477 |
| 8           |              117.680 |                   145.924 |
| 16          |              104.999 |                   119.482 |

All five native900s diagnostic windows met the replay protocol and clean-release gates; that does
not repair the independent failed retrieval check. Native C16 recorded52 preemptions including drain
(BetterScale205); other native concurrency levels recorded0. These are single observations, not a
correctness-qualified speedup or isolated optimization ablation.
