# MuxPD scheduling diagnosis and finite TP4 follow-up

**Preparation draft.** The single-RPC diagnostic has finished. The async plugin candidate and final
fresh-process comparisons are still pending at this draft cutoff. No unmeasured value is filled in
below. Final analysis will be versioned separately and retain this preparation cutoff and all failed
attempts.

## Problem and interventions

The preceding integration used synchronous scheduling and the fusion backend. Its large gap from
native default async execution was also present in pure Decode. This follow-up tests a single-RPC
execute/sample worker first, separates native scheduling/backend factors, then evaluates an explicit
plugin async Decode pipeline. Single-RPC changes alone are not accepted as a demonstrated
performance fix.

The source main is `6fd8f0de9968357e40a90e185b325f02052039f9`. The first atomic worker diagnostic is
`942af268eccfdfabd2e56282d8e69e4f972b4ad1`. Final async candidate commits must come from terminal
controller/source receipts, not a branch name. The native sources remain Core
`d0f22d2bda562156e4dbf433ce645e1769b4f804` and Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`;
the retained Ascend build-file edit is recorded in each run's provenance.

## Actual machine and fixed workload

`runs/main-01/npu-before.txt` reports four idle **910B2** devices with physical IDs **0, 1, 3, 4**.
The child visible selectors are logical `0,1,2,3`. This is a new device allocation; prior campaign
card IDs/timings are not silently pooled with this campaign.

The model is Qwen3.5-35B-A3B, BF16, TP4/EP4, with max model/batch length 16384, max sequences eight,
prefix cache off, Mamba cache mode none, FULL_DECODE_ONLY graphs, and HCCL_DETERMINISTIC=true.
Package versions and the full actual launch and resolved configurations remain in the run artifacts.
Native/backend changes are explicit experimental factors.

The unchanged synthetic token inputs use eight P128/D512 requests for Decode, and four P128/D512
seeds followed by P12000/D128 after each seed delivers exactly token32 for mixed work. Each fresh
process has one excluded warmup per workload, then decode/mixed/decode/mixed measurement windows.
Timing stores actual DELTA tokens and timestamps; there is no profiler or per-step metadata RPC.
These are offline actor-arrival measurements, not HTTP Goodput or an official leaderboard series.
Original producer functions and inputs remain byte-identical across benchmark v1/v2; v2 adds
explicit async role/configuration checks.

## Completed diagnostic: single-RPC worker has no demonstrated benefit

The completed `main-01` and `atomic-01` controllers pass. Within each process, the figures below are
means of two measured windows; they are not population confidence intervals.

| workload       | main token/s | atomic token/s | interpretation                      |
| -------------- | -----------: | -------------: | ----------------------------------- |
| Decode         |      329.089 |        326.866 | no demonstrated gain                |
| Long-P ingress |      182.053 |        181.935 | effectively unchanged in this pilot |

All six windows' complete output-token arrays and termination states match between the two runs,
including warmups. This is a useful negative result: removing the separate sample RPC alone does not
explain or repair the main performance gap. Raw values, full token comparisons and excluded warmups
remain in `analysis/atomic-pilot-v1` and the two original run directories.

## Native factor diagnosis and final comparisons

Native synchronous/npugraphEx and native asynchronous/fusion diagnostics are separate roles. Their
final interpretation must use terminal native-default and plugin results, with actual backend/queue
readbacks. Do not attribute the whole prior gap to async or backend individually without the
relevant matched comparison. Do not turn mixed generated-token trajectory differences into an
identical-math claim.

| final comparison                                           | status at this draft cutoff          |
| ---------------------------------------------------------- | ------------------------------------ |
| Fresh main versus async plugin                             | pending complete terminal collection |
| Fresh native defaults versus async plugin                  | pending complete terminal collection |
| Per-process throughput, long-P TTFT, P99 and maximum pause | pending final selected summary       |
| Complete output-token/termination comparisons              | pending final selected summary       |
| All attempts, source/config identities and cleanup         | pending final manifest               |

The final report must present all process means, drift/repetition limits, failed attempts and actual
token-array differences. Tokens from the same request/step are correlated and must not be treated as
independent statistical samples.

## Quality and claim boundary

[Plugin Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23) remains open. The
earlier fixed task-quality gate failed: both raw/group had three losses on cases answered correctly
by both native controls, despite a higher aggregate score. This scheduling/performance experiment
does not clear that gate, and identical synthetic output arrays do not establish task quality.

This follow-up does not claim HTTP Goodput, physical P/D hardware overlap, Cube/Vector resource
partitioning, production readiness, or universal numerical equivalence. Any default-off experimental
merge must retain these limitations.

## Publication disposition

The planned canonical Release tag is `evidence-muxpd-scheduling-20261026` in
`vLLM-HUST/vllm-hust-benchmark`. Packaging/upload/public readback are pending. Payload archives stay
outside Git. The final companion manifest must bind every included original member and the archive
hash; compiler caches remain excluded.
