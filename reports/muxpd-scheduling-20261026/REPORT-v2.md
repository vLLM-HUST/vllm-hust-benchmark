# MuxPD async scheduling: TP4 comparison and remaining limits

The asynchronous candidate restores Decode throughput close to the fresh native reference while
retaining the substantially shorter worst streaming pause when a long prompt arrives. It does not
establish a throughput advantage over native, and long-prompt TTFT and mixed-workload P99 remain
worse. The earlier single-RPC worker did not provide a meaningful throughput improvement.

**Status:** all nine timed performance processes and the separate three-case hardware lifecycle
diagnostic are terminal PASS. Release upload/public readback remain pending at this report draft
cutoff. The bounded lifecycle results do not establish quality, Manager, default-deployment or HTTP
Goodput qualification.

The earlier [preparation draft](REPORT-DRAFT-v1.md) is retained as a separate historical cutoff.
This report uses this campaign's fresh measurements and does not pool the preceding campaign's
different physical card allocation.

## What changed

The old decoupled integration forced long stretches of pure Decode through synchronous retirement.
The new explicit scheduler/worker pair uses the native asynchronous output and device-token feedback
path for pure Decode, with a separate FIFO retirement record for every queued batch. Before Prefill
work, it drains the Decode queue and returns to the existing Prefill ownership/ACK path.

Prefill-to-Decode handoff supplies the confirmed output history. Preempted Decode-to-Prefill
resumption restores complete confirmed history before synchronous recomputation. Aborted Decode
blocks remain owned until the last queued result retires. Zero-token Prefill progress and
cancellation operations return their required receipt directly from execution. Existing synchronous
configuration remains the default; the async path is explicit and experimental.

The optional atomic worker was evaluated first. It samples in the execution RPC, avoiding a second
synchronous sampling RPC, while preserving the same scheduling and mathematical path. Its negative
result remains in the evidence and is not presented as the performance fix.

## Fixed recipe and experiment design

Actual `npu-smi` records show four **910B2** devices with physical IDs **0, 1, 3, 4**. Child visible
selectors are logical `0,1,2,3`. The model is Qwen3.5-35B-A3B, BF16, TP4/EP4, max model/batch length
16384, max sequences eight, prefix cache off, Mamba cache mode none, FULL_DECODE_ONLY graphs with
capture sizes 1/2/4/8, static kernels off, and HCCL_DETERMINISTIC=true. Native scheduler and
compilation options differ only where explicitly indicated by the role.

Two unchanged synthetic workloads are used:

- Decode-heavy: eight P128/D512 requests.
- Long-P ingress: four P128/D512 seeds; after each seed actually returns token32, enqueue one
  P12000/D128 request.

Each fresh process runs one excluded warmup of each workload, followed by two measured windows per
workload. The original prompt token arrays, fixed output lengths, greedy sampling/seed42, ignore-EOS
setting and timing functions are unchanged. Actual DELTA outputs and timestamps are retained; no
profiler or per-step metadata RPC runs inside performance windows. Per-rank worker/backend readbacks
and queue drain checks occur outside those windows.

Each process contributes a mean of two windows. Two fresh processes each are available for main,
async candidate and native defaults; atomic and the two native factor probes each have one process.
These are descriptive observations, not confidence intervals or a statistically established
equivalence margin. Tokens from the same request/step are correlated and are not independent
samples. This is an offline actor workload, not saturated HTTP service capacity/Goodput.

## Source and role identity

| role                                    | plugin checkout commit used for provenance | async | npugraphEx | scheduling/worker path                     |
| --------------------------------------- | ------------------------------------------ | ----- | ---------- | ------------------------------------------ |
| Main, two processes                     | `6fd8f0de9968357e40a90e185b325f02052039f9` | false | false      | existing decoupled/grouped-P               |
| Atomic diagnostic, one process          | `942af268eccfdfabd2e56282d8e69e4f972b4ad1` | false | false      | single execute/sample RPC                  |
| Async candidate, two processes          | `880173922196f07e96b5ae43fee20335be02efa2` | true  | false      | explicit async scheduler/worker, grouped-P |
| Native default reference, two processes | `6fd8f0de9968357e40a90e185b325f02052039f9` | true  | true       | ordinary native scheduler and NPUWorker    |
| Native synchronous factor probe         | `6fd8f0de9968357e40a90e185b325f02052039f9` | false | true       | ordinary native scheduler and NPUWorker    |
| Native fusion factor probe              | `6fd8f0de9968357e40a90e185b325f02052039f9` | true  | false      | ordinary native scheduler and NPUWorker    |

Native roles do not install plugin scheduling or workers; the plugin checkout is recorded solely for
complete environment provenance. Core remains `d0f22d2bda562156e4dbf433ce645e1769b4f804`, Ascend
remains `03766ac696fde5ab1980d80ca0b8543d3580c989`; the existing Ascend build-file edit and actual
package versions are retained in the raw source/environment receipts. Later documentation-only
commits must be distinguished from measured runtime source, with runtime-byte equality checked
before claiming their coverage.

## Completed single-RPC diagnostic

| workload       | main-01 token/s | atomic-01 token/s |
| -------------- | --------------: | ----------------: |
| Decode-heavy   |         329.089 |           326.866 |
| Long-P ingress |         182.053 |           181.935 |

All six windows' complete output arrays and termination states match, including warmups. This pilot
demonstrates no material throughput gain from removing the second RPC alone. It does not justify a
causal claim that every RPC cost is zero.

## Final performance and native factors

The table uses the independently generated `analysis/final-performance-v1`. Each role mean below
averages its two fresh process means; each process mean uses two measured windows. The maximum-gap
range spans the four measured mixed windows, rather than pooling individual tokens as independent
samples.

| metric                     |  current main | async candidate | native default reference |
| -------------------------- | ------------: | --------------: | -----------------------: |
| Decode throughput, token/s |       328.592 |         475.281 |                  481.183 |
| Mixed throughput, token/s  |       182.951 |         260.244 |                  261.305 |
| Long-P TTFT, ms            |       511.465 |         512.833 |                  436.002 |
| Seed output gap P99, ms    |        49.148 |          47.184 |                   16.799 |
| Worst seed gap range, ms   | 61.359–68.063 |   60.837–62.517 |          419.993–425.464 |

Relative to current main, the candidate improves Decode throughput **44.64%** and mixed throughput
**42.25%**. Relative to fresh native defaults, it remains **1.23% lower** on Decode and **0.41%
lower** on mixed throughput. These small remaining differences do not establish either a
statistically significant loss or a throughput win. The substantive result is recovery of most of
the former gap while preserving the streaming-pause benefit.

The maximum observed mixed seed pauses are about **85% shorter** than native. However, long-P TTFT
is **17.62% worse** and seed-gap P99 is **180.88% worse** than native. Maximum pause and P99 measure
different behavior: the plugin spreads long-P interference across several smaller stalls, rather
than one large stall. The maximum-pause benefit must not be reported without these costs.

Every measured process remains visible:

| process                   | Decode token/s | Mixed token/s | Long-P TTFT ms | Seed gap P99 ms | Worst seed gap range ms |
| ------------------------- | -------------: | ------------: | -------------: | --------------: | ----------------------: |
| main-01                   |        329.089 |       182.053 |        521.269 |          50.036 |           67.394–68.063 |
| atomic-01                 |        326.866 |       181.935 |        516.408 |          48.726 |           64.670–71.941 |
| native-sync-npugraphex-01 |        347.679 |       189.944 |        427.047 |          24.404 |         427.082–427.143 |
| native-async-fusion-01    |        477.735 |       259.777 |        436.763 |          18.156 |         421.963–424.034 |
| native-default-01         |        480.970 |       260.714 |        438.196 |          16.798 |         423.260–425.464 |
| async-01                  |        474.587 |       259.911 |        515.607 |          47.201 |           62.449–62.517 |
| async-02                  |        475.975 |       260.577 |        510.060 |          47.167 |           60.837–61.892 |
| native-default-02         |        481.395 |       261.897 |        433.809 |          16.799 |         419.993–420.146 |
| main-02                   |        328.094 |       183.848 |        501.660 |          48.260 |           61.359–61.843 |

The native probes support asynchronous scheduling as the larger recovered advantage in this recipe:
synchronous native with npugraphEx reaches 347.679 Decode token/s, whereas default asynchronous
native with the same backend averages 481.183. Native async with the fusion backend reaches 477.735,
close to async with npugraphEx. These factor probes have only one process each; they are diagnostic
observations, not population effect estimates or universal backend rankings. The old main/native
difference alone was not a single-factor experiment.

## Actual output trajectories

The independent first-pair review checks every retained request/token against the saved pilot
comparison. Each run has 39 requests and 18,816 generated tokens, including warmups. Every request
reaches its fixed output count and terminates with `length`; arrivals increase within each request.

For main-01 versus async-01, all three Decode windows match exactly. In all three mixed windows, all
four seed requests' complete 512-token arrays also match. Only the long request differs: its first
108 outputs match, output109 differs, and its final 20 paired positions differ. This pattern is
repeated across the warmup and both measured windows of that process pair.

Both roles enqueue long P after the same actor-observed token32 boundary. The async role returns an
additional already-queued seed Decode token before the long P completes: its long first response
accompanies seed token43, compared with token42 in main. This is consistent with queue drain before
Prefill admission, but the retained external timeline does not prove a unique numerical cause for
the long-request divergence. No matched per-step logits or internal device-start trace was
collected. Full arrays and the independent review are retained; equal lengths alone are not
semantic-quality evidence.

## Validation, failures and lifecycle boundary

All nine hardware performance attempts completed successfully and remain in the selection. No
hardware timing failure was dropped. The separate lifecycle run is not a tenth timing sample.

CPU prototype attempts1-4 exposed and corrected explicit class registration, finished-ID maintenance
expectations, resumed-P/preemption handling, and V1/ native patch initialization issues. They remain
in the CPU history. Attempts1-3 have derived outcome ledgers and tool-transcript stdout only; exact
stdout is not reconstructed. Attempts4-5 retain actual redirected logs. The final focused CPU run
passes 347 tests with two inapplicable non-overlap-mode skips and one assert-rewrite warning caused
by deliberate native-worker preload.

The independent finite hardware lifecycle producer tests queued-D abort with actual same-internal-ID
reuse, partial-P cancellation and worker ACK, and native queued-D preemption followed by
full-history P recomputation. It requires empty engine/scheduler queues, no retained cohort/held
allocations, baseline KV free blocks, and empty worker request state after each case. It uses the
native diagnostic ID-randomization-disable option explicitly, allows at most 2000 steps per case and
900 seconds for the fresh process, retains actual outputs/history, and stops on failure without
retry. The actual `async-lifecycle-01` controller passes, collection completes, shutdown returns,
and all three cases pass:

| diagnostic                                   | engine steps | observed result                                                                  |
| -------------------------------------------- | -----------: | -------------------------------------------------------------------------------- |
| Queued-D abort and same internal ID reuse    |           88 | cancelled D held until retirement; survivor and reused ID complete               |
| Partial-P cancellation                       |           71 | one real cancellation ACK at layer4/40; D continues                              |
| Native queued-D preemption and recomputation |           88 | one preemption, worker notification, drained queue and full-history P resumption |

After each case, free KV blocks return to **3257** of a **3258**-block pool (one null block),
engine/scheduler queues and cohort/held allocations are empty, and all four workers report no
pending execution, no cohort and no cached request rows in either runner. All surviving output
tokens are retained and nonnegative with the expected fixed termination. Per-step instrumentation is
intentional here, so these diagnostic wall times are not performance evidence. An unretired queue
entry establishes ownership, not physical device concurrency.

## Remaining limits and publication

[Quality Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23) remains
unresolved. The preceding task-quality gate had three native-correct losses for raw/group despite a
higher aggregate score. This campaign does not rerun or clear that quality gate. Synthetic
output/length checks, lifecycle success and throughput recovery must not be relabelled as
semantic-quality pass.

Only explicit ordinary scheduler/worker classes are under test. The Manager manifest remains
unchanged; there is no new Manager qualification, installed wheel qualification or default-on
deployment recommendation. CLI parsing checks are CPU preparation and are not API-server or HTTP
tests. No HTTP Goodput, physical operator overlap or Cube/Vector resource-partition claim is made.

The canonical evidence destination is the benchmark Release tag
`evidence-muxpd-scheduling-20261026`. Runtime payloads remain outside Git and the plugin repository.
Packaging, upload and anonymous public-byte/member readback are pending at this draft cutoff. The
final publication must provide the exact Release asset names and archive/member manifests before
claiming public custody.
