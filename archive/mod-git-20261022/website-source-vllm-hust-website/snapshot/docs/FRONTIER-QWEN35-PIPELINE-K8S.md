# Qwen3.5-35B-A3B: Pipeline Microbatch container control

This campaign evaluates Pipeline Microbatch against a fresh native PP2×TP2 control in the
user-provided Kubernetes container. Both arms participate on four Ascend 910B2 chips. The pod
exposes eight chips; only the four participating workers enter the throughput denominator.

The model is Qwen3.5-35B-A3B at verified ModelScope revision
`712cf74392b05026a6db2bf213d343747d1f6d45`. All 22 model files match the revision's SHA256 and size
manifest. The public SWE prefix-reuse workload, tokenizer fingerprint and compiled token budgets
match the preceding [container campaign](FRONTIER-QWEN35-MODS-K8S.md).

## Matched configuration

Both arms use TP2, PP2, BF16, 262144-token capacity, 16 server slots, a 4096-token scheduling
budget, APC, Mamba `align`, asynchronous scheduling, natural MTP with two draft tokens, and
FULL_AND_PIECEWISE graph captures 3/6/12/24/48. KV capacity is explicitly 26038239232 bytes per
chip. Thinking is enabled for SWE requests with greedy sampling. There is no host KV offload.

The Pipeline treatment uses `vllm_hust_pipeline_microbatch.policy.PipelineMicrobatchPolicy` at
revision `a15a22961a0e4858da74a0ab806575c82cb254e6`, with two microbatches and measured rank-local
cost profiles. Native uses the builtin scheduler on the same common runtime. The previous two-chip
TP2-only control is not the denominator for a Pipeline speedup.

## Common runtime qualification

The core base is `752a3a504485790a2e8491cacbb35c137339ad34`; the Ascend base is
`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`. Both arms share the neutral batch admission API backport
and PP hybrid-MTP correctness changes:

- Preserve real MTP acceptance feedback by request identity across empty and disjoint batches,
  including earlier pipeline stages.
- Retire a decode fence only with the scheduling output that established it; older intermediate
  prefill cannot release a later final-prefill fence.
- Scatter actual scheduled draft IDs on earlier stages when a continuous batch bypasses the CPU
  input copy.
- Reconcile confirmed, already-retired PP output counts without applying the generic optimistic
  rejection correction twice.

These changes are common integration fixes, not attributed to the MOD. Asynchronous execution, real
accept/reject sampling, MTP, APC and graphs remain enabled. Twenty focused CPU regression checks and
the runtime repositories' complete formatting/static checks passed. Exact executed source hashes and
overlays accompany the measurements; base revision IDs alone do not describe the patched runtime.
These receipts qualify this experimental capsule; they do not certify the standalone package against
arbitrary vLLM releases.

Each measured arm must pass 26 exact marker-retrieval probes: cold/warm at 1024, 8192, 32768, 131072
and 262080 input tokens, followed by 16 concurrent requests. This is a bounded retrieval
qualification, not a general answer-quality or SWE problem-solving certification. Failed diagnostic
attempts remain excluded from performance data.

## Calibration and measurement protocol

Calibration is separate from performance. Rank-local NPU event intervals cover actual execute and
sample calls, including communication and synchronization gaps. Four rank profiles are required. The
fitter uses a chronological 70/30 holdout, with predeclared median relative error at most 35% and
P90 at most 75%, and at least 40 decode rows per rank. With 20 layers fixed per stage, collinear
layer terms are structurally zero; nonnegative request-count, total-context and intercept terms are
fitted to observations.

The final calibration has 2420 valid decode observations per rank, with 726 held out. Median
relative errors are 3.589%, 3.561%, 3.399% and 3.428%; P90 errors are 7.311%, 7.314%, 8.005% and
7.894%. All four profiles pass the predeclared checks. These errors describe held-out observed
timing intervals, not the accuracy of generated answers or proof of an optimization gain.

Performance launchers disable calibration and debug instrumentation. Each arm runs a separate
C2/60-second SWE protocol/prefix-reuse check, followed by C4 and C16 with 900-second measurement
windows. Sessions use fresh salts. Actual streamed tokens inside the window contribute to
throughput; drain tokens do not. Prefix hits and zero request failures are required. The candidate
must show actual policy calls, admissions and completions, with no aborts, failures, invalid
admissions or builtin fallbacks.

The serial pair is supervised in the container with no automatic restarts. A failed gate prevents
measurement; failure of the native arm prevents starting the candidate. Each arm stops its owned
service and verifies that participating devices have no remaining process owners. The pair is one
observation per cell, not peak-capacity or repeatability certification.

## Completed observations

| Arm                 | Concurrency | Output token/s/chip | Decode P90 token/s | Request errors | Policy admissions |
| ------------------- | ----------: | ------------------: | -----------------: | -------------: | ----------------: |
| Native              |           4 |              18.970 |             26.866 |              0 |                 — |
| Pipeline Microbatch |           4 |              19.659 |             28.372 |              0 |             13197 |
| Native              |          16 |              24.799 |              8.797 |              0 |                 — |
| Pipeline Microbatch |          16 |              24.563 |              8.554 |              0 |              9036 |

The observed throughput differences are +3.63% at C4 and -0.95% at C16. These are individual matched
observations, not evidence of a consistent or statistically established speedup. Both positive and
negative differences are retained.

Pipeline actually executed: C4 recorded 13416 calls and 13197 admissions/completions; C16 recorded
9781 calls and 9036 admissions/completions. Both windows have zero aborts, failures, invalid
admissions and builtin fallbacks. These counters span the measured window and its drain; they
demonstrate activity rather than isolate an optimization gain.

All four windows passed protocol validation with zero request errors. All raw requests (Native
C4/C16: 130/228; Pipeline C4/C16: 132/230) have successful responses and exactly the prescribed
number of actual, nonnegative output token IDs. Both arms passed all 26 retrieval probes and their
separate C2/60-second checks. Every measured window has positive prefix-cache-hit deltas. Both
supervised services stopped with exit status 0 and no remaining device owners.

Drain times were 92.69/650.44 seconds for Native C4/C16 and 65.49/591.92 seconds for Pipeline
C4/C16. Long fixed-budget requests explain the extended drain. Its tokens are excluded from the
fixed 900-second throughput denominator. The PP2 results are much slower per chip than the prior
two-chip observations; these points do not establish PP2 as a better deployment topology.

Full original records are retained under `phase2/receipts/{nativepp,pipelinepp}-measured-r1` and
`phase2/receipts/paired-measurement-r2` in the campaign archive. The public SWE evidence rows carry
raw-request SHA256 values, the client summaries, qualification receipts, prefix hit deltas and
policy counters. No calibration or failed diagnostic attempt became a point.

## Reproduction sources

The
[committed experiment capsule](https://github.com/vLLM-HUST/vllm-hust-dev-hub/tree/8eb639c/scripts/frontier_pipeline)
contains exact source overlays, per-file hashes, launchers, measured rank profiles, fit validation,
controller tests and the guarded website importer. Apply the compressed overlays to their recorded
base revisions while retaining the complete official Ascend wheel and its seven shared libraries.
The capsule uses the container's CANN 9.1.0, torch 2.10.0+cpu and torch-npu 2.10.0.post4.

The
[Ascend development commit](https://github.com/vLLM-HUST/vllm-ascend-hust/commit/66350e7b7d8ec68ad68a14fb841678054f2c0f98)
includes the common PP fixes and regression checks. The
[core development commit](https://github.com/vLLM-HUST/vllm-hust/commit/77e6192e7ce7947050f88306d0ae0e85ec864b75)
includes the neutral batch API backport plus post-test formatting and policy metric variable
renaming; exact executed core bytes remain defined by the capsule overlay and manifest, not that
later formatting variant.
