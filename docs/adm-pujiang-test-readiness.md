# ADM × Pujiang: next-test handoff (no dispatch)

Status: **BLOCKED for ADM evaluation**. This document stages a review, not a
new measurement, official baseline change, or permission to run devices.
Sources: [ADM #5](https://github.com/vLLM-HUST/ascend-distributed-metadata/issues/5),
[ADM #7](https://github.com/vLLM-HUST/ascend-distributed-metadata/issues/7),
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254),
[benchmark #403](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/403).
Follow the latest issue and delivery contract if they change.

## Keep comparisons separate

The organization B0/B1 comparison retains its existing definitions and artifacts.
The V5.5 Qwen3.5-35B-A3B BF16, two 910B2, TP2 configuration normally has DP1,
which does not exercise ADM's DP metadata synchronization. Do not label its
score or speed an ADM optimization result.

A distinct ADM OFF/ON comparison must keep the actual source, image, precision,
workload, TP/DP topology, settings and scorer identical, changing only ADM
activation. TP2/DP2 requires at least four exclusively allocated devices;
model fit, topology, runtime support and lease remain to be verified. The current
ADM qualification is a 0.11-family host with a strict method fingerprint, so
the organization 0.18 B0 cannot be presumed compatible. Obtain actual image
identities and an independently reviewed compatible adapter before planning a run;
do not bypass the guard or change B0.

## Reuse inventory and remaining gates

| Dataset | Reuse | Next gate |
| --- | --- | --- |
| MMLU-Pro | [Original-row integration runner](../scripts/run_mmlu_pro_original_samples.py), with pinned checksums and lm-eval 0.4.11 | Its three-row sample is not the fixed-200 or full-set scorer. Freeze the task-ID manifest and qualified scoring pipeline. |
| HLE-Verified | [Native 0.18 completed report](../reports/hle-native018-completed-20261026/README.md) and existing solver/judge/summary workflow | Confirm private source access, frozen Gold/Revision inputs, image, judge, unresolved denominators and scoring ownership. |
| SWE-bench-Pro | Historical sessions and the existing trace program in #403 | #403 currently admits **zero new measurements**; plan review first. No agents, queue, controls, or replay dispatch. |
| FrontierScience | [Frozen input validator](../src/vllm_hust_benchmark/frontierscience_inputs.py) and existing 0.18 paired provenance | Freeze Olympiad/Research inputs and judge provenance; the historical substitute judge is not an official leaderboard result. |
| Terminal-Bench 2.1 | [Harbor contract](../reports/pyramidkv-terminal-harbor-contract-20261009/README.md), historical ARM64 adaptation and #403 traces | Plan review first; no Harbor or agent dispatch. Pin images and controls. Do not assume a new AMD64 backend is needed. |

All five are **not-exercised** on the specified DP1 layout and **blocked** for
an ADM DP2/DP4 ablation until compatibility, resources, worker-path evidence
and the respective scoring gates are satisfied. Existing dataset adapters or a
working endpoint do not prove the ADM fast path. Actual task quality, failure
counts, TTFT, TPOT, ITL and throughput/goodput must retain their prescribed
quality denominator and scorer identity. Replay throughput cannot be used as a
task score. The historical +0.80% ADM result is vs a prior MOD, not native.

## Handoff checklist

1. Benchmark owner freezes task IDs/order, inputs or private trace hashes,
   model/tokenizer/template, seeds, generation parameters, tool images, scorer,
   task-quality denominator and unresolved/failure handling.
2. ADM and runtime owners verify exact actual source/image and fingerprint,
   version guard, TP/DP support, graph mode, fallback cases and the device lease.
   Integrate capture of before/after snapshots **inside each owning worker**;
   bind launch, host, PID/start identity, rank and window. This capture is not
   implemented in the benchmark runner yet. Missing snapshots mean UNKNOWN.
3. Preregister tuned matched OFF/ON commands, warmup and arrival policy, SLO,
   negative control, fresh-service cross-order repeats, measurement and cleanup;
   estimate observer overhead separately and retain private raw artifacts.
4. Budget four devices for each TP2/DP2 window once independently qualified;
   four ABBA windows of T hours each consume 16 × T card-hours per workload,
   excluding setup and judge costs. T and judge costs remain unknown. Do not
   execute SWE/Terminal #403's proposed four fragments (34 total requests,
   139264 output-token cap, 6-hour stop) before that proposal is reviewed; it is
   a B0/B1 proposal, not authorization for ADM OFF/ON.

Owner split: benchmark evaluation owns inputs, runner and scorer; runtime/ADM
owns source compatibility and worker-path evidence; storage and operators own
model access and exclusive device lease; reviewers approve each experiment.
Keep historical raw data immutable and publish only sanitized receipts.
