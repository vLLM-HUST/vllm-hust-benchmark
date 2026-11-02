# Qualification after the standalone release

This plan separates candidate-package acceptance from measured effects. It is owned by @Irisuko and
tracked through [#1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1) and
[#8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8). Execution plans and measured
originals belong in `vllm-hust-benchmark`; the PyramidKV repository owns plugin code, integration
and links to those records.

## Issue #1: current-version quality, net cost and capacity

Freeze the published `0.4.0rc1` wheel hash, source tree, model/tokenizer identities, vLLM `d0f22d2`,
Ascend `03766ac`, native operators, CANN 9.1, actual 910B2 device allocation, BF16/TP2, APC, MTP2,
async/chunked prefill, 32K context and FULL_AND_PIECEWISE before requests. Use aligned 1368/beta-2
and batched metadata for the effect candidate: set `VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA=1` in
both arms and vary only compression activation. Linked historical protocols supply workload and
measurement rules; their old shared-adapter source pins and activation variables are superseded by
this release contract. Do not change budgets or select a metadata path after seeing results.

B0 and B1 install the same wheel and host. Start fresh processes with `PYRAMIDKV_ENABLED=0` versus
`1`; remove old shared activation and competing Manager state. The complete resolved
argv/environment, request hashes and request-matched scheduler/TP acknowledgements prove actual
treatment. Publish all failed, zero and negative outcomes, not only favorable cells.

1. **Quality:** restore the pinned LongBench source/data from the
   [existing protocol](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/main/docs/issue-1-evaluation-protocol.md),
   then use the current `scripts/qwen35/evaluate.py prepare --samples-per-task 100`. Preserve the
   three tasks' development indices 0–49 and holdout 50–99, templates, truncation, output budgets,
   greedy seed and F1 metric. The mean-loss \<=3 and per-task-loss \<=5 gates apply separately to
   both subsets. Run both arms against the same frozen plan; existing shared-adapter outputs remain
   historical comparators.
1. **Service cost:** pre-register separate equal-reservation 8 GiB and 2 GiB series. Each uses
   OFF/ON/ON/OFF fresh lifecycles and the frozen 1K/8K/24K, C1/C4, 128-output workloads and original
   warmup/repetition rules in the
   [net-cost protocol](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/main/docs/benchmark/pyramidkv-net-cost.md).
   The pressure primary cell is 24K/C4, fixed before measurement. Report throughput, TTFT, E2E,
   TPOT, preemptions, cache reuse, MTP, KV ownership and actual HBM together.
1. **Capacity:** first measure admission, completion, latency and preemption at C1/C2/C4 within the
   qualified max-num-seqs=4 profile and at the same reserved budgets in both arms. This establishes
   tested operating points only. Any higher-concurrency sweep must first qualify the larger runner
   buffers and capture sizes in both arms, then freeze its latency/failure criteria. Queued HTTP
   requests and lower KV-pool occupancy do not prove increased resident request capacity or lower
   device HBM.

Preparation deliverables are a frozen case manifest, exact launch files, resource window, total
request/time budget, failure/stop policy and a canonical output destination. The current evaluation
runner can execute the 409-case quality-plus-serving plan per arm. A performance-only selection of
its existing groups and a capacity sweep need explicit runner support/tests before their plans are
frozen; do not invent undocumented command-line flags. No such effect series is claimed completed by
the release smoke tests.

## Issue #8: execution and judge handoff

The five-dataset input audits and SWE V1 / Terminal ARM64 preparation delivered by
[benchmark #546](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/546) are reusable. Rebind
their model/runtime/activation fields to the independent release; old V2/AMD64 audits keep their
original identity. The latest
[#403 coordination scope](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/403) is a
historical-trace plan under review, not an assigned live-Agent window for this plugin. Obtain a
concrete handoff through the existing owners.

| Handoff                   | Owner to confirm                                      | Acceptance material                                                                                                                                                     |
| ------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ARM CPU execution         | profiling / resource coordinator; @Irisuko integrates | Workspace endpoint/access, CPU/RAM/disk/network limits, persistent directory, availability/expiry and task isolation                                                    |
| Images and model route    | storage with CPU owner                                | Selected SWE V1 and derived Terminal image digests, transport/build provenance, actual delivery, task budgets and backend-to-model connectivity                         |
| Execution window          | CPU and NPU owners                                    | Exact resource window, model endpoint/auth via private channel, fresh-sandbox controls, verifier and tool/retry policy                                                  |
| HLE/FrontierScience judge | canonical evaluation owner                            | Actual judge model/revision/endpoint, access and budget, fixed grading instructions/parser, calibration set identity and an independently checked semantic pass receipt |

The judge handoff must preserve the user's calibrated-judge requirement. Freeze mandatory
correct/incorrect/equivalent/refusal/malformed cases and their expected behavior before calibration;
all mandatory checks must pass, and judge failures remain visible. Substitute or uncalibrated
self-judging scores from other MODs are not PyramidKV evidence. Do not post credentials or
restricted reference answers in issues.

HLE full image-bearing subsets require an explicitly qualified image-aware execution path; this
release qualifies text only. A text-only subset must be separately named. FrontierScience Olympiad
and Research retain separate scores. MMLU-Pro and short FrontierScience prompts remain not-exercised
if they do not cross the original compression threshold; do not pad tasks or lower admission to
manufacture an optimization. SWE/Terminal calls must preflight against the qualified context window
without silently truncating tasks or tool histories.

Resource and judge tasks are complete only when the handoff is usable and its checks pass. A posted
request or prepared job file is not a resource allocation, calibration result or task-quality score.
Canonical evidence precedes website synchronization and any closure of the parent issues.
