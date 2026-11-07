# Adaptive Quantized KV: five-dataset applicability candidate

Date: 2026-10-20. Status: **candidate; maintainer review pending**. Tracking:
[AQKV #10](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/10) and
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). The machine-readable
record is [applicability.json](applicability.json). This planning report adds no measured cell,
quality score or optimization claim.

## Scope and decision rule

The target is an executing AQKV provider for cached multi-token attention over paged INT8 KV, not
generic BF16 attention or a token-count compression rule. Status describes readiness to evaluate
this target end to end on each dataset. A missing execution contract, qualified runtime or
observed-path coverage is `blocked`. `not-exercised` requires a concrete frozen configuration with a
justified exclusion or trustworthy absence observation; missing evidence is not zero activation.
`applicable` requires a traceable supported path, not proof of speedup.

Sources were checked at benchmark commit `fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca`. Five names and
source revisions come from the
[pinned program](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca/leaderboard-data/dataset-validation/dataset_program_v1.json).
Its status descriptions are older snapshots: the subsequent MMLU-Pro, HLE and FrontierScience
development reports exist. They do not supply an AQKV B0/B1 pair.

## Five-dataset matrix

Every row also depends on the common runtime and precision blockers below. Different rows do not
imply that a complete shared evaluator is absent or that another MOD's result is ours.

| Dataset            | AQKV status | Specific reason and next evidence                                                                                                                                                                                                                                       | Primary task metric                   |
| ------------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| MMLU-Pro           | `blocked`   | A shared scorer and full development report exist. Freeze the AQKV task/runtime pair and observe actual C8 dispatches, including batch-induced chunks. Short prompts alone do not rule out continuing prefill.                                                          | Accuracy under the frozen scorer      |
| HLE-Verified       | `blocked`   | A Gold/Revision development report exists with same-model substitute judging. Select the applicable subset, rights, multimodal and judge contract for this evaluation, then verify the qualified C8 path. Long-input rarity is not an exclusion proof.                  | Verified score on the named subset    |
| SWE-bench-Pro      | `blocked`   | Confirm executable task/repository/image mapping, a fresh sandbox with model connectivity, agent/tools/budget/regrader and task rights. Growing agent history is a hypothesis until target calls are observed.                                                          | Fresh-sandbox resolved rate           |
| FrontierScience    | `blocked`   | The two-track development report exists, with an explicitly unpassed strict-final calibration and separately authorized extraction policy. Decide the appropriate frozen judge/scoring contract and verify C8 execution; do not inherit that exception or those scores. | Separate Olympiad and Research scores |
| Terminal-Bench 2.1 | `blocked`   | Confirm Harbor image digests, fresh-sandbox/verifier execution, model connectivity, task rights and agent/tool budgets; then observe target calls. A successful HTTP request alone is not a resolved task.                                                              | Harbor task resolution / pass@1       |

Each row records source revisions, blocker IDs, responsible roles and the exact next action in the
JSON. Unassigned dependency contacts remain null; naming a coordination role is not a claim that it
has accepted the work.

## Common blockers and scope boundaries

1. **Formal precision scope.** The
   [active V5.5 declaration](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca/docs/ACCEPTANCE_V5_5.md)
   and its linked semantic mirror retain the mandatory BF16 line. Mirror table appendix 2,
   `kv_cache_dtype`, requires requested=auto and effective=baseline dtype, rejecting a mismatch.
   Consequently a compliant BF16-KV configuration does not select this INT8 target. This is a
   configuration-level scope finding, not five dataset executions. The separately identified
   canonical final binary remains unverified by this report. INT8 admission needs a traceable
   scope/configuration decision; absence of an answer is not an exception.
1. **Internal midterm is separate.** The pinned
   [internal v1 protocol](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca/reports/vllm-hust-midterm-20261019/INTERNAL-ACCEPTANCE-v1.md)
   selects the best measured eligible B1 per workload/metric and permits recorded runtime/MOD
   differences while requiring the same precision. It does not explicitly admit INT8 KV.
   Interpretation of that requirement remains unconfirmed; this report neither substitutes midterm
   thresholds for V5.5 nor requests such a decision by silently changing a profile.
1. **Qualified execution is missing.** The public
   [manifest at 9f9844e](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/blob/9f9844ec7d73bdd627edf3930f3ec6cb5ef5f325/src/vllm_ascend_adaptive_quantized_kv/vllm-hust-extension-v0.3.json)
   is `import_only` with empty activation entry points.
   [Host PR #35](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/35), checked at
   `4f2de34002f1af09463ce8dc8e06bacaaac25dae`, is open Draft, not accepted runtime delivery. A
   revision-bound calibrated C8 profile/loading route and Manager/worker/graph/correctness/
   fallback/cleanup qualification remain prerequisites under the
   [recorded restoration conditions](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/1#issuecomment-5856268495).
1. **The bounded direct-call candidate was rejected.** The
   [published 2026-10-18 failure](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/1#issuecomment-6244057335)
   records native V3 rejection of the frozen multi-query five-dimensional NZ call. Direct output and
   timing are absent. It does not disprove every INT8 representation, but changing layout,
   quantization or kernel requires another reviewed contract; a shape-only view is not a fix.

## Existing shared progress, without borrowing results

- [MMLU-Pro development precheck](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca/reports/mmlu-pro-qwen35-b0-precheck-20261012/README.md)
  contains task accuracy and a separate serving run; it has no AQKV treatment pair.
- [HLE-Verified report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca/reports/hle-verified-qwen35-20261015/README.md)
  separates Gold/Revision scoring from four serving workloads and retains unresolved grading.
- [FrontierScience report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/fc0a09b56d3e7eb6438afc76d2ddaf9818b1deca/reports/frontierscience-qwen35-b0-precheck-20261013/README.md)
  retains two tracks, judge limitations and its distinct serving workload.
- The
  [2026-10-19 #254 follow-up](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254#issuecomment-6266705600)
  reports another MOD's still-missing calibrated judge and agent backends. These are coordination
  inputs, not a verified assignment or proof of our own environment's current contents.

No per-dataset C8 execution receipt or matched AQKV quality/performance pair is supplied here.
Scores, paired telemetry and unknown measurements remain null, not zero. A BF16 path observation
cannot promote a C8 cell. There is no evidence-based frequency estimate for our target across the
five datasets. Do not inherit KVCompress's 9216-token threshold, pad tasks, remove short cases,
change judge extraction or reset agent budgets to manufacture activation.

## Review and eventual execution

@XilingGao owns this matrix and extension-side evidence. Host maintainers review the reachable
carrier; @ShuhaoZhangTony is the previously recorded technical decision contact. Benchmark #254
coordinates the named dataset/evaluator/resource roles; specific unassigned owners must be
confirmed.

Review the status semantics, scope distinction and assignments before execution. If a row is
unblocked, freeze exact tasks, model/tokenizer/runtime/image identities, inputs, topology, scoring,
agent/tools/budgets and the permitted B0/B1 differences; then retain path coverage, worker and graph
qualification, raw outcomes/failures, quality first and supporting telemetry. Preserve positive,
zero, negative, blocked and not-exercised results. Publish evidence here first and back-link
#10/#254. Neither this candidate nor its eventual merge checks the Issue #10 reviewed box on its
own.

Verify this planning package from the repository root:

```bash
cd reports/aqkv-five-dataset-applicability-20261020
sha256sum -c SHA256SUMS
python -m json.tool applicability.json > /dev/null
```
