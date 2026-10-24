# Next work: execute the blind-attribution contract

## Current status

Profiler PR #17, vLLM-HUST PR #236, and vLLM-Ascend-HUST PR #216 are merged.
The runtime observation path and Issue #19 evidence harness are available.
Development G1--G6 cases remain non-blind inputs and do not prove M0.

Issue #19's corrected paired M0 is an evidence-valid, scoped mechanism
`NO_GO`: resource pathology occurred in 1/10 repetitions and matched latency
pathology in 0/10, below the preregistered 8/10 gate. Preserve the result and
sanitized evidence. Do not rerun it, change its workload/capacity/gate, or
implement the rejected reconciliation treatment.

The durable boundary is:

> Issue #19 mechanism `NO_GO` is not Request Lifecycle topic `NO_GO`.

## Immediate local work

1. Keep the Issue #19 scoped decision boundary unchanged.
2. Review and freeze `docs/blind_attribution_contract.md` before any opaque
   case or ground truth is revealed.
3. Use the pinned `third_party/traceloom` structured SQLite evidence. Do not
   add a second generic execution-tree/profiler-row model.
4. Implement only the adapter, coarse-prefill lifecycle links, five-arm
   renderer, scorer, and development fixtures required by the contract.
5. Run CPU tests and package checks before any hardware capture.

## Blind execution order

1. The custody side creates two fresh positive cases and one valid
   insufficient-evidence/no-single-root-cause negative case.
2. The diagnosis side receives only the normalized case bundles and fixed time
   budget, then locks top-1/top-3, confidence, residual, abstention,
   counterfactual proposal, and time-to-localize before reveal.
3. Compare aggregate metrics, flat timers, timed raw-log/manual diagnosis,
   lifecycle spans with causal rules disabled, and the full lifecycle DAG.
4. Reveal the oracle and score every method and case.
5. Run one rank-one counterfactual for every non-abstained top-1.
6. Run the separate four-workload trace off/on overhead matrix.
7. Update the claim ledger with success, tie, failure, or downgrade. Do not
   alter thresholds or rescore revealed cases as blind successes.

## Claim and performance boundary

A longest stage is localization, and trace completeness is evidence validity;
neither is causal proof. The project remains `NOT_M0_PROVEN` until the genuine
blind matrix demonstrates an advantage over the strongest non-DAG baseline and
the counterfactuals validate every non-abstained top-1.

Any later runtime performance implementation claim must separately follow
<https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/95>. The blind
diagnosis comparison and Issue #19 result do not replace benchmark base/head
performance evidence.
