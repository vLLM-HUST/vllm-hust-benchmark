# kvcompress-ascend: five-dataset applicability candidate

This is the canonical planning and blocker record for
[MOD issue #17](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/17) under
[dataset program #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). The
machine-readable matrix is [`applicability.json`](applicability.json). Maintainer review is pending.
It contains no task-quality result, B0/B1 measurement, or optimization claim.

The five names are the Pujiang-specified scope. Their source snapshots were revision- and
checksum-frozen in benchmark [#257](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/257); that
does not make the evaluators or sandboxes executable. This environment does not mount
`/root/vllm-hust-eval-data/pujiang-five`. The existing 21-dataset B0 workbook contains only MMLU-Pro
serving telemetry and does not provide task accuracy.

| Dataset            | MOD status      | Current reason                                                                                                                                                                                                      | Primary task metric                 |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| MMLU-Pro           | `not-exercised` | The independently audited reference five-shot Qwen prompts have at most 2,910 tokens. With the candidate 2,048-token output budget, total possible length is 4,958, below this MOD profile's 9,216-token threshold. | Accuracy under pinned scorer        |
| HLE-Verified       | `blocked`       | Rights, chosen subset, multimodal handling, judge and runtime contract pending.                                                                                                                                     | Pinned verified score               |
| SWE-bench-Pro      | `blocked`       | Rights, task/repository/image mapping and fresh sandbox/regrader pending.                                                                                                                                           | Fresh-sandbox resolved rate         |
| FrontierScience    | `blocked`       | Olympiad and Research graders and final output budget pending. Candidate input audit alone does not settle live compression.                                                                                        | Separate Olympiad / Research scores |
| Terminal-Bench 2.1 | `blocked`       | Harbor image digests, sandbox, agent/scorer contract and task rights pending.                                                                                                                                       | Harbor task resolution              |

The MMLU-Pro bound uses the
[12,032-task length audit](../pyramidkv-five-dataset-adaptation-20261009/length-audit.json) and
[candidate generation contract](../pyramidkv-five-dataset-adaptation-20261009/plan/contract.json),
which were frozen for another MOD. It is an applicability bound for exactly that rendering and
budget, not a kvcompress run or a whole-program task score. This MOD's existing
[Qwen3.5 V3 configuration](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/master/.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json)
sets `kv_budget=8192`, `recompute_window=1024`, and `min_output_tokens_for_compression=512`. Its
code computes the threshold as budget plus window. If the prompt, generation budget, or method
config changes, the MOD owner must audit the exact frozen tokenized manifest again. Eligible token
budgets require live B1 scheduler commits and worker acknowledgements before any compression
assertion.

The
[FrontierScience candidate input audit](../pyramidkv-frontierscience-inputs-20261009/summary.json)
measured 100 Olympiad and 60 Research problem-only prompts. It did not freeze the two grading
contracts or solver output budget. The other blocked rows inherit their execution gates from the
canonical [dataset program](../../leaderboard-data/dataset-validation/dataset_program_v1.json).
Owners and concrete next steps are recorded per row in the JSON. No dataset has been marked
`applicable` without a frozen final workload that can reach the control path.

## Publication gate

Keep this candidate separate from measured cells. Before any future optimization claim, freeze task
IDs, scorer, model/runtime/container/image digests, topology, hardware, agent/tools and budgets; run
matched B0/B1 on the same manifest; retain per-task outputs, failures, task-quality scores,
telemetry, server commit/ack logs and SHA-256 hashes. Publish positive, zero and negative outcomes
here first, then synchronize the website. The MOD owner is @Jiawan23; benchmark #254 maintainers
coordinate dataset rights, evaluators and sandboxes. This report does not close either issue.
