# Issue #3: existing-output and source audit

This is a read-only audit of the
[installed-wheel inference and rollback diagnostic](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/benchmarks/results/issue2_inference_rollback_20261019/README.md).
It did not launch a model or rerun the 5,000-request performance replay. The tested Qwen3.5-35B-A3B
W8A8, Ascend 910B3 TP1 service generated only 16 tokens per request.

[`audit.py`](audit.py) joins every enabled-arm cache reuse log event to the client request ID and
cached source ID. It compares the retained decoded output text against the disabled baseline and
confirms the rollback arm matches the baseline. [Machine-readable comparisons](comparisons.json)
retain all 30 baseline and plugin output fragments, source IDs, reuse spans, cosine values, and
equality flags. The original full server logs and outputs remain in the pinned issue #2 source
commit above. The six consumed client-output and server-log files match the SHA-256 entries in that
source report's `SHA256SUMS`.

| Diagnostic                                        |  Result |
| ------------------------------------------------- | ------: |
| Paired requests                                   |      30 |
| Approximate reuse events                          |      24 |
| Skipped prompt tokens on hits                     |  53,704 |
| Hits whose 16-token output text equals baseline   |  4 / 24 |
| Misses whose 16-token output text equals baseline |   6 / 6 |
| Reuse sources from a different workload group     | 16 / 24 |
| Rollback outputs equal baseline                   | 30 / 30 |

This establishes **text divergence in the observed short completions**, not a task-error rate. A
changed 16-token fragment may be a harmless wording change or the beginning of a factual error; the
record cannot decide which. The 5,000 request latency replay used one output token per request. Its
preserved client JSONL records request IDs, timing, token counts and matching flags but omits
decoded output text. Its 4,752 reuse events therefore cannot be graded for semantic quality from the
retained client outputs. The card assignment in that replay was fixed, and native APC was disabled,
so performance claims remain limited to the recorded synthetic prefill workload.

For accuracy-sensitive tasks, disable SimLLM or explicitly select the guarded
`HYBRID_REUSE_MODE=exact` path and validate that mode on the intended host and task. `exact`
requires a complete source prefix and correct recurrent snapshot on Qwen3.5; it was not the mode
measured in this audit. The latency-oriented default `tail_anchor` uses approximate matching,
restores attention KV from a similar source, resets recurrent state and recomputes the target
suffix. High cosine similarity, a matching topic anchor and a 512-token suffix do not prove that a
changed number, negation, subject or instruction inside the skipped span was recomputed correctly.

Reproduce this audit without model weights or hardware:

```bash
python reports/simllm-issue3-output-audit-20261029/audit.py \
  /path/to/vllm-ascend-simllm-hust/benchmarks/results/issue2_inference_rollback_20261019 \
  /tmp/issue3-comparisons.json
```

The next quality experiment should compare complete outputs for paired prompts that change a number,
negation, subject or instruction inside a long common background. Include repeated approximate
source turnover and a native APC arm. For each arm retain request-to-source mapping, hit rate,
skipped token counts, full outputs, task grading and latency/throughput. The current evidence does
not establish those outcomes.
