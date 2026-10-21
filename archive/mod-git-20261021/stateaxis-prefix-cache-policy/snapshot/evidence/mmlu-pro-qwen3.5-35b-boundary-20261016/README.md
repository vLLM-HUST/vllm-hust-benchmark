# MMLU-Pro Qwen3.5-35B-A3B applicability boundary

This directory records a successful OFF/ON/OFF TP2 smoke bracket and an
offline eligibility audit of all 200 official materialized prompts. It is a
real-online negative applicability result: the workload cannot exercise a
complete prefix-cache page for this hybrid model/runtime combination.

The leading OFF arm was cold and must not be used as the sole comparator. The
ON arm looked faster than that cold arm, but recorded zero local-cache hits and
was slower than the warmed trailing OFF arm. No positive or negative throughput
effect is attributed to the MOD from this bracket.

Raw logs and retained failures remain in the external campaign evidence roots
listed in `RESULT.json`. Nothing was deleted or overwritten.

Reproduce the offline gate with the recorded dataset and model paths:

```bash
python analyze_prefix_eligibility.py \
  --dataset /root/stateaxis-workloads/plans/mmlu-pro-qwen35-b0-1bfd58a4a208/bench-prompts.200.jsonl \
  --model /models/modelscope_cache/Qwen/Qwen3___5-35B-A3B \
  --page-tokens 2048
```

The output must match `ELIGIBILITY.json`.
