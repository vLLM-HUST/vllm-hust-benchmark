# TOP_TIER_PUSH_PLAN

## Target Claim

LLM serving optimization needs causal request lifecycle evidence, not only
aggregate TTFT/TPOT numbers and scattered runtime logs.

## Current State

Follow `docs/next_agent_task.md` for the current NPU6 execution checklist. The
first evidence gate is complete: the repo has a stable trace schema,
synthetic-fault coverage, client-observed NPU6 traces, slow-stream
client-visible diagnosis, and a pinned vLLM-HUST runtime hook carrier.

The current runtime-hook gate is
`.benchmarks/results/npu6_runtime_hook_pair_plan/summary.json`: hook-enabled
and hook-disabled source runs are paired, every observed request chain has all
nine lifecycle stages, and small-smoke latency overhead is bounded. This is the
starting point for live fault attribution, not the final paper result.

## Next Push

1. Run controlled NPU6 fault attribution one class at a time: slow streaming,
   long-prompt prefill pressure, decode-heavy output, KV-pressure boundary, and
   cleanup stall.
2. Preserve client-observed proxy anchors while collecting internal runtime
   JSONL so each claim can state which layer produced the diagnosis.
3. Compare against raw logs, simple stage timers, and causal rules disabled.
4. Add memory and TPOT overhead for hook-enabled versus hook-disabled modes.
5. Refresh `make top-tier-readiness`, `make paper-assets`, and the paper after
   each valid evidence batch.

## Required Baselines

- raw vLLM/vLLM-HUST logs;
- simple stage timer aggregation;
- manual post-hoc diagnosis;
- profiler with causal rules disabled.

## Claim Discipline

Do not claim a performance optimization unless a policy also changes runtime
behavior. Profiler-only results should be framed as observability, diagnosis,
or artifact contributions.

## Submission Gate

This project is only ASPLOS-ready if causal lifecycle traces correctly
attribute controlled faults, expose a non-obvious real serving bottleneck, and
show why raw logs or simple timers would lead to weaker optimization decisions.
The current repo satisfies the trace-capture readiness gate; the remaining
top-tier gate is live controlled-fault accuracy plus a decision-impact story.
