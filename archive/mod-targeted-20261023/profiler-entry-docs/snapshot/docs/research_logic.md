# Seven-Step Research Logic

## 1. Problem

How can an LLM serving system turn per-request lifecycle traces into causal
bottleneck attribution rather than a pile of correlated logs?

## 2. Importance

Optimizations often fail because the apparent bottleneck is not the causal
bottleneck. Aggregate TTFT/TPOT and scattered logs cannot reliably distinguish
queueing, prefill, KV pressure, decode jitter, streaming stalls, and cleanup
delays.

## 3. Existing Gap

Serving systems expose metrics and logs, but they rarely produce request-level
causal evidence packets that connect events across tokenizer, scheduler, KV
manager, executor, and streamer.

## 4. Key Idea

Represent each request as an ordered lifecycle trace and classify the dominant
bottleneck chain using explicit event spans and controlled fault-injection
ground truth.

The current implementation has two trace layers: client-observed proxy anchors
and internal runtime hook events. Claims must name the layer they use, then
show when internal spans confirm or overturn the client-visible hypothesis.

## 5. Feasibility

The first implementation runs on NPU6 with optional runtime hooks in the pinned
`third_party/vllm-hust` submodule. The smoke gate already shows complete
runtime chains and bounded latency overhead; the next feasibility question is
whether known live faults are attributed correctly.

## 6. Evaluation

Baselines: raw logs, simple stage timers, manual diagnosis, and profiler with
causal rules disabled.

Metrics: attribution accuracy, false positives, false negatives, tracing
overhead, event coverage, memory overhead, TPOT/TTFT deltas, and
time-to-root-cause.

## 7. Takeaway

This work shows that request lifecycle traces can become auditable causal
evidence for LLM serving optimization. To become a top-tier systems paper, it
must show that this evidence changes the next optimization decision compared
with raw logs or aggregate timers.
