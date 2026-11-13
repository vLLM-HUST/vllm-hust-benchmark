# Prefill Admission

Short first-token latency changed from 126.746 to 40.342 ms (68.17% lower); batch completion changed
from 505.131 to 515.347 ms (2.02% higher).

One 2048-token prompt arrives at 0 ms; three 8-token prompts at 50 ms; all request 32 output tokens.
Qwen3-0.6B FP16, TP1, Ascend910B2; five independent process pairs and 60 unprofiled batches.
Corresponding output tokens matched.

This is a latency/throughput tradeoff: batch completion increased by 2.02%. The reserved capacity is
workload-specific. Keep the capture sizes unchanged and compare short-request TTFT with total batch
completion.

The source result digest and process-level results are in [`result.json`](../evidence/result.json).
The frozen workload and independent process count are in
[`protocol.json`](../evidence/protocol.json). These are historical measurements of the exact stock
settings, extracted into this configuration package. The new package is verified on CPU for inert
defaults, context checks, conflict rejection and preservation of all other engine kwargs. The smoke
example is for installation, not a benchmark result.

For device requalification, use five fresh process pairs in alternating AB/BA order, the protocol's
inputs/arrival schedule/warmup/timed-batch counts, and identical model and graph settings. Record
the exact command, engine and package revisions, dirty diff, hardware/software identity, effective
engine configuration, all timings and full token vectors in fresh output directories. Compare
corresponding outputs exactly and report every pair. Acquire the device allocation before starting
any engine.

Canonical code repository: https://github.com/vLLM-HUST/vllm-hust-prefill-admission
