# Graph Capture Coverage

At a fixed 32-token step budget, expanding [1,2,4] to [1,2,4,8,16,32] reduced median batch
completion by approximately 57.0% in the reference workload. A separate 128-token-budget factorial
campaign measured 38.7%; the two campaigns are distinct.

Four simultaneous prompts of 8/48/96/160 tokens, 16 output tokens each; Qwen3-0.6B FP16, TP1, Ascend
910B2. Ten fresh engine processes and 80 unprofiled batches in the 32-token reference campaign;
corresponding output tokens matched.

Larger capture coverage uses more startup time and graph memory. At budget128 the separate campaign
observed about +1.30 s capture time and +10 MiB graph memory with compiler caches retained. Measure
both startup and service behavior for a new workload.

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

Canonical code repository: https://github.com/vLLM-HUST/vllm-hust-graph-coverage
