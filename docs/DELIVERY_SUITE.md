# Proposed vLLM-HUST delivery suite

`delivery_suite_registry.json` is a proposal for the next generation of fixed acceptance targets. It is intentionally additive and does not modify, replace, or inherit identity from the repository's current official targets. Promotion into the official target registry is a later review action after the proposal, adapters, model artifacts, and runners are admitted.

## Terms

- **Application scenario**: a user-facing workload class. The suite fixes nine: general QA, code, reasoning, multimodal, KV reuse, structured output, long context, Agent, and AI4Science.
- **Engine capability target**: an L0 or L1 system behavior such as protocol conformance, overload handling, prefill/decode balance, preemption, KV eviction, OOM boundaries, structured decoding, fault recovery, fairness, or soak stability.
- **Evaluation asset**: a typed input to evaluation. `quality_benchmark`, `serving_trace`, `serving_harness`, `microbenchmark`, `enterprise_replay`, and `design_reference` are deliberately distinct.
- **Fixed target**: the resolved model artifact, data revision, runtime profile, client profile, thresholds, and receipt. A scenario name or benchmark name alone is not a fixed target.

The registry keeps nine scenarios but does not require nine unique benchmark frameworks. A scenario can reuse a serving harness or checkpoint when that improves causal attribution.

## Three-layer coverage

| Layer | Purpose | Examples |
| --- | --- | --- |
| L0 | Deployment and protocol conformance | artifact startup, tokenizer/processor/chat template, stream, content parts, tool call, response format, clean restart |
| L1 | Engine mechanism and boundary | open/closed/burst/overload, prefill/decode mix, preemption, KV reuse/eviction, OOM, TP/EP, cancellation, fairness, two-hour soak |
| L2 | Application credibility | the nine application scenarios plus public quality assets and enterprise replay |

L0 and L1 close engine coverage. L2 closes scenario credibility. Passing only L2 does not prove mechanism coverage, and passing only L1 does not prove user-facing quality.

## Model set

The main suite uses six pinned candidate checkpoints for nine scenarios:

| Checkpoint | Scenarios | Rationale |
| --- | --- | --- |
| Qwen3-32B | General QA, KV reuse, structured output | Dense general baseline and clean causal reuse across mechanisms |
| Qwen3-Coder-30B-A3B-Instruct | Code, AI4Science | Code-specialized MoE with affordable repeated deployment |
| DeepSeek-R1-Distill-Qwen-32B | Reasoning | Decode-heavy reasoning without a multi-node 671B model |
| Qwen3-VL-30B-A3B-Instruct | Multimodal | Native image-text pipeline with an official Ascend single-node path |
| Qwen3.6-27B | Long context | Official Ascend 128K path and sufficient HBM headroom for long KV |
| GLM-4.5-Air | Agent | Agent-oriented 106B-A12B MoE representing the enterprise GLM class |

GLM-5 and Kimi K2/K3 are model-class extensions, not main single-node targets. They can be promoted only after a customer-accessible exact artifact and production-equivalent multi-node topology are frozen.

Every main model record pins a full repository commit SHA, tokenizer/processor revision, chat-template source, dtype, TP/DP/EP topology, and maximum model length. `pinned_candidate` means identity is fixed but runtime admission is still required; only an actual load and preflight receipt can promote it to `admitted`.

## Acceptance logic

Formal performance uses five paired restart blocks. Each block randomizes A/B or B/A order, warms for 120 seconds, measures for at least 600 seconds, and obtains at least 1,000 successful online requests when the workload is not task-count bounded. The primary endpoint must meet its scenario-specific minimum effect and its paired-bootstrap 95% confidence interval must exclude zero in the beneficial direction.

All nine scenarios must produce valid runs and pass quality non-inferiority. At least seven of nine primary performance endpoints must pass; general QA, KV reuse, structured output, and long context are mandatory. Global guardrails allow at most 5% regression on secondary metrics, 0.1% load error rate, 0.05 percentage-point error-rate increase, zero silent fallbacks, and zero unclassified failures.

Four configurations establish causality:

1. unmodified upstream stock;
2. vLLM-HUST with target optimizations disabled;
3. vLLM-HUST with the preregistered optimization set enabled;
4. feature-on minus one optimization.

The feature-on versus feature-off pair supports the mechanism claim. Feature-off versus stock exposes integration overhead. Feature-on versus stock is the product-level result. Each claimed optimization must also pass one preregistered L1 target and its minus-one ablation.

Inspect the proposal with:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.delivery_suite
PYTHONPATH=src python -m vllm_hust_benchmark.delivery_suite --workload-id long_context
```

No command in this proposal registers a current official target or changes the existing leaderboard.
