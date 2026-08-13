# vLLM-HUST delivery suite

The package registry `delivery_suite_registry.json` is the machine-readable acceptance contract layered above the existing official leaderboard targets.

It fixes nine non-overlapping workload classes:

| Workload | Benchmark | Fixed single-node model |
| --- | --- | --- |
| General QA | ShareGPT serving | Qwen3-32B |
| Code | LiveCodeBench | Qwen3-Coder-30B-A3B-Instruct |
| Reasoning | AIME 2025 | DeepSeek-R1-Distill-Qwen-32B |
| Multimodal | VisionArena serving | Qwen3-VL-30B-A3B-Instruct |
| KV reuse | vLLM prefix repetition | Qwen3-32B |
| Structured output | JSONSchemaBench | Qwen3-32B, non-thinking mode |
| Long context | LongBench v2 | Qwen3.5-27B |
| Agent | BFCL V4 | GLM-4.5-Air |
| AI4Science | NatureBench | Qwen3-Coder-30B-A3B-Instruct |

The existing 7B/14B official targets remain unchanged for leaderboard continuity and PR/nightly gates. The delivery suite adds 27B–106B single-node acceptance targets. GLM-4.6 W8A8 is conditional on load and KV-capacity preflight. GLM-5 and Kimi K2/K3 remain customer multi-node extensions and cannot satisfy a fixed single-node target.

Inspect the contract with:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.delivery_suite
PYTHONPATH=src python -m vllm_hust_benchmark.delivery_suite --workload-id long_context
```

Every formal comparison must keep model, workload, hardware, runtime parameters, and client parameters identical between baseline and candidate. Run status, benchmark quality, throughput, median latency, tail latency, and error rate remain separate gates.
