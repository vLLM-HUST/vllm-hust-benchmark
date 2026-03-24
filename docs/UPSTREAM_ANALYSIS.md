# Upstream vLLM Benchmark Analysis

## Goal

This note explains what the official vLLM benchmark actually looks like today, and what should be mirrored in an independent benchmark repository.

## Official Boundary

The old `benchmarks/*.py` entry scripts in upstream are mostly deprecated wrappers. The real public boundary is now the CLI:

- `vllm bench serve`
- `vllm bench throughput`
- `vllm bench latency`
- `vllm bench sweep ...`

The CLI glue lives in:

- `vllm/entrypoints/cli/benchmark/main.py`
- `vllm/entrypoints/cli/benchmark/base.py`
- `vllm/entrypoints/cli/benchmark/serve.py`
- `vllm/entrypoints/cli/benchmark/throughput.py`
- `vllm/entrypoints/cli/benchmark/latency.py`
- `vllm/entrypoints/cli/benchmark/sweep.py`

The implementation behind those commands lives in:

- `vllm/benchmarks/serve.py`
- `vllm/benchmarks/throughput.py`
- `vllm/benchmarks/latency.py`
- `vllm/benchmarks/datasets.py`
- `vllm/benchmarks/lib/`
- `vllm/benchmarks/sweep/`

## What Matters Structurally

### 1. CLI-first benchmark entrypoints

Upstream has already standardized on benchmark subcommands. That means an independent repo should not be built around legacy one-off scripts.

### 2. Shared dataset sampling layer

The most important upstream benchmark component is the dataset layer in `vllm/benchmarks/datasets.py`.

It centralizes:

- dataset classes
- parser wiring
- request sampling
- prompt-length and output-length shaping
- multimodal and custom dataset support

Representative upstream dataset types include:

- `ShareGPTDataset`
- `RandomDataset`
- `RandomMultiModalDataset`
- `PrefixRepetitionRandomDataset`
- `CustomDataset`
- `CustomMMDataset`
- `SonnetDataset`
- `BurstGPTDataset`
- `HuggingFaceDataset`
- `InstructCoderDataset`
- `VisionArenaDataset`
- `AIMODataset`
- `MTBenchDataset`
- `ASRDataset`

### 3. Standardized serving metrics

The official serving benchmark reports:

- request throughput
- output token throughput
- total token throughput
- TTFT
- TPOT
- ITL
- E2EL

This is the metric contract an independent repo should preserve when adding new scenarios.

### 4. Backend-aware online benchmarking

Upstream serving benchmarks route requests through backend-specific request functions in `vllm/benchmarks/lib/endpoint_request_func.py`.

That means scenario definitions should not hardcode only one transport or only one endpoint style.

### 5. CI uses explicit scenario definitions

Upstream CI and performance automation already maintain explicit serving and throughput scenario definitions under `.buildkite/performance-benchmarks/`.

That is the right direction for an independent repo too: define scenarios as data.

## Recommended Independent-Repo Shape

An independent repo should preserve these upstream ideas while improving extensibility:

1. Keep the CLI surface close to `vllm bench`.
2. Record official scenarios in a registry file.
3. Build commands from scenario definitions rather than hand-written shell snippets.
4. Add extension tags such as `agi4s`, `domestic-hardware`, `long-context`, `tool-calling`, `structured-output`, or `multimodal` without modifying the base command builder.
5. Keep future result-export logic separate from command construction.

## Why This Repo Does Not Copy Upstream Files Blindly

Blindly copying upstream benchmark implementation code into a separate repo would create a hard-to-maintain fork.

Instead, the first phase of this repo mirrors:

- the official command surface
- the official scenario vocabulary
- the official benchmark categories
- the official metric expectations

while isolating our additions in:

- a scenario registry
- a command builder
- explicit metadata for future new scenarios

That keeps the independent repo useful now, while still leaving room to vendor or wrap more upstream code later if the maintenance tradeoff is justified.