# PR 154 Prefix-Routing Benefit Benchmark

This submission measures the routing policy itself. It compares two
independent vLLM replicas behind a deterministic random proxy with the same
two replicas using longest-prefix routing and ZMQ snapshot/replay events.

## Provenance

- vllm-hust: `68cf893a4abb95a72f9378634410077df51c1dcc`
- vllm-ascend-hust: `03a12f9bddd944952bd029c6b62e23d68fa3a28e`
- vllm-hust-benchmark: `42f17ebb79483ae9d0f119760cf9527c0f2e009a`
- Model: `Qwen/Qwen2.5-7B-Instruct`, BF16
- Hardware: one host, two Huawei 910B2 NPUs

## Workload

- Dataset: `prefix_repetition`
- 400 prompts, 200 prefixes, two prompts per prefix
- 1,536 shared-prefix tokens, 64 suffix tokens, 64 output tokens
- Request rate 4, maximum concurrency 16
- No artificial cache warmup
- Three repetitions with alternating phase order
- Fresh vLLM processes and empty caches for every phase

The baseline uses a seeded-random external proxy and does not enable the
global prefix scheduler. The candidate sends every request to the integrated
prefix-routing entrypoint. Both modes enable the native local prefix cache.

## Median Results

| Metric | Random baseline | Prefix routing | Improvement |
| --- | ---: | ---: | ---: |
| Request throughput | 3.9542 req/s | 3.9538 req/s | -0.01% |
| Total token throughput | 6,579.91 tok/s | 6,579.25 tok/s | -0.01% |
| Mean TTFT | 136.89 ms | 120.99 ms | 11.62% |
| P95 TTFT | 223.80 ms | 174.80 ms | 21.89% |
| P99 TTFT | 281.72 ms | 233.26 ms | 17.20% |
| Mean TPOT | 21.06 ms | 19.86 ms | 5.69% |
| P95 E2EL | 1,756.33 ms | 1,604.47 ms | 8.65% |

All six benchmark phases completed 400/400 requests with zero failures and
identical total input/output token counts. The aggregate native prefix-cache
hit rate increased from 24.44% to 47.88%. Candidate routing was balanced at
204 requests on node0 and 196 on node1 in every repetition.

## Reproduction

```bash
PYTHONPATH=/workspace/myproject/vllm-prefix-bench-68cf893a4:/workspace/vllm-ascend-hust \
VLLM_USE_V1=1 \
/workspace/myproject/vllm-hust/.venv/bin/python \
  tests/distributed/run_prefix_routing_benchmark.py \
  --expected-sha 68cf893a4abb95a72f9378634410077df51c1dcc \
  --model /data/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --devices 2,3 \
  --gpu-memory-utilization 0.60 \
  --result-dir /workspace/myproject/prefix-routing-benefit-68cf893a4
```

`comparison.json` contains the complete phase configuration, cache counters,
routing counts, raw results, and median comparisons. The `raw/` directory
contains all six original `vllm bench serve` JSON files.

`scripts/backfill_single_gpu.py validate` passes for the submission. Public
snapshot changes are intentionally not included because the current public
matrix rejects BF16 entries. The measured runtime dtype is BF16, so the
submission preserves the real precision rather than relabeling it as FP16.
