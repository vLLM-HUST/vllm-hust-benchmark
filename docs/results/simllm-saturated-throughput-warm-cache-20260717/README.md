# SimLLM saturated warm-cache throughput reference result

This directory archives the local reference run used to define the SimLLM
official benchmark requirement. It is reproducibility evidence, not an
official baseline or a canonical public submission.

## Run configuration

- Run date: 2026-07-17 UTC
- Device: one Huawei Ascend 910B2, visible device 5
- Model: `Qwen/Qwen2.5-14B-Instruct`, FP16
- Input/output length: 4096/32 tokens
- Requests: 32
- Measured request rate: `inf`
- Maximum concurrency: 16
- Temperature: 0
- Warm-cache pass: one pass at 1 req/s with the same prompt seed
- SimLLM KV cache entries: 32

Reference command:

```bash
cd /workspace/vllm-hust-benchmark
ASCEND_RT_VISIBLE_DEVICES=5 \
CURRENT_MODEL_PATH=/data/shared_models/Qwen2.5-14B-Instruct \
bash scripts/run_simllm_saturated_throughput_warm_cache.sh
```

The path and device in this command describe the machine used for the local
run. An official runner must allocate the device and resolve the model path
rather than hard-code either value.

## Result

| Metric | Baseline | SimLLM warm cache | Improvement |
| --- | ---: | ---: | ---: |
| Successful requests | 32/32 | 32/32 | -- |
| Requests/s | 1.4054897373 | 4.7896510134 | 240.78% |
| Output tokens/s | 44.9756715951 | 153.2688324294 | 240.78% |
| Total tokens/s | 5801.8616357621 | 19771.6793833923 | 240.78% |
| Mean TTFT | 3616.746 ms | 547.197 ms | -84.87% |

See `throughput_comparison.json` for the machine-readable comparison and the
two `raw_benchmark_result.json` files for the unaggregated benchmark output.

## Provenance and caveats

- vLLM-HUST commit: `f3f23914074764619c290694e00908fec1954664`
- vLLM-Ascend-HUST commit: `93f005f555a96d813875118658d083ba0c114e05`
- The benchmark client emitted a non-fatal optional Triton import warning.
- The measured requests all completed and the archived server logs contain no
  runtime OOM, `AssertionError`, `EngineDeadError`, or fatal engine error.
- A larger 256-request asynchronous-scheduling test exposed a scheduler
  placeholder assertion during development. The official requirement includes
  a separate stability test so this local 32-request result is not presented as
  a general scalability claim.
- Files under `submission/` are generated exporter examples from this local
  run. They must not be copied into canonical official submissions.
