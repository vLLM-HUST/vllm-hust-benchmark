# Native MTP4 gamma4 reproduction capsule (2026-10-27)

These files were restored byte-for-byte from the evidence archive in
`vLLM-HUST/vllm-hust-benchmark` commit
`0ce4365692c335f31e2cbbb043d251e7f3d185ec`.

The ordered archive has SHA-256
`306bb0a5b29251078a57b9071e9c7a558c5db106a844eadf181661f66e78f76e`.
The original report is
`reports/runtime-mod-review-20261027/research/speculative-mmlu-pro/gamma4/RESULTS.md`.

## Important protocol distinction

The reported 796.26–825.71 output tokens/s values are **not** measurements
with every request fixed to 1024 output tokens. They are calculated as the
sum of the actual output-token counts from a frozen 56-question MMLU-Pro
burst divided by the timed `llm.generate` wall time. Each request has
`max_tokens=4000`; requests may stop or truncate at different lengths.

The frozen native arm uses Qwen3.5-35B-A3B, TP2+EP, BF16, B16,
`max_model_len=16384`, `max_num_batched_tokens=8192`, APC, async scheduling,
chunked prefill, and `FULL_AND_PIECEWISE` graphs. Native MTP uses gamma 4.
See `gamma4-configs.json` for the complete engine configuration.

## Original native-arm invocation

Create an environment at `venv/` containing the matching vLLM and
vLLM-Ascend builds, make `/models/Qwen3.5-35B-A3B` available, and create an
empty `evidence/` directory. From this directory, the original native arm is:

```bash
HCCL_BUFFSIZE=1024 \
HCCL_OP_EXPANSION_MODE=AIV \
VLLM_NO_USAGE_STATS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
VLLM_PLUGINS=ascend \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_AOT_COMPILE=0 \
PYTHONHASHSEED=0 \
TOKENIZERS_PARALLELISM=false \
LUMI_TIMEOUT_SECONDS=1800 \
./venv/bin/python launch_gamma4.py \
  0,1 gamma4-performance-q35-mtp4-repro \
  screen_gamma4.py q35-mtp4 perf56
```

`launch_gamma4.py` locks and verifies the two requested physical NPUs and
refuses to overwrite an existing evidence label. `screen_gamma4.py` performs
the 14-prompt × 16-token warmup outside the timed region, then times the
56-question burst. It writes the complete result and metrics under
`evidence/`.

To compute the comparable throughput from the result JSON, divide the sum of
all `rows[*].output_tokens` lengths by `elapsed`. A fixed-1024 workload is a
different protocol and should be reported separately rather than compared
directly with the historical 800+ values.

## Files

- `screen_gamma4.py`: frozen workload and timing harness
- `launch_gamma4.py`: NPU locking, mapping, placement, timeout and cleanup
- `campaign_gamma4.py`: original three-repeat alternating campaign
- `gamma4-configs.json`: native MTP4 and vSpec candidate engine configs
- `q35-environments.json`: frozen environment variables
- `perf56.jsonl`: frozen 56-question performance subset
- `pilot.jsonl`: 14-prompt warmup set
- `gamma4_witness.py`, `gamma4-guard/`, `guard/`: metadata/placement receipts
- `partial_results.py`, `scoring.py`: completion custody and answer parsing
