# Qwen2.5-14B current-plugin reproduction

The user requests an actual OFF/ON test of Qwen2.5-14B-Instruct with 7168
input tokens, 512 output tokens and concurrency 32, to compare with the
historical +41.38% throughput result. The user explicitly chose the current
CANN 9.1 / Torch 2.10 runtime, identical in both arms, rather than restoring
the historical CANN 9.0 / Torch 2.9 stack. Other historical launch parameters
and the 512/beta20 compression budget are retained. Runtime differences from
the old report remain visible.

## Approach and boundaries

The released 0.4.0rc1 currently rejects this model at its method compatibility
gate. The algorithm and adapter already represent dense GQA, 128-token pages,
TP1 and eager per-layer metadata. Restore only the exact Qwen2.5-14B geometry
and eager profile through the current independent adapter, with regression
tests and live NPU verification. This is a new candidate derived from the
current plugin, not a claim that unmodified 0.4.0rc1 supported the workload.
Reinstating the removed schema-v1 runtime would create a second implementation;
changing the workload to Qwen3.5/C4 would not answer the user. Neither is used.
The exact Qwen3.5 profile remains admitted by its existing checks.

The user authorized the test and autonomous repository work; the narrow
adaptation and evaluation details below are agent implementation decisions.
No new issue is needed. Results will be written to existing issue #1.

## Runtime and method contract

- Official Qwen2.5-14B-Instruct: qwen2, 48 full-attention layers, Q/KV heads
  40/8, head dimension and rotary dimension 128, no RoPE scaling.
- BF16, TP1/PP1, standard Ascend backend, eager/NONE, no speculation,
  APC, async scheduling, chunked prefill, kernel/logical block size 128.
- Reject changed geometry, graph mode, TP, quantization or speculative paths.
  Host-level restrictions continue to apply; no generic model whitelist bypass.
- Configuration: max_capacity_prompt=512, min_compression_prompt_tokens=4096,
  window_size=8, kernel_size=7, maxpool, beta=20, kv_head/mean GQA, no merge.
  Maximum physical prompt allocation is 1024 tokens / eight pages; per-layer
  retained lengths still differ. Source blocks stay immutable with APC.
- Reuse the same current pinned core/Ascend/native operators for OFF and ON.
  Only activation changes. Both load the same frozen candidate package and
  explicit host worker. Extra host bridges must be common to both arms and
  regression tested if a real startup failure requires one.
- Single allocated 910B2 physical device 6 (container-visible ordinal 1).
  Existing shared model files have the same config/index/tokenizer SHA256 as
  the historical report; read them without modifying or duplicating weights.

## Measurement contract

Use max-model-len=8192, max-num-seqs=32, max-num-batched-tokens=2048,
gpu-memory-utilization=0.8, block-size=128, seed=0, generation-config=vllm,
APC/chunked/async enabled, enforce-eager, no MTP. Do not choose a smaller KV
pool to manufacture pressure. Record actual allocated blocks/bytes in each
process, plugin staging overhead and sampled HBM.

Recover the historical random workload generator at core 4861aab3 and
freeze its seed-0 request text/token identity. Preserve any tokenize/decode
round-trip difference explicitly; do not silently substitute current random
generator semantics. Historical long benchmark has 32 prompts, request-rate
infinity, maximum concurrency32, ignore-eos, 7168/512 nominal lengths. Its
preceding short workload is 48 prompts, 1024/256, rate1, seed0. Reproduce this
sequence in each lifecycle, with any client preflight/warmup separately
identified and the same in both arms.

After correctness checks, freeze six fresh lifecycles in historical order:
OFF1/ON1/OFF2/ON2/OFF3/ON3. Every failure and incomplete run remains visible;
no configuration tuning or selective retry after viewing formal results.
Startup repairs occur only during separately labelled preflight. No local
tests, tokenization, lint or audits during measured requests. Run one server
at a time and stop all owned processes when finished.

Record raw requests/SSE/output usage, benchmark wall time and mean/p50/p95
TTFT/TPOT/E2E, output-token and request throughput, cache usage, preemptions,
cached tokens, actual HBM, and scheduler/worker transaction correspondence.
For published aggregate ratios, preserve the historical mean-of-three
per-process metric convention and also expose per-process values. A positive
performance result does not certify unchanged task quality.

## Verification and delivery

1. Prove the published gate rejects Qwen2.5, then test the narrowly restored
   profile and negative cases; run the full plugin suite.
2. Run real 40-to-8 GQA selection oracle, short/boundary/long serving smoke,
   source/destination and layer-order checks, and concurrent request cleanup.
   A metadata-only or HTTP-only pass is insufficient.
3. Freeze the candidate and launch/input identity before formal measurement;
   compare all six lifecycles without dropping negative outcomes.
4. Publish code/tests/docs through a reviewed PR and benchmark evidence with
   offline recomputation, then update #1. Report OFF/ON numbers in plain terms
   beside the historical result, with the runtime-version difference explicit.
5. Preserve originals outside the clean workspace, stop the owned service,
   and remove temporary checkout/test copies after canonical custody exists.
