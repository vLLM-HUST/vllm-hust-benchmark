# Runtime, artifacts and replay provenance

Scope: Qwen3.5-35B-A3B quantisation evidence for the 2026-11-17 tech-outcomes submission. No new
model requests were made while assembling this archive; the MMLU-Pro GDN-W8A8 run of 2026-11-14 is
the only new measurement included.

## Artifacts

| Arm           | Local path (controlled)                | Weights                      | Note                              |
| ------------- | -------------------------------------- | ---------------------------- | --------------------------------- |
| BF16          | `/data/jxd/Qwen3.5-35B-A3B`            | 71,927,122,971 B = 66.99 GiB | 14 shards, revision `712cf743...` |
| Official W8A8 | `/data/jxd/Qwen3.5-35B-A3B-W8A8`       | 39,810,890,505 B             | experts + attn q/k/v W8A8         |
| W4A8/GDN      | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot` | 24,096,662,520 B = 22.38 GiB | W4A8 experts + GDN W8A8           |
| gcy W4A8      | `/data/gcy/Qwen3.5-35B-A3B-W4A8`       | 25,099,795,178 B             | experts int4, GDN bf16            |

Model weights stay controlled (see `controlled-location.json`); shard hashes are in the artifact
record `vllm-ascend-quant-hust/docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md`.

## Runtime

- vLLM: `0.28.1.post1.dev143+gf18cf803c` (commit `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`)
- vLLM-Ascend: `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27` (commit
  `74f0c0a272376412b51e1c1864803d5f3a0f1b5f`)
- CANN: 9.1.0 (per-process `/data/jxd/Ascend-9.1.0/{ascend-toolkit,nnal/atb}`; system 8.5.0
  untouched)
- Quantisation env: msmodelslim 26.1.0, torch 2.9.0, torch_npu 2.9.0, transformers 5.2.0
- Hardware: Ascend 910B3, 1 card (64 GiB HBM) for TP1; BF16 baseline is published TP2

## Measurement plans

- SWE prefix reuse: `prepared/qwen35.json`, sha256
  `ac1fbf9885076c369fd25c5525d1163ff8b7945972cdf4c0c6d2d523ab7c1172`
- MMLU-Pro: 12,032 cases, frozen plan contract sha256
  `0a208ebbeec1afc9dea6d3b5e13d12863995d630ba34f60e510046292a63b5ca`; adapter
  `c2237c69f848e2358abe71c68f54143b5c090a81bd4f7915088b19b8ae157801`; runner
  `6c02a8bef434d59cb6688cbc0c6a6ec788b2749f6a6a5187b0798ee5c2c9ba18`; /v1/completions, max_tokens
  2048, temperature 0, seed 17, concurrency 16, no retries, 32K context, no APC, no MTP.
- Long context: pre-registered P2-3, 8K/32K/128K + 260K task-level; 260,000-token prompt + 1024
  generation; BF16-anchored PPL at 4K/32K/128K; C1 long-text performance at 8K/32K/128K/260K.
- Full-context appendix (2026-11-16): capacity smoke at prompt 262,143 + 1 token for both arms;
  task-level NIAH body 261,000 and longgen body 260,800; 19.22 GiB same-KV appendix for W8A8/W4A8.

## Evidence locations

- Small raw files: `data/` in this report (see `originals-manifest.json`).
- Large raw files: GitHub Release `tech-outcomes-20261117-ascend-quant` (see
  `release-manifest.json`).
- Controlled originals: `controlled-location.json`.
- Every committed file is bound by `SHA256SUMS`; every reported number is mapped in
  `evidence-map.csv`.
