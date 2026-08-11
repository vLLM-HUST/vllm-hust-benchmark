# Official target registry changelog

> Generated from `official_target_versions.json`. Do not edit manually.

## 1.4.0 — 2026-08-11

- English: Add 7 Ascend 910B3 full-graph-parallel inplace specialty specs
  (cudagraph_mode=FULL_DECODE_ONLY + split_batch inplace_parallel, temperature 0.0) as provisional
  profiles, one per workload (random-online, random-latency, sharegpt-online, sharegpt-throughput,
  sonnet-throughput, prefix-repetition-online, instructcoder-online). Registry classifier now gates
  public-leaderboard/active on hardware_chip_model == 910B2; 910B3 specs are always
  specialty/provisional.
- 中文：新增 7 个 Ascend 910B3 full-graph-parallel inplace 专用 spec（cudagraph_mode=FULL_DECODE_ONLY +
  split_batch inplace_parallel、temperature 0.0）作为 provisional profile，每个 workload
  一份（random-online、random-latency、sharegpt-online、sharegpt-throughput、sonnet-throughput、prefix-repetition-online、instructcoder-online）。registry
  分类器新增硬件门禁：仅 hardware_chip_model == 910B2 可判为 public-leaderboard/active；910B3 一律为
  specialty/provisional。
- Source set: `af85e0f896670e1e7a270d2f14a32898f6de9fba702ce44c349977eff67a6b39`
- Supersedes: `1.3.0`

## 1.3.0 — 2026-08-03

- English: Revert non-executable specialty specs to non-executable drafts: remove unsupported vLLM
  CLI parameters (enable_alltoall, compression_plugin, enable_simllm_scheduler, etc.) and
  placeholder model names (Qwen2.5-MoE-A14B-Instruct, Qwen2.5-14B-Instruct-EAGLE,
  Qwen2.5-14B-Instruct-slicegpt-75pct). Add parameter validation test covering MoE/EPLB, spec
  decode, SliceGPT/KNorm.
- 中文：将不可执行的专用 spec 回退为非可执行草案：移除不支持的 vLLM CLI
  参数（enable_alltoall、compression_plugin、enable_simllm_scheduler
  等）和占位模型名（Qwen2.5-MoE-A14B-Instruct、Qwen2.5-14B-Instruct-EAGLE、Qwen2.5-14B-Instruct-slicegpt-75pct）。添加覆盖
  MoE/EPLB、投机解码、SliceGPT/KNorm 的参数验证测试。
- Source set: `20e94aab3fdb09b47aa388d87bee1464e129b39bad97af34db810437a9d31c9a`
- Supersedes: `1.1.0`

## 1.1.0 — 2026-08-02

- English: Add 12 specialty Ascend v0.18.0 baseline scenarios (KV pressure, attention boundary, MoE
  alltoall, spec decode, etc.) as provisional profiles.
- 中文：新增 12 个 Ascend v0.18.0 专用基线场景（KV 压力、注意力边界、MoE alltoall、投机解码等）作为 provisional profile。
- Source set: `7ab4d0175091d372b913eec159f9e8e24c1b5e39ce8dfdf4cc54cca1acde0c31`
- Supersedes: `1.0.0`

## 1.0.0 — 2026-07-31

- English: Initial public 14B text, 14B code, and 7B multimodal target contract.
- 中文：首次发布 14B 文本、14B 代码和 7B 多模态公开固定靶契约。
- Source set: `54d31914a6ceae39fb5a58b972b979f417dd654ff51aa72d1b4b4b8535dc0d36`
- Supersedes: `none`
