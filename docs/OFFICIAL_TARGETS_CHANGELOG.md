# Official target registry changelog

> Generated from `official_target_versions.json`. Do not edit manually.

## 1.3.1 — 2026-08-11

- English: Align the provisional prefix-repetition workload with its request count by setting eight
  repeated prefixes for eight prompts.
- 中文：将 provisional prefix-repetition workload 与请求数量对齐：8 个请求使用 8 个重复前缀。
- Source set: `d44a084792b2a33218f19ea070870405c7684cae518a2e7ccbee77d69fda2e76`
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
