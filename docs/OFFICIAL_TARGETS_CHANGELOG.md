# Official target registry changelog

> Generated from `official_target_versions.json`. Do not edit manually.

## 1.7.0 — 2026-08-03

- English: Pins exact model revisions and workload data identities for all 11 strict official Ascend
  targets, including run-scoped input hashing for nondeterministic latency prompts.
- 中文：为 11 个严格 Ascend 官方靶固定精确模型 revision 与负载数据身份，并要求对非确定性延迟输入记录单次运行哈希。
- Source set: `9d1637f5a06c59936243bc90917961e821d750be170ce83be076a63a79e90c04`
- Supersedes: `1.6.2`

## 1.6.2 — 2026-08-03

- English: Pins SHA-256 prefix-cache hashing for both SimLLM targets so the immutable runtime does
  not depend on an undeclared xxhash package.
- 中文：为两个 SimLLM 靶固定 SHA-256 前缀缓存哈希，避免不可变运行时依赖未声明的 xxhash 包。
- Source set: `4293db1fb645aa0a39944362818e254bb62d5736d73a2237d5484cb802b8f6a9`
- Supersedes: `1.6.1`

## 1.6.1 — 2026-08-03

- English: Pins a six-hour per-request timeout in the TraceLab production-trace contract so
  long-tail coding-agent outputs are reproducible and cannot silently inherit a short client
  default.
- 中文：在 TraceLab 生产轨迹契约中固定单请求六小时超时，确保长尾 coding-agent 输出可复现且不会静默继承过短的客户端默认值。
- Source set: `abeb1df62d983f57acdccbe7c4e491cbb5a640978c774b88b3878a8402d5e95e`
- Supersedes: `1.6.0`

## 1.6.0 — 2026-08-03

- English: Registers exact SimLLM random-online and saturated-throughput warm-cache A/B targets with
  pinned source commits and immutable runtime provenance; no local reference measurements are
  promoted as official results.
- 中文：注册精确的 SimLLM random-online 与 saturated-throughput warm-cache A/B
  靶，固定源码提交和不可变运行时来源；不将本地参考测量提升为官方结果。
- Source set: `49daa78b4d3745231e80ecc441f030a4fe19c720a8b7ca5f063508f690b524e1`
- Supersedes: `1.5.0`

## 1.5.0 — 2026-08-03

- English: Replaces the incompatible GLM MoE production-trace target with
  DeepSeek-R1-Distill-Qwen-32B on the immutable official runtime, pinning its supported
  batch-invariant PIECEWISE compilation path and disabling unsupported norm-quant fusion.
- 中文：将不兼容的 GLM MoE 生产轨迹靶替换为不可变官方运行时上的 DeepSeek-R1-Distill-Qwen-32B，固定其受支持的 batch-invariant PIECEWISE
  编译路径，并关闭不受支持的 norm-quant 融合。
- Source set: `02d40ec87d2a0a00b9d736755d4f9e8d2bc814eb9162bbb8c7f5903423c62434`
- Supersedes: `1.4.2`

## 1.4.2 — 2026-08-03

- English: Pins and attests the upstream batch-invariant compatibility mode required by the
  immutable v0.22.1rc1 image for production-trace execution.
- 中文：固定并证明不可变 v0.22.1rc1 镜像执行生产轨迹所需的上游 batch-invariant 兼容模式。
- Source set: `80846fea984c9b29291cb5ec84e5a4cb11b765476df7bd1803992c3a571e17fd`
- Supersedes: `1.4.1`

## 1.4.1 — 2026-08-03

- English: Pins the Ascend compilation compatibility setting required by the immutable v0.22.1rc1
  image when AddRmsNormBias is unavailable.
- 中文：固定不可变 v0.22.1rc1 镜像在缺少 AddRmsNormBias 时所需的 Ascend 编译兼容配置。
- Source set: `bc2c8c1b2f3f4f1a75f6bdec644d04e8ed3ae7e786855fc0f901b07ef45a05d5`
- Supersedes: `1.4.0`

## 1.4.0 — 2026-08-03

- English: Migrates GLM-4.7-Flash production-trace targets to the smoke-verified official
  vLLM-Ascend v0.22.1rc1 OpenEuler image pinned by digest.
- 中文：将 GLM-4.7-Flash 生产轨迹靶迁移到已通过 smoke 验证、按摘要固定的官方 vLLM-Ascend v0.22.1rc1 OpenEuler 镜像。
- Source set: `0dc53a59fcf7c4a80e7f0aa9b7003f28bd23cb496e6136f9cd1a7b40bcb72d97`
- Supersedes: `1.3.0`

## 1.3.0 — 2026-08-02

- English: Pins the reproducible Transformers runtime recipe required by GLM-4.7-Flash
  production-trace targets.
- 中文：固定 GLM-4.7-Flash 生产轨迹靶所需的可复现 Transformers 运行时配方。
- Source set: `7bf3e63d8e9518aa78f3d15ed2b60d9735433768992e32816f04b7a138aca1a6`
- Supersedes: `1.2.0`

## 1.2.0 — 2026-08-02

- English: Hardens production-trace targets with exact source and model revisions, variable-length
  cohort semantics, and replay evidence binding.
- 中文：强化生产轨迹靶：固定源码与模型 revision，采用变长 cohort 语义，并绑定完整回放证据。
- Source set: `d41fc696bfab5e88795fcb2ea7012f689dbc37350cbf0ed176a1e5d072322516`
- Supersedes: `1.1.0`

## 1.1.0 — 2026-08-02

- English: Adds a dedicated GLM-4.7-Flash 30B-A3B, two-chip, 131K production-trace profile for
  BurstGPT and TraceLab only.
- 中文：新增仅供 BurstGPT 与 TraceLab 使用的 GLM-4.7-Flash 30B-A3B、双卡、131K 生产轨迹官方靶。
- Source set: `7b4cddad1754f9b53bb0666ecf472e447d27de6f2742b53962017c2418cae1c8`
- Supersedes: `1.0.0`

## 1.0.0 — 2026-07-31

- English: Initial public 14B text, 14B code, and 7B multimodal target contract.
- 中文：首次发布 14B 文本、14B 代码和 7B 多模态公开固定靶契约。
- Source set: `54d31914a6ceae39fb5a58b972b979f417dd654ff51aa72d1b4b4b8535dc0d36`
- Supersedes: `none`
