<!-- 性能证据 checklist 共享 partial（issue #95 Layer 3，增强建议 #8）
     core/Ascend 仓库的 PR 模板通过 include 引用此文件，单点维护避免漂移。
     源文件：vllm-hust-benchmark/.github/perf-evidence-checklist.md -->

## Performance Evidence (issue #95 merge gate)

> 性能相关 PR 合并前必须提供一组可比的 real-online paired 性能证据。 纯文档/测试/网站 PR 可用受控 label 跳过（见下方例外）。

### Target Declaration（必填）

- [ ] `official-fixed-target` — 跑官方固定靶（14B 单卡主线默认）
- [ ] `specialty-target` — 跑 specialty 靶（须填下方理由）

### Official Fixed-Target Evidence（选 official-fixed-target 时必填）

- target_id: `<!-- 从 #104 registry 读取，如 official-ascend-jan-2026-v0.18.0 -->`
- target_version: `<!-- 如 v0.18.0 -->`
- profile_id: `<!-- 如 core-text-14b -->`
- base artifact: `<!-- base commit (PR fork point) 的 run_leaderboard.json 路径/链接 -->`
- head artifact: `<!-- head commit (PR 最新 push) 的 run_leaderboard.json 路径/链接 -->`
- spec_id: `<!-- paired base/head 的 same_spec.spec_id，必须一致 -->`
- model: `<!-- 如 Qwen/Qwen2.5-14B-Instruct -->`
- hardware: `<!-- 如 910B2 x1 -->`
- gpu_memory_utilization: `<!-- 14B 文本线=0.6, vision 线见 registry -->`
- max_model_len: `<!-- 14B 文本线=32768, vision 线=30720 -->`

### Specialty Target Evidence（选 specialty-target 时必填）

- specialty spec: `<!-- 独立 spec 名称，必须能在 registry 匹配到 specialty profile -->`
- specialty reason: `<!-- 为什么不跑主线，跑这个 specialty 的理由 -->`
- paired base/head artifact: `<!-- 同上 -->`

### Docs/Test/Website-only Exception（跳过性能证据时必填）

- [ ] 已添加受控 label：`perf-skip:docs-only` / `perf-skip:test-only` / `perf-skip:website-only`
- 审批人: `<!-- @reviewer -->`

### Checklist

- [ ] data_source 以 `real-online` 开头（不接受 smoke/replay/derived/截图）
- [ ] base/head 的 CI 状态均为 accepted（missing/cancelled/skipped/resource_busy 一律 fail closed）
- [ ] 3B perfgate 通过 **不等于** 14B public-target evidence 通过
- [ ] registry hash 匹配（target_id 在 #104 registry active 列表）

> 判定工具：`python -m vllm_hust_benchmark.cli merge-gate-check`
> Registry：`vllm-hust-benchmark/src/vllm_hust_benchmark/data/fixed_target_registry.json`
