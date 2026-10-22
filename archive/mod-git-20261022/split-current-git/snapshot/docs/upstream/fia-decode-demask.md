# 上游 PR 素材：vllm-ascend decode FIA 冗余掩码省略

> ⚠️ **本文件已冻结（2026-09-11）——不做上游 PR。**
> 决定与纪律见工作区 `AGENTS.md` 硬性约束与本目录 [README.md](README.md)。
> 下面 §"论据链" 仍作本地 patch 维护的技术参考；§"PR 标题草案"/§"上游化形态"/§"验证清单（上游侧）"
> **不再维护、不提交**，仅留档说明当时的推导。

> 目标仓库：vllm-ascend（上游）。形态：decode-only 小 patch，非行为变更。
> 本文件是 PR 描述草稿 + 技术论据索引，数字全部来自本仓验收证据。

## PR 标题草案

`perf(attention): skip the inert causal template for pure-decode FIA calls (Q_S=1)`

## 论据链

1. **冗余的构造性论证**：decode 路径（`AscendAttentionBackendImpl.full_graph_fia`
   与 `update_graph_params`，attention_v1.py :998/:1143/:800-1035）以
   `input_layout="TND", sparse_mode=3, pre_tokens=next_tokens=INT_MAX` 调 FIA，
   并携带 `get_splitfuse_attn_mask()`（attention_mask.py:50-55）生成的
   2048×2048 int8 上三角模板。CANN 文档：`sparse_mode/pre_tokens/next_tokens`
   在 Q_S==1 时为 no-op（算子自身在 Q_S==1 分派到 IncreFlashAttention 分支），
   且单 token 查询 attend 全部有效 KV——模板在此域恒为恒等。
2. **性能**：msprof 图 replay sweep（B=32、kv 128→8192、108 cell）：去掩码使
   FIA device 时间 −8…−23%（中位 −12%）；serving e2e（Qwen2.5-14B-Instruct，
   910B2，FULL 图模式）decode TPOT p1600×B32 +3.10%、B64 +4.61%（3+3 轮，
   首轮预热离群对称剔除，腿内散布 0.2%）。
3. **数值等价**：fp32 golden 对照 rel_l2=bf16 底噪；greedy 生成 8/8 逐 token
   bit 级全等（temperature=0）。
4. **守卫面**：仅当 `input_layout=="TND" && sparse_mode==3 && pre/next==INT_MAX
   && atten_mask 为方形 int8 && 累计 actual_seq_lengths==T（= 每请求恰 1 查询
   token）` 时改写；prefill / chunked-prefill / 混批 / 滑窗 / spec-decode 一律
   原样。实现参考：本仓 `fia_demask_plugin.py`（同谓词的 op 面包装，可作
   上游 in-tree 化的底稿）。

## 上游化形态（建议）

在宿主两处调用点（capture 与 update_graph_params）增加同一谓词：匹配则不传
`atten_mask` 且 `sparse_mode=0`。比插件包装更干净（无代理层），且宿主已在
`batch_invariant.py` 有同符号 patch 先例，不引入新契约。

## 验证清单（上游侧需重跑）

- 精度：greedy parity（任一 causal 模型）+ fp32 golden 对照
- 性能：msprof decode 图 replay（sm3 vs sm0）+ 端到端 TPOT
- 边界：chunked prefill / 滑窗 / spec decode / MLPa（若引入）回归
