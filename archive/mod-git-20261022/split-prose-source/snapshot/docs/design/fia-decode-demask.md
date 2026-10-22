# Design note: decode FIA de-mask (drop the redundant 2048² int8 causal template)

- 状态：**新能力，`import_only`**（未进 manifest `active`；三项证据见
  `../../docs/progress.md` 与
  `/vllm-workspace/profiles/qwen14b-instruct-hotspot-20260910/probe-fia/E2E-mask-removal.md`）
- 开关：`VLLM_HUST_FIA_DEMASK=1`（默认 0，不 patch、不注入、日志零输出）
- 诊断：`VLLM_HUST_FIA_DEMASK_TRACE=1`（observe-only：只记录、不改调用；
  用于 OFF 腿取"mask 证据"，同样默认关）
- 落点：`src/vllm_ascend_split_batch/fia_demask_plugin.py`
- 入口：`vllm.general_plugins` → `vllm_ascend_split_batch.fia_demask_plugin:load`

## 1. 宿主读码结论（vllm-ascend-hust main @ `74f0c0a27`）

### 1.1 调用点与实参来源

生产 decode 路径（`--compilation-config '{"cudagraph_mode":"FULL"}'`）
`AscendAttentionBackendImpl.forward_fused_infer_attention`
(:1415) → `_EXTRA_CTX.capturing` → `full_graph_fia` (:998) →
`torch_npu.npu_fused_infer_attention_score.out(...)` (:1143)；
图重放前的再参数化是 `AscendAttentionBackendImpl.update_graph_params`
(:800-1035) 的同一算子 `.out` 调用。两处传的是同一组字段，均由
`AscendMetadata` 派生：

| 参数 | decode 生产值 | 来源 |
|---|---|---|
| `input_layout` | `"TND"` | :1020 / :996 |
| `sparse_mode` | **3** | :1021 `4 if sliding_window else 3 if attn_metadata.causal else 0` |
| `pre_tokens` / `next_tokens` | **`SWA_INT_MAX`(2147483647) / `SWA_INT_MAX`** | :1022-1023 `self.sliding_window or SWA_INT_MAX` / `0 if sliding_window else SWA_INT_MAX` |
| `atten_mask` | **2048×2048 int8 上三角** | :1020 ← `attn_metadata.attn_mask` ← Builder :368 `get_attention_mask(causal, model_config)` ← `attention_mask.py:50-55 get_splitfuse_attn_mask()`（`torch.triu(ones(2048,2048), 1).to(int8)`） |

即：**mask 是硬编码的单例模板，sparse_mode/pre/next 是由 `attn_metadata.causal`
与 `self.sliding_window` 推导的，没有任何用户可配面**。

### 1.2 为什么是 `sparse_mode=3` + `INT_MAX`？（是否 workaround）

- 由 `sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0`
  引入于 `ac357ed4b`（2026-05-15，*Fix sliding window attention computation*，
  upstream #8508）。该 PR 的动机是**滑窗**（`sparse_mode=4` + `pre=window/next=0`）；
  非滑窗的因果分支取 `3` 是"TND 因果的默认模式"，**不是 CANN 旧版本的
  workaround**，也没有任何注释/blame 指向历史 bug。
- CANN 官方接口文档（`torch_npu.npu_fused_infer_attention_score.__doc__`，原文摘录）：
  - "当 Query 矩阵的 S 为 1，进入 **IncreFlashAttention** 分支，其余场景进入
    PromptFlashAttention 分支" → 算子自身就在 `Q_S==1` 处按 decode/prefill 分流，
    与本次改动限定的域完全一致；
  - "sparse_mode … **Q_S 为 1 且不带 rope 输入时该参数无效**"；
    `pre_tokens`/`next_tokens` 亦标注 "**Q_S 为 1 时该参数无效**"；
  - "sparse_mode 为 3 时，代表 **rightDownCausal** 模式的 mask，**需要传入
    优化后的 atten_mask 矩阵 (2048\*2048)**"；
  - "sparse_mode 为 0 时 … **如果 atten_mask 未传入则不做 mask 操作**，
    忽略 pre_tokens 和 next_tokens（内部赋值为 INT_MAX）"。
- 探针实测（`probe-fia/REPORT.md` §2.4）：`sparse_mode=3` **且不传 mask** 被
  tiling 拒绝（`561002`）。所以 mask 不是"多余的可选输入"，而是宿主声明
  `sparse_mode=3` 时必须附带的模式模板 —— 宿主没有为 decode 特判，
  于是 decode 也付了这条模板路径的钱（8–23%）。**结论：不是 workaround，
  是"同一条 TND 因果路径被 decode 复用"的顺带成本。**

### 1.3 prefill 是否同用该调用？

是。`full_graph_fia` 的 `sparse_mode`/`pre_tokens` 是**按 metadata 通式**算的，
`_EXTRA_CTX.capturing` 下 prefill/chunked-prefill 形态同样会走到这里
（`sparse_mode=3` + mask 对 `Q_S>1` 是有意义的 rightDownCausal）。
因此 patch **必须只作用于 decode 形态**；探针证据也只在 decode（`S_q=1`）域成立。
判别式取"**每个请求恰好 1 个 query token**"（见 §2.2），它精确等于算子内部的
IncreFlashAttention 分支条件（`Q_S==1`）。

### 1.4 官方面是否有"关 mask"的开关？

**没有。** 全量核查：
- `vllm_ascend/envs.py`（完整 `env_variables` 表，仅 9 项，无 mask 相关）；
- `ascend_config.py` 的 `from_additional_config` 各配置对象（eplb / finegrained_tp /
  xlite_graph / scheduler / sparse_kv_offload / dump_config …）无 FIA mask 项；
- `attention_mask.py::get_attention_mask` 只在 `causal=False`（双向）时返回
  `None` —— 因果 LM 用不到；
- `attention_v1.py` 内没有 `atten_mask=None` 的可达分支（除 C8/BNSD 与
  non-causal 分支，均非本 profile 形态）。

→ 官方开关不存在，落到**插件 wrapper patch**（优先级 2）。

## 2. 选型与实现

### 2.1 为什么 patch 在算子面（`torch_npu.npu_fused_infer_attention_score`）

`full_graph_fia`（:1143）与 `update_graph_params`（:1010）是宿主内两段
大体积私有方法，二者唯一的公共交汇点就是**同一个算子模块属性**
`torch_npu.npu_fused_infer_attention_score.out`。因此：

- 覆盖**捕获**与**重放再参数化**两条必经之路（否则捕获无 mask、重放有 mask
  会直接对不上）；
- patch 面积 = 一个 duck-typed 代理对象，不重写任何宿主逻辑，宿主重构
  `full_graph_fia` 内部结构不影响本插件；
- 宿主自身就有先例（`vllm_ascend/batch_invariant.py:120` 直接改
  `torch_npu.npu_fused_infer_attention_score`），说明这是被宿主认可的接缝。

同时包住 `torch_npu._npu_fused_infer_attention_score_get_max_workspace`：
workspace 必须按**同一份**实参算出来（探针两臂各自用匹配的 ws），
否则可能出现 ws 与实参不匹配。

**不做的事**（明确排除，避免面积蔓延）：
- 不碰 `npu_fused_infer_attention_score_v2`（`sinks`/`learnable_sink` 路径，
  本 profile `sinks is None`；后续如需另开一档）；
- 不碰 `DeviceOperator.npu_fused_infer_attention_score`（eager prefill/vision 分支）；
- 不改宿主 `attention_v1.py` / `attention_mask.py`；不改 `attn_metadata` 字段。

### 2.2 判别式（decode 域限定）与等价性

全部条件同时成立才改写，否则**原样透传**（fail-open）：

1. `input_layout == "TND"`；
2. `sparse_mode == 3`（`:1021` 的因果分支）；
3. `pre_tokens == next_tokens == INT_MAX`（无滑窗、无 pre/next 窗口）；
4. `atten_mask` 存在、`dtype == torch.int8`、2 维且方阵（宿主的 2048² 模板指纹）；
5. **纯 decode**：`actual_seq_lengths` 存在且 `len(actual_seq_lengths) == query.shape[0]`，
   且 `actual_seq_lengths[-1] == query.shape[0]`。

第 5 条即"每个请求恰好 1 个 query token"：`actual_seq_lengths` 是正增量的
累加长度，长度等于总数 T 时所有增量只能是 1。它同时排除了 prefill、
chunked-prefill、mixed 批与 spec-decode（`Q_S>1`）——正是算子文档里
"PromptFlashAttention 分支"的补集。

改写内容只有两项：`atten_mask → None`、`sparse_mode → 0`
（`pre_tokens/next_tokens` 保持 `INT_MAX`，与 sparse_mode=0 的文档默认一致）。

**构造性等价论证**（与探针独立）：
`Q_S==1` 时（a）文档明确 `sparse_mode`/`pre_tokens`/`next_tokens` 无效；
（b）decode 的单个 query 位于序列末位，rightDownCausal 模板对齐到右端后
该行**全通过**，模板在这一形态下不排除任何 kv 位置，而 `actual_seq_lengths_kv`
负责截断 padding；（c）`sparse_mode=0` 且不传 mask 的文档语义就是"不做 mask 操作"。
→ 去 mask 与带 mask 数学上精确等价。探针在 B=32 / kv∈{128…6400} 上对
fp32 dense 参考的 `rel_L2` 两臂相同（0.00237 = bf16 底噪）；本插件另做
e2e greedy 逐 token 对拍（见证据文件）。

### 2.3 default-off 与守卫

- `load()`：`VLLM_HUST_FIA_DEMASK != "1"` 且 `VLLM_HUST_FIA_DEMASK_TRACE != "1"`
  时**立即返回**——不 import torch/torch_npu、不 patch、不打日志；
- 使能判定在 `load()` 时固化（不在调用点读 env），图捕获安全；
- **签名漂移 fail-closed**：算子面缺失（`torch_npu` 无该符号 / 无 `.out`）
  → 拒绝安装 + 一条 warning，进程保持原生行为；
- **调用级 fail-open**：判别式抛异常 → 计数 + 一条 warning + 原实参原样透传；
- 失败/拒绝均计数（`stats()`），便于 smoke 断言。

## 3. 已知偏差与未尽事项

- **v2 算子（sinks / learnable_sink）未覆盖**：`full_graph_fia_v2` 与
  `update_graph_params` 的 sinks 分支用的是
  `npu_fused_infer_attention_score_v2`（`sparse_mode=3, next_tokens=0`），
  本 profile 不走；若要覆盖需另做一档（`next_tokens=0` 与 v1 的
  `next_tokens=INT_MAX` 语义不同，判别式不能共用）。
- **`Q_S>1` 不复用**：prefill/chunked-prefill 保留 mask（mode 3 在
  `Q_S>1` 有意义），故本插件对 prefill 无任何影响。
- **收益上限**：探针预测 decode FIA device 时间 −8…−23%（中位 −12%）
  ≈ **3.1% TPOT**（范围 1.7–6.1%）。若 e2e 落在噪声带内，按噪声如实上报。
- 本插件不改任何数值语义，故**不**需要 new kernel；同样不触碰
  `manifest.host.version_range`（仍钉 `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`）。

## 8. activation 面决策（2026-09-11，独立 bundle）

**决策：独立 bundle `org.vllm-hust.fia-demask`，不并入
`org.vllm-hust.split-batch-full-graph`。**

理由：

1. 单能力独立开关——并入意味着 `enable` 同时打开 cascade 与 demask，
   "关 A 开 B"在 manager 面不可表达，违背 default-off 单能力语义；
2. demask 与 cascade 无共享契约（protocols 为空），没有同 bundle 的技术理由；
3. 生命周期不同步：cascade 已 active，demask 刚过验收，独立 bundle 让两者
   各自走各自的 §3/翻 active 节奏。

发现机制约束（`extension-manager/src/vllm_hust_ext/discovery.py`）：

- 每个 `vllm_hust.extension_bundles` entry point 必须指向一个**静态模块目录**，
  目录内**恰好一份** manifest（文件名白名单 `vllm-hust-extension-v0.2.json` 等）；
- 同一 distribution 可注册多个 bundle（各自模块目录独立）。

落地形态：

- 新模块目录 `src/vllm_ascend_split_batch/fia_demask/`（`__init__.py` +
  `vllm-hust-extension-v0.2.json`，bundle_id `org.vllm-hust.fia-demask`，
  `activation.environment = {"VLLM_HUST_FIA_DEMASK": "1"}`，protocols 空数组）；
- 旧 bundle manifest 移除 demask 的 component + implementation 项（避免双注册）；
- pyproject 增加 entry point `"org.vllm-hust.fia-demask" = "vllm_ascend_split_batch.fia_demask"`；
- `tests/test_manifest.py` 新增 `test_fia_demask_bundle_is_import_only_until_activation`
  （守 import_only + 注入键 + version_range 与主 bundle 一致）+ 旧测试补"demask 不残留"断言。

注意：新增 entry point 需要**本地 editable 重注册**才对 importlib.metadata 可见
（`pip install --no-deps -e .`，不触碰任何依赖，torch/triton 配对不受影响）——
这是对该仓库唯一安全的 pip 用法，执行后须复核 `torch==2.13.0` 未动。
