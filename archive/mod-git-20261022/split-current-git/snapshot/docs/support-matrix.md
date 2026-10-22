# 支持矩阵（能力 × 准入条件 × 代码位置）

> 用途：回答「某个能力在我的配置下到底会不会生效、不生效是因为哪一条」。
> 每行都给出**判定代码位置**，可直接对照。状态口径：
> **`active`** = manifest 允许启用且已过启用验证；**`import_only`** = 描述符在、但无验收证据或缺少宿主 seam，
> 管理器拒绝 enable；**default-off** = 启用后仍需运行时开关。
>
> 本表是**功能维度**矩阵；宿主/环境/硬件/已验证版本域见 [release.md](release.md) §0.1。
> 逐项数字与原始证据见 [evidence/cascade/](evidence/cascade/README.md)。

## 1. 状态总览

| 能力 | bundle / entry point | manifest 状态 | 运行时开关 | 一句话结论 |
|---|---|---|---|---|
| dual-pad planner（split-batch 规划） | `org.vllm-hust.split-batch-full-graph` / 无 entry（纯函数 `plan_dual_pad`） | **import_only** | `SplitBatchConfig.enabled` | **不可用**：宿主未提供 `vllm.graph.runtime-key.v1` 等 seam，只有纯逻辑 + 预检 |
| cascade 两段式 decode（eager） | 同上 / `vllm.general_plugins: cascade-attention` | **active** | `VLLM_ASCEND_ENABLE_CASCADE_DECODE=1` | 可用，受 §2 全部条件约束 |
| cascade 图孪生（aclgraph） | 同上 / 同一 `load()` 内的 `cascade_graph_plugin.install()` | **active** | 再加 `VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1` | 可用，另需 §3 的图条件 |
| fia-demask（decode FIA 去冗余掩码） | `org.vllm-hust.fia-demask` / `fia-demask` | **active** | `VLLM_HUST_FIA_DEMASK=1` | 可用；只对"纯 decode + Q_S==1/请求"形态生效 |
| rope-fix（fork RoPE 三缺陷覆写） | `org.vllm-hust.rope-fix` / `rope-fix` | **import_only** | `VLLM_HUST_ROPE_FIX=1` | 三证据未齐，管理器拒 enable |
| zerocost-wiring（prefill `.out` + 跳过 cos/sin） | `org.vllm-hust.zerocost-wiring` / `zerocost-wiring` | **active** | `VLLM_HUST_FI_PREFILL_OUT=1`、`VLLM_HUST_SKIP_COS_SIN=1` | 可用 |
| gate-fp32（MoE gate 权重 fp32 外提，D4 const-hoist；归零每步 `Cast[256,2048]`） | `org.vllm-hust.gate-fp32` / `gate-fp32` | **active** | `VLLM_HUST_GATE_FP32=1`（默认关） | 可用；证据在 **0.28/Qwen3.5 基准**，cohort 见 [cohort-and-ownership.md](cohort-and-ownership.md) §2 与 [evidence/d4-gate-fp32-cohort.md](evidence/d4-gate-fp32-cohort.md) |
| mlp-chunk（MLP token 分块） | 无 bundle manifest / `mlp-chunk` | — | `VLLM_HUST_MLP_CHUNKS`（默认 1 = 关）、`VLLM_HUST_MLP_CHUNK_MIN_TOKENS` | 插件可加载；**未收录 manifest**，即不经管理器发布面 |
| gdn-merge-inproj（GDN `in_proj_ba`→`in_proj_qkvz` 加载期权重合并，F2 迁移载体） | `org.vllm-hust.gdn-merge-inproj` / `gdn-merge-inproj` | **import_only** | `VLLM_HUST_GDN_MERGE_INPROJ=1`（默认关） | **当前不生效且不改任何权重（fail-closed）**：宿主 `AscendGatedDeltaNetAttention.forward` 没有可安全消费合并权重的 seam，载体只打一条 warning；已验收的 F2 数字属于运行树补丁，**不随迁移继承**，须同口径重新 L4 验收（判定代码 `gdn_merge_inproj_plugin.forward_consumer_available`，blocker 见 [../HOST_CONTRACT.md](../HOST_CONTRACT.md) GDN 段） |
| aot-cache-guard（AOTAutograd 缓存 guard） | 无 bundle manifest / `aot-cache-guard` | — | **默认开**（`VLLM_HUST_AOT_CACHE_GUARD=0` 关） | 验收前置修复；宿主自修后变 no-op |
| W1R GMM `tuning_config`（**提案补丁**，非插件 entry） | 无 bundle manifest | **未发布（tier-2 proposal，未提交上游）** | `VLLM_HUST_GMM_TUNING_CONFIG`（默认关） | 只改 `apply_gmm1`/`apply_gmm2` 两处 GMM 的 `tuning_config`；承载目录 `host-proposals/W1R-gmm-tuning-config/`；**单值**（列表被拒）；未设时无设备差异、零 shape/IO/import，但每次 GMM 仍有两次早退 Python 调用（非字节等价） |
| W1R GMM counts dump（**提案补丁**，非插件 entry） | 无 bundle manifest | **未发布（tier-2 proposal）** | `VLLM_HUST_GMM_COUNTS_DUMP=<path>`（默认关） | 只记录 group_list counts/形状/provenance，不改变输出；capture 内按 `sample_source` 跳过；**dump 产物不是计时产物**；active FULL graph 下 M=128 真实 decode counts **取不到** |
| fi-gelu / fi-sampling | 无 bundle manifest / `fi-gelu`、`fi-sampling` | — | `VLLM_HUST_FI_GELU`、`VLLM_HUST_FI_SAMPLING` | 能力在、证据显示 e2e 收益在噪声带内，保持默认关 |

## 2. cascade 准入条件（任意一条不满足 → 回原生 FIA）

判定函数：`cascade_plugin._use_cascade_attention`（`src/vllm_ascend_split_batch/cascade_plugin.py`）。

| # | 条件 | 代码位置 / 默认值 |
|---|---|---|
| 1 | `VLLM_ASCEND_CASCADE_STRICT=0` | `cascade_plugin.py`（STRICT 分支） |
| 2 | `VLLM_ASCEND_ENABLE_CASCADE_DECODE=1` | `_cascade_gate_env()`；未置 1 时 `load()` 只打启动标记行即返回 |
| 3 | kernel wheel 可用（`ascend_kernel` 可导入且 `torch.ops.npu` 注册了 `fa_fp32_stage1` / `lse_merge`） | `_probe_kernel_wheel()`；缺失 → 整体禁用 + 单条 warning |
| 4 | 无 microbatching（`use_ubatching` 为假） | `_use_cascade_attention` 的 ubatch 分支 |
| 5 | `VLLM_ASCEND_CASCADE_PRECISION ∈ {bf16, fp32}` | 非法值 → 单条 error + 禁用 |
| 6 | 非 alibi / 非滑窗 / 非 local attention / `dcp_world_size == 1` | 同上 |
| 7 | **每请求恰好 1 个 query row**（MTP 校验、chunked prefill 不满足） | `any(int(length) != 1 for length in query_lens)` |
| 8 | `speculative_config is None` | 同上（投机解码整体 fail-closed） |
| 9 | shared prefix ≥ `VLLM_ASCEND_CASCADE_MIN_PREFIX`（默认 **8192**） | 同上 |
| 10 | 请求数 ≥ `VLLM_ASCEND_CASCADE_MIN_REQS`（默认 **32**） | 同上 |
| 11 | 自适应 gate 对该 (batch, prefix) 桶判 `on` | `cascade_gate.py`；未 bench 的桶默认 on；`P ≤ 4096` 桶要求 ≥25% 余量 |

> 由 9/10 可见：**当前并发 `< 32` 或共享前缀 `< 8192` 时 cascade 一定不触发**。
> e2e 收益区间（共享前缀 8k/16k × B32/64，Qwen2.5-Coder-14B 替身，910B2 单卡；
> **历史 Qwen2.5/0.25 域**，勿与 0.28/Qwen3.5 基准互推）：
> TPOT −14.5% / −18.5% / −32.5%（`evidence/cascade/section3-performance.md` §13.5）；
> 4.4k 小前缀是亏区，由 11 的 gate 回落覆盖。

## 3. cascade 图孪生额外条件

| # | 条件 | 说明 / 代码位置 |
|---|---|---|
| 1 | `VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1` | `_graph_plugin_enabled()` |
| 2 | `cudagraph_runtime_mode == FULL` **可达** | `cascade_runner_patch` 捕获包装器；`cudagraph_mode` 写单值 `"FULL"` 会因 `separate_routine()` 为假而静默跳过全部桶（见工作区 `AGENTS.md` 陷阱 8），用 `FULL_AND_PIECEWISE` |
| 3 | 非 sparse / 非 compress / 无 ubatching / 无投机解码 | `_capture_cudagraphs` 包装器（`_spec_decode_active()` 新增） |
| 4 | capture sizes 建议收到 `[32,64,128]`（`--gpu-memory-utilization 0.85` 时） | 默认网格会在 twin 捕获期 NPU OOM → capture 级 fail-open（`evidence/cascade/section4-active-enablement.md`） |

## 4. dual-pad planner 为什么不可用（不是配置问题）

`planner.plan_dual_pad` 是纯函数，预检在 `planner.precheck_reason`（`src/vllm_ascend_split_batch/planner.py`）：

| 预检项 | 返回值 |
|---|---|
| 未启用 / mode ≠ `dual_pad` | `mode_disabled` |
| 非 uniform decode | `non_uniform_decode` |
| graph_mode ∉ {full, piecewise} | `graph_mode_not_supported` |
| 投机解码 | `speculative_decode_conflict` |
| LoRA / MLA / M-RoPE | `lora_conflict` / `mla_conflict` / `mrope_conflict` |
| 请求数 < `min_batch_size_for_split` | `batch_too_small` |

即使全部通过，**也没有执行面**：manifest 的四个协议（`vllm.graph.runtime-key` / `vllm.forward.split-context` /
`vllm.ascend.graph-pool` / `vllm.worker.split-executor`）`version_range` 均为 `null`，宿主不存在对应的 typed
contract（见 [release.md](release.md) §0）。这是"能力预览"而非可用能力，管理器据此拒绝 enable。

## 4.1 宿主窗口（2026-09-27 起）

`host.version_range = >=0.25.1rc2.dev125,<0.25.2`（**有界兼容窗口**，非点钉）。窗口两端点均已真机验证
（`dev125` 历史；`dev605` = `fbe4911bb`，证据 `VERIFY-PROGRAM-20260927.md`）；窗口内其它 build
**声明兼容**，在那里出问题按宿主侧发现处理（裁定见 [release.md](release.md) §0.3）。
窗口外（`<0.25.1rc2.dev125` 或 `>=0.25.2`）判 `incompatible`，`vllm-hust-ext run` 拒启。
⚠️ 该区间只覆盖 **vllm-ascend**，不含 vllm core：asc `fbe4911bb` 必须配 core `0aee727ff6`（§0.3 配对警告）。

## 5. 兼容禁区（两侧一致 fail-closed 的部分）

| 形态 | dual-pad planner | cascade |
|---|---|---|
| 投机解码 / MTP | 拒（`speculative_decode_conflict`） | 拒（条件 7/8） |
| MLA / M-RoPE / LoRA | 拒 | 未显式判（cascade 只判 alibi/滑窗/local/dcp）→ **未验证域**，不要据本表推断可用 |
| 多 query row（MTP 校验、prefill） | 拒（非 uniform decode） | 拒（条件 7） |
| microbatching（DBO / ubatch>1） | 未判 | 拒（条件 4） |

## 5.1 证据的 cohort 边界（2026-10-13 基准切换；勿跨域外推）

> **权威源**：[cohort-and-ownership.md](cohort-and-ownership.md)（含四条引用纪律与四项归属）；
> 本节只给"某配置下会不会生效"的读者一个入口。
> 上游 2026-09-28 审计（issue #2 评论 `5867233862`，对 `f77dc2214727ab1448874f1aee787a67328f1433`）
> 原文与"未发现形式化成功点 / 不得外推"结论保留在权威源 §1（逐字历史收据）；
> **owner 已于 2026-10-13 把基准切到 0.28/Qwen3.5**，下表为现行口径。

| 本仓证据覆盖（0.28 基准） | **不覆盖**（不得外推） |
|---|---|
| 0.28 镜像栈：vllm `0.28.1.post1.dev143+gf18cf803c` + vllm-ascend `0.25.1rc2.dev125+hust…`，CANN 9.1.0，2×910B2 **TP=2** | 实测 0.28 镜像之外的其它宿主 build |
| Qwen3.5-35B-A3B（40L：30 GDN + 10 MoE）hybrid **TP2** | native **MTP2** 的 k+1 verification rows / 任何投机解码 |
| chunked prefill **on**、APC **off**、`cudagraph_mode=NONE`、bf16 | 未跑过的 `async` 组合 |
| — | **已退役的 Qwen2.5 / 0.25 域**（历史独立专项，见 [evidence/cascade/](evidence/cascade/README.md)） |

⇒ 投机解码（含 MTP2）在两个域都 **fail-closed**；那上面"cascade ON"跑的是**原生路径**，
**不构成 cascade 效果**。要进统一 cohort，前置仍是"支持 k+1 verification rows 的 dual-pad/graph
bucket 合同 + 该配置上的真实 replay correctness"—— 本仓**未立项**（与 OPEN-05 裁定一致）。

## 6. 扩展 `configuration` 字段：本仓**不消费**（边界声明）

`vllm-hust-ext configure <id> --file <json>` 会把 JSON 原样存进该扩展的
`ExtensionConfig.configuration`：上游只校验"文件顶层是 object"
（`extension-manager/src/vllm_hust_ext/cli.py:185-187` 的 `isinstance(..., dict)`），
**没有** per-bundle schema，未知键既不拒绝也不解释。

本仓的准入开关**全部是 env**（上表第 4 列；`src/**` 里 35 处 `os.getenv`/`os.environ`，
无任何 `.configuration` 访问）⇒

- 放什么进 `configuration` **不改变任何判定**（含 `extension check` 的 states）；
- 因此本仓**没有**"未知配置拒绝"要加 —— schema 属上游框架，本仓不是它的消费方；
- 若将来某个载体真要读 `configuration`，必须同时改
  `tests/test_extension_config_boundary.py`（它会红）并在本表登记该键。

机械判据：`tests/test_extension_config_boundary.py`（3 例：env-only 门控面、
未知键下 check 输出逐字相同、configure→enable→disable→forget 只写元数据且不 import 实现）。
服务级 disable/uninstall→重启的验收仍未做（需占卡，登记见 `release.md` §8）。
