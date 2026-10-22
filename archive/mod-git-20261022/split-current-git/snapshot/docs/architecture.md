# 架构与契约

## 1. 一句话定位

本仓库是 **vllm-ascend 宿主的插件壳**:以 default-off 方式为 vLLM 注入
cascade 两段式 decode(调度 + 图模式)能力,并承载 split-batch 规划契约的
提案描述符。**编译型算子(CCE/Ascend C)实现在仓库外的 `ascend_kernel`
wheel 里**;**Triton / 纯 Python kernel 载体则在本仓内**(见 §6),本壳统一
只做适配、门控与调度决策。

## 1.1 仓库命名约定(团队沟通用语)

| 简称 | 仓库 | 远端 |
|---|---|---|
| **插件库**(本仓库) | `vllm-ascend-split-batch-hust`,唯一对外交付面 | `vLLM-HUST/vllm-ascend-split-batch-hust` |
| **算子库** | `cascade-merge-op`(ascend-kernel 工程 + S1 锚点数据,wheel 工厂) | `Raing5Days/vllm-hust-cascade-kernel` |

**插件库 ⊃ 算子库**(逻辑包含):本库经 `kernels` extra 钉版消费算子库
wheel,单向依赖;算子库对 vllm-hust-ext 与插件机制零感知。日常工作流见
算子库 `docs/workflow.md` Loop B。

## 2. 三层关系

```text
┌─ 宿主(只读,禁止修改)────────────────────────────────────────────┐
│  vllm 0.28.1.post1.dev143+gf18cf803c (VLLM_TARGET_DEVICE=empty) │
│  vllm-ascend 0.25.1rc2.dev125+hust.20260903.4                   │
└──────────────┬──────────────────────────────────────────────────┘
               │ 唯一官方入口: vllm.general_plugins
               │ entry `cascade-attention` → cascade_plugin:load()
┌──────────────▼──────────────────────────────────────────────────┐
│  本壳 vllm-ascend-split-batch(纯 Python)                        │
│  · bundle manifest(0.2-experimental, cascade active)            │
│  · env 门控 + monkeypatch(见 HOST_CONTRACT.md)                  │
│  · 纯逻辑 planner / gate(CPU 可测)                              │
└──────────────┬──────────────────────────────────────────────────┘
               │ 单向依赖: torch.ops.npu.*
┌──────────────▼──────────────────────────────────────────────────┐
│  ascend_kernel wheel(CCE 算子, 仓库外)                          │
│  fa_fp32_stage1 / lse_merge;对本壳零感知                        │
└─────────────────────────────────────────────────────────────────┘
```

> 图中的版本号只是层级示意,**现行验证域以 [release.md](release.md) §0.1 为准**:
> vllm `0.28.1.post1.dev143+gf18cf803c`(vllm-hust release v1)/ vllm-ascend
> `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`。旧基线 `vllm 0.23.0` /
> `vllm-ascend 0.23.0rc1` 仅在 `docs/evidence/cascade/EVIDENCE.md` §8 留档,
> 其参考源码树已删,只作历史数字对照,不继承。

依赖方向铁律:**本壳 → kernel 包**单向;kernel 包禁止 import 本壳或任何
vllm 模块;宿主源码不可修改,宿主能力只走公开面(entry point、
`--additional-config`、`--worker-cls`、官方 CLI/env)。

## 3. Extension Bundle 语义(vllm-hust-ext 0.2-experimental)

| 概念 | 本仓库的取值与含义 |
|---|---|
| Bundle ID | `org.vllm-hust.split-batch-full-graph`,注册于 `vllm_hust.extension_bundles` entry point,value 指向包目录(含 manifest JSON) |
| `implementation[].status` | `active` 才可 `enable`。当前 cascade 两个 carrier(`cascade_plugin:load` / `cascade_graph_plugin:install`)为 `active`;`planner:plan_dual_pad` 仍是 `import_only`(无验收证据,review F7) |
| `activation.environment` | **enable 时注入的值**,不是文档。当前声明 `ENABLE_CASCADE_DECODE=1` + `ENABLE_CASCADE_GRAPH=1`;不 enable 则什么都不注入 |
| `protocols[].version_range` | 四个协议均 `null`(**不独立版本化**):宿主不存在 `vllm.plugins.contracts`,manager 探测不到版本;声明具体区间只会让 `run` 以"version is unavailable"拒启。约束由 `host.version_range` + 验收证据承担,理由详见 [release.md](release.md) §0 |
| `host.version_range` | 声明**有界的已验证域**(2026-09-27 起 `>=0.25.1rc2.dev125,<0.25.2`,见 release.md §0.1/§0.3);两端点均已真机验证,窗口内 build 声明兼容 ⇒ 窗口内失败归因宿主;禁止 `>=0` 与任何无上界写法(由 `tests/test_manifest.py` 机械守住),且 packaging 不允许有序比较符内出现 `+local` 标签 |
| `components[].contracts` / `execution_planes` | `dual-pad-planner` 与 `cascade-two-stage-decode` 声明在 worker/device plane,permissions 含 `device_access` |
| 发现 vs 导入 | `extension list/inspect/check` 只读元数据不 import 实现;import 实现的只有 vllm 的 `general_plugins` 加载(`load()` 内部再按 env 门控) |

manifest 与 entry point 打包完整性由构建检查保证,流程见 [release.md](release.md)。

## 4. default-off 的双层保证

1. **门控层**:所有 env 开关默认缺省,`load()` 与被 patch 方法首行检查
   gate,未开启时逐字委托原实现——宿主行为 bit-identical(README 有验证证据)。
2. **bundle 层**:bundle 的 cascade carrier 虽为 `active`,但 `enable` 只是
   允许 manager 注入 `activation.environment`;不 enable 则什么都不注入,
   即使用户手工装了 wheel,不改 env 也零差异。

附加互斥:任何 microbatching(`use_ubatching`: DBO 或 `ubatch_size>1`)下
强制回落标准路径,与官方 core gate 对齐。

## 5. monkeypatch 面(弱契约,重点维护区)

本壳对宿主的实际控制面 = 以下方法/对象的运行时包装(权威清单与约束见
[HOST_CONTRACT.md](../HOST_CONTRACT.md)):

- `vllm.v1.cudagraph_dispatcher.add_cudagraph_key` / `dispatch`
- Runner 私有方法:`_capture_cudagraphs`、`_warmup_and_capture`、
  `_determine_batch_execution_and_padding`、`_model_forward`、
  `_update_full_graph_params_if_needed`
- vllm-ascend:`update_full_graph_params` / `get_graph_params` / `GraphParams`
- `ACLGraphWrapper` variant-entry 侧表(按标准 `BatchDescriptor` 键)
- `ACLGraphWrapper._updatable_graph_replay`(**只在新图重放机制的宿主上存在**;插件在宿主
  重放**之前**做 cascade 重参数化,旧宿主自动 no-op —— 机理见 [pitfalls.md](pitfalls.md) §2.4)
- active runner 的 `load_model`(D4 `gate-fp32` 与 F2 迁移载体 `gdn-merge-inproj`
  都只包这一个方法,各自幂等标记 + 各自 `uninstall()`;后者当前是 **fail-closed**,
  见 HOST_CONTRACT.md 的 GDN 段)
- **提案补丁**（非插件入口）：`W1R-GMM` 的 `tuning_config` / counts dump
  只改宿主 `routed_experts.apply_gmm1/apply_gmm2` 两处调用，承载目录
  `host-proposals/W1R-gmm-tuning-config/`，见 HOST_CONTRACT.md 末节

`gdn-merge-inproj`(`gdn_merge_inproj_plugin.py`)是已验收 F2(加载期
`in_proj_ba`→`in_proj_qkvz` 权重拼接)从运行树补丁**迁移**到插件库的载体:加载期权重转换
已实现且有 CPU 单测,但宿主 forward 没有可安全消费合并权重的 seam,故载体 default-off 且
**拒绝改写任何权重**(fail-closed,一条 warning;blocker 见 HOST_CONTRACT.md)。其性能数字
**必须**在当前口径 `mmlu-pro-eval-traffic` 上重新 L4 验收,已验收的 F2 数字属于运行树补丁,
不随迁移继承。

这是与 bidkv 类"宿主主动调插件"的 typed contract 最大的不同:**宿主没有
给我们的 seam,契约靠我们单方面遵守 + `host.version_range` 钉死**。宿主
升级时必须按 [release.md](release.md) 第 4 节逐项核对签名。

## 6. 与 kernel 库的边界

- **形态边界(2026-10-13 明确)**:CCE / Ascend C / CATLASS(编译成 `.so`)→
  仓库外 `ascend_kernel` wheel;**Triton / 纯 Python kernel → 本仓 default-off
  载体**(先例 W3.1 `fi_gelu`、`fi_sampling/kernels.py`),`triton` lazy import、
  关闭时零 import。不得把 Triton 塞进 kernel 库,也不得改宿主只读的
  `vllm_ascend/ops/triton/`。
- 调用面只有 `torch.ops.npu.fa_fp32_stage1` / `torch.ops.npu.lse_merge`
  (`import ascend_kernel` 即注册);wheel 是软依赖,仅在 `[kernels]` extra
  声明,基础依赖不含。
- wheel 缺失/注册失败 → cascade 功能**整体自动禁用**(fail-open 到标准
  full-KV 路径,单条 warning,`load()` 不抛错)。守卫点:`load()` 探针
  (`_probe_kernel_wheel`)、eager gate(`_use_cascade_attention`)、graph
  twin capture(`cascade_runner_patch._cascade_wheel_ready`)、gate bench
  子进程(`cascade_gate_self` 导入守卫,exit 3 → gate neutral)。
- kernel 侧的兼容四元组(平台 tag / PyABI / torch_npu / CANN)、链接纪律、
  已知危害,见 kernel 仓库 README §4-§5(工作区
  `ops/kernels/ascend-kernel/`),本仓库不重复维护。
