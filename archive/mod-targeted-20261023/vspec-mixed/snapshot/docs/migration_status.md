# vSpec 迁移状态

更新时间：2026-09-03<br>
插件版本：0.12.0

## 结论

Draft、线性 EAGLE 和 EAGLE3 的主要运行时实现已经迁入插件目录。插件不再只把
环境变量转发给当前工作区中的定制源码；在基础提交的干净 worktree 上也能注入
active-vocab、W8A16、Target projection 和 rejection sampler 并完成 NPU 推理。

插件仍依赖 vLLM-HUST 和 vLLM-Ascend-HUST 提供基础 scheduler、attention、KV
cache 和 model-runner ABI。固定宽度 tree/Target-width 等跨 scheduler 的功能尚未
变成纯插件实现。

## 最新宿主迁移

| 组件 | Revision | Worktree |
|---|---|---|
| vLLM-HUST | `762f85b311fbab0bcf8921dd216f5093cd58b9b8` | `/root/data/vllm-hust-latest` |
| vLLM-Ascend-HUST | `4e57439e58ed3d78e675f9fd7b4614fb183c5394` | `/root/data/vllm-ascend-hust-latest` |

Ascend 主线通过 `.github/vllm-main-verified.commit` 固定上述 vLLM-HUST revision；
Ascend 版本为 `0.25.1rc1+hust.20260903.4`。插件已迁移到该配对版本，并保留旧宿主
`_draft_runtime_compilation_context` ABI 的兼容路径。

最新 vLLM-HUST 已移除内置 `qwen2_eagle.py`，因此 vSpec 现在自带
`Qwen2ForCausalLMEagle` 并通过 `ModelRegistry` 延迟注册。新版 Ascend 也移除了
`_draft_runtime_compilation_context`，vSpec 会在 Draft 模型加载、`dummy_run` 和
`_propose` 时显式切换到独立 Draft TP group。

旧开发分支的 `VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL` 私有开关没有进入新版主线；
新版已在原生 padded proposer 中传递 `num_rejected_tokens_gpu` 完成异步状态修正，
因此 relaxed 预设不再启用该旧开关。

最新版依赖已安装在独立环境 `/opt/vllm-hust-cann91`：CANN 9.1、
torch/torch-npu 2.13 和源码构建的 triton-ascend 3.6。宿主默认 CANN 9.0 未被覆盖。
新环境已通过 torch-npu、Triton NPU、自定义算子、插件和 Manager 验证；14B 单卡
端到端复测因当前所有 NPU 仅余约 8 至 9.5 GiB 显存而等待资源。完整记录见
`docs/latest_stack_validation_20260903.md`。

本轮最新版迁移验证结果：

- 最新 CANN 9.1 最终 wheel 安装态完整插件测试：`149 passed, 15 warnings`；
- torch-npu 矩阵乘法和 Triton-Ascend vector-add NPU 冒烟通过；
- vLLM-Ascend 自定义算子完成源码构建并可动态加载；
- 新版 multi-layer KV-cache 平台检查已由可自动失效的插件兼容补丁覆盖；
- 单卡 TP1 Target-only、串行 self-Draft gamma2 和 GSM8K 评分入口 NPU 冒烟通过；
- 新环境 `pip check` 无依赖冲突；
- 旧宿主回归：`146 passed, 1 skipped`；
- Draft、EAGLE strict、EAGLE relaxed、EAGLE3 四个预设均完成真实 patch 注入；
- Qwen2 EAGLE 模型通过 `ModelRegistry` 隔离子进程导入；
- Extension Manager inspect/enable/run/disable/forget 生命周期通过；
- `manage.sh` wheel/editable 安装、完整准入与渲染、托管启动、精确升级/回退和受控
  卸载流程通过；
- `_version.py`、独立 manifests 包、wheel/sdist 审计与 tag 发布门禁已按 0.2 打包指南补齐；
- ruff lint 和 format check 通过。

## 插件所有

| 能力 | 实现位置 | 验证 |
|---|---|---|
| Draft merged FULL/compact PIECEWISE、padding、KV slot mapping | `backends/draft.py` | 单测、既有 NPU smoke |
| 新版 multi-layer KV-cache 平台能力声明 | `backends/kv_cache.py` | 单测、最新版宿主 self-Draft NPU smoke |
| 线性 EAGLE/EAGLE3 无逐行 D2H 验收 | `backends/eagle_rejection.py` | 单测、NPU smoke |
| Draft/Target FP16 active-vocab | `eagle_draft.py` / `eagle_target.py` | 干净宿主 NPU |
| Draft W8A16/W8A8 分块 LM-head | `backends/eagle_draft.py` | W8A16 干净宿主 NPU；W8A8 CPU/静态 |
| strict/relaxed/prefix acceptance | `eagle_target.py` / `eagle_rejection.py` | 单测、relaxed NPU |
| ACL Graph event ordering | `backends/eagle_graph.py` | 当前宿主 NPU；干净 ABI 注入检查 |
| speculative metadata cache | `backends/eagle_metadata.py` | 干净 ABI CPU 行为测试 |
| Draft compile control、共享模块隔离、Target hidden 保留 | `backends/eagle_host.py` | 当前/干净 ABI 注入检查 |
| KV 首次清零和诊断 trace | `backends/eagle_runtime.py` | 单测/静态检查 |
| Qwen2 EAGLE QKV bias 初始化 | `backends/eagle.py` | 插件加载检查 |
| vSpec Adaptive profile/Online-UCB Goodput 控制、当前轮逐请求置信停止、逐位置 AC、异步状态、gamma=0 和多 gamma Graph | `adaptive/` | 单测、Qwen2.5 Draft/EAGLE NPU A/B、置信停止 NPU A/B |
| DFlash 并行 proposer 动态 gamma/slot width hook 和并行 latency profiler | `adaptive/` | 当前宿主 ABI/单测；缺 checkpoint，未做 NPU E2E |
| 宿主版本/API 检查 | `compatibility.py` | 当前/干净 ABI doctor |

## 宿主边界

以下能力仍要求配套宿主源码，插件会在显式启用且宿主缺失时立即报错：

1. `VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL`：内核调用嵌在完整 input preparation
   状态机中，涉及 async accepted-token correction 和多个持久 buffer。
2. 固定宽度 EAGLE tree、Target-width 和 tree Graph commit：同时改变 vLLM
   scheduler output、spec metadata、KV ownership 和提交路径。
3. `VLLM_ASCEND_EAGLE_DRAFT_IO_TRACE_DIR`：采集单层 EAGLE 的 QKV/RoPE/attention
   中间值，依赖模型内部逐层位置；它是诊断能力，不在吞吐热路径中。
4. vSpec Adaptive 的 DFlash 并行动态 gamma 已接到 vLLM-Ascend 原生 proposer，
   但本机没有匹配 checkpoint，尚缺 NPU 端到端验证。仍缺 PLD 独立
   proposal/verification 长度，以及同一模型实例内切换 gamma 的低成本 profiler。
   `gamma=0` sticky 是零 Draft 计算但按 cohort 恢复；sync 可同请求恢复但会运行
   gamma=1 shadow Draft。连续请求下同时满足零计算和周期探测需要 per-request
   混合 Draft 状态，当前没有实现。
5. `FULL_DECODE_ONLY` 的 Target graph params 按 query width 隔离；组合 `FULL`
   必须使用共享 Target table，否则共享 mixed event family 在重复回放时可能死锁。
   Draft graph params 在两种模式下都按 gamma 隔离。组合 `FULL` 的 gamma=0 width-1
   step 仍因静态 FIA metadata 冲突安全回退 eager。

不依赖离线 profile 的 `online` 策略已在 Qwen2.5 Draft/EAGLE 上完成 FULL Graph、
async scheduling、GSM8K B128/N200 验证。完整探索 1..4 四个 arm 的 Draft
Online-UCB 相对固定 gamma3 加速 1.0417x；叠加当前轮置信停止后加速 1.0164x。
固定 gamma2 仍是该组扫描的 oracle；EAGLE Online-UCB 比固定 gamma2 慢 1.65%，
因此 EAGLE B128 当前仍推荐固定 gamma2。详见
`docs/vspec_online_adaptive_npu_report_20260902.md`。

这些边界不能通过单个公开类方法的轻量 wrapper 完整表达。直接复制整个 scheduler
或 model runner 到插件会形成第二套不可维护的运行时，因此当前采用受控 ABI 依赖，
由 `vllm-hust-vspec-doctor` 和启动时检查共同约束。

## 检查命令

```bash
./manage.sh install
./manage.sh status
vllm-hust-vspec-doctor --method draft
vllm-hust-vspec-doctor --method eagle
vllm-hust-vspec-doctor --method eagle3
vllm-hust-vspec-doctor --method dflash
/root/.venvs/vllm-hust-latest/bin/python -m pytest -q
./manage.sh uninstall
```
