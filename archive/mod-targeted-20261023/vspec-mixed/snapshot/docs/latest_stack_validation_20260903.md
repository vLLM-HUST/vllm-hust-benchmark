# 最新软件栈验证记录

更新时间：2026-09-04 UTC<br>
插件版本：0.12.0

## 验证范围

本轮在独立环境 `/opt/vllm-hust-cann91` 中验证最新版 vLLM-HUST、
vLLM-Ascend-HUST 和 vSpec。宿主默认 `/usr/local/Ascend/cann` 保持 CANN 9.0，
没有被覆盖。

所有投机解码配置保持单卡串行：Target
`tensor_parallel_size=1`，Draft `draft_tensor_parallel_size=1`。本轮没有使用 TP2。

| 组件 | 版本或 revision |
|---|---|
| Python | 3.11.16 |
| CANN | 9.1.0 |
| PyTorch | 2.13.0+cpu |
| torch-npu | 2.13.0rc1 |
| Triton-Ascend | 3.6.0+git148a35c0 |
| vLLM-HUST | `762f85b311fbab0bcf8921dd216f5093cd58b9b8` |
| vLLM-HUST distribution | `0.17.2rc1.dev5871+g762f85b31.empty` |
| vLLM-Ascend-HUST | `4e57439e58ed3d78e675f9fd7b4614fb183c5394` |
| vLLM-Ascend-HUST distribution | `0.25.1rc1+hust.20260903.4` |

## 已通过

- CANN 9.1 环境下 torch-npu 矩阵乘法通过。
- 源码构建的 Triton-Ascend NPU vector-add 通过，最大误差为 `0.0`。
- vLLM-Ascend 自定义算子完整构建，`vllm_ascend_C` 可从新环境动态加载。
- vSpec 会在新版 Ascend 尚未声明时补齐多 attention 层 KV-cache 平台能力，允许
  Target 和 Draft 使用相同层号但按完整层名绑定各自 cache。
- `pip check` 返回 `No broken requirements found.`。
- vSpec Draft doctor 的 11 项能力全部可用且兼容。
- vSpec EAGLE doctor 的 15 项能力全部可用且兼容。
- Adaptive EAGLE 所需 Scheduler、Graph dispatcher、async output、GraphParams 和
  NPUModelRunner 动态接口全部可用。
- Extension Manager 的 list、validate、inspect、check、plan、render、enable、status
  和两种预设的 managed-run dry-run 通过。
- 最终 wheel 安装态完整插件测试：`154 passed, 15 warnings in 62.50s`。
- Ruff 0.16.6 静态检查通过。

## 单卡小模型冒烟

为在每张卡仅余约 9.3 GiB 时验证真实引擎，使用
`Qwen2.5-0.5B-Instruct`、eager、B2、N2、`max_tokens=8`、
`gpu_memory_utilization=0.10` 运行。该配置只验证执行链路，不作为性能结论。

| 路径 | Target TP | Draft TP | Gamma | 输出 token | 生成耗时 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Target-only ShareGPT | 1 | - | 0 | 16 | 0.714 s | 通过 |
| vSpec self-Draft ShareGPT | 1 | 1 | 2 | 16 | 6.347 s | 通过，输出哈希逐条一致 |
| Target-only GSM8K | 1 | - | 0 | 16 | 0.715 s | 通过，评分字段成功落盘 |

self-Draft 让同一个 0.5B 模型同时承担 Target 和 Draft，并包含首次 speculative
sampler 编译，不能与正常的 14B + 0.5B 组合比较吞吐。它确认了插件注册、两套模型
加载、Draft proposal、Target verification、rejection sampler 和输出提交路径。

首次 self-Draft 启动发现 vLLM 新增了
`check_runner_kv_caches_multi_layer` 平台能力检查，而最新版 Ascend 尚未覆盖该方法。
vSpec 现只在 `NPUPlatform` 没有原生实现时补充能力声明；Ascend 后续原生实现后插件
自动不再覆盖。修复后两套 KV cache 成功绑定，且生成输出与 Target-only 一致。

## NPU 端到端状态

Qwen2.5-14B 单卡串行的 Target-only、Qwen2.5-0.5B Draft 和 EAGLE 冒烟均已通过。
B2/N2/`max_tokens=16` 下两种投机路径均生成 32 个 token，且输出与 Target-only
逐条哈希一致。

GSM8K B128/N200/`max_tokens=512`、BF16、FULL Graph、async、prefix caching 的
正式性能组也已完成。所有路径保持 Target TP1、Draft TP1，没有使用 TP2：

| 路径 | 输出吞吐 (tok/s) | 相对 Target-only | GSM8K |
|---|---:|---:|---:|
| Target-only Graph | 1573.13 | 1.0000x | 94.0% |
| Draft fixed gamma=2 | 1762.50 | 1.1204x | 93.0% |
| Draft Online gamma=1-4 | 1715.70 | 1.0906x | 93.5% |
| EAGLE fixed gamma=2 | 2176.62 | 1.3836x | 92.0% |
| EAGLE Online gamma=1-2 | 2296.65 | 1.4599x | 93.0% |

详细配置、耗时、输出 token、动态选择分布和结果路径见
`docs/latest_stack_gsm8k_graph_performance_20260904.md`。

本轮同时修复最新版宿主的 EAGLE metadata、Scheduler 动态 batch hook 和
`ACLGraphWrapper.__init__` 三处 API 变化。完整测试、Ruff、release audit、`pip check`
和 Extension Manager inspect/check 均已通过。

## 产物

- vSpec wheel：`dist/vllm_hust_vspec-0.12.0-py3-none-any.whl`
- vSpec sdist：`dist/vllm_hust_vspec-0.12.0.tar.gz`

两个产物均已通过 `release.sh check`，最终 SHA256 由构建审计输出记录。
- Triton-Ascend wheel：
  `/root/data/triton-ascend-hust-3.6/dist/triton_ascend-3.6.0+git148a35c0-cp311-cp311-linux_aarch64.whl`

发布到 PyPI 不属于本轮验证的前置条件。本地 wheel 安装、卸载和 Manager 生命周期
均不需要 PyPI 凭据；正式发布时应使用 Trusted Publishing 或项目级 token。
