# 发行构建关联：第二轮 TP4 功能资格

第二轮硬件控制器于 2026-10-25 20:02:10–20:11:02 UTC 完成，记录 `passed=true`、`errors=[]`。 它重新验证了安装后的普通 ECPA
激活、输出、取消、停止和下一次原生启动路径。 **最终发行 Manager wheel 与直接上卡的 wheel 不同；下文通过运行时包文件字节建立有限关联，没有改写实机身份。**
本页不将所有环境的回归合并为全通过，也不声明数值等价或当前包性能提升。

本轮独立原件为 `ecpa-tp4-20261025-release`；[自动提取事实](release-qualification-facts.json)
来自[发行资格提取器](extract_release_facts.py)。 [首轮资格](README.md)和[独立 72 请求数值研究](NUMERICS.md)保留原有构建、数值与文件摘要。

## 哪个包直接运行，哪个包通过字节关联

| 角色             | 源码 commit                                | wheel SHA256                                                       | 证据范围                                                  |
| ---------------- | ------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------- |
| 插件             | `658b4d0b4e8982bd281499005500ac4c3e0100f5` | `7cc6b1d9fc8a2afa90526495a7f33e4d08901001d7c7195fbac3bf9ad06ecd7f` | 两轮 TP4 都实际安装并运行                                 |
| 本轮实机 Manager | `894c109993a001e15ea5c4b0c6bf091ca40a4709` | `ac0fb620a862314cd6f8510e1d498f61dceff39de9eb94bfba8bc330f8030781` | 第二轮 TP4 直接运行                                       |
| 最终发行 Manager | `b7dc1393069e2699f0bd3149347ef7e32a758379` | `b78dd61775e2959adf111e0b531a80e3d5e1bbc10c9df62f5f0e644a640b436e` | 独立安装 CPU 检查；全部 42 个运行时包文件与实机包字节相同 |

`build-identities.json` 把插件 18 个、Manager 42 个包文件与不可变 Git object 逐一比对。 这不声称当前工作树干净。最终 CPU
回执进一步比对源码、wheel、sdist、实际 site-packages， 并记录实机 `894c109` 与发行 `b7dc139` 的 42 个运行时文件完全相同。
发行资格提取器另直接读取两份 wheel，再次核对全部 42 个运行时成员的集合、大小、SHA256 和字节。

两份 Manager wheel 的差异为 `.dist-info/METADATA` 与 `.dist-info/RECORD`，源于 README/规格政策文字订正。 完整包 SHA
不同且始终分别保存；这个关联只覆盖明确列出的运行时文件，不能泛化为所有分发元数据或第三方依赖等价。 首轮 `c27d14c / 194e18ce` 的实机结果没有被改标为任何后续包。

## 本轮功能结果及数值限制

模型和宿主保持 [首轮报告](README.md)记录的 Qwen3.5-35B-A3B BF16、TP4+EP、910B2、pinned Core/Ascend 范围。 使用 raw-P
q4、D-first、isolated 通信组、独立 decode graph；候选调度 trace 开启。 没有使用本轮诊断计时来计算吞吐、时延加速或 Goodput。

| 检查                                    | 观察结果                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------ |
| candidate / after-cancel / native smoke | 各 9/9 项达到检查条件                                                                      |
| 完整 prefill 生命周期                   | 20 个 cohort，最大完整 prompt 12,000 tokens；25 个 completed 与 1 个 cancelled retirement  |
| P 完成前 D 进度                         | host trace 最多记录 9 个 D steps；层进度回执 378 条                                        |
| decode graph                            | 每个 rank 的最终快照：4 个 captured entries、350 次 replay、0 次 forward failure；包含预热 |
| Manager 运行生效                        | healthy 尚无 effective；成功输出后 3 组件回执齐；停止后撤销                                |
| disable 后原生回滚                      | 新原生进程无 MuxPD 选择器，smoke 成功，无 MuxPD effective                                  |
| 停止与设备进程                          | 两个 Manager 在受控 SIGINT 后返回 130；选定设备 1、2、3、6 的进程表均为空                  |
| 有限自然语言                            | 两侧各 8 条 HTTP 200、非空输出；6/8 文本精确一致，case 2 与 3 不同                         |

这些自然语言请求的 `passed` 只检查 HTTP 成功与非空，没有语义 oracle。 未保存输出 token IDs 或 logits；不能据此声称答案正确或逐 token 等价。
停止日志中的 `EngineDeadError` 和 `resource_tracker` 告警也保留，不能把受控退出与设备进程释放写成“完全无告警关闭”。 trace/graph 计数是 host
路径回执，不是本轮 NPU kernel 交叠的 profiler 证明。

独立数值研究观察到两侧各自的文本变化，但不能解释或排除插件引入的影响；限制见
[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)。 只有受限功能范围可标为硬件验证，maturity
保持 experimental alpha。

## 最终 wheel 的独立 CPU 检查

回执 `validation.json` SHA256 为 `a1496f2174fc871c10ceff888b653be4110b172221b059510333feff2ee9eaa7`，
`receipt-files.json` SHA256 为 `de13a929b937910862cd39bc3564f5e9373c249c675a2d5eda52c550bd181549`。
它们来自没有 Torch、torch-npu、vLLM 或 vLLM-Ascend 的独立安装环境，包含 48 份顶层回执文件与 9 份复制测试源码。

- 18 个 CLI 命令的实际返回码与预先规定的返回码一致；缺宿主 doctor/serve 的非零退出是预期拒绝。
- 插件 installed-wheel 纯 CPU 测试：219 passed、4 skipped。
- Manager installed-wheel 的类载体、模块回执、旧 entry-point、CLI/provider 范围：175 passed。
- 插件 18 个、Manager 42 个包文件均在源码、wheel、sdist、安装目录之间逐字节一致。
- `manager_hardware_runtime_bridge` 保存实际 ac0 与发行 b78 的 42 文件对应和分发元数据差异。

## 完整回归与已发布软件

[Manager 终态 CPU/CI 回执](manager-validation.json)分别保存完整回归结果，不能用上述 175 个定向测试替代：

- [最终远端 CI](https://github.com/vLLM-HUST/extension-manager/actions/runs/40229694768)在源码 `b7dc139` 的
  Python 3.10 和 3.12 上各记录 **2483 passed、36 skipped、6 subtests passed**，wheel 与其他检查也成功。
- 本地相同最终源码完整回归记录 **1 failed、2482 passed、36 skipped、6 subtests passed**；唯一失败为
  `test_original_full_cpu_window_new_epoch_and_rebinding_negatives` 的既有 H3 500 ms 派发门槛。失败原日志和最后一轮原始
  H3 回执保留。
- 相同 H3 测试源码与阈值的单项复测在较早窗口通过，使用 894 源码；其 H3 测试与相关运行时文件和 b7 相同。该次仍与另一 pytest CPU
  进程同时运行，未做隔离因果实验，没有调整门槛。远端与单项通过不抹去本地全套失败。
- 较早的环境错误、中断回归与真实 phase-handoff 回归也保留，逐项范围见终态回执。已过期的早期临时 H3 原始文件明确不声称已留存。

本节订正较早准备摘要中“随后”或“隔离/低负载通过”的措辞；这些原始摘要与日志按原字节保留，不把旧措辞作为因果证据。

Manager 的 `0.3.1.dev1` 组织级发布政策仍为 **no-go**，不是全部插件/宿主的通用生产资格。这里的硬件验证只覆盖本报告限定的 MuxPD 安装与 TP4
路径；不改变其发布政策门槛。

插件源码 `658b4d0` 合入
[`7c9033d`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/commit/7c9033d16119c3e64aff7fc1e8bc7c60987bdaae)，Manager
源码 `b7dc139` 合入
[`a78c968`](https://github.com/vLLM-HUST/extension-manager/commit/a78c9681ee428589dc7666c0c159cb24c3f0c58a)；各自源码树与构建源码树相同，见
[Git object 记录](merged-source-identities.json)。[插件远端 CI](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/actions/runs/40227254719)的原回执亦已保留。

实际软件预发行 [v0.2.0a1](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases/tag/v0.2.0a1) 已包含两份
wheel、两份 sdist、LICENSE 与 SHA256SUMS。[下载复验回执](software-release-download-verification.json)确认全部六个资产的
API 状态、摘要与下载字节；这与另行发布的原始证据 tarball 分开。

## 归档范围与复核

新硬件目录、最终 CPU 回执和两份 Manager wheel 以独立成员前缀追加；旧资格、失败与数值研究原件保持不变。 CPU
虚拟环境不归档；实际使用的测试源码、stdout/stderr、审计脚本、安装包和绑定清单保留。 归档的每个成员可在不执行请求的情况下用大小和 SHA256 核对。

```sh
python reports/muxpd-ecpa-20261025/extract_release_facts.py \
  --hardware /path/to/ecpa-tp4-20261025-release \
  --cpu /path/to/muxpd-release-wheel-cpu-receipts \
  --output /new/path/release-qualification-facts.json
```

上述提取器读取 CPU 回执所记的 wheel 原始路径并核验 SHA；在其他机器复算时应按归档清单恢复这些路径， 或先只进行不依赖原始路径的 archive
`--verify`。归档验证不能替代重新执行硬件资格。 历史
[TP4 Goodput 原件](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-tp4-goodput-20261011)
是另一软件身份下的场景测量，既不复制成当前性能数据，也不作为当前数值差异的豁免。
