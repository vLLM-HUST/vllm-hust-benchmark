# MuxPD 0.2.0a1：安装包与 TP4 ECPA 接入验收

**本轮功能资格控制器已结束，记录 `passed=true`、`errors=[]`；自然语言配对仅 7/8 文本精确一致。** 本报告记录实际安装 wheel
后的功能、运行生效、取消及资源生命周期检查，不是性能实验，也不把历史 MuxPD Goodput 结果转记为当前安装包的收益。功能验收通过不等于数值等价；本报告本身不表示目录已经上线。

本次初轮失败与最终功能资格原件均已冻结，仍由执行端保管。后续已完成的数值研究作为独立附录，不覆盖首轮事实。 文中原件名相对最终硬件资格目录或独立 CPU
回执目录，发布时由清单绑定到固定字节；不复制正在进行的新包资格日志。 [自动提取的资格事实](qualification-facts.json)保留 79 个输入文件的路径、大小和 SHA256；
自然语言一致性由原始响应文本重算，没有将控制器的总 `passed` 作为精确一致的替代。

后续修复 Manager 后重新上卡及最终发行包的运行时字节关联，见[发行资格附录](RELEASE-QUALIFICATION.md)。 本页继续保留首轮 `c27d14c / 194e18ce`
的原始身份和结果，不能把附录中的新包替换进本页表格。

2026-10-25 的目录与网站发布已完成，实际线上页面、数据、安装链接及独立证据资产回验见[上线附录](LIVE-PUBLICATION.md)；该发布工作没有新增模型性能或数值资格。

## 安装身份与软件边界

| 组件                 | 本轮身份                                                                    |
| -------------------- | --------------------------------------------------------------------------- |
| 插件源码             | `658b4d0b4e8982bd281499005500ac4c3e0100f5`，构建时 tracked tree clean       |
| 插件 wheel           | `vllm_ascend_muxpd-0.2.0a1-py3-none-any.whl`，82,326 bytes                  |
| 插件 wheel SHA256    | `7cc6b1d9fc8a2afa90526495a7f33e4d08901001d7c7195fbac3bf9ad06ecd7f`          |
| 插件 sdist SHA256    | `c070ea79c320d18684c755ec46926f25169402bc14888e70134b8b721d41c793`          |
| Manager 源码         | `c27d14cd92658e732fa336cb1a453e64f73371cd`，构建时 tracked tree clean       |
| Manager wheel        | `vllm_hust_ext-0.3.1.dev1-py3-none-any.whl`                                 |
| Manager wheel SHA256 | `194e18ce6fd9f9cb8b95d8af7ee7312232dc273f98b3bf4c4c9a05159f95e366`          |
| Manager sdist SHA256 | `98693b88488918924a14ae5ed8c791239063c6e17f0af8c59095bf7607d9d59c`          |
| Core                 | `d0f22d2bda562156e4dbf433ce645e1769b4f804`，`0.1.dev1+gd0f22d2bd.empty`     |
| Ascend               | `03766ac696fde5ab1980d80ca0b8543d3580c989`，`0.1.dev1+g03766ac69.d20261010` |
| Python 依赖          | Torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、greenlet `3.5.6`              |

部署检查同时核对 native editable 源码 commit、实际 import origin 和 tracked 修改； Ascend 仅允许部署文档记录的
`csrc/build_aclnn.sh` 构建修改。仅有 `vllm` 公共版本号、 admission API `1.1` 或 Manager enable 状态，不足以放宽这个软件边界。

CPU 安装审计逐文件比对源码、wheel、sdist 和实际 site-packages：插件 18 个文件、 Manager 42
个文件字节相同。该断言只覆盖审计清单中的包文件，不是所有运行依赖的 可复现构建证明。构建身份见 `build-identities.json`，逐文件审计见 CPU
`validation.json`。

模型为 Qwen3.5-35B-A3B，BF16，单节点 TP4+EP，DP/PP 均为 1。模型配置、tokenizer 等 六份文件有 SHA256；14 份权重仅记录大小和
mtime，**未计算完整权重 SHA256**。 当前 cgroup 的 `cpu.max=1600000 100000`，即 16 CPU 配额；内存上限 137,438,953,472
bytes。不得把本轮环境等同于历史 32 CPU 配额的 TP8 实验。 模型和资源边界来自 `model-host-provenance.json`；芯片及选定设备状态以本轮 NPU 原件为准。

## 安装与 CPU 检查

在没有 Torch、torch-npu、vLLM 或 vLLM-Ascend 的独立环境中，实际安装最终两份 wheel， 从 site-packages 执行 18 个有限命令。静态
manifest、configuration、plan、Manager list/inspect/validate/configure/enable/disable
均完成；没有以宿主替身伪造设备兼容。 缺少 native 环境时 doctor 返回 1、serve 返回 2，符合预期拒绝。

安装包纯 CPU 测试为 **219 passed、4 skipped**。独立环境中的 Manager 状态保留 `degraded`，没有
`runtime_effective`，不会把没有宿主的静态安装测试标为运行成功。 最终源码在既有 pinned runtime 的完整测试为 **784 passed、2
skipped**；两组计数有 不同环境和范围，不合并为设备测试数。

CPU 回执 `validation.json` SHA256 为
`2249148fd8783a323bd27a8afd49ddd7834f6bb4b74a737b77cb65687867ea39`。 `receipt-files.json` 索引的 42
份文件已逐一核对大小和 SHA256。 归档仅需要回执、18 个命令的 stdout/stderr、审计脚本和测试源码，不包含虚拟环境。

## 真实 TP4 候选配置

Manager 从安装 manifest 投影三个实际组件：

- `vllm.scheduler-class.v1`：`DecoupledMuxPDScheduler`；
- `vllm.worker-class.v1`：`DecoupledMuxPDNPUWorker`；
- `vllm.batch-admission-policy.v1`：`MuxPDPolicy`。

类接口仍由固定 native 版本约束，不将它们描述为跨版本稳定的 public Scheduler ABI。 插件不通过一个空 general-plugin 回调来冒充已接入。

本轮为 raw-P 四层切片、D-first、P/D isolated 通信组、`stream_mode=overlap`、 独立 D
graph；`compile_full_prefill=false`、`adaptive_layers=false`、
`prefill_first=false`、`decode_budget_guard=false`、`prefill_layer_token_budget=0`。 完整 token budget 为
16,384，最大并发为 8；关闭 chunked prefill、async scheduling 和 prefix caching，Mamba cache mode 为 `none`。D
graph 模式 `FULL_DECODE_ONLY`， capture sizes 为 `[1,2,4,8]`，P 保持完整 prompt、raw 分层执行。

**本轮开启了调度 trace，只用于资格诊断，不能作为无记录开销的性能窗口。** 其协议和传输日志不独立证明 NPU kernel 同时执行，也没有移植 CUDA SM/Green Context
资源分区；`overlap` 配置名不等于设备交叠的实测证据。

## 已完成的候选阶段

| 检查                 | 当前已观察结果                                                                     | 原件                                                                       |
| -------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 安装、配置、启用     | doctor、Manager inspect/validate、配置及三组件投影完成                             | `installed-doctor.log`、`installed-validate.log`、`plan.log`、`render.log` |
| 初次服务 smoke       | 9/9 项通过                                                                         | `candidate-smoke.json`                                                     |
| 长 P 期间 D 持续输出 | 12,000-token P 完整响应；trace 中 P 完成前最多 9 个 D steps                        | `overlap.json`、`trace-audit.log`                                          |
| 未完成 P 取消        | 取消 probe 通过，出现 cancelled ACK                                                | `prefill-cancel.json`、`trace-audit.log`                                   |
| 取消后复用           | 再次 smoke 9/9 项通过                                                              | `after-cancel-smoke.json`                                                  |
| 有限自然语言配对     | 每侧 8/8 HTTP 请求通过当前检查；相同 payload 的文本 7/8 精确一致                   | `candidate-natural.json`、`native-natural.json`                            |
| 候选停止             | 监督器收到中断并返回 130；已观察进程退出、runtime_effective 撤销，选定四卡无进程   | `candidate-stopped-status.log`、`npu-after-candidate.log`                  |
| 停用后原生服务       | 无 MuxPD 选择器的新原生进程完成 smoke 9/9；Manager 中没有 runtime_effective        | `native-dry-run.log`、`native-smoke.json`、`native-running-status.log`     |
| 原生停止             | 监督器返回 130，选定四卡无进程；保留停止时的 EngineDeadError/resource_tracker 告警 | `native-rollback-service.log`、`npu-after-native.log`                      |

每轮 9 项 smoke 包括初始健康检查、greedy 算术、三条并发中文请求、2,305-token prompt、 长生成取消、取消后健康与后续算术。请求通过不能替代完整模型语义或逐 token
数值等价。

trace checker 返回 passed：20 个 P cohort、最大完整 prompt 12,000 tokens、 25 个 completed 和 1 个 cancelled
prefill retirement、379 个 layer-progress steps； 模型总层数为 40。retirement 数不是 HTTP 请求总数，也不是 cohort
总数，不能互换。 该审计范围为 host 生命周期和层进度回执；不把它扩展成设备 profiler 证明。

四个 rank 的最终 graph 计数分别显示 4 个 captured entries、351 次 decode replay、 0 次 decode forward failure 和 0 次
uncaptured FULL call；每 rank 有 4 次 capture forward 和 4 次 eager fallback。计数器明确包含启动预热，不能把这些数字当作测量窗口 的
token 数或推理吞吐。P forced-eager 调用为每 rank 21 次，P compiled 调用为 0。 这些是实际 native wrapper 路径的 host 计数，来自
`candidate-service.log`。

## ECPA 状态不是装饰性标记

| 实际边界               | 状态中的关键差异                                                            |
| ---------------------- | --------------------------------------------------------------------------- |
| 安装但停用             | installed/discovered/compatible/configured，没有 enabled/effective          |
| 已 enable、尚未启动    | 增加 enabled，没有 runtime_effective                                        |
| 服务 healthy、尚未请求 | 仍没有 runtime_effective                                                    |
| 成功采样且调度提交后   | 出现 runtime_effective；三组件齐，观察到 EngineCore 与四 worker 共 5 个 PID |
| 停止候选后             | enabled 意图仍在，runtime_effective 撤销                                    |
| 显式 disable 后        | enabled 与 runtime_effective 均不在                                         |

原件依次为 `installed-disabled-status.log`、`enabled-before-start-status.log`、
`healthy-before-request-status.log`、`candidate-effective-status.log`、 `candidate-stopped-status.log`
和 `disabled-status.log`。Manager 以 manifest SHA、 class/ref、role、launch ID、PID 出生身份绑定回执；仅成功输出路径产生有效记录。
这表示这些真实进程经过了已实现的完成边界，不代表每个 TP rank 的独立设备正确性认证。

## 数值差异、停止告警与后续构建

最终 `qualification.json` 覆盖 2026-10-25 19:40:11–19:49:16 UTC；27 个控制器步骤均达到该脚本的检查条件， 候选与原生监督器均在受控
SIGINT 后返回 130。原生回滚 dry-run 没有 MuxPD scheduler、worker 或 policy 选择器， 新进程采用原生 chunked
prefill。`npu-smi` 原件逐一记录物理设备 1、2、3、6 没有运行进程； 这个结论不是 HBM 占用归零的断言。

8 条自然语言请求的两侧 payload 完全相同，temperature=0、seed=42。case 2 要求用中文一句话解释计算机缓存， 两侧均 HTTP 200、正常
`stop`，但措辞不同，候选与原生分别生成 35 与 34 tokens；其余 7 条文本相同。 这些响应没有提供可核对的输出 token IDs，因此不能将 7/8 文本一致写成 token
IDs 等价，也不能把单次差异归因于浮点误差、缓存污染或插件错误。 随后独立完成的 72 请求研究见[数值重复性附录](NUMERICS.md)：并发配对 23/32 文本一致、隔离配对 2/4。
两侧内部均观察到文本变化；原生变化不能解释或排除插件影响。该研究与本节首轮 8 请求保持独立， 客户端的 `passed` 仅代表 HTTP 成功且非空，不是语义正确性的 oracle。限制持续追踪于
[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)。

原生请求检查全部结束后，SIGINT 停止阶段出现 `AsyncLLM output_handler failed` / `EngineDeadError`， 并有
`resource_tracker` 报告 4 个 semaphore 与 6 个 shared-memory 对象待清理。
对应日志及行号保留在资格事实中。实际进程退出且选定卡没有运行进程，但不能描述为“完全无告警关闭”。

另一次 Manager 完整 CPU 回归发现，新增模块绑定字段影响了既有五字段 formal phase-handoff 协议。 后续窄修恢复旧协议，并继续拒绝尚未支持的
module-carrier phase handoff。该修复不改变本轮已执行的普通 Manager 监督路径，但会产生不同 Manager wheel；**本报告的实机身份始终是上表
c27d14c / 194e18ce 构建， 不能把它改标为后续修复包的实测结果。** 后续包的回归及任何补充运行必须独立记录。

本次之前的启动失败必须保留：执行 harness 错误覆盖了 CANN 所需 `PYTHONPATH`， 导致 `acl` 导入失败。该失败没有被当前成功阶段覆盖或伪装为通过；最终报告需引用其
独立目录及修正说明。它不用于证明插件算法不支持，也不能从成功结果中删除。

## 提取与归档方法

[事实提取器](extract_qualification_facts.py)只读原始 JSON 与日志，复核 CPU 清单、9 份复制测试源码和自然语言原始响应， 提取 Manager
状态、graph 最终快照与停止告警。它不重新发请求或导入 native runtime：

```sh
python reports/muxpd-ecpa-20261025/extract_qualification_facts.py \
  --final /path/to/ecpa-tp4-20261025-final \
  --failed /path/to/ecpa-tp4-20261025 \
  --cpu /path/to/muxpd-final-wheel-cpu-receipts \
  --output /new/path/qualification-facts.json
```

[归档工具](archive_qualification.py)默认仅生成精确成员清单，显式 `--pack` 才压缩；输出目录必须不存在。 纳入初轮失败、最终资格目录、CPU
清单原件与复制测试、被实际审计的 wheel/sdist、三份实际客户端/checker源码及两个归档分析脚本。 客户端源码必须与最终资格记录的插件 commit
对应字节一致。虚拟环境和活动运行不在输入范围，完成的独立数值研究可通过 `--numerics` 显式加入。 原件不改写；gzip 时间戳、tar 元信息固定，每个解压成员按大小和 SHA256
验证；发生失败时保留输出。

```sh
python reports/muxpd-ecpa-20261025/archive_qualification.py \
  --final /path/to/ecpa-tp4-20261025-final \
  --failed /path/to/ecpa-tp4-20261025 \
  --cpu /path/to/muxpd-final-wheel-cpu-receipts \
  --producer-root /path/to/muxpd-plugin-checkout \
  --output-directory /new/path/muxpd-qualification-release --pack
python reports/muxpd-ecpa-20261025/archive_qualification.py \
  --verify /path/to/muxpd-ecpa-qualification-20261025.tar.gz \
  --index /path/to/archive-index.json
```

首次资格归档计划（137 个文件、2,602,772 bytes）与当时工具保留不改。新增独立数值研究后的 v2 计划为 166 个文件、2,908,073 bytes
原始字节，当时未压缩或上传，不包含后来完成的新包资格。 使用 `--numerics` 显式纳入附录原件，忽略 Python `__pycache__`，不忽略原始请求或失败字段。 实际发布仍需上传新
Release 附件并下载回验，再绑定 canonical 数据和网站部署 commit。 11
项小型合成测试只验证文本差异口径、确定性归档、篡改拒绝、活动输入拒绝、原件保持与安全路径，不是设备资格证据。

发行资格和最终 CPU 关联追加后的 v3 草稿已在本地压缩并逐成员验证：284 个成员、5,426,908 bytes 原始字节， tar.gz 为 2,256,022 bytes，SHA256 为
`6ae14c674528881e589c4e0b69afa7f9fe55e5b8a33cd5f31fd31931503510fe`。 这是未上传的归档草稿，不是已发布资产；最终 CI
回执等补充证据应生成后续版本，不能覆盖既有草稿或声称已下载回验。

最终 v4 已纳入终态 CPU/CI 清单并在本地完成压缩与两次逐成员验证：[成员索引](archive-index.json)、[验证回执](archive-verification.json)记录
**347 个成员、9,376,187 bytes 原始字节**，tar.gz 为 **3,680,786 bytes**，SHA256 为
`5c16db263554c1d1538ca773790b91b4780ec4b9fe5cc6c5a30003f186aaebbb`。两份 Manager 实机/发行 wheel、42
文件运行时桥接、远端成功、本地完整回归失败、早期失败与数值原件均保留。没有虚拟环境或活动日志；v1/v2/v3 草稿也未覆盖。

软件
[v0.2.0a1 预发行](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases/tag/v0.2.0a1)已发布且六资产下载字节复验通过；原始证据
tarball 的上传/下载与目录发布是后续独立步骤，遵循[发布顺序](PUBLISHING.md)。

## 明确不作的推论

- 未测当前安装包的无 trace 配对性能，因此没有新的 Goodput、吞吐或时延加速结论。
- 历史 TP4 Goodput/TP8/group-FX 记录仍属于各自冻结版本，不合并到本轮验收。
- 软件预检和 installed-wheel CPU 通过不等于上游最新 Torch/NPU 2.13 栈已在设备验证。
- smoke、有限自然语言和 lifecycle 通过不等于所有模型、采样参数、动态形状或并行度通用。
- 只有 benchmark 数据 PR 合并、固定 commit 部署成功并核对线上目录，才算网站实际登记完成。
