# 四个主数据集验收证据索引

本索引区分已有实测结果、部分完成和未执行范围。指定 B0 为官方 vLLM 0.18 / vLLM-Ascend 0.18；开发栈 Native 不替代 B0。B1 按数据集、工况和指标分别确定，不将
MMLU-Pro 的结果转移到其他数据集。

## 已有结果与覆盖范围

| 数据集           | 已有证据                                                                                                                                                                                                                                              | 尚未完成的验收范围                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| HLE-Verified     | 开发栈 Native：Gold 156/668，32 份评分未决；Revision 313/1143，40 份未决。vLLM-HUST school 配置：Gold 173/668，18 份未决；Revision 316/1143，28 份未决。vLLM 0.25.1 优化配置：Gold 168/668，20 份未决；Revision 320/1143，27 份未决。均包含图文输入。 | 指定 0.18 基线的完整图文运行及适用配对；已有完整优化方案质量可复用。未决评分保留原状态，不等同全部未测。 |
| FrontierScience  | 指定0.18 B0：Olympiad 66/100、Research 3/60，未决4/1。原配对vLLM-HUST组合：62/100、2/60，未决5/3。另一BetterScale配置60/100、3/60独立保留。                                                                                                           | 160题配对已完成；该组合两个赛道判对率均低于B0，不宣称质量提升或达标。                                    |
| SWE-bench-Pro    | V1 单任务官方 Agent 已提交补丁，独立官方评分为 resolved 0/1、unresolved 0；评分阶段未调用模型。另有新 CPU 工作区的 Submitted 补丁等待评分。                                                                                                           | 单任务结果不代表全量任务通过率；尚缺指定 B0/B1 的任务级配对与完整约定范围。                              |
| TerminalBench2.1 | 已冻结 89 任务、Harbor/Terminus 版本及 AMD64 镜像；已有本地 API 传输验证。                                                                                                                                                                            | 尚无真实任务沙箱及评分结果；API 成功不能代替任务通过。                                                   |

HLE-Verified 和 FrontierScience 的上述评分使用已披露的 Qwen
替代裁判、单次试验协议，不宣称官方榜单复现。评分未决保留在完整分母中作为未成功，不等同已确认答案错误。各部署的精确预算、版本和截断记录以对应原始规范报告为准。

HLE 的已测最优质量按子集分别列示：Gold 为 school 配置 173/668，Revision 为 vLLM 0.25.1 优化配置
320/1143。两者不是同一部署，不合并计算为一个方案的全量质量，也不与另一部署的吞吐或时延拼接。school 配置的实际 vLLM 为
`0.28.1.post1.dev143+gf18cf803c.empty`；vLLM 0.25.1 优化配置源码为
`33ccc94da6ed5035aafaebb1d0a982a4de6b3c2f`。

## 证据

- HLE 开发栈 Native：[规范报告](../../website-publication/data/hle-verified-20261015/report.json)。
- HLE 与 FrontierScience 的 BetterScale
  配置：[规范报告](../../website-publication/data/betterscale-school-runtime-20261016/report.json)。
- HLE 的 vLLM 0.25.1
  优化配置：[规范报告](../../website-publication/data/betterscale-hle-20261015/report.json)，SHA256
  `7acc1ef79a759bd58656e0674fefee22d79918352de24bd0ee5471e9aa7ae57f`。
- FrontierScience
  组合方案：[汇总](../frontierscience-combination-quality-20261022/summary.json)及同目录逐题派生表、协议和运行身份。
- SWE-bench-Pro
  已评分单任务：[报告](../swe-bench-pro-native-context-20261020/report.json)。该任务的代码修复未通过；不因失败重跑择优，不将其归为评分缺失。
- SWE-bench-Pro CPU 工作区及新补丁状态：[执行说明](../../docs/SWE_BENCH_PRO.zh-CN.md)。V1、V2 与 hard 子集不混用。
- TerminalBench2.1：[冻结合同](../pyramidkv-terminal-harbor-contract-20261009/README.md)及[传输诊断](../pyramidkv-harbor-local-transport-20261009/README.md)。

## 补测与复用范围

FrontierScience
已保留原160份solver和128份judge记录，仅补32份judge（31有效、1未决）。原128份字节一致，原4份invalid保留。完成审计SHA256为`41592e715f344b8d47ba9b67e092cd4802c52074b8acdbd2c40512e5891f678c`；[规范配对证据](../frontierscience-native018-paired-20261025/provenance.json)保留原失败回执，裁判栈不冒充B0求解栈。

HLE-Verified 已恢复 Gold 668 和 Revision 1143 输入，分别包含 93、118 份图像题；文本专用服务不能冒充图文全量覆盖。0.18 多模态配置和处理器 CPU
构造已通过，但不代替图像推理。最小补测为一条原图像资格验证，以及指定 0.18 基线的 1811 题回答和对应原 strict_json 裁判；已有完整优化方案无需重测，旧未决评分不选择性重判。

SWE-bench-Pro 继续使用独立 CPU 工作区隔离任务执行，优先对已提交但未评分的补丁执行原官方 grader，不重新生成补丁。当前 NPU storage 不具备可用
Docker/Harbor 后端，不能由此推断组织没有可用 CPU 模板。TerminalBench2.1 的既定 AMD64 镜像不能未经验证转用 ARM 工作区。

### SWE-bench-Pro 执行器复用边界

2026-10-25，现有 CPU 执行器负责人交接了 `admin/swe-pro-arm-cpu-20261019` 工作区及环境修复。该工作区正在执行另一批 GenNova/vLLM
八题三轮配对，须协调空闲窗口，不并发启动评分、Agent 或镜像构建。本轮尚未在该执行器执行新任务；交接中的 vLLM 0.23 / Ascend 0.23 数据不是指定 0.18 / 0.18
基线。

复用前核验以下条件，不把其他线程的环境诊断直接当作本轮失败原因：

- Agent 和 grader 使用已验证的独立 MTU 网络，不修改全局 Docker 网络。
- Docker sidecar 与客户端的 `/root` 不共享；评分脚本和补丁必须放入双方可见的独立目录。评分后将完整证据复制到持久存储并逐文件验证 SHA256。
- 派生干净镜像必须保留父镜像和 tracked source 身份，不覆盖原始 tag；镜像内未跟踪的 Redis 数据不能继承为任务初态。复用对应 empty/gold 控制证据前核对镜像身份。
- 官方原评分与测试名称映射的补充评分分别保存，不静默改分。`Submitted` 不等于 `resolved`，时间预算耗尽时保留的部分补丁也应评分。
- 只有原 traceback 明确包含固定 wall budget 触发、事件时间落在预算至预算加 5 秒内、单次查询异常、容器清理及独立评分证据完整时，才可派生标注
  `budget_censored`。保留原 failed 回执，不仅凭 `RetryError` 或耗时重分类，也不将截断时长当完整解题延迟。

固定 V1 parquet SHA256 仍为
`c8cd7115496ad4e9a8b21d088cef576a65bf821bb542b24336f13f714cef13f8`；已有镜像及控制覆盖八题，不等于全部 731
题可执行。通用入口仍使用仓库[运行指南](../../docs/SWE_BENCH_PRO.zh-CN.md)，不重建正在使用的 CPU 工作区。

本索引不改变已完成的 M01 V0.81 有效算力结论，也不将任务评分与服务吞吐合并为同一验收指标。
