# 四个主数据集验收证据索引

本索引区分已有实测结果、部分完成和未执行范围。指定 B0 为官方 vLLM 0.18 / vLLM-Ascend 0.18；开发栈 Native 不替代 B0。B1 按数据集、工况和指标分别确定，不将 MMLU-Pro 的结果转移到其他数据集。

## 已有结果与覆盖范围

| 数据集 | 已有证据 | 尚未完成的验收范围 |
| --- | --- | --- |
| HLE-Verified | 开发栈 Native：Gold 156/668，32 份评分未决；Revision 313/1143，40 份未决。vLLM-HUST school 配置：Gold 173/668，18 份未决；Revision 316/1143，28 份未决。vLLM 0.25.1 优化配置：Gold 168/668，20 份未决；Revision 320/1143，27 份未决。均包含图文输入。 | 指定 0.18 基线的完整图文运行及适用配对；已有完整优化方案质量可复用。未决评分保留原状态，不等同全部未测。 |
| FrontierScience | vLLM-HUST 组合方案：Olympiad 62/100，5 份未决；Research 2/60，3 份未决。另一个 BetterScale 配置为 60/100、3/60，分别保留，不拼成一个部署。 | 指定 0.18 solver 已完成 160 题；已恢复的 128 份 judge 记录包含 124 份 ok 和 4 份 invalid_judge，仅余 32 份未执行。 |
| SWE-bench-Pro | V1 单任务官方 Agent 已提交补丁，独立官方评分为 resolved 0/1、unresolved 0；评分阶段未调用模型。另有新 CPU 工作区的 Submitted 补丁等待评分。 | 单任务结果不代表全量任务通过率；尚缺指定 B0/B1 的任务级配对与完整约定范围。 |
| TerminalBench2.1 | 已冻结 89 任务、Harbor/Terminus 版本及 AMD64 镜像；已有本地 API 传输验证。 | 尚无真实任务沙箱及评分结果；API 成功不能代替任务通过。 |

HLE-Verified 和 FrontierScience 的上述评分使用已披露的 Qwen 替代裁判、单次试验协议，不宣称官方榜单复现。评分未决保留在完整分母中作为未成功，不等同已确认答案错误。各部署的精确预算、版本和截断记录以对应原始规范报告为准。

HLE 的已测最优质量按子集分别列示：Gold 为 school 配置 173/668，Revision 为 vLLM 0.25.1 优化配置 320/1143。两者不是同一部署，不合并计算为一个方案的全量质量，也不与另一部署的吞吐或时延拼接。school 配置的实际 vLLM 为 `0.28.1.post1.dev143+gf18cf803c.empty`；vLLM 0.25.1 优化配置源码为 `33ccc94da6ed5035aafaebb1d0a982a4de6b3c2f`。

## 证据

- HLE 开发栈 Native：[规范报告](../../website-publication/data/hle-verified-20261015/report.json)。
- HLE 与 FrontierScience 的 BetterScale 配置：[规范报告](../../website-publication/data/betterscale-school-runtime-20261016/report.json)。
- HLE 的 vLLM 0.25.1 优化配置：[规范报告](../../website-publication/data/betterscale-hle-20261015/report.json)，SHA256 `7acc1ef79a759bd58656e0674fefee22d79918352de24bd0ee5471e9aa7ae57f`。
- FrontierScience 组合方案：[汇总](../frontierscience-combination-quality-20261022/summary.json)及同目录逐题派生表、协议和运行身份。
- SWE-bench-Pro 已评分单任务：[报告](../swe-bench-pro-native-context-20261020/report.json)。该任务的代码修复未通过；不因失败重跑择优，不将其归为评分缺失。
- SWE-bench-Pro CPU 工作区及新补丁状态：[执行说明](../../docs/SWE_BENCH_PRO.zh-CN.md)。V1、V2 与 hard 子集不混用。
- TerminalBench2.1：[冻结合同](../pyramidkv-terminal-harbor-contract-20261009/README.md)及[传输诊断](../pyramidkv-harbor-local-transport-20261009/README.md)。

## 补测与复用范围

FrontierScience 保留原 160 份 solver 输出和 128 份 judge 记录，按原提示、预算、评分规则只续缺失的 32 份，不重新判定原失败或截断记录。恢复审计 SHA256 为 `81608b9a517e23c7ec70fa99877e8d30057473806988afecace01e406e23d305`。

HLE-Verified 已恢复 Gold 668 和 Revision 1143 输入，分别包含 93、118 份图像题；文本专用服务不能冒充图文全量覆盖。0.18 多模态配置和处理器 CPU 构造已通过，但不代替图像推理。最小补测为一条原图像资格验证，以及指定 0.18 基线的 1811 题回答和对应原 strict_json 裁判；已有完整优化方案无需重测，旧未决评分不选择性重判。

SWE-bench-Pro 继续使用独立 CPU 工作区隔离任务执行，优先对已提交但未评分的补丁执行原官方 grader，不重新生成补丁。当前 NPU storage 不具备可用 Docker/Harbor 后端，不能由此推断组织没有可用 CPU 模板。TerminalBench2.1 的既定 AMD64 镜像不能未经验证转用 ARM 工作区。

本索引不改变已完成的 M01 V0.81 有效算力结论，也不将任务评分与服务吞吐合并为同一验收指标。
