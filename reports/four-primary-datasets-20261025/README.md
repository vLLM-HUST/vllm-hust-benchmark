# 四个主数据集验收证据索引

本索引区分已有实测结果、部分完成和未执行范围。指定 B0 为官方 vLLM 0.18 / vLLM-Ascend 0.18；开发栈 Native 不替代 B0。B1 按数据集、工况和指标分别确定，不将
MMLU-Pro 的结果转移到其他数据集。

## 已有结果与覆盖范围

| 数据集            | 已有证据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 尚未完成的验收范围                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| HLE-Verified      | 指定 0.18 B0：Gold 157/668、Revision 316/1143，评分未决23/31。school配置：173/668、316/1143，未决18/28；0.25.1优化配置：168/668、320/1143，未决20/27。开发栈Native的156/668、313/1143及未决32/40另行保留。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 指定基线1811题求解及评分已完成、释放封存；未决保留完整分母。不同部署最好成绩不拼接为一个B1。 |
| FrontierScience   | 指定0.18 B0：Olympiad 66/100、Research 3/60，未决4/1。原配对vLLM-HUST组合：62/100、2/60，未决5/3。另一BetterScale配置60/100、3/60独立保留。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 160题配对已完成；该组合两个赛道判对率均低于B0，不宣称质量提升或达标。                        |
| SWE-bench-Pro     | 指定0.18 B0已尝试18/731题，官方原记录2/18；同一答案依赖恢复后3/18。Tutanota未提交部分补丁应用成功但编译失败，功能测试未执行；旧失败及限制保留。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 713题及B1同题配对未完成；[规范目录](../swe-native018-tutanota-20261027/catalog.json)。       |
| TerminalBench 2.1 | TerminalBench2.1已尝试2/89题，87题未执行。OpenSSL完整答案恢复评分6/6通过、reward=1；cancel-async-tasks在提交和评分前因ContextLengthExceededError终止，原reward=null。未提交的原部分代码单独恢复评分，1/6通过、reward=0，不计第二次Agent或完整提交评分；不构成全量或B0/B1配对验收。 Cython原控制仍为nop 1/11、oracle 10/11，reward均为0。一次固定源码与依赖重建取得单项完整trace：planarity.networkx_graph(pg)之后，representation.py:338读取pos触发KeyError；实际属性为vertex_end、vertex_position和vertex_start。该结果定位本次重建的属性接口不匹配，不冒充原终止容器的完整trace，不证明架构因果或版本升级历史。原wheel及环境字节不可恢复，本地Git基线与no-build-isolation差异已披露。setup退出0、诊断测试退出1，容器已释放；未改测试或oracle，不新增控制成绩或模型覆盖。Terminal仍为2/89项已尝试、87项未执行，本题正控制仍未通过。 | 87题及B1同题配对未完成                                                                       |

HLE-Verified 和 FrontierScience 的上述评分使用已披露的 Qwen
替代裁判、单次试验协议，不宣称官方榜单复现。评分未决保留在完整分母中作为未成功，不等同已确认答案错误。各部署的精确预算、版本和截断记录以对应原始规范报告为准。

HLE 的已测最优质量按子集分别列示：Gold 为 school 配置 173/668，Revision 为 vLLM 0.25.1 优化配置
320/1143。两者不是同一部署，不合并计算为一个方案的全量质量，也不与另一部署的吞吐或时延拼接。school 配置的实际 vLLM 为
`0.28.1.post1.dev143+gf18cf803c.empty`；vLLM 0.25.1 优化配置源码为
`33ccc94da6ed5035aafaebb1d0a982a4de6b3c2f`。

## 证据

- HLE 开发栈 Native：[规范报告](../../website-publication/data/hle-verified-20261015/report.json)。
- HLE 指定0.18完整基线：[报告及封存证据](../hle-native018-completed-20261026/README.md)。
- SWE
  指定0.18单题：[已评分报告](../swe-native018-single-agent-20261026/README.md)；[100步停止原因独立复核](../swe-full-continuation-20261026/README.md)。
- SWE指定0.18当前结果：[跨仓库实测及评分覆盖](../swe-native018-continuation-20261026/README.md)、[可复算任务目录](../swe-native018-continuation-20261026/catalog.json)；[前三题历史归档](../swe-native018-incremental-20261026/README.md)。
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

HLE-Verified
指定0.18完整图文求解和原strict_json评分均已完成。接续过程复用1811份求解和668份Gold评分，仅完成缺失的Revision评分；原控制器退出码不可恢复的字段保持null。原失败回执、截断和23/31份invalid_judge保留，服务已排空释放。已有完整优化方案继续复用，不重跑求解或选择性重判。

SWE-bench-Pro 继续使用既有独立 CPU
工作区隔离任务执行。指定0.18单题部分补丁已完成原官方grader评分，不重新生成补丁；旧工作区未评分补丁需先核对身份，不能直接算作新增指定B0结果。当前 NPU storage 不具备可用
Docker/Harbor 后端，不能由此推断组织没有可用 CPU 模板。TerminalBench2.1 的既定 AMD64 镜像不能未经验证转用 ARM 工作区。

### SWE-bench-Pro 执行器复用边界

2026-10-25，现有 CPU 执行器负责人交接了 `admin/swe-pro-arm-cpu-20261019` 工作区及环境修复。该工作区正在执行另一批 GenNova/vLLM
八题三轮配对，该历史工作区不计作本轮指定基线的新任务执行器。当前指定0.18任务在既有独立持久工作区完成；交接中的 vLLM 0.23 / Ascend 0.23 数据不是指定 0.18 / 0.18
基线。

复用前核验以下条件，不把其他线程的环境诊断直接当作本轮失败原因：

- Agent 和 grader 使用已验证的独立 MTU 网络，不修改全局 Docker 网络。
- Docker sidecar 与客户端的 `/root` 不共享；评分脚本和补丁必须放入双方可见的独立目录。评分后将完整证据复制到持久存储并逐文件验证 SHA256。
- 保留父镜像和 tracked source 身份，不覆盖原始
  tag；上轮修改过的数据不得继承。固定上游镜像自带的未跟踪文件须按实际身份及定向允许列表披露，不能混称为上轮缓存。第二题提交这些文件导致的评分应用证据限制单独保留，不推广到初始工作树干净的第三题。
- 官方原评分与测试名称映射的补充评分分别保存，不静默改分。`Submitted` 不等于 `resolved`，时间预算耗尽时保留的部分补丁也应评分。
- 只有原 traceback 明确包含固定 wall budget 触发、事件时间落在预算至预算加 5 秒内、单次查询异常、容器清理及独立评分证据完整时，才可派生标注
  `budget_censored`。保留原 failed 回执，不仅凭 `RetryError` 或耗时重分类，也不将截断时长当完整解题延迟。

固定 V1 parquet SHA256 仍为
`c8cd7115496ad4e9a8b21d088cef576a65bf821bb542b24336f13f714cef13f8`；镜像准备与已执行结果分别统计，已测范围以规范任务目录为准，不等于全部731题可执行。通用入口仍使用仓库[运行指南](../../docs/SWE_BENCH_PRO.zh-CN.md)，不重建正在使用的
CPU 工作区。

本索引不改变已完成的 M01 V0.81 有效算力结论，也不将任务评分与服务吞吐合并为同一验收指标。

- Terminal ARM64原任务控制：[OpenSSL nop/oracle](../terminal-openssl-arm64-controls-20261026/README.md)。
