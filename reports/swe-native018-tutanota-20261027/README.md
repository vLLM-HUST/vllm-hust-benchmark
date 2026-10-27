# SWE-bench-Pro Tutanota增量实测

## 对象与方法

Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend
0.18.0，TP2。原任务、Agent、官方评分器及预算不变：100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试；Agent与评分使用不同新容器。

## 结果

Tutanota沿用原协议仅执行一次Agent，100次模型调用、437.454803秒，步数耗尽且未主动提交。原7862字节部分补丁成功应用于4个路径，应用后与测试前实际待评分状态一致。官方评分false：TypeScript编译失败，目标测试尚未执行。编译器记录OperationProgressTracker模块缺失、仅类型导入被用作运行时值及构造参数数量不匹配；原解析器产生1条ERROR，必要F2P目标缺失1项，P2P集合为空。该结果不是功能断言失败，也不代表完成提交。未追加Agent、改写补丁或重复评分，评分阶段模型调用0；资源已释放。

累计18/731个不同任务，官方原记录2/18；保留同一Protonmail答案依赖恢复后为3/18，未执行713题。恢复评分没有增加Agent尝试；不是全量通过率或B0/B1配对验收。旧应用失败、应用证据不足、未提交部分补丁、编译失败和测试依赖限制均保留。

## 证据

[规范目录](catalog.json)、[当前聚合](current-aggregate.json)、[本次原派生摘要](summary-tutanota.json)、[独立复核白名单](storage-review.json)、[释放回执](release-tutanota.json)。公开材料不含题目、答案、补丁或轨迹正文；受控原件以SHA256索引。应用前、应用后及测试前实际状态分别保留。

`python3 verify.py`可离线复核公开聚合和绑定；`derive_private_tutanota.py.source`需受控原件，不能仅凭公开摘要重跑完整评分。
