# BidKV：历史 Work 负载的 EOS 双卡补充实验（运行前封存）

这份计划用于核验真实抢占时 BidKV 是否调用、选择是否合法，以及吞吐是否改善。不是五数据集正式成绩。原五项已完成，但全部 solver 的实际引擎抢占计数均为零，因此不能证明 BidKV
机制收益。此前补充压力实验的失败与未完成记录另行保留。

使用原 EvoScientist first12 的全部 12 条提示、顺序和逐条输出预算，输入 SHA256 为
`53588fc1a06bbe59b6737b434a0664b58fd673a7dfe38a493749b3faed921d8d`。历史 TP4、强制长度的正收益只能提供负载设计线索。本次
TP2、保留 EOS、默认 thinking、MTP2、官方 0.23 引擎，是独立补充实验；历史数字不能充当本次成绩。

两臂共同使用 1600 MiB KV/rank、4096 上下文、并发 12、固定 4 req/s 到达和全部共同默认优化；两臂共同配置历史
`utility_preempt_weight=5.0`，只有目标策略启用状态不同。不能宣称默认 0.3 参数收益。64 GiB 卡上显式缩小 KV 的适用条件与额外冷启动费用均需披露。实际双
rank 几何、token hash、CPU 绑定及全部源码必须先通过准入。

预先固定四次独立冷启动顺序：Native→BidKV→BidKV→Native，每次 12 请求，共计划 48。无重试、换题、补齐、强制长度或挑最好轮。首个 Native
完成后若真实抢占为零，保留为无压力资格检查，其余不执行；若 B1 有真实抢占但策略调用为零，记录接入异常。符号为负的有效配对同样保留。

全部源/配置/输入身份公开并读回后才开始首次冷启动和新的 9000 秒总时限；不让模型等待发版、不重置时钟。只运行有界实验进程，不创建保活、cron、停机策略修改或自动重启。

冻结 [22 成员清单](source-manifest.json) 的原 Python 仅追加 `.source`，原 README 映射为
[original-README.md.source](original-README.md.source)，字节均不变。[实际 Linux CPU 核验](linux-cpu-admission-v2.json)
为 38 项通过、零模型请求，包括 POSIX 信号清理验证；它不是模型运行结果。[恢复摘要](cpu-restoration-summary.json) 记录与前次相同的 SDK 和源码。v1
在模型启动前发现传输状态分类与信号清理问题，未运行模型；v2 修正后重新封存。

[执行计划](execution-plan.json)、[输入 token 身份](input-token-receipt.json)、[内存与历史配置依据](historical-memory-rationale.json)
分别给出边界。完整原始输入、响应、SSE 和时间戳受控保存；公开数字报告在运行完成后附上，并区分调用、合法选择、实际改变内置选择、吞吐、未决回答和费用。
