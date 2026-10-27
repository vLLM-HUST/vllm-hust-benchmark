# Circuit-fibsqrt 原始上下文终止结果

本目录保留同一个原始 Agent 尝试的公开投影，未重跑 Agent、控制或评分。Terminal 模型覆盖为 4/89，85 题未尝试；原前三题记录保持不变。

实际 11 次 HTTP 包括 10 次成功和 1 次上下文拒绝。轨迹记录 9 个 Agent 步骤、10 个 episode 和 9 项成功延迟；另一次成功来自原输出截断处理中的递归请求。API
与轨迹 token 差额分别为输入 5449、输出 4096，不将 episode 或 turn 视为 HTTP 计数。

原 32768 上下文中，至少 28673 输入 token 与请求的 4096 输出 token 合计超限。Agent 约 331 秒后终止，并非耗尽 100 turn 或 3600 秒。原
verifier、reward 和 CTRF 均为 null，客户端退出码 0 不表示任务成功。

有限原命令审查未发现答案文件写入，原 artifact 清单为空，终态答案文件 SHA 未采集。不得从打印的信号计数生成答案，或重跑、补完生成器进行所谓恢复评分。

`whitelist.json` 绑定四份原公开 JSON；受限原件仅保留哈希，不包含答案、命令或请求标识。`python verify.py` 只做离线一致性校验。
