# 35B Request Throttling Controller 在一题 SWE-bench Pro V1 上的运行记录

[交接 issue #4](https://github.com/vLLM-HUST/request-throttling-controller/issues/4) 要求比较官方 0.18
引擎直连（B0）、优化引擎直连（B1-engine）、同一优化引擎经原版限流层（B1-system）。本目录收录同一个 SWE-bench Pro V1 任务在两条 B1 路径上的六次真实 Agent
执行。模型产生工具调用，Agent 提交补丁，官方评分器独立判定结果。

| 运行条件                                | B1-engine，模型调用 / 墙钟秒 | B1-system，模型调用 / 墙钟秒 |
| --------------------------------------- | ---------------------------: | ---------------------------: |
| 11 月 4 日，原工具消息格式              |                113 / 509.677 |                 80 / 298.385 |
| 11 月 4 日，共同采用 tool_response 格式 |                 60 / 247.250 |                 90 / 316.014 |
| 11 月 16 日，第六版逐请求重算配置       |                105 / 496.295 |                 90 / 403.246 |

六份提交的补丁在该题的官方评分均为 false。两次评分器对照中的参考补丁均通过，空补丁均未通过。三次经限流层运行都回读到实际 program
释放，六次工具容器清理退出码都为零。两臂各自生成的模型输出和工具工作量不同。各次墙钟只用于追溯单题执行，目前缺少预先固定任务集合及缓存初态的匹配批次、B0 同题评分与独立重复。

11 月 24 日的另一项 MMLU-Pro 任务已有双卡 B0 成功启动回执。运行时 vLLM 与 vLLM-Ascend 均为 0.18.0，使用原有文本适配、TP2，启动耗时 704.672
秒。其归档及运行身份单列于 `separate_b0_startup`，仅作为后续 B0 启动的来源。

## 复算

[evidence.json](evidence.json) 提供逐次调用数、耗时、补丁字节数、官方判定、program 释放与对应原件
SHA-256。私有题目、补丁、工具内容及模型回答保存在原归档。

持有原件时，从本目录执行以下命令。参数为 Controller 仓库保存 `standardization-20261104`、`standardization-20261116` 和
`standardization-20261124` 的 `results` 目录。

```bash
python3 recompute.py /path/to/request-throttling-controller/results > /tmp/issue4-single-task.json
diff -u evidence.json /tmp/issue4-single-task.json
```

脚本按任务 ID 关联 Agent 与评分器、核对两组评分对照以及请求释放，并验证 11 月 16 日 CPU 归档摘要与 11 月 24 日 B0 归档的完整摘要和 415 个清单成员。B0
启动时间单独来自引擎回执，运行时包版本、张量并行度和任务类型均从包内回读。
