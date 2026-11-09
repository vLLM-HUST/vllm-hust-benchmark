# PegaFlow 原生 vLLM 调度适配验收（2026-11-09）

在一个容器的两张 910B2 上采集真实 Qwen2.5-7B-Instruct 运行轨迹，再以实测输入、输出及 连接器通知重放原生调度器。36 次真实请求、687 轮调度的 token
选择、computed/output 数、 状态、抢占数及物理 KV 页编号全部一致。该结果验证调度接口与状态衔接；本轮未重新评估 TTFT/TPOT 预测误差。

## 配置与结果

| 真实运行                                         | 请求数 | 调度轮次 | 结果           |
| ------------------------------------------------ | -----: | -------: | -------------- |
| 单卡模型：冷缓存、混合上下文、共享前缀、并发 4/8 |     24 |      364 | 决策与状态对齐 |
| 本机双卡 P/D 的 P                                |      4 |       33 | 决策与状态对齐 |
| 本机双卡 P/D 的 D：恢复和 APC 复用               |      8 |      290 | 决策与状态对齐 |

模型 revision：`a09a35458c702b33eeacc393d103063234e8bc28`；BF16、TP=1、graph FULL_DECODE_ONLY、每实例 16GiB
NPU KV、16 序列、2,048 tokens/轮、128-token 页、同步 调度。输入由此前固定 ATIF 标定轨迹的可见历史截取；构造与客户端参数保存在采集脚本中。 上下文覆盖约
0.5K～24K tokens，输出长度覆盖 1、16、32、64、128 tokens。

双卡运行使用同一原生服务的 16GiB CPU 池。P 的 2,049/8,193/16,385/24,577-token 请求 分别发布 16/64/128/192 块缓存；D
首轮成功恢复，每请求只计算一个输入后缀 token，第二轮 复用 D 的 APC。D 的空调度轮次也计入对照。该本机缓存交接未执行跨主机 RDMA。

vLLM 安装版本 `0.23.0+empty`，源码提交 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；vLLM-Ascend 0.23.0，源码提交
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。本环境含 full-ISL reservation 扩展，实际 Scheduler SHA256 为
`9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9`。 芯片型号及物理卡 5、6 来自
[inventory.json](inventory.json) 的 npu-smi 实测。

6 项原生接口测试通过：分块 prefill/APC、异步恢复屏障、小池抢占与排空、1P1D/2P2D 发布交接、整块输入后缀。无法装入 KV 池的请求明确拒绝。原有便携测试为 72 passed。

## 证据与复核

- [validation.json](validation.json)：最终原生测试状态、三组逐轮重放结果和源码身份。
- [manifest.json](manifest.json)：包摘要、成员清单及边界说明。
- [raw-capture.tar.gz](raw-capture.tar.gz)：原始请求、调度 JSONL、服务日志、测试 XML、 采集/重放脚本及执行时的模拟器源码快照。

包内 `baseline-v2/` 和 `local-pd/` 是成功真实运行；`validation-accepted.*`、
`parity-*-accepted.json`、`native-tests-accepted.xml` 是最终验收。`baseline/` 保留首轮 遥测字段版本不匹配导致的中断；早期 parity
日志保留适配期间的构造参数/HMA 接口错误。 最终验收使用修正后的适配器重放同一批原始轨迹，没有用重复实测覆盖首次失败。

首版采集器直接记录了外部查询、分配和收发完成通知；HMA finish 回调的延迟释放标志由 更新后的请求保留状态推导，最终 JSON 明确记录其来源。后续采集器已补全该回调的直接观测。
重放提供实测模型输出及连接器输入，因此验证的是原生调度器对这些输入的响应；不能据此 宣称模拟器已独立预测恢复完成时刻或真实 token 内容。

解包到新的目录后，可在匹配的 vLLM 运行时中使用包内 `code/replay_native_trace.py`， 指定一份
`process-*.jsonl`、本地模型配置目录和新的输出路径重放。固定 `PYTHONHASHSEED=17`。 包内源码为执行快照；维护中的实现位于
[PegaFlow 工作分支](https://github.com/vLLM-HUST/pegaflow-hust/tree/feat/issue23-four-card-calibration/scripts/issue23)。

## 适用范围

原生后端支持同步调度、TP=1、单 full-attention KV group。外部离散事件模型继续提供 计算、CPU 发布、网络与恢复耗时。未建立新规模、多节点或其他模型的时间误差保证；历史
恢复慢尾仍需独立标定。本集合属于研究诊断及调度适配证据。
