# MuxPD：完整 prefill 的层级分时执行

研究日期：2026-10-26。只读源码与作者保存的报告，未运行设备实验。源码快照 `6fd8f0de9968357e40a90e185b325f02052039f9`；该 HEAD
含后续文档更新，不等于新设备验收。最终采用结论：PASS，见下方共同评审。

## 解决什么问题

长 prompt 的 prefill 会阻塞正在出词的 decode。常规 chunked prefill 沿 token 维度切分 prompt；MuxPD 保留完整 prompt，沿
transformer 层边界让出执行，让其他请求先推进下一轮 decode。它追求减少长 P 到达时的出词停顿，而不是减少模型 FLOPs。

不是跨设备 PD 分离，不需要在 P/D 设备间传 KV，也不是 PP 层分片。两条 runner 在同一设备组共享模型权重和 KV，隔离输入、metadata、workspace 和 graph
buffer。TP/EP 仍有原生通信。

## 执行路径

- `serial`：整段 P 或 D，一轮只跑一种；完整 P 仍会阻塞 D。
- `layered`：同轮内 P/D 按层交错，但必须等两者完成再提交本轮结果；长 P 内只有一轮 D。
- `decoupled`：P 的中间状态跨引擎轮次保留，D 每轮独立产生结果；P 完整网络完成后才产出首 token。

默认 fenced 的 decoupled 路径示意（不是物理并行）：

```text
D 完整 forward → P 第 1～k 层 → 本轮 D 采样/返回
D 完整 forward → P 第 k+1～2k 层 → 本轮 D 采样/返回
…… P 到末层后才产出 P 首 token
```

`lanes.py` 的 `LayerInterleaver` 用单 host 线程上的 greenlet 保存 Python 调用栈；decoder post-hook 调用
checkpoint，达到层数配额时 fence、保存 lane 状态并切回。中间 hidden states/residual 等保留，不重做 embedding 或已执行层。

`decoupled_worker.py::_advance_lanes` 在 fenced 模式先推进完整 D，再推进规定 P 层数。调度器保留最多一个 pending P
cohort，避免其被当成已完成 prefill 进入 decode；取消必须等 worker 确认不再访问 KV 后才能释放。

配额可以固定，也可按完整 P token 数、耗时 EWMA、TTFT deadline 和可选 decode budget guard 调整。最小推进粒度仍是一层，单层过慢时无法提供硬出词时延保证。

## 同步、并发与成本

- `fenced`：跨流 event 建立执行依赖，不能称为 P/D kernel 真并行；不等于每个 event 都是 host 全设备 synchronize。
- `overlap`：实验路径放松普通层边界等待，仍保留初始化、通信、采样等依赖。共享 communicator 会串联通信；isolated 配置创建独立 P 通信组。源码含独立 decode
  graph 的实验分支，不能只凭较旧设计文档断言所有 overlap 都不支持 graph。
- 当前配方没有论文式 Cube/Vector 资源划分，不能从两 stream 或协程推导空间隔离、物理重叠。
- 切换带来 hook、状态切换、event、workspace 和调度管理成本；P 中间激活与完整 prompt 的 KV 分配持续占用内存。
- 完整 P 容量要求 `max_num_batched_tokens >= max_model_len`；当前接入要求同步 EngineCore，禁用 chunked prefill/async
  scheduling，示例关闭 prefix caching。应与实用原生后端比较，不能只与被限制的同步对照比较。

## 最新已读证据：改善极端停顿，不是整体加速

中央 v5 报告的设备候选为 `f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`，限定 TP4/EP4、910B2、BF16、Qwen3.5；不把源码 HEAD
当作这轮 run 身份。以下为作者数据，非我们的实测。

| 指标                               |       实用原生参考 |                Group-P |
| ---------------------------------- | -----------------: | ---------------------: |
| Decode output token/s              |            368.844 |    218.525（−40.754%） |
| Mixed output token/s               |            194.555 |    120.129（−38.254%） |
| 长 P TTFT                          |         455.777 ms | 616.651 ms（+35.297%） |
| Mixed seed gap P99                 |          32.575 ms |  60.640 ms（+86.157%） |
| 四个 mixed 窗口最大 seed token gap | 423.349–445.610 ms |       75.486–86.950 ms |

解释：极少数长停顿被削平，但常规速度与尾部指标更差。实用原生开启默认 async/npugraphEx，但仍保留固定实验配置；非厂商最优配置，也不是单因素消融。Mixed
输出轨迹不同，不能把全部差距归因到调度开销。该测试不是 HTTP SLO goodput 或容量测量。

质量：200 题中原生两个新进程均 165 正确，raw/group 均 166；但丢掉了两原生共同答对的 3 题，新增答对 4 题。按预先冻结的“不新增错误”门槛判
FAIL。总分略高不等于无损，也不能仅凭文本不同判答错；该失败不证明层级分时算法必然有损，数值/实现原因仍需区分。

Raw/group 混合 profiler 没有显示 group 减少原生 kernel launch：每 rank launch 2854→2924，record/wait event
数不变。不能把插件内部 +3.466% 吞吐改进解释成 launch 减少，更不能当成超越原生。

## 固定来源

- [README 与当前边界](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/6fd8f0de9968357e40a90e185b325f02052039f9/README.md)
- [设计与 pending P 协议](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/6fd8f0de9968357e40a90e185b325f02052039f9/docs/design.md)
- [greenlet 层切换](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/6fd8f0de9968357e40a90e185b325f02052039f9/src/vllm_muxpd/lanes.py)
- [跨轮 worker](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/6fd8f0de9968357e40a90e185b325f02052039f9/src/vllm_muxpd/decoupled_worker.py)
- [v5 质量、实用原生参考与 profiler](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ebddb585390c333ac1f8c6ea40cfeead3c4321af/reports/muxpd-followup-20261025/quality-reference-v5/REPORT-v5.md)

## 共同评审结论

**PASS，不继续投入：当前吞吐不合格，没有看到放弃 mixed batching 的充分理由。** 优先采用 mixed/chunked prefill，在容量和延迟预算内调整 mixed
batch，而不是保留整段 P 中间激活并支付层级切换与额外调度成本。

这是我们的工程投入判断，不是本轮独立消融得出的普遍性能定理。Mixed 本身不保证长 P 永不阻塞 D，扩大 batch
也不是无条件降低延迟；但已读该实现的吞吐、常规延迟与质量边界不足以支持采用。[完整 comment](../comments.md)
