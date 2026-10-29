# 固定 B0 与纠正 B1 长上下文检索配对

Qwen3.5-35B-A3B、BF16、TP2，同两张 Ascend 910B2；C1 串行，temperature=0、请求 seed=0、输出上限 8192。 沿用既有六条合成档案输入，在填充段
10%/50%/90% 位置植入六位检索码。实际输入分别为 16,457 和 32,843 token，各三条。 双侧服务分词的 token ID SHA 与冻结输入一致；严格按最终 content
的完整字符串评分，reasoning 分开记录。

| 方案 | 实际运行栈                                                                                           | 完成 | 严格正确 | 请求失败 |
| ---- | ---------------------------------------------------------------------------------------------------- | ---: | -------: | -------: |
| B0   | vLLM 0.18.0 / vLLM-Ascend 0.18.0，Qwen3.5 text-only 兼容补丁，CANN 8.5.1                             |  6/6 |      6/6 |        0 |
| B1   | vLLM 0.28.1 开发版 / Ascend 0.25.1rc2 开发版，纠正后的 E24/R36 + KV 逐请求重算 + Microflow，CANN 9.1 |  6/6 |      6/6 |        0 |

完整版本、配置及逐题结果见 [summary.json](summary.json)。双侧均使用独立服务和新空缓存，正常共享只读权重及 OS 页缓存；每条请求原有唯一 salt 保留。 B1
七次实际重算准入包含一条 parser 资格题，策略与引擎准入逐项对应，reused_tokens=0；两 rank 的 Microflow 分块资格及实际回放事件均核验。
这证明本组六条输入的检索结果和组件触发，不外推为所有跨请求状态均无污染。parser 资格题不计入 6/6。

B0 首次启动因入口缺少多进程 spawn 保护而在提交请求前退出；失败记录和源码保留。第二次仅修入口，输入、预算、评分及运行栈不变。 最终 B0 客户端/服务退出码为 0/0，B1 为
0/130（Manager 收到预期关闭信号）；两侧所有保留设备均已释放后才封存。

原始测试数据：[raw.tar.gz](raw.tar.gz)；[SHA256 与范围](capsule.json)。包内含三次启动回执、原始 SSE、逐题结果、服务日志、设备采样、配置及执行器源码。
完整第三方运行时不随包重复分发；必要源码身份记录在各 plan 中。已有五个 0.18 混合长输入窗口另见
[离线覆盖审计](../long-context-fixed018-20261029/README.md)。

```sh
python scripts/audit_fixed018_retrieval.py
```

受控 4K/16K 连续窗口及五项任务稳定性仍未闭合；本次检索配对不替代这些结论，也不构成吞吐排名。正式报告/网站 V1.01 尚未同步本轮新数据。

## English summary

Fixed B0 (vLLM/vLLM-Ascend 0.18.0) and corrected B1 (0.28.1/Ascend 0.25.1rc2 development stack) each
retrieved all six codes correctly with no failed requests. Both used the same model weights,
BF16/TP2, two reserved 910B2 devices and identical tokenized inputs of 16,457/32,843 tokens. The
[raw capsule](raw.tar.gz) retains the failed initial B0 startup, successful runs, streamed responses
and activation evidence. Controlled continuous 4K/16K stability and five-task coverage remain
separate, unfinished requirements; this is not a throughput or general state-isolation verdict.
