# 0.28 源码候选固定配置报告

Qwen3.5-35B-A3B、BF16、两卡原测：固定五项、4K/16K/32K 真实上下文、四租户及三次原生 T。200 请求独立 companion 已封存；公开 CPU 入口在封存 JSON
预检拒绝前缀算量合同，F 与效率为空。原整轮报告保持 failed；较早完整 SDK 准入失败也保留，预检复算不冒充完整准入。

[公开原件、曲线与复算](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/official-b0-0.28-source-full-report-20261123-v1)、[归档清单](archive-manifest.json)、[原报告](original-report.json)、[自动拒绝](automatic-ft-rejection.json)、[来源链](source-chain.json)、[匿名下载复算](anonymous-replay.json)。

这是 vLLM 0.28.0 / Ascend 0.28.0.rc1 源码候选，保留原模型、构建和执行身份。九组观测包括完整质量分组与未解析人口；C8
仅保留原计数、时钟，未从预算推算吞吐。元数据包选择原封存中的 76 个文件，其余 9217 个仅保留摘要，另附最小前缀反例包，不是完整 SSE 归档。独立 B0 不自动组成新版 B1 正式配对；CPU
拒绝复算不是上游推理错误、全套通过或人工/NPU见证。
