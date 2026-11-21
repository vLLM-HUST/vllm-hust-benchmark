# 官方 0.30 固定配置报告

Qwen3.5-35B-A3B、BF16、两卡原测：固定五项、4K/16K/32K 真实上下文、四租户及三次原生 T。独立 companion 完成 200 请求后，自动核验拒绝算量合同，F 与效率为空；原整轮报告保持 failed。

[公开原件、曲线和 CPU 复算](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/official-b0-0.30-full-report-20261122-v1)、[归档清单](archive-manifest.json)、[原报告](original-report.json)、[自动拒绝](automatic-ft-rejection.json)、[来源链](source-chain.json)、[匿名下载复算](anonymous-replay.json)。

元数据归档保留 74 个原文件，其余 9042 个只保留摘要；另附最小前缀反例复算包。C8 合成负载保留原请求数和时钟，本次未复算 SSE，吞吐和时延不从输出预算推断。CPU 复算不代表全套通过、人工见证或新的 NPU 测量。
