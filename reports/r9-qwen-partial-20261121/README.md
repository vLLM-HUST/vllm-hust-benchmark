# R9 Qwen 已完成阶段与续跑记录

固定 B0 Original018 与冻结 R9 的 Qwen3.5-35B-A3B / BF16 / TP2，物理卡 1、6。这里只发布已完成的真实阶段，不新跑推理。

| 工况        | B0 → R9 吞吐（tok/s） | 请求 / 每臂 | 正确数 / 固定分母 |
| ----------- | --------------------: | ----------: | ----------------- |
| MMLU200     |      506.24 → 1494.71 |   200 / 200 | 139 → 143         |
| Frontier160 |       388.59 → 988.13 |   160 / 160 | 69 → 65           |
| 4K / C8     |       126.24 → 384.29 |   152 / 455 | —                 |
| 16K / C8    |        59.57 → 159.54 |    73 / 189 | —                 |

Frontier 两侧使用同一个 R9 裁判，未决 B0 9、B1 8，保留在 160 分母内。该质量与性能报告独立于书生模型的 R9 集成验证；不替代完整五项、有效算力或上下文验收。

HLE 的 B1 服务在首个正式请求之前发生驱动锁页内存分配错误 `aclrtMallocHostWithCfg 207001`，后续阶段因此未执行。控制器持有的可选 16 GiB
预留未按实际剩余内存释放，修复见
[DeepLinkEngine #37](https://github.com/DeepLinkNext/DeepLinkEngine/pull/37)。原始失败保留。当前容器卡号及 PCI
地址改变，续跑需要恢复原卡，不能跳过同设备准入或自动重跑 B0。

[公开统计](summary.json)保留阶段、原日期、原 B0、封存摘要与缺项原因。原始流和记录位于
[R9 Release 附件](https://github.com/vLLM-HUST/vllm-hust-engine/releases/download/full-answer-b1-20261119-r9/r9-qwen-partial-originals-20261121-v1.tar.gz)，SHA256
`a73f661e451f5fe606094978416330a42bca1c45777d01487cf6532e7cd9226c`。附件重新下载核验通过。共 7,292 文件；8
个环境文件仅保留原哈希，避免泄露凭据，原件未修改。

下载并解包附件不需要模型服务。以下命令核验全部公开文件，逐流复算四组吞吐、TTFT、TPOT 和阶段时钟；不启动模型：

```bash
python recompute.py --capsule r9-qwen-partial-originals-20261121-v1.tar.gz --check summary.json
```

参考分母保持原同工况完整统计；R9 尚未执行本轮原生分子，不用 R8 的效率百分比补写。
