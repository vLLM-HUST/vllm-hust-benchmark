# INT8 KV 动态量化：功能、容量与快速测试补证

关联：[MOD issue #3](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/issues/3)、[总入口 #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)。维护者新增提供 2026-11-16 快速测试，完整原件和独立复算见 [MMLU-Pro 快速测试补证](quick-test-20261116/README.md)。本轮保留该单轮固定子集的输出速率观察；旧 A3 长上下文增幅仍撤回，不主张稳定性能或质量无损。更新日期：2026-11-17。

## 可引用结论

> 在双卡 910B2、TP2 的 MMLU-Pro 固定 200 题单轮配对中，INT8 输出 token 速率观察值增加 2.177462768170302%，正确题为 145/200（默认 KV 为 142/200）；整批耗时及平均 TTFT 同时增加，不能主张请求吞吐、稳定性能或质量无损收益，且不替代旧长上下文质量证据。

## 容量、HBM、并发与吞吐分别记录

| 项目 | 已有记录与本轮结论 |
| --- | --- |
| 理论容量估算 | PR #9 及旧文档报告 8192 上下文的 90.14→126.20，约 +40%。本轮仅将它保留为待核验的历史引擎估算：所存短提示 JSON 不含该估算的启动日志或容量计算字段，不能确认其配置推导，更不能记作实际并发提升。 |
| 实际 HBM | 旧归档缺少配对 HBM。新测试保存设备时序，独立复算纠正原报告将未匹配字段误记为零的问题；采样覆盖及两臂负载限制见新报告，不据此主张同负载 HBM 节省。 |
| 实测并发 | PR #9 的短提示批次只有 8 条请求；max-num-seqs=16 是服务上限，不是 90/126 路实测。后续 A3 档案包含多个固定客户端并发档，但没有同工况长上下文质量结果，不能升级为稳定并发或项目验收结论。 |
| 请求吞吐 | 单位 requests/s；新固定题目批次为 0.2986673051902859→0.29795253265346056，略降，不从 tokens/s 或理论容量推算。 |
| 输出吞吐 | 单位 tokens/s；新增 MMLU 固定子集单轮观察及同批质量原件，见新报告；输出量也增加、耗时略增。新 C8 负结果同时公开；旧 A3 增幅不恢复。 |

容量缩减机制为以 INT8 保存 K/V 并保留每通道 scale 等额外存储。不能由存储位宽直接推出整卡 HBM 减半；模型权重、固定状态、工作区与缓存分配预算均影响实际值。

## 代码与功能身份

| 内容 | PR / 固定提交 | 结论范围 |
| --- | --- | --- |
| 原始动态 INT8 插件 | [60cd123](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/commit/60cd123df73b1d0af8b161c4cbdd8faf565cc02f) | 原始插件提交，作者 hustcui；现有发行的来源见 MOD 仓 PROVENANCE。 |
| INT4 / Qwen3.5 集成 | [PR #9](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/pull/9)、[2f0b37b](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/commit/2f0b37b63dc1aba25f035a0dbf620f4abb73b2b3) | 包含多人的历史贡献，不能将合并作者视作所有底层算法与实测的唯一作者。 |
| 当前宿主初始化适配 | [PR #11 / baec6d1](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/commit/baec6d1d31c838848ccbf0aedcca21d3cf82e717) | 动态 INT8 初始化与静态 C8 配置隔离；支持范围以该提交说明为准。 |
| Graph 捕获与校准修复 | [PR #13 / e80fe07](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/commit/e80fe07db24cd52394479fa056e89ac27affd594)、[原提交 1380cfc](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/commit/1380cfcf82e1eb7475cf29af9bd6828ff5ab9785) | 保持 scale 缓冲地址、排除无效校准槽位并拒绝未校准真实回放；CPU 契约测试不等于新设备实验。 |

历史功能修复检查以 e80fe07 为身份；新快速测试实际插件版本为 0136fde6000384f4a25cec6bacbf68b4d48156e8。宿主 SHA/dirty diff、后采模型权重哈希及设备采样见新原件。新测试身份不能替换旧 A3 实验身份；仍有编译缓存未隔离、部分单测未跟踪文件未封存等限制。

## 已归档原件：固定链接，不重复上传

- [完整历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/README.md)、[配置](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/configuration.json)、[逐轮结果](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/results.json)。实验：Qwen3.5-35B-A3B，TP2，BF16 权重，auto/int8 两臂，2×910B2；合成重复文本，30720 输入 / 2048 强制输出、1800 秒窗口，包含 drain。原始请求、服务日志、控制器和启动脚本见清单。
- [性能原件清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/evidence-manifest.json)：三个原始包、分片以及逐文件 SHA256/字节数均已归档于同一固定提交，历史证据提交由 [专用归档分支](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/archive/int8-kv-evidence-20261105) 保留。依清单链接原文件；不复制到本目录。初始轮只统计完成请求，后续轮采用窗口内 continuous usage 增量，不能混算。
- [独立 GSM8K 固定 200 题报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/quality-gsm8k-200-20261105/README.md)、[逐题标量](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/quality-gsm8k-200-20261105/summary.json)、[原件清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/87b2c10325f9ca763925a98c05941741a0b0ea84/reports/a3-int8-kv-20261104/quality-gsm8k-200-20261105/manifest.json)：输入 761–1654 token，并发 8；两臂各 200 个计划样本，正确题 194/195，长度截断 1/2，截断保留分母。完整请求/输出、错误与截断样本、参考答案及源码快照现已归档于本目录的 [GSM8K 原件包](quality-originals/README.md)，对应 [逐文件清单](quality-originals/manifest.json) 与 [独立校验器](quality-originals/verify.py)。读者下载分片后可离线校验和复算，不需要访问实验机器；原始清单中的本机路径仅是来源记录。

GSM8K 是独立短上下文子集，不能当作旧长上下文实验的联合质量验收。旧 A3 30 组均未通过约定延迟门槛，实验时插件/Ascend 完整 SHA 与 dirty diff、权重哈希和设备采样未完整冻结；这些限制保持可见。

所有固定引用及文件 SHA256 见 [evidence-index.json](evidence-index.json)。本机已重新运行旧归档校验器，验证三个包、30 组、7463 请求的完整性与计数；亦核对 GSM8K 清单中的标量及外部原件哈希。[verification.json](verification.json) 只证明原件校验，不证明新实验通过。

## 实际贡献与后续安排

历史 A3 与 GSM8K 实验实际执行人为 hustcui（维护者于本轮确认）；归档提交 [6c90e778](https://github.com/vLLM-HUST/vllm-hust-benchmark/commit/6c90e778db674dffa70674c0dcd945e6d675e323) 与 [87b2c103](https://github.com/vLLM-HUST/vllm-hust-benchmark/commit/87b2c10325f9ca763925a98c05941741a0b0ea84) 均保留原作者。插件贡献分工由 MOD 仓贡献说明关联实际提交；WenjuZhao 负责 MOD 整体技术方向把控与团队协调（组长职责，由维护者在本轮确认），不据此推定具体代码或设备实验作者。

本次只离线整理与复算已完成的新快速测试，不申请设备、不重跑模型。若以后申请旧长上下文吞吐或稳定并发收益，须另立冻结计划：同时保存模型/运行时/插件身份与 diff、两臂完整命令、HBM 时序、实际 Running/Waiting、未四舍五入请求与 token 计数、同工况质量及错误/截断样本。未完成前不得将本次固定子集观察外推为上述收益。新原件完整公开、原始报告按字节保留；纠错结果由独立入口生成。
