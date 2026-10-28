# MuxPD 原 batch6 反事实实验，2026-10-28

[COUNTERFACTUAL-v1.md](COUNTERFACTUAL-v1.md)继续检验
[前一轮质量复核（固定提交1c70f51a）](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/1c70f51a70e90943851e246a41fa13e9afc5332c/reports/muxpd-quality-review-20261028/REVIEW-v1.md)
发现的10672第17个输出分歧。本轮四个新进程均完成有界采集和清理；q40、原生分步执行、单次分页 attention-state 干预与新 q4 对照，各自144个捕获向量均与原 raw q4
逐字节相同。

这组实验缩小了定位范围，没有得到有效的质量缓解。原200题的逐题答案不退化与截断门槛保持
**FAIL**；[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)保持
**OPEN**。本轮没有重跑完整答案、产生新吞吐/Goodput 数据或赋予当前安装包新资格。

独立审计覆盖四个新终态、三个既有有效对照、全部21组配对和1008个向量文件。原件、冻结 producer、完整分析与小型审计的归档入口为
[evidence-muxpd-counterfactual-20261028](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-counterfactual-20261028)，资产名
`muxpd-counterfactual-20261028-evidence-v1.tar.gz`。归档
SHA256：`1ea0a4861112593f111ad9e67ca3edb66fe60f1ad5d2e91555c10167d7604ff8`；三个公开附件匿名 HTTP 200、大小与 SHA
校验通过，归档内全部 738
个原始成员校验通过。见[下载回执](public-download-verification-v1.json)和[成员回执](public-member-verification-v1.json)。
