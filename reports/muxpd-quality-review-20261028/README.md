# MuxPD 质量复核与三题数值定位，2026-10-28

[REVIEW-v1.md](REVIEW-v1.md)回应
[Issue #23 的新复核留言](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23#issuecomment-6446722727)：分别记录安装包、同步
raw/group、历史 streaming 和异步候选的实测身份，保留特定负载的 Goodput 收益，同时保持既定答案不退化门槛 **FAIL**。

本轮已完成三道历史丢失题的原件追踪，以及10672原八请求组的实机定位。保存8192-token采样预算，在目标第18个输出后主动中止，共五次 fresh
尝试：四个有效采集、一个原始诊断清理校验器失败，全部保留。没有完成新答案、重跑200题、修改运行时代码或新增性能结果。

两次 native 的144对捕获向量字节相同；raw observer/plain 的八条保存前缀相同。10672前16个 token 相同，第17个输出的两项 logits 差从 native
的+0.250变成 raw 的−0.125，唯一 argmax 随之翻转。目标首个 P 输出向量已不同，但完整 batch 上下文并不匹配；这是数值定位进展，尚非唯一根因或质量修复。

原输入副本、派生脚本、producer v1/v2、五次尝试（含原始失败）、完整 tokens/logits
与日志已发布到[本轮 benchmark Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-quality-review-20261028)。三个公开资产均通过[匿名 HTTP 200、大小及 SHA256 校验](public-download-verification-v1.json)，下载归档的全部746个原始成员通过[逐成员验证](public-member-verification-v1.json)。Git
仅保存本报告、小摘要、[manifest](MANIFEST-v1.json)、[校验和](SHA256SUMS-v1)及验证回执；原始大文件保存在 Release。

上一轮异步调度报告的 publication-pending
文字残留由[单独订正](../muxpd-scheduling-20261026/PUBLICATION-STATUS-CORRECTION-v1.md)说明。其原始测量、报告和验证回执保持原字节，不把文字订正当作新实验。
