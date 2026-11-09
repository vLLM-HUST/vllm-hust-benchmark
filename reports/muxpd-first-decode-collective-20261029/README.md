# MuxPD 首个 D TP collective：payload形状改变了相同输入的规约结果

本次固定边界的四rank结果均为：**实际native
mixed输出A与原形状重放C逐字节相同；相同输入的单行重放B有625/2048个输出元素不同**，最大绝对差0.00390625，RMS为9.40213873039046e−05。重复、跨rank一致性、历史前缀复现和observed/plain桥全部通过。

这是首层、首个seed D通信边界的相同输入形状敏感性证据。B/C为排空后的原TP组私有重放，**真实prefix
graph尚未观察**；没有证明完整答案修复或新的性能收益。原full200质量门仍FAIL，运行时PR27保持实验草稿。

- [结果报告](REPORT-v1.zh-CN.md)、[完整分析](analysis/analysis-v1.json)、[独立raw审计](independent-audit/summary-v1.json)：独立核验28对raw/NPY、32对历史前缀、8对新桥及16份maps/RPC/helper回执。
- [observed终态](observed/controller.json)、[plain终态](plain/controller.json)：exit0、无剩余owned进程；两次executor均有grace期后SIGTERM，无SIGKILL或controller额外强制清理，不表示所有worker自然退出。
- [冻结producer与准备清单](protocol/freeze-manifest-v1.json)、[库与通信源码身份](protocol/operator-contract-v1.json)、[conv与DTP沿革](protocol/protocol-lineage-v1.json)、[此前失败保留](protocol/prior-failure-reference-v2.json)。
- [原始包身份](publication/archive-record-v1.json)、[逐成员manifest](publication/archive-manifest-v1.json)、[本地完整核验](publication/local-verification-v1.json)。tar为38,652,433字节，SHA256
  `4af1329bebae0ae494a5b880682c33cf56271bd7e7860769b7fd334fc213e24a`，218个payload加一份内嵌manifest，全部219成员及gzip
  EOF/CRC本地通过。
- [48项外部文件依赖](publication/external-dependencies-v1.json)含两native、raw、prefix完整历史控制及conv失败/成功沿革；新包不包含所有模型、运行时或历史完整run。原[归档准备计划](publication/archive-plan-v1.json)保持其创建时状态，不追改为后续完成事实。
- [复制来源](copied-sources-v1.json)绑定小型原件字节。仓库报告允许排版整理，原封包/独立附件报告字节由归档manifest绑定。

[含完整原始包的9个公开附件](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-first-decode-collective-20261029)已于2026-10-29完成上传和匿名读回。[API原件](publication/release-api-v1.json)与[匿名下载回执](public-readback/final-status-v1.json)分开保留：全部9项单次GET均HTTP
200、TLS验证通过、大小和SHA256一致；完整tar的219成员及gzip
EOF/CRC通过。[下载期望值](public-readback/expected-truth-v1.json)和[读回源码](public-readback/readback.py.source)提供复核入口。此前本地记录中的`remote_verified:false`是其创建时状态，未追改原件。公开可用性不替代答案不退化门。

附加的[CPU规约顺序假设](supplementary-cpu-order/result.json)枚举120种逐次BF16
RNE舍入的有序二叉树：A/C可由`((r0+r1)+r2)+r3`匹配，B可由`((r0+r1)+r3)+r2`匹配，各有8种交换等价形式。[独立CPU复核](supplementary-cpu-order/independent-review.json)重算全部候选，并用326,400个有限FP32样本逐位验证RNE公式与PyTorch
CPU
BF16转换相同。[原始脚本](supplementary-cpu-order/analyze.py.source)保留原字节；`.py.source`后缀避免格式化改写。这是事后数据匹配假设，未观察HCCL实际算法，不是生产修复、真实prefix图观测或质量证据。**这3份附加文件仅在仓库中，不属于原219成员归档；归档和原REPORT附件未重新封装或修改。**[新增复制清单](copied-sources-publication-v1.json)绑定其来源和SHA256。
