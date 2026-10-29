# MuxPD 实际首 D 图：相同输入下归约输出与单行重放一致

真实TP4观测中，四rank各自的首mixed D GDN0 TP输入与历史native输入逐字节相同；实际图output与此前private B单行结果逐字节相同，与native
A/原大形状private
C均有625/2048个元素不同，最大绝对差0.00390625，RMS为9.40213873039046e-05。全部8请求历史完整保留前缀和新observed/plain桥通过。

**当前物理卡2/3/4/6与native历史0/1/2/7不同。**
这是同输入归约边界的描述性结果，不能唯一归因geometry或graph模式，硬件拓扑仍是混杂。没有证明07338唯一根因、答案修复或性能收益；full200答案不退化仍FAIL。

- [完整报告](REPORT-v2.zh-CN.md)、[主分析](analysis/analysis-v2.json)、[独立raw审计](independent-audit/summary-v2.json)及[主分析交叉复核](independent-audit/primary-crosscheck-v1.json)：独立核验图8条raw/8条NPY和native16条raw/16条NPY；16次历史前缀比较与8对新桥通过。
- [observed终态](observed/controller.json)、[plain终态](plain/controller.json)：exit0、errors空、owned成员空、卡空闲，无controller额外清理。两次executor均有grace后SIGTERM，无SIGKILL，不能写成全部worker自然退出。
- [本版freeze](protocol/freeze-manifest-v1.json)、[源码/库合同](protocol/operator-contract-v1.json)、[图源码合同](protocol/graph-source-contract-v1.json)和[构造兼容源码](protocol/graph_scheduler_compat_v2.py.source)：仅LLM构造期间一次可逆worker类名映射，原scheduler类、selector和所有调度方法保持。数值观测组件及输入按原字节复用。
- [v1失败终态](legacy-v1/controller.json)、[原失败分析](legacy-v1/analysis-v1.json)、[取消plain的处置](legacy-v1/root-failure-disposition-v1.json)、[独立失败审查](legacy-v1/failure-review-v1.json)：四rank启动捕获后LLM构造被原worker类名guard拒绝，0请求，plain未启动，原矩阵保持FAILED_INCOMPLETE_MATRIX。不是挑选成功重试。
- [归档身份](publication/archive-record-v1.json)、[457项payload清单](publication/archive-manifest-v1.json)、[本地核验](publication/local-verification-v1.json)：tar为32,138,688字节，SHA
  `e59947c1f566f632526445b8b675d965e0b4d2716e1bbbaedd38ceff08ff5e95`；457
  payload加内嵌manifest，458成员及gzip
  EOF/CRC本地通过，其中含[197项完整v1失败payload](publication/legacy-v1-payload-v2.json)。
- [135项历史外部依赖](publication/external-dependencies-v2.json)保留native DTP、conv失败/成功和full
  controls来源；模型/运行时和全部历史控制不在新包中，不是完全独立可移植重跑包。[归档准备计划](publication/archive-plan-v2.json)保留其创建时状态，不追改历史记录。
- [复制来源](copied-sources-v2.json)绑定仓库小型原件字节；Python原件使用`.py.source`避免格式化改写，仓库报告只允许排版整理。

[公开Release的11项附件](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-prefix-first-decode-graph-20261029)已完成上传及匿名读回。[原API快照](publication/release-api-v2.json)、[最终读回回执](public-readback/final-status-v2.json)、[下载期望值](public-readback/expected-truth-v2.json)和[读回源码](public-readback/readback.py.source)分别保留：11项各单次GET均HTTP200、TLS验证通过、大小及SHA256一致，完整tar的458成员和gzip
EOF/CRC通过。9项基础附件中的SHA256SUMS覆盖8项文件；新增launch-path
review由独立SUPPLEMENT-SHA256SUMS覆盖。本地归档记录中的remote_verified=false仅是创建时状态，原件未追改。原封包REPORT仍保留封包时待公开验证的文字；本README记录此后完成的公开状态。公开可用性不替代答案不退化门。

[执行路径偏差独立复核](supplementary-launch-path/launch-path-review-v1.json)是事后补充，**不属于已封装458成员**，原archive未重新封装。冻结matrix/dry-run记录runs-v1，实际root
CLI使用runs-v2；原矩阵未逐字遵循，这不是事前修订。复核确认两固定labels、24个producer、plan/config字节和实际helper身份不变；仅输出、派生module及compiler-cache路径前缀替换，无新增attempt或择优重试。controller保存child
argv与destination，未保存完整parent argv，因此不声称完整parent命令字节一致。该偏差复核不取消跨硬件与质量/性能限制。

[新增复制清单](copied-sources-publication-v2.json)绑定API、完成的读回回执、11份curl元数据及补充校验和；这些发布后证据不属于原458成员归档，未重新封装原包。
