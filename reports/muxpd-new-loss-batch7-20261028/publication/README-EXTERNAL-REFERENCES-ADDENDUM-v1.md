外部依赖补充：raw 历史原件仍是完整 analyzer 重跑的必需输入

本补充纠正封存 `external-references-v1.json` 中 `raw_reference` 的适用范围。四个新实验只有 native/prefix × observed/plain，没有 raw 实验角色；但冻结 analyzer 的 `verify_historical_references()` 会遍历全部四份历史 full-run 引用，包括 `raw-quality-v2-hccl-true-01`。因此 raw 虽只用于历史上下文，其原始 `controller.json`、`results/run.json`、`process.log` 仍必须具备并通过哈希/大小校验，才能完整重跑 analyzer。不能把“本轮不运行 raw”理解为“重算不需要 raw 历史文件”。

所需旧 raw 归档：

- 下载：[muxpd-raw-quality-v2-hccl-true-01.tar.gz](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-raw-quality-v2-hccl-true-01.tar.gz)
- 大小：5,103,766 字节。
- SHA256：`bbfdef527e51194205b7f7cf1e5d5613f2b63e3c7c2e3e8482e075dc8cfabf8a`。
- full-run 成员：`runs/raw-quality-v2-hccl-true-01/results/run.json`，3,397,246 字节，SHA256 `75aa8495902fd795dbe845da35eb1bf06bb3adea7c81b515e8c3375b07a13f9e`。

全部三个必需 raw 成员的路径、大小和 SHA，以及原发布 manifest/readback 引用，见 `external-references-addendum-v1.json`。它与原外部依赖清单合用，使历史依赖覆盖两次 native、一次 raw、一次 prefix full-run；此处针对“是否必须提供 raw 历史原件”的说明优先于原清单的歧义措辞。

本次没有修改或重建已封存的 84,817,778 字节 v1 tar，也没有修改原报告、manifest 或原 `SHA256SUMS`。补充两文件由独立 `SUPPLEMENT-SHA256SUMS` 绑定；原 v1 分析失败及 v2 修正记录仍完整保留。本补充没有下载历史附件、执行新实验或验证公开可用性，不增加质量、性能或因果结论。
