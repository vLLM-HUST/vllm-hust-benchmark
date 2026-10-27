# MuxPD v5：质量、实际原生参考与混合 profiler

**当前插件在保存负载中的价值是缩短长 prefill 到达时的最坏 decode 停顿；吞吐、TTFT 与答案不退化有明确代价，不能宣称全面超过原生 vLLM。**
[最终报告](REPORT-v5.md)同时保留收益、失败和适用范围。

- 两个原生质量控制各 165/200，raw/group 各 166/200；raw/group 全部 200 题的 token、文本和终止记录相同，但相对原生共同正确集合均新增 3
  错。原严格门槛和独立预冻结补充答案门槛都 **FAIL**；collection
  和实际混合覆盖通过不等于质量通过。[逐题结果](analysis/group/review.json)、[严格结果](analysis/group/strict-result-v1.json)、[补充结果](analysis/group/supplementary-result-v1.json)。
- 与两个实际 async/npugraphEx 原生参考相比，group 混合吞吐低 38.254%，长 P TTFT 高 35.297%，seed gap P99 高 86.157%；但四个测量
  mixed 窗口的最大 seed 间隔从原生 423.349–445.610 ms 降为 group 75.486–86.950
  ms。[完整原报告](originals/NATIVE-REFERENCE-REPORT-v2.md.source)、[全部窗口](analysis/native-reference/all-window-summaries.json)。参考单独描述，不加入既有
  ABBA，不称所有默认或厂商最优配置。
- Group 相对 raw 的原混合吞吐增量 +3.466% 保持 [v4 的原定义](../performance-lifecycle-v4/README.md)。本次真实混合 profiler
  每 rank 原生 kernel launch 反而增加 70（2854→2924）；不能用更少 launch 解释增量，也没有证明物理
  overlap。[capture 对比](analysis/mixed-profile/comparison-v1.json)、[完整 API 清单](analysis/mixed-profile/all-runtime-api-comparisons-v1.csv.source)。

七个终态运行和一个支持胶囊已于 2026-10-26 04:52 UTC 完成独立匿名读回：13 个新资产、8 个胶囊、2,463 个原始成员全部一致，旧 57 个资产 API
身份保持不变。[公开回执](PUBLIC-READBACK-v5.md)仅证明这次下载和 custody，不改变实验结论。下载计划及完整 SHA 入口为
[Release 资产清单](release-assets-v5.json)、[归档索引](archive-index-v5.json)、[上传清单](upload-selection-v5.json)和
[SHA256SUMS](SHA256SUMS-v5)。小文件 Git 副本不代替约 729 MiB 的完整解析 profile 树或原 run；大文件全部留在胶囊。

原 v2/v3/v4/预冻结质量协议的 400 个 canonical 文件保持原字节。最初准备时的
[草稿](REPORT-DRAFT-v1.md)、[pending 状态](staging-state-v1.json)与[原导航快照](originals/STAGING-README-v1.md.source)保留为历史，不覆盖本次完整终态。原
default-v1 首窗口前失败、旧严格失败、native offline analyzer
首次失败均未改写；[另行版本的精确默认展开适配](originals/native-default-offline-adapter/README.md.source)仅修正离线识别三个 native
默认键。

实际设备 source 为 f7；source98 的既有桥接证据范围保持不变，后续 docs HEAD 不充当实测身份。#11 的有界调查关闭不代表质量通过；#23 仍有已知质量限制。本目录不是合并
PR10、解除默认关闭或做出普遍数值等价承诺的证明。[源码支持边界](originals/BATCH-INVARIANT-SOURCE-AUDIT-v1.md.source)和[最终处置范围审查](originals/final-scope/RAW-QUALITY-FAILURE-SCOPE-REVIEW-v1.md.source)保留完整限定；Issue
评论草稿仍是草稿。

```sh
python reports/muxpd-followup-20261025/quality-reference-v5/verify_v5.py
python reports/muxpd-followup-20261025/quality-reference-v5/verify_public_v5.py
```

此离线检查核验精确小文件、终态选择、原失败状态和旧文件保护；它不重新评分、不执行胶囊代码、不运行模型，不证明远端当前可下载。历史副本的原相对/绝对路径保留，不能直接假定归档重定位后可执行。
