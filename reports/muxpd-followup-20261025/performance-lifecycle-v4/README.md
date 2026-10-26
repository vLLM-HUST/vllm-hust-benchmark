# MuxPD TP4 性能与生命周期追加 v4

**分组 P 相对 raw 插件的混合负载吞吐提高3.466%，但仍比匹配的同步原生参考低2.325%。** 较短的最大seed暂停来自明确标注的补充分析，同时存在长P TTFT及seed gap
P99代价。实用默认原生参考、实际profiler与成对任务质量尚未完成；不能据此宣称普遍优于原版vLLM。

[完整报告](REPORT-v4.md)保留全部指标、六个角色、相邻配对与漂移，并明确mixed模式下native与插件生成token不同对成本归因的限制。[原primary分析](analysis/primary/group-abba-with-native.json)、[全部36窗口](analysis/primary/all-window-metrics.csv)和[独立审查](analysis/primary-review/review.json)可逐项复核；12个warmup窗口明确排除于性能估计。

本批恰有12个终态运行：五项生命周期实验、六项主性能实验和一次默认原生v1的计时前失败。生命周期结果支持有限P8取消/ACK持块/释放与复用、真实D抢占时pending
P块保持；原20块套件失败与SIGTERM/SIGKILL回退记录均保留。默认原生v1失败有0个测量窗口，v2仅有准备身份和CPU检查，未混入性能估计。

[归档索引](archive-index-v4.json)记录13个本地已验证包、1181个原件成员与96,892,412压缩bytes；[明确终态选择](terminal-selection-v4.json)、[support选择](support-selection-v4.json)和[上传准备清单](upload-selection-v4.json)固定本批范围。**18个新增资产已于2026-10-26
02:05:57 UTC完成独立匿名公开回验。**
[当前公开回执](PUBLIC-READBACK-v4.md)核验全部13个归档/1181个成员及前后57项公开清单，旧39项仅比较API身份。上传前报告及准备清单继续保留原状态。
所有原有248个报告/协议文件及旧公开回验保持原字节与时间身份。

完整运行日志、编译缓存和24528个seed间隔只在Release归档中；Git保存可审阅报告、小元数据和源码snapshot。[全间隔原件定位](analysis/primary/stalls-release-location.json)给出SHA、大小与support成员名，原件定位文件仍保留准备时的字段；实际下载证明见新增公开回执。

```sh
python reports/muxpd-followup-20261025/performance-lifecycle-v4/verify_v4.py
```

该命令仅检查离线保存的身份及有限结论，不运行模型、不访问网络、不执行归档代码。此次公开回验通过新增回执呈现；后续实验继续追加，不能改写本准备清单。

[追加的发布范围独立复核](reviews/publication-scope/independent-scope-review-v1.json)确认固定12项身份、原primary表格及生命周期/失败边界；该复核没有重新检查归档或执行实验。
