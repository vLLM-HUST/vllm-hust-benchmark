# r8 有效算力参考发布订正（2026-11-19）

此前完整结果已归档到 DeepLinkEngine，benchmark Release 目录没有接入，网站展示代码仍在草稿 PR714。本次补接已有封存认证，不新增推理、不更改冻结引擎与原始测量。

B0效率下界22.2269749464%，B1效率下界89.7838253325%。B1的50%里程碑达成，90%最终目标未达成。MMLU200/out256/EOS，每臂三轮，BF16/TP2/双910B2；完整回答五项成绩、上下文与四租户数据保持原协议和身份。

每臂所有轮次先求和，再计算：

- C_current_lower = ΣF_current_lower / (2 × ΣT_current × 10^12)
- C_ref_upper = ΣF_ref_upper / (2 × ΣT_ref_lower × 10^12)
- η_lower = 100 × C_current_lower / C_ref_upper

参考时间下界由独立组件标定和保留资源依赖图最长路径求得：finish(v)=duration(v)+max(finish(parent))。本轮与原参考算量不同，保留各自F/T，不相消、不复用旧效率百分比。B0原参考上界26.8182576764809057、B1原参考上界19.3403833468717505
TFLOP/s/卡。

[配对认证及全部六轮来源](../../website-publication/data/r8-compute-reference-20261119-v2/compute-reference/matched-reference-pair.json)
·
[完整参考分母统计](../../website-publication/data/r8-compute-reference-20261119-v2/compute-reference/qualified-full-reference-statistics.json)
· [网页报告](../../website-publication/data/r8-compute-reference-20261119-v2/report.html) ·
[公开原件与复算入口](https://github.com/DeepLinkNext/DeepLinkEngine/blob/72a677b591f9c5c0d911e9a280759278bc0f4a74/benchmark/reports/r8-compute-reference-20261119-v1/README.md)。

数据生成与CPU核验使用本目录的prepare_publication.py；其中固定公开来源文件清单并逐项核对12份工作/时钟原件、全部三轮和参考统计。原证据保持原字节，订正仅补充目录和发布投影。
