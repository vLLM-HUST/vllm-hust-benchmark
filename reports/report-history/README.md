# B0/B1 历史报告索引

[网站按日期浏览](https://vllm-hust.sage.org.ai/report-history.html) ·
[机器可读索引](../../website-publication/data/report-history-v1.json) ·
[最新一轮对比](https://vllm-hust.sage.org.ai/b0-b1.html)

索引包含本轮 r6、旧 r4 固定五项归并、完整回答联合测量、M01 历史参考、Controller 探索及所有现有中期报告发布版本。每个入口绑定原报告的固定提交与
SHA256；指标与原始结论不重写。

日期统一按北京时间倒序展示，并区分：

- `experiment_date`：原件明确记录的测试日期。
- `report_date`：报告或归并日期，不代表所有复用原件均在当天运行。
- `archive_label_date`：原归档目录的日期标识，未另行证明实际请求执行日期。
- `source_revision_date`：原报告来源提交的修订日期，不代表新增推理；同时保留 UTC 提交时间。

旧五项报告的 v1、稳定性解释订正 v2 及 HLE 退化记录均保留。历史 M01/out256 结果不用于替代本轮完整回答参考测量。 `current`
只指网站选择的最新报告，不表示四项目标全部验收通过；旧报告的通过或未通过判定保留在各自工况内。

更新报告时先在本仓库追加证据，再更新索引中的当前入口与历史状态；网站通过固定提交及发布清单校验后读取。
