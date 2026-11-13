# 固定五项测试 · 2026-11-13 封存结果归并

本轮通过 Engine 根目录 `one_click.py` 执行全部五项冻结范围，显式复用四批封存原件。 15 阶段重新核验，新增推理阶段 0；不是一次新连续推理，也不是五个上游数据集原始全集。
原批次日期、失败/停止状态、预算、裁判未决与负收益均保留。首次归并尝试因传入归档副本 而被原始流路径校验拒绝，未启动模型；第二次使用原目录通过。两份回执都保留在私有原件中。

固定范围：MMLU200；HLE Gold文本64/图像10、Revision文本113/图像13； Frontier
Olympiad100/Research60；SWE公开trace32任务64请求/臂； Terminal公开trace32任务48请求/臂。后两项只测推理，不执行工具，不报告任务解决率。

## 四目标与限制

- 3.4.1：复算历史独立 out256 工况参考，不将其改写成本轮新增 F/T 测量。
- 3.4.2：MMLU固定200配对性能数值达标；这不是五项质量等价或所有工况时延达标。 HLE首token均值退化，MMLU正确数下降1/200，Frontier
  Research成功数下降2/60，均保留。
- 3.4.3：固定五项执行完成，整体验收未通过。HLE两臂各106/200裁判未决， Frontier裁判未决7/5；SWE/Terminal输入≥16384的实际请求数均为0。独立C8的
  4K/16K、60秒预热/300秒正式窗口不能替代真实任务的16K尾延迟稳定性验收。
- 3.4.4：MMLU同费率单位token成本估算与同窗口双卡采样利用率在披露范围内达标， 费用沿用2026-10-29类似机型公开报价，不是本机账单，也不宣称该报价仍为当前价。

[完整公开指标与来源摘要](summary-public-v1.json)包含每项两臂吞吐、均值/有限样本分位数、
输入覆盖、长度终止、错误、未返回、利用率、成本估算、分组评分及独立judge阶段；不分发题目、答案或原始对话。

复算入口为
`scripts/export_five_dataset_results.py --run <封存一键结果目录> --engine-tools <固定Engine工具目录> --aggregation-commit <固定提交> --output <新公开JSON路径>`，依赖原始同机路径可读及已核验的一键报告。

新增复用入口与私有完整派生报告见
[B1 r4.3.10](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4.3.10)。
冻结数据标准保持 [five-datasets-v1](../../docs/dataset-standards/five-datasets-v1/README.md) 原字节。
