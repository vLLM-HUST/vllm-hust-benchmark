# 官方 0.23 固定全套报告存档

Qwen3.5-35B-A3B，BF16，Ascend 910B2 × 2。包含固定五项、4K/16K/32K、四租户、三次原生计时与自动 F/T 核验报告。此处 full
指固定配置，不是所有上游题目。

4K/16K/32K 长上下文与 4K 四租户 C24 均回放 Terminal-Bench 2.1 的五条任务历史。该轮封存的输入 SHA-256 与
[r9 专项](../r9-qwen-run29-20261121/README.md) 的冻结输入一致，为
`5156a6ae436c7e32364a5ecbb028b52f26da1808be883c02e8e230defd556559`。合成 C8 使用独立输入配方。

MMLU 为 152/200；Frontier 的 olympiad、research 分别为 64/100、4/60，未解析保留。HLE 固定 200/1811，已评分 101，未解析
99；不得作为完整准确率。SWE/Terminal 为公开轨迹推理，不测 Agent 任务成功率。

自动算量适用性核验未通过：观察到的第二步 KV 长度超出所审查的 selected-prefix 计数域。保留反例和原三次 T，F
与效率为空；未观察内核实际访问或槽值，不能据此宣称引擎正确性缺陷。没有第三方人工见证。

归档含 67 份经原封存逐项核验的报告/身份元数据；8646 份其余原始成员仅保留摘要。它不含完整 SSE、权重或 SDK。CPU 入口验证所选 33
份报告依赖并重建报告与曲线，不重算原始流、不启动模型。原 C8 存档保持原样。

解压 portable-full-report-metadata.tar.gz，获取固定 CPU 报告源码
[778426ce](https://github.com/DeepLinkNext/DeepLinkEngine/tree/778426cef90ed398019fd7a92119eba418ee14de)，执行：

```sh
python run_all.py --official-baselines --report-only --config /解压目录/report-only.example.json
```

配置固定原报告与版本封存摘要，输出须为新目录。绘图依赖由脚本探测和安装；本轮首次下载超时留档，后用已校验报告环境完成。自动 F/T 拒绝报告来自独立公开入口 0b3450f9，见
automatic-ft-execution.json；不把报告源码重标为设备执行源码。

[公开附件与摘要](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/official-b0-0.23-full-report-20261122-v1)；[归档清单](archive-manifest.json)、[原报告](original-report.json)、[自动拒绝报告](automatic-ft-rejection.json)、[匿名复算](anonymous-replay.json)。
