# vSpec 0.15.2：FrontierScience / HLE-Verified 两轮复测证据

本目录保存两组冻结题单下的真实双卡 910B2 评测：每组均有 vSpec 2/4/6/8 与原生 MTP2/4/6/8 两轮求解、独立离线评分，以及一轮单列的逐步 gamma trace 诊断。它是专题实验报告，不是官方榜单或公开 wheel 的原样跨机验收。

| 数据集 | 冻结题数 | 报告 | 公开证据 |
| --- | ---: | --- | --- |
| FrontierScience | 160 | [两轮复测报告](vSpec-v0.15.2/FrontierScience-v0.15.2/FrontierScience两轮复测报告.md) | [证据清单](vSpec-v0.15.2/FrontierScience-v0.15.2/证据清单.md) |
| HLE-Verified | 200 | [两轮复测报告](vSpec-v0.15.2/HLE-Verified-v0.15.2/HLE-Verified两轮复测报告.md) | [证据清单](vSpec-v0.15.2/HLE-Verified-v0.15.2/证据清单.md) |

两轮中，vSpec 的输出 token/s 均高于同轮四个原生 MTP 深度；自然停止造成各臂输出 token 总数不同，因此该吞吐倍率不能当作同工作量端到端加速比。质量判分存在波动，FrontierScience 的 Olympiad 正确数两轮均低于同轮 MTP4，不能宣称质量无损。

公开候选包只含冻结题号、参数、客户端/运行脚本、脱敏吞吐与逐题号评分、独立 trace 以及文件哈希；不再分发题目、图像、请求或回答正文。本地运行环境还含尚未并入公开安装包的兼容补丁，详见各报告“复现与限制”及候选包 `README.md`。两个数据集的原始材料须由读者按上游条款自行取得。
