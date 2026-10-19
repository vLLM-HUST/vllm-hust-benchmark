# vLLM-HUST中期测试记录

主报告：[中期测试报告](%E4%B8%AD%E6%9C%9F%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A.md)。测试对象为vLLM-HUST推理引擎，涵盖数据集性能、长上下文与设备标定。当前整体验收尚未完成。

第五章列出数据集性能结果。下述合成长输入实验采用不同运行栈及缓存配置，作为独立诊断归档，不纳入第五章方案的验收判定。

2026年10月19日完成Qwen3.5-35B-A3B BF16、TP2、C8合成长输入B0补测： 每条实际输入16,384 token，60秒资格检查后测量300秒，窗口内输出79,437
token， 264.79 token/秒。312条请求全部成功，其中304条在窗口内完成、8条在排空完成。 B1也已完成：窗口内57,585 token、191.95
token/秒，230条请求全部成功。 本轮吞吐下降27.51%，平均TTFT、TPOT及资源系数均退化，如实保留。
这是不同运行栈的整体部署对照，不是单组件消融、主测数据集质量结果或900秒Frontier点。

`long-input-verification.json`是B0独立复核结果，`long-input-comparison.json`是B0/B1对照； `results.json`是既有报告复算，
二者不可拼接成一次配对实验。计时来自AsyncLLM输出事件，不是HTTP端到端计时。

`synthetic-b0-evidence.tar.gz`保留三次启动的原始日志、输入、脚本、逐请求事件、 设备采样、配置及SHA256清单，不含编译缓存。前两次失败均保留。成功运行退出码0，
自建进程已释放，析构与共享内存清理告警详见主报告。

`synthetic-b1-evidence.tar.gz`保留B1同类原始证据及组件加载检查，
`calibration-evidence.tar.gz`保留双卡BF16计算、逻辑内存复制和HCCL标定。 合成输入输出序列并不完全一致，不能宣称质量无损；标定不等于整模型有效算力。

在空目录解压归档后，可只用CPU执行复核，无需启动模型：

```bash
tar -xzf synthetic-b0-evidence.tar.gz
python verify_long_run.py midterm-run-20261019-r4
cd midterm-run-20261019-r4
sha256sum -c SHA256SUMS
```

在报告目录解压B0和B1归档后运行：

```bash
python compare_long_runs.py midterm-run-20261019-r4 midterm-b1-20261019 comparison.json
```

原始运行脚本中的绝对路径用于追溯本机环境，不应直接在其他机器执行。 主报告列出的既有来源依赖原报告仓库或保留的私有证据，未声称全部包含在本归档中。
