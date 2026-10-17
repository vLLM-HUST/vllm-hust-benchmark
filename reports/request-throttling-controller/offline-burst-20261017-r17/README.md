# SWE-smith-tranjactory 离线窗口实验当前结果

目标：整批96个多轮Agent program在time0同时到达，客户端不限流，验证Controller准入/暂停/恢复在KV压力下的收益。每条program保留原调用依赖和工具等待。

| 配置维度 | 值 |
|---|---|
| 模型 / 精度 / 设备 | Qwen3.5-35B-A3B / BF16 / Ascend910B2 / 单个TP2后端 |
| 输入 | 96完整输入program、2318轮；input SHA0d95fa124751067178645e561e5ec99b9442908242ded8da680c383f8bbdc256 |
| 到达与客户端限制 | 全部time0到达；无客户端slot上限 |
| 引擎 | APC开启；KV512MiB；max_model_len16384；max_num_seqs16 |
| 容量适配 / 周期 | 源码审计49152 / 每program buffer14335；0.5秒 |
| 测量 | 每侧180秒；缓存分别重置；同代理传输路径调度开/关；源码冻结 |

| run | seed | OFF tokens/s | ON tokens/s | ON/OFF | OFF/ON完成调用 | OFF/ON抢占 |
|---|---:|---:|---:|---:|---:|---:|
| r15 | 11 | 15.6576 | 27.7241 | 1.7706x | 48/72 | 246/0 |
| r16 | 11 | 15.7807 | 28.3378 | 1.7957x | 48/83 | 249/0 |
| r17 | 17 | 15.7981 | 30.2828 | 1.9169x | 49/80 | 250/0 |

当前结论：已核验轮次同方向窗口收益，OFF KV峰值100%、ON75%，OFF有抢占、ON零抢占；公共完成轮次prompt/usage一致，零请求/token/history错误。r15 OFF→ON与r16 ON→OFF是seed11顺序复测；r17为seed17 OFF→ON；r18 seed23 ON→OFF已冻结，真实状态单列。不能把seed11两个顺序当两个独立seed；三seed中位数须等seed23终态与校验后计算。

边界：输入完整不代表窗口内全部轨迹完成；窗口终止后取消的调用不计完成，不能据此声称96条完整批次makespan下降。APC/512MiB是独立实验配置，不是原矩阵配置，非任务修复准确率、非原矩阵资格；网站尚未发布。历史C32 r12/r13单列保留，不混入本结果。

归档纠正：r17首个采集metadata误指r16，错误包和public保留为INVALID，未公开。正确包重新回收且校验run/job/root/report/seed；采集工具新增3项回归，导出拒绝report/job跨轮。有效r17私有包SHA96760bd309f7f7f294ef7382b2073d5bba369f9c04519fb4884859735c298ad4。测量源码和配置未改。

每个validated-export独立包含verify.py、SHA256SUMS、输入、轮次哈希和metrics；测量源码/容量审计公开备份见benchmark reports/request-throttling-controller/offline-burst-protocol-20261017。公开数据不含内部地址、凭据和token payload。代码PR和网站交付待完成。
