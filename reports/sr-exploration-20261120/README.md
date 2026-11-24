# SR 系列探索台账与后续 MOD 选择

评估日期：2026-11-20。该目录是已有证据的汇总和候选处置，不是新实验结果，不增加任何实测轮次或有效算力结论。

## 先看结论

本次核验范围内，没有另一个已经证明“相对冻结 r8 提升完整模型性能且质量合格”的可直接开启 MOD。已包含模块不重复计收益，已测无收益、退化、质量或兼容失败的原实现不再推荐重测。

| 顺序 | 仍值得推进的工作                           | 推进条件                                                                                                    | 已有结果与边界                                                                                      |
| ---- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1    | 继续 restore＋门控 RMS 融合0.2.1的既有接入 | 复用C12/C14组件验证与C16/C17源码，补实际服务形状、NPU数值、非连续Z、图角色/流及一次归一化资格，再比较同一r8 | 属于正收益组件的后续资格工作；未证明r8吞吐、纯decode或任务质量收益。旧0.1.0大形状负例不抹掉。       |
| 2    | BidKV 的当前瓶颈检查                       | 先从当前r8证明KV压力、驱逐/重算及可改变的有效抢占机会；没有机会即停止接入筛选                               | 旧TP4/512MiB KV单元的13.83%不能赋给TP2/20GiB r8。现有RTC记录的preemption为0也不等同所有状态无淘汰。 |
| 3    | OpSet 的实际调用路径检查                   | 先确认Qwen3.5 MoE底座命中受支持Linear–SwiGLU路径、shape和ABI；没有命中即移除                                | 旧dense Qwen2.5-7B Graph正例不是r8收益。该项先做源码/调用资格，不先跑大模型加速试验。               |

这不是三项“开关打开即可加速”的清单。后两项只有前置检查通过才有理由安排新的对照。vSpec不再以整个项目名称进入首选；已测局部argmax无收益，动态专题的质量/后端差异单独记录。PegaFlow、MuxPD、KV量化等保留其历史正例和代价，当前不排入直接叠加优先项。

## 基线与身份

- 组合基线：冻结 `full-answer-b1-20261117-r8`，Qwen3.5-35B-A3B、BF16、双Ascend 910B2、TP2/PP1。
- 配方：E36/R54、MTP2、FULL、每rank
  State4GiB及KV20GiB。完整包含清单来自[固定源码](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/mods/active-stack-r8.json)。
- 源码主入口默认profile仍为r6，网站当前选择r8；二者不混写。后续以新冻结/已部署身份更新本台账，不凭版本称呼推定实际worker。
- SR1是DiffSpec，已发布SR2是PyramidKV。一个未合并Layered Prefill原型分支标题也使用过sr2；本表保留其独立原型身份，不改写已发布编号。
- C/E/K/M旧实验保留原编号和基线，只建立交叉索引；没有给旧运行补造新的SR版本。
- 本次覆盖固定canonical目录的69个分类条目和下表31条机制/阶段记录。它不声称收齐未提交或未归档的私有运行。

## 已探索机制与阶段

`exclude_unchanged`等排除针对原实现、原配置及其已测范围，不是声称一个研究方向永久无效。启动前拒绝、CPU准备、组件通过、完整模型对照、插桩诊断分别记录，未知指标不填0。

| 记录                               | 机制                               | 阶段／结果                               | 已有结论                                                                                                                                 | 处置             | 证据                                                                                                                                                                                             |
| ---------------------------------- | ---------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| sr1-diffspec                       | DiffSpec 与 r8 组合                | 整模型／退化                             | 同卡各一次配对，MMLU200 吞吐1499.31→131.86 token/s，4K/16K吞吐和尾时延均退化；未替换r8。                                                 | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/23d9903d635ec86d980c9d211e9050b2ef724d8e/reports/b1-sr1-20261118/README.md)                                                       |
| sr2-pyramidkv                      | PyramidKV 与 r8 组合               | 整模型／无合格增益                       | v3单次同卡16K为159.69→160.80 token/s，P99变长；v2 MMLU150→144/200；v4因主机4GiB余量保护中止。不能作为稳定提速候选。                      | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/23d9903d635ec86d980c9d211e9050b2ef724d8e/reports/b1-sr2-20261119/README.md)                                                       |
| vspec-local-argmax                 | vSpec 局部 argmax                  | 整模型／无收益                           | 三对独立服务ON/OFF吞吐几何平均变化−0.009906%；没有超过当时B1；未重新评分。该记录仅覆盖局部argmax，不冒充整个vSpec动态控制。              | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/23d9903d635ec86d980c9d211e9050b2ef724d8e/reports/vllm-hust-midterm-20261019/vspec-school028/README.md)                            |
| vspec-adaptive-topic               | vSpec 动态投机专题                 | 其他配方整模型／质量未合格               | Frontier/HLE两轮输出token/s高于各固定深度，存在逐臂后端差异、自然输出量差异；Frontier Olympiad低于同轮MTP4，不能称质量无损或r8叠加收益。 | 暂停泛化推荐     | [原记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/23d9903d635ec86d980c9d211e9050b2ef724d8e/reports/tech-outcomes-20261117/vspec/README.md)                                          |
| E02-capacity32                     | 执行容量24→32                      | 整模型／存在取舍，未晋级                 | 两对吞吐+20.98%/+18.67%，平均TPOT+9.22%/+8.07%；未过原筛选。                                                                             | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E02-RESULTS-v1.md)                                        |
| E03-mtp-depth                      | 上下文并行开关与MTP3               | 启动／准入／当前契约不支持               | 前两臂MTP2/CP1和CP0完成，CP关闭吞吐仅+0.675%未过筛选；MTP3在正式请求前被两步元数据契约拒绝，旧自动重复被中止，剩余两臂撤销。             | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E03-mtp-v3/README.md)                                     |
| E04-capacity26                     | 执行容量24→26                      | 整模型／存在取舍，未晋级                 | 两对吞吐+9.83%/+7.73%，正确148→138、149→144；性能与质量分开判定，未晋级。                                                                | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E04-RESULTS-v1.md)                                        |
| E05-state-policy                   | 关闭State维护及partial reclaim     | 整模型／退化                             | 单次开关配对吞吐−2.80%，并非已证明减少维护I/O就可加速。                                                                                  | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E05-RESULTS-v1.md)                                        |
| E06-native-multistream             | 原生共享专家多流                   | 整模型／未过原筛选                       | 吞吐+1.70%，P95 TTFT+5.18%；未过原5%筛选，不能列作现成稳定增益。                                                                         | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E06-RESULTS-v1.md)                                        |
| E08-graph-padding                  | 缩小decode图桶80→72                | 整模型／未过原筛选                       | 吞吐+2.09%，未过原5%门槛；输出量和评分变化保留，不把TTFT变化归为单算子贡献。                                                             | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E08-RESULTS-v1.md)                                        |
| E09-E11-layout-failures            | 全局NZ及标量兼容启动失败           | 启动／组件／启动失败                     | 保留原NZ启动/图内Cast不接纳及后续诊断；E11正式请求为0，不能写成性能为0或已完成模型对照。                                                 | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E11-RESULTS-v1.md)                                        |
| E13-nz-cache                       | 专家NZ＋路由缓存                   | 整模型／未过原筛选                       | 吞吐+0.463%，未过原筛选；不是新鲜重复配对，质量/稳定性未合格。                                                                           | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E13-RESULTS-v2.md)                                        |
| E14-router-cache                   | ND路由预转换缓存                   | 整模型／未过原筛选                       | 吞吐+0.146%，未过原筛选。                                                                                                                | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E14-RESULTS-v1.md)                                        |
| E15-layout-no-cache                | 专家NZ／门控ND／关闭显式缓存       | 整模型／退化                             | 吞吐−0.683%；四格布局/缓存结果全部保留，不选其中正例拼接收益。                                                                           | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E15-RESULTS-v1.md)                                        |
| E16-gmm-layout                     | GMM布局组件诊断                    | 组件／组件正负例并存                     | 7个case组件比值1.0238–1.0961，rank0一case0.9070；不是全部通过或模型加速。                                                                | 仅保留组件线索   | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/README.md)                                                |
| E17-selective-layout               | 专家NZ／普通Linear保留ND           | 整模型／未过原筛选                       | 相对原ND吞吐+0.895%，未过原筛选；保留与E15的次要比较，不换较弱主参考。                                                                   | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E17-RESULTS-v1.md)                                        |
| E18-E20-joint-layout               | 容量、布局和多流联合探索           | 整模型／组合取舍未验收                   | 存在约5%量级局部配方正收益，同时保留低收益、质量/截断与代价；不是独立组件因果或r8新增收益。原配方不作为新MOD再测。                       | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/E18-RESULTS-v1.md)                                        |
| C02-journal-spool                  | 日志落盘方式                       | 整模型／无收益                           | 吞吐变化−0.070686%；不推荐原日志落盘改法作为加速项。                                                                                     | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/C02-journal-spool-20261115-r3/analysis-v1.json)           |
| C03-C04-capacity36                 | E36/R54容量组合                    | 整模型／r8已包含                         | 完整回答两对吞吐+25.07%/+29.10%，TPOT+14.17%/+12.87%；已进入r8组合，原收益/取舍不重复计为新增模块。                                      | 已纳入r8         | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/README.md)                                                |
| C05-resident36                     | 固定E36、驻留54→36提案             | 源码准备／未实测                         | 已冻结原四臂方案和静态检查；查到的记录未启动模型，不把它写成实测失败或新收益。                                                           | 延期未测提案     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/C05-resident-budget-20261116-r1/README.md)                |
| C06-execution48                    | 48执行席位源码准入                 | 源码准备／未实测                         | 只有主机/源码检查，未启动NPU；不是已验证可用容量或性能。                                                                                 | 延期未测提案     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/C06-execution48-audit-20261116-r1/README.md)              |
| C07-mc2                            | prefill矩阵乘＋AllReduce融合       | 组件／未过原筛选                         | r1原生控制图失败保留；r2数值通过，但8个正反计时对中只有3对同时过wall/device门槛，整体未通过。                                            | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/C07-prefill-MC2-component-20261116-r2/analysis-v1.json)   |
| K01-K02-restore-tile               | 状态恢复行块R16→R32                | 整模型／无稳定收益                       | K01组件正例后，K02两对整模型吞吐+1.999%/−2.314%，未过双向筛选；保留R16。                                                                 | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/K01-restore-row-tile-20261116-r2/RESULTS-v1.md)           |
| K03-K04-async                      | 受保护进度复制异步化及生命周期修复 | 整模型／r8已包含                         | 两对完整模型吞吐+14.16%/+12.04%，质量取舍保留；原生异步和初始化修复已纳入r8，不再重复作为新增叠加。                                      | 已纳入r8         | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/README.md)                                                |
| K05-cache-relocation               | Triton编译缓存索引迁移             | 主机工具／非稳态加速                     | 修复部署索引、原payload保持SHA；这是恢复工具工作，不列稳态吞吐加速MOD。                                                                  | 非性能MOD        | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/K05-triton-cache-relocation-20261116-r1/RESULTS-v1.md)    |
| K06-gdn-tile                       | GDN value tile 16／64／128         | 组件／未过原筛选                         | BV16/64未通过双卡整体性能筛选；BV128因UB超限编译失败，余项按协议未执行。                                                                 | 原实现不重测     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/K06-GDN-value-tile-20261116-r2/analysis-v1.json)          |
| C10-C15-r8-combination             | 容量＋异步＋短prefill尾段联合组合  | 整模型／r8已包含，保留取舍               | C10/C15完整结果和失败都保留；r8有真实吞吐收益，同时原反向4K TTFT及反向MMLU TPOT筛选未全通过。已包含项不重计新增收益。                    | 已纳入r8         | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/mods/active-stack-r8.json)                                                                  |
| restore-rms-v010                   | 旧版状态恢复＋门控归一化融合       | 组件／未过原筛选                         | 小mixed形状正收益，大prefill负例；原完整48对门槛失败。旧版本保持排除。                                                                   | 排除旧版         | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/mods/README.md)                                                                             |
| restore-rms-v021                   | 状态恢复＋门控归一化融合0.2.1      | 组件通过／源码接入／组件通过，整模型待验 | packed32/64各32数值配置/48性能对通过；C16/C17与M15完成源码/CPU准备。尚无该实际服务适配器的整模型收益。                                   | 继续既有资格验证 | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/mods/restore-gated-rms-fusion/README.md)                                                    |
| M16-M17-tail-diagnostics           | 短尾预算及HTTP时间线诊断           | 诊断／非性能验收计时                     | 固定状态预算复算不能证明闭环收益；M17插桩和原缺失臂接续用于定位4K退化，不是新干净性能验收。                                              | 继续既有诊断     | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/blob/44fefd07b014c85a14bf425092c5fe103ba471a5/experiments/optimization-20261115/M16-fixed-state-tail-budget-replay-20261117-r1/README.md) |
| layered-prefill-unmerged-prototype | Layered Prefill未合并原型          | 源码准备／未合并原型，未实测             | 分支标题曾使用B1-sr2，但已发布SR2实际是PyramidKV。此原型只有CPU parity，Ascend调度器接入和设备资格待完成，没有完整模型性能记录。         | 延期未合并原型   | [原记录](https://github.com/vLLM-HUST/vllm-hust-engine/pull/37)                                                                                                                                  |

## 完整 MOD 目录处置

[逐项目录表](CATALOG.md)覆盖上述固定目录全部条目，保留已测结果、组件/功能证据和未完成资格之间的区别。[机器台账](exploration-ledger-v1.json)提供稳定ID、别名、原基线、处置和重新接纳条件；[源文件快照](sources-v1.json)记录固定提交及Git
blob身份。

同一机制可能同时有旧负例和新修订正例。例如restore/RMS旧版完整组件筛选失败，新连续布局与角色不变版本通过；只继续新版本尚未完成的实际服务验证，不重复已合格的组件池。

## 再推荐或再实验前必须核对

1. 先用MOD名、别名、SR编号及具体机制查本台账，并回读所引原件/原始分析；不能只看插件网页摘要。
1. 已包含项不算新增叠加；历史Native正例、不同模型/设备/拓扑、组件计时或功能通过不自动成为r8增量。
1. 原实现无收益、退化、质量/兼容失败或控制动作未触发的，保持排除。重新接纳必须指出实质实现变化或新增真实瓶颈证据，并保留原负结果。
1. 对当前固定基线写清资源预算、worker/状态所有权、MTP/FULL/缓存冲突及质量边界，再前瞻性确定对照与停止条件。不得为重复测试缩预算、换弱参考、挑最好轮次或覆盖旧原件。
1. 新结果追加到原实验族/新版本，更新本台账和来源索引；有资格的新冻结组合才进入发布版本。该排序本身不证明性能验收，也不启动或授权新的硬件运行。

## 核验与范围

本次核对已发布SR标量、vSpec三对局部选择证据、Engine主分支机制记录/包含清单以及benchmark固定目录，并核对引用文件在源快照中的Git
blob身份。没有重新运行推理，也没有把“读到报告”写成重新下载全部私有raw并完成独立复算。

benchmark原始流、冻结分析及私有Engine Release继续保留原址；本目录只追加汇总，不覆盖SR1/SR2数据或历史实验文件。网页数据和引擎运行时没有因本次文档整理改变。

## B1-sr4 共享专家融合

[SR4 场景实验与五套配对](../b1-sr4-20261120/README.md)记录同卡复现的 4K 端到端尾延迟改善、16K 吞吐改善以及 16K 首字 P95
和任务准确率回退。作为场景实验冻结，当前实现按有回退处置。

## R9 书生集成后续族

[R9原生模型验证](../r9-intern-native-20261120/README.md)追加书生35B的MMLU200及C8连续4K/16K结果、公开R8参考、完整取舍与可恢复原件。继承的10项R8
MOD不重复计增量，不补造SR编号，也不把跨模型差异当成独立MOD因果收益；先前失败和原筛选记录保留。

## R10 插件与状态路径复核

2026-11-23。冻结 R10 以 Qwen3.5-35B-A3B BF16、Ascend 910B2 TP2、E36/R54、FULL 和 MTP2
为本次对照身份。各复核保留自己的负载及统计分母。[R10 插件复核数据](../../website-publication/data/r10-mod-followup-20261123/summary.json)

| 机制                                  | 证据范围                             | 结果与处置                                                                                                                                                    |
| ------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PegaFlow KV 传输                      | 历史整模对照、离线前缀扫描、CPU 夹具 | 历史 C1 −28.43%、C8 +24.16%。固定 Terminal/SWE 输入的整块前缀上限为 20.35%／1.61%。直接叠加暂缓，R10 还需验证 GDN/MTP 状态恢复。                              |
| Graph Coverage                        | R10 图分发实机诊断和 CPU 重放        | 并发 6／8／10 的 24 请求、78 次 FULL 分发均为零 padding。当前窗口不增补图规格。                                                                               |
| BF16 直接贪心采样                     | R10 整模筛选和 v2 组件修订           | 原实现新增 190 题吞吐 −0.45%、答对 141→139。v2 不同场次同卡十题观察到吞吐 +19.67%，最长请求输出从 8,192 token 截断变为 6,687 token 正常结束，归因待配对验证。 |
| vSpec 减步                            | R10 原生 MTP2 逐请求离线诊断         | 190 题共 254,360 轮，第二位置接受率 77.32%。以最近 32 轮低于 40% 为条件仅触发 632 轮，其中 379 轮仍接受第二位置。保留固定 MTP2，不重复跑这一减步规则。        |
| BetterScale 已接受 token 状态写回融合 | R10 图外状态准备组件                 | 8／10 活跃行同步完成 162.89／158.30 → 69.22／67.50 μs，144 组 CPU 一致性与 7 种 NPU 形状通过。保留组件候选，整模收益及质量待验证。                            |
| OPset persistent linear               | R10 启动与源码核对                   | 两 rank 未启用该路径必需的 batch-invariant，暂不进入性能筛选。                                                                                                |

## R10 两项独立接入筛选

2026-11-24 · 两项方案分别对照冻结 R10。两臂各启用同一对 Ascend
910B2。[数据及配置](../../website-publication/data/r10-unintegrated-mods-20261124/summary.json)

- 短请求预填充准入。R10 机制移植，4,096-token 总预算，长请求单步上限 3,072。合成混合批次的 C4／C8 短请求首字 P50 +26.32%／-0.32%，整批耗时中位
  +11.31%／+20.31%。输出 token ID一致 65／65。每规格五批，同卡各一次服务。该档位未通过性能筛选。
- BetterScale 状态写回融合。R10 整模十题筛选（并发 10，双 rank 活动）：输出吞吐 -6.39%，P95 首字 -0.09%，P95 每 token 耗时 +3.22%，答对
  10→6。同卡各一场，未通过集成筛选。 2026-11-24 · 冻结 R10 MMLU-Pro 前 10 题，同一对 910B2，输出 token 总量 28592→25888，截断
  1→0，从对转错 4 题。复算 20 段原始流，生成文本完全一致 0／10。新内核在封存前由双 rank 预编译。相同 R10 启动和题单的两场原生对照分别答对 7 和 10 题。
  输出量变化会影响整批吞吐的归因。
- StateAxis 流水化权重加载。R10 准入复核中，当前扩展清单限定 StateAxis 宿主和 Qwen2，冻结 R10 使用 Qwen3.5 MoE，未运行候选。

## 已接受 Token 状态写回融合

上文的 BetterScale 状态写回融合现以独立实验 MOD 登记。该试验依赖冻结 R10
的宿主状态路径。代码包默认关闭，沿用已入库的组件结果和整模未通过结论。[独立仓库](https://github.com/vLLM-HUST/vllm-hust-accepted-state-fusion)
· [公开证据](../../website-publication/data/accepted-state-fusion-20261124/summary.json)

## BF16 直接贪心采样 v2

2026-11-24 · [同卡冻结 R10 对照与逐请求数据](../greedy-bf16-v2-20261124/README.md)。 MMLU-Pro 200 题吞吐
+1.88%、平均首字时延 +5.83%、答对 149→147。4K／16K 闭环并发 8 吞吐 +0.54%／ +0.01%。保持实验身份，暂缓集成 R11。
