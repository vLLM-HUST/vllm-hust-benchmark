# SWE-bench Pro真实Agent对照：配置审查后的结果与问题

## 结论先说清楚

本次比较了Native组和Controller组，各128个Agent，整批同时到达，不在客户端限并发；每组观察20分钟。模型及两组预算一致，Controller组额外启用已明确标注的暂停容量修正。

Controller组在窗口内的返回输出量相对Native减少0.5%；它有0个Agent连第一次模型调用都没有正常返回。提交情况和等待分布需同时核对，不能把输出量变化当作相同工作量加速。

读下面的图时，蓝色表示Native，橙色表示Controller。先看任务是否推进，再沿请求链路解释等待转移到了哪里。

## 验证状态：20分钟机制验证：本轮零回复和准入超时均为0；未证明性能改善，任务均未提交

两组各128个真实Agent、8题16副本、time0整批到达，客户端不设并发限制，均使用1200秒/200步/4096输出/262144上下文。Native每个9–24次正常回复，Controller11–23次，两组均0零回复；进展分布需以图为准。

本轮Controller计数按结束减开始：1411次aged恢复、1691次请求边界让出容量、0准入超时。此为本轮观测，不是全局公平性或所有输入的保证。

Native2298次正常回复/139724输出token；Controller2212次/139070输出token。Controller输出量减少约0.47%，正常返回调用减少约3.74%；p95为111.66秒，Native97.40秒。本轮未观察到性能改善，生成工作不同，不能作同工作量因果加速结论。

两组均128份终态、Submitted0，评分未决。Native127个总时限退出，另1个代理下游httpx.ReadTimeout导致HTTP500/InternalServerError；Controller128个总时限退出。模型与两卡保留，终态请求和缓存均排空。

严格逐事件核对：1200秒后无新模型请求或Agent工具步骤。预算后git diff和git status来自冻结runner收集结果，独立列在strict-budget-audit-private.json，未当成继续Agent步骤。

计划文字继承错误保留：300秒、a4和PENDING旧文本与typed1200秒/d082实际stage/资格回执不符，另存protocol-clarification与plan-text-audit；未修改运行或历史计划。Native精确终态现场UI漏采；Controller开始/10min/截止/终态现场与自动逐秒状态保留，组件非同时采样。

本地软件检查：两个干净3.11/3.12各206PASS5可选SKIP，标准CLI/build通过；benchmark相关28项通过，全benchmark同环境前后69失败集合相同。未声称远端LinuxCI或全benchmark通过。

## 系统链路与实验边界

Agent产生第一次模型请求 → Controller决定放行或等待 → vLLM引擎排队、处理输入、生成输出 → 请求返回 → Agent执行工具并继续下一轮。Controller等待、引擎排队、模型计算和工具执行是不同阶段。

本轮两组各128个独立Agent；每个Agent保留多轮模型与工具依赖。两组模型采用相同8192 token调度预算，async_scheduling=True，上下文262144 token；Agent上限200步、1200秒、单次输出4096 token、工具60秒。一次Native→Controller顺序比较，不足以证明稳定因果收益。

## 第一层：Controller——哪些任务在引擎外等候？

本轮有0个没有正常回复的Agent；这些任务中最短可确认的队列观测时长约0.0秒。逐任务状态与后端分配需结合原件核对，不能把代理计数当作模型调用已执行成功。

![首次调用准入与全部Agent进展分布](figures/controller-first-call.png)

左图把全部Agent按正常回复次数排序，避免平均数掩盖完全没推进的任务；右图只展示没有正常回复的Agent及其被观测到的排队时段。记录开始前的缺口没有补零，也没有当作完整逐事件追踪。

![Controller状态与准入队列](figures/controller.png)

这张图展示Controller登记的活动、暂停和等待状态。活动program数不是引擎实际执行数；逐次动作采样文件缺失的面板保持缺失，不能解释为没有暂停或恢复。

本轮Agent总时限为1200秒。准入和恢复的逐次选择须结合本轮启用的策略、等待配置和实际日志核对，不能仅凭队列快照宣称每个调度时刻的完整判断。

## 第二层：引擎和设备——队列减少，是否真的处理得更好？

![引擎队列、缓存池和输入输出处理](figures/engine-load.png)

Native的引擎内部平均排队时间为29.49秒，Controller为4.56秒。需结合Controller等待队列判读，不能把“引擎不排队”理解为“所有任务都不等”。

缓存图表示每个rank缓存池的占用，不是设备总HBM。输入计算、输出生成和缓存命中分别画在各自面板，不能互相替代。

![引擎输入处理、生成阶段与批次观测](figures/engine-latency-and-batches.png)

![引擎首token和每token耗时](figures/engine-token-latency.png)

上述阶段延迟来自各窗口内直方图sum/count增量；不是逐kernel耗时。分位桶只给区间，iteration指标也尚未被证明等同每一次真实调度步。

![两张物理NPU卡的设备观测](figures/engine-device.png)

设备图来自本轮实际两张卡的记录，不用缓存占用推测设备利用率。AI Core用实际scrape时间去重，其他指标保留查询时观测点；纵轴保留exporter原生读数，不猜测未核实的单位。这些曲线没有提供kernel或通信profile，仍不能据此宣称算力、内存带宽或通信已经饱和。

## 第三层：Agent——每个任务是否都在推进？

![全部Agent的推进过程和时间分配](figures/program.png)

Native每个Agent获得9–24次正常回复；Controller为11–23次，其中0个为0。分布差异须结合上图判断，不能仅从最小值、最大值或平均返回延迟推断所有Agent的进展。

终态原件显示，Native组提交0个，Controller组提交0个。异常分类和评分状态以终态原件为准，不能把未决评分填成0分。任务完成数不能被引擎输出量替代。

## 第四层：请求——调用者实际拿到了多少，等了多久？

![正常返回数量、输出吞吐和调用延迟](figures/request.png)

Native正常返回2298次、输出139724 token；Controller正常返回2212次、输出139070 token。返回次数、回复长度与后续生成轨迹需分别核对，不能直接解释为同工作量完成得更快。

正常返回调用的平均端到端延迟为58.30/61.57秒，p95为97.40/111.66秒（Native/Controller）。窗口末各127/128个未返回调用不在正常返回延迟分布中。

![不同阶段的平均延迟与长尾](figures/request-latency-by-interval.png)

要同时看前段、后段和完整窗口。总体均值下降不代表后段长尾没有恶化，也不代表未获得第一次返回的任务被照顾到了。

## 当前判读和下一步

判读需同时核对输出交付、引擎内部队列和全部Agent的进展，局部指标不能证明完整批次完成更快。新实验前核对首轮推进、等待上界、失败归档和报告交付。

现场截图覆盖、逐请求client→proxy→engine关联、原子选择理由及kernel profile以本轮取证清单为准；缺失项不得用推测补齐。

## 复算与来源

本文数字来自规范四层报告data.json、normal-client-returns.csv、逐Agent终态、零回复审计和设备适配器；生成脚本不启动任何推理或配置变更。来源哈希、图文件哈希和生成入口记录在manifest.json。

公开数据见data，生成器见reproduction；原题、答案、工具输出及补丁不公开。远端CI未查，不声称通过。
