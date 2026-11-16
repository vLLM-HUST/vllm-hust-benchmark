# PegaFlow 双卡冻结五数据集执行源

本目录为旧CPU绑定false的准备源，等待门闩时已被新MOD默认true配置替代；没有按此源启动新C8或正式五项（0生成）。保留源/计划作为准备记录，不作为当前MOD执行配置。

生成前固定[执行计划](execution-plan.json)和[源码清单](source-manifest.json)。本目录为项目自行编写的控制、SSE、统计适配代码；不分发私有 Engine
或原始题目/答案/对话。`.py.source` 去掉`.source`后为执行原字节。

`python -B c8_campaign91.py`先等待独立3GiB容量probe终态、两卡释放和公共发布门闩；随后按本机KV16GiB/卡、context65536、保留图像、MTP2和FULL_AND_PIECEWISE配置分别完成Native/PegaFlow两臂C8。每臂实际16个C8与112个trace
token ID均须匹配离线冻结哈希，独立逐原始SSE复算后释放设备，才自动调用`python -B campaign91.py --out <durable>/formal-01`。

正式顺序预注册为Native全五项、PegaFlow全五项，再用全新Native共同judge。每臂672 solver请求，两臂HLE/Frontier共计划720
judge输入；原评分器不变，失败/length/未决保留。所有正式五项保留EOS。C8原合成资格单独固定256
tokens、ignore_eos=true、各60秒预热/300秒提交窗口，完整排空，不替代32K稳定性或四目标验收。

Native/Candidate框架、SDK、模型、并行、服务预算相同；Candidate另加固定PR49源码的PegaFlow连接器与30GiB host
pool。不使用BidKV桥。正式请求唯一salt与内部/外部prefix0、Pega实际load success0交叉佐证零复用；该load计数来自vLLM
API，原始sidecar指标也保留。零恢复工况不能证明缓存卸载收益。

每阶段检查源码/实际输入plan哈希、core6精确依赖和原框架/插件artifact身份；NPU进程必须属于自己的server/sidecar祖先。启动期只对模型分片做file-cache
advice，计时内不做清理。公开前Linux3.12编译10源和六核心依赖校验通过，属于CPU准入，尚不代表正式结果。

此配置与此前KV20GiB原SWE负结果和独立3GiB容量probe分别封存；不存在用其中一个原件替代另一项资格或成绩。缺少同工况独立算力参考、>=32K长稳、报价适用性或多租户证据时，四目标对应项标未验证，不套用Engine历史通过结论。
