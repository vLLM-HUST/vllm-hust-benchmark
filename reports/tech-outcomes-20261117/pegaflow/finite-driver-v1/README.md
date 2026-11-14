# BidKV 冻结五数据集完整执行源

本源对应旧CPU绑定false自定义配置。其首个完整cohort在Native
MMLU200已发/44成功时被主agent过早停止，156未完成，全部原件保留；不能算完整五项结果。新MOD要求保留默认绑定true，因此另立新profile和资格，不补齐本残批。

本目录是项目自行编写的服务控制、SSE 传输和统计代码，不含私有 Engine 源码或原始题目。`.py.source` 去掉 `.source` 后使用；执行还需获得原冻结私有工具和数据的授权。

完整执行命令为
`python -B campaign81.py --out /data/shared_upload/cybber695/bidkv-five-datasets-20261113-01/results/five-datasets-01`。首次正式生成前冻结
[execution-plan.json](execution-plan.json) 和 [source-manifest.json](source-manifest.json)。两臂各 672
solver 请求；HLE/Frontier 两臂共计划 720 个共同 judge 输入。原 MMLU 提取器、HLE/Frontier
提示词及评分器保持原字节，运行时路径和实际传输适配器身份在派生 contract 中另记。

顺序在观察成绩前固定为 Native 全五项、BidKV 全五项，再使用全新 Native 的同一固定 judge
为两臂评分。各臂保持相同原样本及顺序；无重试、填充、换题、按成绩过滤或改变预算。此顺序减少重复加载，但未随机交错，时间漂移和缺少独立重复的限制保留。所有五项正式请求保留 EOS，与单独固定长度合成
C8 资格分开。

复用本机 C8 前，独立重新检查两臂四个 phase 的全部原始 SSE、时间戳、固定请求、usage、结束原因、唯一盐及零 prefix-hit 计数。C8 资格/CPU
检查不是正式五项结果；32K 稳定性、单芯片有效算力与成本目标也不能自动宣布通过。

默认服务与刚取得 C8 的配置完全相同：双卡 910B2、TP2、BF16、KV16GiB/卡、context65536、maxseq16、batch4096、保留 HLE 图像、MTP2 与
FULL_AND_PIECEWISE。两臂相同官方 0.23 隔离 scheduler；Native 关闭策略，BidKV
启用当前策略，策略错误或回退使该实验无效。若没有调度决策/调用，则不能宣称证明 BidKV 收益。

完整原始问题/答案/对话/SSE、评分和时间戳受控保存；公共交付只含可公开数值、配置、来源身份、哈希与受控位置索引。此前两次输入准备失败记录保留：初次Windows默认GBK写入失败、后续
server 准备脚本语法错误均发生在正式推理前，已在独立 attempt 修正，没有改任务或生成预算。
