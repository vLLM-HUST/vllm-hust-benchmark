# MMLU-Pro Controller完整质量检查点

本轮独立TP2服务，run
request-throttling-datasets-20261018-mmlu-r3；全12032题、5shot、max2048、固定client16。9599/12032＝79.778923%，66无效答案、1536长度截断、0传输失败；完整分母保留。客户端测量23372.755431秒，不当独立短性能或调度收益。

独立私有核验12058文件hash、全部12032条assembled输出/usage/固定计划绑定、冻结源码评分重解析、score.json逐字节与summary评分及请求均值复算PASS，usage缺失0。原客户端未保存实际HTTP请求体和原始SSE，不能宣称复核未留存的传输字节或独立重建整批墙钟。新包直接物化计划内容，核验拒绝所有符号链接。

匹配Native为9632/12032，分数保留99.657392%；contract/cases/tasks/source/generation/model/client配置hash或值全部一致。独立性能尚未完成，不授予成本阶梯资格。普通单轮没有program
ID，不把透传差异归因于节流。成对报告仍待完整性能核验，网站暂不回填。

公开`python verify.py`只核验公开哈希和算术；私有`python verify.py --archive raw-and-plan.tar.gz`完成原始记录核验。私有包SHA见checkpoint.json，原始提示/答案/token-ID/地址不公开。private-verification.json为实际离线核验回执。
