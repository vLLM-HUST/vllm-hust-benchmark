# MMLU-Pro Native完整质量检查点

本轮独立TP2服务，run
request-throttling-datasets-20261018-mmlu-r3；全12032题、5shot、max2048、固定client16。9632/12032＝80.053191%，62无效答案、1520长度截断、0传输失败；全部保留完整分母。客户端测量23828.671923秒，不当Controller收益或独立短性能。Controller成对质量/独立性能尚未完成，不回填Controller网站缺项。

私有独立核验：12032份assembled输出记录与results一致、冻结token-ID计划/源码hash、usage计账、源评分器重解析，score.json逐字节复算、summary评分统计及请求均值复算PASS，usage缺失0。cases
SHA
f77a7e5e6fd96fbbf46be7ab11f360309554105e97921805f02d1c93fc05fcf3与团队输入一致。原客户端未保存实际HTTP请求体或原始SSE字节；本检查不能证明未留存的传输内容，也不能独立重建整批墙钟。

原始包保留输入目录符号链接，发现后原包保持不变；另补固定输入的实体文件包，逐文件hash与原归档manifest/contract完全一致，不重跑评测。完整私有核验需要raw-and-plan.tar.gz和frozen-plan-materialized.tar.gz两个包（SHA见checkpoint.json与plan-supplement.json）。符号链接不提取执行，仅验证预期链接并使用安全提取的补包。

公开`python verify.py`核验公开哈希/算术，不冒充私有检查。私有两个包、提示/答案/token-ID和地址不公开；private-verification.json是实际离线核验回执。
