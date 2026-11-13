# BidKV 冻结五数据集补跑与受控证据索引

**当前按MOD专用提示词执行。**
已发现旧配置覆盖了上游CPU绑定默认值，新profile恢复默认true并重新取得资格；旧记录保留限定配置，残批不补齐。见[MOD身份/默认优化/中止记录审计](../pegaflow/mod-evaluation-audit.md)。

这是用户新增指定`five-datasets-20261113-v1`后的独立实验。Issue664对历史固定成果免于重测的归档范围保留；历史EvoScientist/抢占压力成果不能替代这次五项。原历史汇总可见[固定公共记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a3d96977005f7257502ded2938a051c70de60b02/website-publication/data/dataset_validation_qwen35_bidkv.json)。

两张910B2，在重启后的81物理6/7、逻辑0/1，使用官方0.23隔离实验载体[BidKV PR28](https://github.com/vLLM-HUST/vllm-hust-bidkv/pull/28)，非已发布的通用兼容路径。两臂相同prepared
scheduler与模型预算，仅启用/关闭策略。新配置完整五项计划与执行源见[默认True驱动v4](driver-mod-v4/README.md)，旧False源见[历史驱动](../pegaflow/finite-driver-v1/README.md)，每臂672
solver、另共同Native judge720计划输入；模型与标准身份见计划。

[旧False配置双臂C8数值原件与受控索引](c8-results-v1/README.md)已通过685条原始SSE的独立复算：Native342、Candidate343，四phase零失败，两臂均零策略调用/决策，不能据此宣称BidKV机制收益。此阶段为单独合成C8资格，不能作为新默认True配置资格、完整五项或32K稳定性通过。

旧False完整流程已中止，Native发出200条、44成功、156未完成，后续两臂与judge未执行；见[逐项计数与中止原因更正](aborted-cohort01/README.md)。新默认True配置尚未完成C8或正式五项；共同稀疏卡绑定修复的首个运行包在0生成预检漏复制CANN算子资源，[失败原件与真实退出状态](native04-failure-v1/README.md)保留。资源复制修复使用原依赖与二进制，随后[Native05实际健康、128个token ID与CPU绑定](native05-admission-v1/README.md)已通过；该0生成准入不等于正式成绩。

问题、答案、对话和完整SSE受控保存；公共只提交配置、数值记录、校验和及受控位置索引。所有失败及尚未执行部分保留。
