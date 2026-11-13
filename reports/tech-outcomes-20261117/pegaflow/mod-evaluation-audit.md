# MOD 提示词替换与 B0/B1 身份审计

用户已明确用 MOD
专用提示词替换此前文件。本轮`test_target=own_mod`、`selected_scope=full`；冻结五数据集、标准/指南SHA、任务顺序、生成预算、EOS、原评分器与统计口径不变。有效用户文件SHA为`43bc925a6ad749971a64e5244b86451f5d4cf3361e87423b1eda8ea0b0d04461`。参考Engine/SDK只是可选实现，不定义本轮B1，也不要求安装固定Torch2.13
Engine替代待测MOD。

## 实际 MOD 与共同基线

| 项目     | B0                                                       | B1                                                                                            | 必须披露的适配或开销                                                                                                                                                          |
| -------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BidKV    | 官方vLLM/Ascend0.23的同一隔离prepared host，关闭目标策略 | 同一host启用`bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy`与其固定配置             | 实验载体PR28源码53eed4dd65bc4ba12efa37f9f1e334fa1ae51076，非已发布通用支持；scheduler hook两臂相同，Native负责状态/回滚；策略初始化、实际调用、选择和fallback journal分别核验 |
| PegaFlow | 同一官方0.23框架、SDK、模型与资源预算，无KV连接器        | 加载固定PR49源码e2cdc2f7b32d8294b54a74545c05d103e5a5ecaf的PegaKVConnector与30GiB host sidecar | MOD为HMA安全关闭其本地prefix命中，原B0 APC不能随之关闭；真实外部load/save、额外RAM/传输和init成本单列                                                                         |

共同vLLM提交0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665，Ascend提交5cb98caaadeff42b5b62b996e34bb2aaa29d20fd，core
SDK为Torch2.10.0+cpu/torch-npu2.10.0.post4、Transformers5.5.4/tokenizers0.22.2、CANN9.1。双卡910B2/TP2、同源Qwen3.5-35B-A3B
BF16；各自机器内关闭/开启自己的MOD，不跨机器排名。离线冻结trace tokenizer仍为5.14.1，实际服务逐项token
IDs一致才允许生成，不能静默重算。共同APC、MTP2、Mamba align、异步和图编译配置保留。

## 必须修正的共同配置

现场源码与[上游固定源码](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/ascend_config.py#L162)均确认`enable_cpu_binding`默认true；旧源显式设false。实际affinity/cpuset为0–191，8CPU只是quota，不能据此声称必须关闭默认绑定。新MOD
profile两臂恢复true，先核验真实绑定/worker线程affinity和是否warning
skip，再取得新的同配置双臂C8资格，沿原冻结五项完成配对。其余资源预算和任务参数不变，不把恢复默认优化后的结果套在旧false profile上。

旧false C8
Native342/Candidate343条数值、原SWE十个900秒点和所有失败完整保留，限定为已披露自定义配置，不冒充保留默认优化的新资格。81旧false完整流程曾发出200个Native
MMLU请求，44条成功返回、156在途未完成；主agent在读完新文件前过早发了停止指令，该残批已停止并封存，不是完整标准成绩，也不是用户要求补齐/重试。不能恢复补齐这个残批；新默认配置使用独立namespace、缓存和输出目录。91旧false新五项控制器在等待阶段被替代，0模型启动/0生成；独立3GiB旧SWE压力probe因实际需3.07GiB才能容纳262144上下文而启动准入失败，未产生吞吐值，不记成0。

## 评价与交付

质量由独立答案与原评分器判断；不同token不能直接判质量下降，相同token也不证明答案正确。报告固定配对样本数、失败/截断/无法解析/未决及不确定性；不把不显著写成等价。HLE四层、Frontier两轨单列；SWE/Terminal只测推理回放，不给解决率。C8为原单独合成资格，五项正式全部保留EOS。

MOD加载、调用、触发与收益分开：旧BidKV C8实际启用但0决策/0调用，不能据小幅时间差宣称收益；新五项fresh
salt/零KV复用也可能不触发Pega外部恢复，需如实记录。初始化/编译/准备成本、主机/设备内存、额外sidecar开销与solver窗口分开，成本写明单位/费率/范围；未测或不可比写未验证，不填零。

MOD收益评价与项目3.4四目标验收分开。新提示词只在同时申请项目验收时要求相应正式基线与门槛，不把Engine历史成绩或中期/最终门槛混贴给MOD，不把90%/2x/32K/成本30%当作本轮MOD执行的隐含前置条件。

## 默认绑定的实际准入与共同兼容修复

81默认True的Native03完成健康和128个实际token ID核验，仍因上游topo namespace混用而两rank
warning/skip，0生成。现场物理6/7分别亲和CPU96–119/48–71，全部在allowed0–191内；91物理3/4对应逻辑0/1、亲和CPU0–23/96–119，也在允许集合内。这不是需要关闭绑定的真实资源限制。上游parse_topo_affinity保存物理ID，而running列表已转换逻辑ID，后续以逻辑查物理字典得到空集。

计划在独立copied
Ascend0.23载体内加窄physical→logical解析修复，原安装树与二进制/依赖保留；每个项目B0/B1共用同一修复、同默认True，仅目标MOD开关不同。公开原/新源码SHA、补丁、导入路径和identity/sparse/unknown映射测试，再核验实际线程绑定。共同兼容变化与MOD机制收益分开，不把相对于旧false配置的组合差异归因单MOD。新同配置C8和全五项尚不能在此审计页宣称完成。

91另一份True v2准备在权重receipt字段守卫发生KeyError，0模型启动/0tokenize/0生成；原版编译通过未证明真实准入。修正将按实际91
receipt的固定revision、14个分片size/hash与完成字段逐项CPU执行验证，作为新的v3；不改原始权重或补造旧成功。
