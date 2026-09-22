# 双卡 MTP2 官方数据集性能复测

两臂：Qwen3.8-27B BF16，Ascend 910B2 ×2（同机6/7），MTP2，graph，32K上下文、8并发、2048调度预算、6GiB
KV/rank。固定冻结源，不是未修改的公开发行 wheel。

ABBA各两轮，算术均值，不剔除慢轮。以下均为输出tokens/s。

| 场景                     | Native MTP2 | BetterScale MTP2 |   提升 |
| ------------------------ | ----------: | ---------------: | -----: |
| instructcoder-online     |     202.425 |          362.288 | 78.97% |
| agent-research-online    |     260.200 |          317.240 | 21.92% |
| prefix-repetition-online |     129.011 |          176.132 | 36.53% |
| random-online            |     214.371 |          300.986 | 40.40% |
| sharegpt-online          |     212.159 |          361.688 | 70.48% |
| sharegpt-throughput      |     182.823 |          267.840 | 46.50% |
| sonnet-throughput        |     180.211 |          247.158 | 37.15% |

## 验证与边界

- 在线3328/3328成功，失败0；四臂逐请求输入/输出长度一致，Agent逐记录输出预算保持原样；每组保留实际MTP提案计数。
- 离线两场景共8轮全部完成，输出token数取CLI实际计数，不混用输入+输出吞吐。
- 两臂32K冷/暖、8×4097并发容量资格通过；不是8条完整32K并发或模型精度评测。
- 原生保持FULL_AND_PIECEWISE，仅使用必要的SD/V1 Mamba ABI桥；候选使用FULL、device continuation、lookahead APC及已资格通过的GDN
  layout fusion。
- random-latency未支持：APC-off在上游将mamba_cache_mode归一化为none，两份冻结适配要求align。未开启缓存伪造冷延迟，也未放松数值保护。原始失败保存在offline-abba.excluded-apc-off。
- 准入观察65秒超时的无worker尝试另存offline-abba.excluded-admission-timeout；扩展外围等待上限至210秒覆盖16次有界采样，准入标准不变。
- 原生两次在线InstructCoder/ShareGPT吞吐存在波动，均保留；不是统计显著性或普遍加速保证。
- 与无MTP历史成绩不能解释为孤立MTP因果效应：源实现、APC策略也有变化。
- 结果以独立 MTP2 实验配置提交；不是已发布 wheel 的性能承诺。

原始记录：online-abba/、offline-abba/；汇总：summary.json；源身份：source-identity.json；精确参数：各receipt.json。

原始提交位于
`submissions/betterscale-qwen27-tp2-mtp2-20260922-*`。每个目录含两轮原始结果、真实配置、数据集与运行时清单、冻结源身份和协议校验清单；在线另含两轮
MTP/APC 计数器。没有候选 Git commit 可准确代表冻结组合源，故保留空值而不借用发行版 commit。与既有无 MTP 的16条记录分离。
