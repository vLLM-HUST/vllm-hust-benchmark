# KV-tiering：完整 SWE 历史回放补证

**状态：待公共原件归档的草稿。** 这是对已完成历史实验的离线复核与补充， 没有重跑实验。全部原件在本地存在并已校验；Release/tag 尚未获准发布，
`raw-archive-manifest.json` 的下载地址暂为空。原件可公开访问前，本 PR 保持 Draft， 不视为已完成成果验收。见
[归档状态](RAW_PUBLICATION_STATUS.md)。

对应 [MOD PR #11](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/pull/11)、
[分项 Issue #7](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/issues/7#issuecomment-6846017531) 和
[补证总表 #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)。 这是 separately labelled
eager diagnostic，不进入官方榜单、网站成绩快照或任务质量 验收。以本完整实验更新成果口径，旧实验保留历史身份，不追认成最新版本结果。

## 配置与源码

- Qwen3.5-35B-A3B，BF16，2×Ascend 910B2 64GiB，TP2。
- eager、APC、async scheduling；无 MTP。双方自动 HBM 分配0.92，无固定KV字节 限额、无block
  override，运行记录的实际KV容量约21.02GiB/卡。
- context262144、max-num-seqs128、max-num-batched-tokens4096。
- 分层CPU总预算14GiB，`secondary_tiers=[]`，是HBM↔CPU两层实验。
- HBM压力门限0.8/0.6，CPU门限0.9/0.7；工作集窗口600秒/64000条；每请求新增 写入预算64MiB；idle TTL和按年龄降级均关闭。
- core源码 `3bf4ba0b0ee547577b4e6bb772ccd45aa5793670`，Ascend源码
  `b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9`。
- 原插件是 `1b6c2b43fd0a0f8d7b01bf5cbe5a7695eceb7fb5` 加未提交revision-v2，
  不能仅checkout该提交替代完整快照。独立核对测量开始/结束的18个Python/JSON 运行时payload哈希，与MOD PR #11原HEAD
  `09a52b6082b5e1a689416e91552073fef54f82e1`及本轮清理工作区一致。 清理仅移除结果与整理文档/复现入口，没有改变插件实现。
- 环境与模型身份见 `provenance/`。权重只记录分片大小/mtime，没有全量字节哈希。 原环境保留的distribution字符串不同于实际源码版本，以真实导入路径和源码身份
  为准，不能通过改metadata伪装同版本。

## 负载与计时

项目所列SWE-bench Pro公开 `claude-sonnet-4-paper` 轨迹；冻结协议中的32任务， 加其余已发布任务按预定哈希排序取96任务。选样早于测量，不按结果换任务。
保留完整合法历史，不截断、填充、加盐或人为对齐；模板关闭thinking并使用冻结 工具定义。按轮次屏障回放，下一轮输入仍用原历史，不执行工具。

三个完整臂均128任务、2215请求、26213057输入token、70880输出token；每请求
32token、temperature0、ignore_eos，客户端最大并发128，单请求超时1800秒。 Native→正常压力分层→提前写入分层，每臂独立新引擎，单次探索，不是重复确认。

保留所有冷请求和收尾。阶段吞吐为全部输出token/完整轮次墙钟之和，不含引擎启动 与轮次间指标采样；另列包含客户端启动及指标等待的全过程吞吐。TTFT含排队，
TPOT按首个非空流块至请求结束除以输出token数减一计算，含客户端/网络尾部。 这不是纯设备逐token算子耗时，也没有SLO合格吞吐判定。

## 完整结果

| 指标                             |     Native | 正常压力分层 | 提前写入分层 |
| -------------------------------- | ---------: | -----------: | -----------: |
| 完整请求                         |       2215 |         2215 |         2215 |
| 输出吞吐 token/s                 |  13.489935 |    15.324565 |    14.745492 |
| 包含指标等待的全过程输出 token/s |  13.419765 |    15.233499 |    14.660995 |
| 平均TTFT 秒                      | 112.104045 |   102.989586 |   101.551543 |
| 平均TPOT 毫秒                    | 425.596700 |   409.595398 |   438.428249 |

正常压力分层的阶段输出吞吐提升13.60%，平均TTFT降低8.13%，平均TPOT降低3.76%；
全过程输出吞吐提升约13.52%。实际计算prefill从24824513降至22954689，少计算
1869824token。D2H逻辑payload9866444800bytes，H2D为64822312960bytes。 三个完整臂无无效请求、输入usage不符或记录到的抢占。

提前写入增加恢复但未超过正常门限吞吐，TPOT更差；因此不能把“恢复更多”直接 等同于更快。详细指标/分位数/压力采样保留在原始analysis与附件中。

## 首轮与全流程成本

该SWE实验不是“每卡4GiB、8192前缀、三轮合成”的旧实验，两者不可替代。 本组按全部历史轮次计分，首次冷填充未剔除。可在附件中逐轮查看 `swe-turn-000-summary.json`
等以及D2H/H2D的before/after；不只挑重访窗口。 归档中同时保留每臂整体client-wall与启动/退出回执。

首次历史轮次（每臂128请求、输出4096token）的原始记录为：

| 模式         | 轮次wall秒 | 输出tok/s | 平均TTFT秒 | 平均TPOT毫秒 |  D2H bytes | H2D bytes |
| ------------ | ---------: | --------: | ---------: | -----------: | ---------: | --------: |
| Native       |    31.4051 |  130.4246 |    12.2311 |     326.8127 |          0 |         0 |
| 正常压力分层 |    31.1291 |  131.5810 |    12.2414 |     321.9087 |          0 |         0 |
| 提前写入分层 |    32.8512 |  124.6834 |    13.0360 |     341.8436 | 2551971840 |         0 |

首轮没有恢复收益，正常门限尚未产生写入；提前写入已经付出写入代价。 这些是完整历史回放的首轮，不是旧合成实验首轮，不能混用。

## 零/负结果、失败和质量边界

- 32任务完整三次独立配对的输出吞吐变化为−4.55%、−1.39%、+0.64%，没有 外部恢复或prefill计算节省，不能称稳定加速。原始摘要在analysis中，完整请求与
  源码快照在待发布的swe32附件中。
- CPU128GiB容量探测在共享内存preflight阶段失败，未启动模型，没有性能结果。 原始pressure附件保留此失败附录。
- 本次排除按用户要求缩短的半程复测，以及服务器到期中断的新全量任务；不把它们 作为第二次完整实验。不删除外部原历史记录，但它们不参与本报告结论。
- 没有Agent任务解决率、人工语义正确性或官方fresh-salt/零复用评分；固定输出 长度与HTTP成功不能证明通用答案质量。本报告不宣称正式B0/B1任务质量验收。
- 没有文件层启用，FS读取不是本次加速来源，H2D不等于磁盘回读。
- 插件支持三层机制，但本结果只支持该两层前缀复用配置，未证明普遍加速或独立 重复的完整128任务收益，也未比较旧PR实现与新策略的独立贡献。

## 原件、复核与复现

原归档名称、大小与SHA256见 `raw-archive-manifest.json`。文件字节保持不变， 未重新打包、编辑日志或重算后覆盖旧指标。大附件按仓库Release方式发布，
下载地址只有在实际发布并核验后才填写。

`audit_existing.py`独立重读全部三个臂的输入、SSE、usage、DONE、输出hash、 计时公式和原始Prometheus计数；6645条响应通过，三臂输入逐请求相同，18个插件
payload哈希一致。复核输出为 `analysis/offline-re-audit.json`，不是新硬件测试。

下载并核对原件后，可在仓库外解压pressure附件并执行：

```bash
python reports/tech-outcomes-20261117/kv-tiering/audit_existing.py \
  --extracted-root /path/to/extracted-pressure-archive \
  --plugin /path/to/matched-plugin-checkout \
  --output /tmp/kv-tiering-offline-audit.json
```

原始脚本保留原服务器绝对路径，见reproduction目录和附件。参数化复现入口、 完整服务命令及数据来源见MOD PR #11的 `docs/reproduction.zh-CN.md`；该入口是
维护整理，不能冒充它已经跑出本报告结果。

## 可引用结论（公共原件归档后使用）

> 实现HBM、主存及文件系统之间的KV分层缓存与异步回读，并增加工作集准入和 生命周期保护。在Qwen3.5-35B-A3B、双910B2、128任务池的完整SWE历史回放中，
> HBM—主存两层缓存的单次配对输出吞吐提升约13.6%，平均TTFT降低约8.1%、 平均TPOT降低约3.8%。结果仅适用于该前缀复用配置，不代表文件层加速、正式 Agent任务成绩或普遍收益。

历史分析、provenance与reproduction的原始副本不做自动格式化；其逐文件哈希见
`preserved-members-sha256.json`。新增审计程序和报告照常接受源码及格式检查。
