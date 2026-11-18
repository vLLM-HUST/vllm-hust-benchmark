# KV-tiering：SWE 历史回放与分层压力诊断

**状态：Draft，公开数值可离线复算，完整原件受控保存。** 2026-11-18新增的
[SWE前8轮压力诊断](diagnostics/swe-pressure-8turn-20261118/README.md)包含一次新的Native/MOD设备运行；下方完整SWE章节仍是此前历史实验的整理，未重新执行全量。此前公开逐请求数值、分轮记录、压力采样和来源清单见
[public/](public/README.md)； 完整输入、轨迹及 SSE 的保管人、位置、哈希和获取方式见
[controlled-location.json](controlled-location.json)。 原件本地可用性已核验，审核者受控取件尚未独立验证；本报告不自行宣布验收通过。

对应 [MOD PR #11](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/pull/11)、
[分项 Issue #7](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/issues/7#issuecomment-6846017531) 和
[补证总表 #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)。这是单独标注的 **eager
性能诊断**，不进入官方榜单、网站成绩快照或任务质量验收。

## 2026-11-18新增：SWE前8轮CPU＋FS诊断

固定此前128任务池的前8轮、1016请求/组，双方并发128、自动HBM0.92、BF16/TP2、eager、APC、async，固定每请求32 token，保留冷轮与轮间成本。MOD CPU14
GiB、FS64 GiB；两组输入完全一致、各32512输出token，2032条请求全部成功。该诊断与原graph/EOS五项、此前两层完整历史实验分别报告。

| 指标            |    Native |       MOD |   MOD/Native−1 |
| --------------- | --------: | --------: | -------------: |
| 全流程输出tok/s |   22.4879 |   26.4517 |        +17.63% |
| 全流程耗时秒    | 1445.7536 | 1229.1099 |        −14.98% |
| 平均TTFT秒      |   86.6485 |   65.0409 |        −24.94% |
| 平均TPOT毫秒    |  415.1693 |  443.4472 | +6.81%（变慢） |

观测H2D33.47 GB、FS逻辑读取125.48
GB，实际计算prefill减少21.69%；其中既有外部恢复，也有本地缓存命中变化，不能全归因于FS。文件层后半段耗尽配额，累计capacity_rejections4206、store_errors2148、load_errors27、queue_rejections72；推理请求成功不等于后台分层无错误。收益已包含这些成本，不扣除失败时间，也不宣称无错误容量验收。

完整的正负分轮、配置偏差、失败启动、源身份、原件索引和CPU复算见[诊断报告](diagnostics/swe-pressure-8turn-20261118/README.md)。该组仅支持特定压力子集下的净收益，不证明全量SWE、正式五项、质量无损、独立重复稳定性或新版相对旧版MOD的独立贡献。

## 验收要求与证据入口

完整历史实验映射见 [evidence-map.csv](evidence-map.csv)。以下表格及后续“完整结果”属于此前全量实验；新增子集的FS回读、后台错误和限制只见上方诊断，不回填旧证据：

| 要求                                          | 当前证据                                                                                     | 结论与限制                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 128 并发分轮记录                              | [公开数值清单](public/manifest.json)、三个 `pressure-*-phases.json.gz` 和逐请求记录          | 128 任务池、客户端最大并发 128；完整 26 轮、每臂 2215 请求。不是每个时刻都有 128 个正在执行的请求 |
| 首轮开销与全流程收益                          | 下文首轮表、全部轮次墙钟和 client-wall                                                       | 保留冷请求、写入和收尾；报告阶段与全客户端两种吞吐，不挑选重访窗口                                |
| 实际回读                                      | 每轮 Prometheus before/after 中的 load/store 计数、external-prefix 命中和实际 prefill 计算数 | 证明 HBM↔主存传输；文件层关闭，**不证明文件系统回读**                                             |
| 首词元与逐词元时延取舍                        | 逐请求 TTFT/TPOT、正常压力与提前写入两候选、32 任务三组完整配对                              | 提前写入的 TPOT 变差；32 任务没有稳定收益，不只提交正结果                                         |
| 旧 16 并发合成实验的文件回读及 14.67→38.73 ms | 当前材料未建立该组原始逐请求记录的完整证据链                                                 | 撤下这组数字作为本次可引用成果；不拿新 SWE 结果冒充旧原件恢复                                     |
| 旧 128 并发合成实验的分轮和文件回读           | 本次 SWE 与其输入、HBM 容量及层级配置不同                                                    | 未覆盖；新 SWE 不能回填旧合成实验的缺项                                                           |
| 分工、版本、兼容与适用范围                    | MOD PR #11 的评测索引及文档；下文固定源码链接                                                | benchmark 保存实测证据；MOD 保存安装源码与说明。最终验收及 Issue 收口由维护者确认                 |

本报告以完整 SWE 历史实验更新当前成果口径，旧实验保留历史身份，不追认为最新版本结果。 三层机制的源码支持与本次两层实测收益分别表述。

## 配置与源码

- Qwen3.5-35B-A3B，BF16，2×Ascend 910B2 64GiB，TP2。
- eager、APC、async scheduling；无 MTP。双方自动 HBM 分配 0.92，无固定 KV 字节限额、 无 block override，运行记录的实际 KV 容量约
  21.02GiB/卡。
- context 262144、max-num-seqs 128、max-num-batched-tokens 4096。
- 分层 CPU 总预算 14GiB，`secondary_tiers=[]`，是 **HBM↔CPU 两层实验**。
- HBM 压力门限 0.8/0.6，CPU 门限 0.9/0.7；工作集窗口 600 秒/64000 条；每请求新增写入预算 64MiB； idle TTL 和按年龄降级均关闭。提前写入臂仅关闭
  `tiering_hbm_pressure_aware`，其余见冻结命令。
- core 源码 `3bf4ba0b0ee547577b4e6bb772ccd45aa5793670`，Ascend 源码
  `b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9`。
- 原插件是 `1b6c2b43fd0a0f8d7b01bf5cbe5a7695eceb7fb5` 加未提交 revision-v2，不能仅 checkout 该提交替代完整快照。测量开始/结束的
  18 个 Python/JSON 运行时 payload 哈希见 [源码等价核对](provenance/source-equivalence.json)，与 MOD PR #11 原 HEAD
  `09a52b6082b5e1a689416e91552073fef54f82e1` 一致。
  [可安装源码固定提交](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/tree/00d6a3ef2ade4ef7a105ee37458b84a5853b7d3f)
  随后清理结果文件、整理文档与测试，未改变上述运行时 payload；它不是测量时直接 checkout 的版本。
- 基线和候选的原始服务命令、导入路径、退出状态见 [provenance/](provenance/)，完整计划见
  [full128-plan.json](provenance/full128-plan.json)。权重只记录分片大小/mtime，没有全量字节哈希。 原环境保留的 distribution
  字符串不同于实际源码版本，以真实导入路径和源码身份为准。

## 负载与计时

使用项目所列 SWE-bench Pro 公开 `claude-sonnet-4-paper` 轨迹：冻结协议中的 32 任务， 加其余已发布任务按预定哈希排序取 96
任务。选样早于测量，不按结果换任务。 保留完整合法历史，不截断、填充、加盐或人为对齐；模板关闭 thinking，使用冻结工具定义。
按轮次屏障回放，下一轮输入仍用原历史，不执行工具，不把本臂生成答案接回轨迹。

三个完整臂均为 128 任务、2215 请求、26213057 输入 token、70880 输出 token；每请求 32 token、temperature
0、ignore_eos，客户端最大并发 128，单请求超时 1800 秒。 顺序为 Native→正常压力分层→提前写入分层，每臂独立新引擎。只有一次完整探索，不是重复确认。

保留所有冷请求和收尾。阶段吞吐为全部输出 token / 全部轮次墙钟之和，不含引擎启动与轮次间指标采样； 另列包含客户端启动及指标等待的全过程吞吐。TTFT 含排队；TPOT 为
`(请求完成时刻 - 首个非空流块时刻) / (输出 token 数 - 1)`，含客户端/网络尾部， 不是纯设备逐 token 算子耗时。没有 SLO 合格吞吐判定。

## 完整结果

| 指标                             |     Native | 正常压力分层 | 提前写入分层 |
| -------------------------------- | ---------: | -----------: | -----------: |
| 完整请求                         |       2215 |         2215 |         2215 |
| 输出吞吐 token/s                 |  13.489935 |    15.324565 |    14.745492 |
| 包含指标等待的全过程输出 token/s |  13.419765 |    15.233499 |    14.660995 |
| 平均 TTFT 秒                     | 112.104045 |   102.989586 |   101.551543 |
| 平均 TPOT 毫秒                   | 425.596700 |   409.595398 |   438.428249 |

正常压力分层的阶段输出吞吐提升 13.60%，平均 TTFT 降低 8.13%，平均 TPOT 降低 3.76%； 全过程输出吞吐提升约 13.52%。实际计算 prefill 从 24824513
降至 22954689，少计算 1869824 token。 D2H 逻辑 payload 为 9866444800 bytes，H2D 为 64822312960 bytes。
三个完整臂没有无效请求、输入 usage 不符或记录到的抢占。

提前写入增加恢复但未超过正常门限吞吐，TPOT 更差；不能把“恢复更多”直接等同于更快。 分轮计数由公开 before/after 重新求差，公开压力采样保留原采样时刻与错误标记。
采样峰值不是连续设备峰值，通用 `kv_cache_usage_perc` 与插件 `kv_tiering_device_cache_usage` 口径不同，不相互替代。

## 首轮与全流程成本

该 SWE 实验不是“每卡 4GiB、8192 前缀、三轮合成”的旧实验，两者不可替代。 本组按全部历史轮次计分，首次冷填充未剔除。首轮为每臂 128 请求、4096 输出 token：

| 模式         | 轮次 wall 秒 | 输出 tok/s | 平均 TTFT 秒 | 平均 TPOT 毫秒 |  D2H bytes | H2D bytes |
| ------------ | -----------: | ---------: | -----------: | -------------: | ---------: | --------: |
| Native       |      31.4051 |   130.4246 |      12.2311 |       326.8127 |          0 |         0 |
| 正常压力分层 |      31.1291 |   131.5810 |      12.2414 |       321.9087 |          0 |         0 |
| 提前写入分层 |      32.8512 |   124.6834 |      13.0360 |       341.8436 | 2551971840 |         0 |

首轮没有恢复收益，正常门限尚未产生写入；提前写入已付出写入代价。 这些数值可由公开的 `swe-turn-000` 请求与分轮记录复算，不能与旧合成实验首轮混用。

## 零/负结果、失败和质量边界

- 32 任务完整三次独立配对的阶段吞吐变化为 −4.55%、−1.39%、+0.64%，没有外部恢复或 prefill 计算节省，不能称稳定加速。六臂共 3240
  条请求均有公开数值记录，原摘要保留在 analysis 中。
- CPU 128GiB 容量探测在共享内存 preflight 阶段失败，未启动模型，没有性能结果。 原始 pressure 归档保留此失败附录，没有将它导出为成功测量。
- 本次排除按要求缩短的半程复测，以及服务器到期中断的新全量任务；它们不参与本报告结论。
- 没有 Agent 任务解决率、人工语义正确性或官方 fresh-salt/零复用评分；固定输出长度与 HTTP 成功不能证明通用答案质量，不宣称正式 B0/B1 任务质量验收。
- 文件层没有启用，FS 读取不是本次加速来源，H2D 不等于磁盘回读。
- 插件支持三层机制，但本结果只支持该两层前缀复用配置；未证明普遍加速、独立重复的完整 128 任务收益，或旧 PR 实现与新策略各自的独立贡献。固定运行顺序和单次配对也限制归因。

## 原件、复核与复现

两种复核分别提供，不混淆它们的证明范围：

1. \*\*公开数值复算：\*\*下载本目录即可，用 Python 3.10+ 标准库执行，无需模型、NPU 或完整轨迹。 校验公开文件哈希、9 臂/9885
   条请求、全部轮次、输入身份、计时公式、分轮计数和压力采样， 与保留的历史 analysis 逐项对照。导出字段和原始成员摘要见
   [public/README.md](public/README.md)。

   ```bash
   python reports/tech-outcomes-20261117/kv-tiering/public_evidence.py --check
   ```

1. \*\*受控原件复核：\*\*按照 [原件索引](controlled-location.json) 向保管人取得完整归档，先核验
   [归档清单](raw-archive-manifest.json) 的大小与 SHA256。可检查公开派生过程：

   ```bash
   python reports/tech-outcomes-20261117/kv-tiering/public_evidence.py \
     --check-originals /path/to/original-archives
   ```

   原有 `audit_existing.py` 可独立重读 128 任务三个臂的输入、SSE、usage、DONE、输出 hash、 计时公式及原始 Prometheus 计数。此前 6645
   条响应通过，三臂输入逐请求相同，18 个插件 payload 哈希一致；结果保留在
   [offline-re-audit.json](analysis/offline-re-audit.json)，不是新硬件测试。

   ```bash
   python reports/tech-outcomes-20261117/kv-tiering/audit_existing.py \
     --extracted-root /path/to/extracted-pressure-archive \
     --plugin /path/to/matched-plugin-checkout \
     --output /tmp/kv-tiering-offline-audit.json
   ```

公开派生数值不包含题面、prompt token 数组、生成文本或完整 SSE，不能独立重验其内容及任务质量。 完整归档保持原字节，不重新打包；当前没有公开大附件下载端点。

设备重现实验的参数化入口、完整服务命令和数据来源见
[固定版本复现说明](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/00d6a3ef2ade4ef7a105ee37458b84a5853b7d3f/docs/reproduction.zh-CN.md)。
本目录 reproduction 下保留测量时脚本，含原服务器绝对路径；后续整理的参数化入口不能冒充已经产生本报告结果。 硬件复跑不等于公开数值复算，本轮补证不要求重新跑设备。

## 当前可引用成果口径

> 在 Qwen3.5-35B-A3B、双 910B2、128 任务池的完整 SWE 历史回放中，HBM—主存两层缓存 的单次配对输出吞吐提升约 13.6%，平均 TTFT 降低约 8.1%，平均
> TPOT 降低约 3.8%。 冷请求、分层写入和收尾均计入测量。结果仅适用于该前缀复用配置，不代表文件层加速、 正式 Agent 任务成绩或普遍收益；32 任务三组完整配对没有稳定加速。

这更新当前填报口径，不追认旧 16 并发的 14.67→38.73 ms 数字，也不补造旧文件系统回读原件。 最终成果收口由维护者结合固定提交与受控核验决定。

历史 analysis、provenance 与 reproduction 的原始副本不做自动格式化，逐文件哈希见
[preserved-members-sha256.json](preserved-members-sha256.json)。新增报告、派生材料与复算代码接受常规检查。
