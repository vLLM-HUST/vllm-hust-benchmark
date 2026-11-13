# 新五项标准的适用性核对

本页早期准备状态已被用户新MOD提示词修订；[当前MOD绑定与默认配置审计](mod-evaluation-audit.md)优先适用。旧CPU绑定false的C8不作为默认true的新资格，81残批中止和91旧准备停止均另记。

用户在 2026-11-13 另行指定 `five-datasets-20261113-v1`，并要求不符合时直接安排新实验。 本页区分历史证据的原有适用范围、新标准的输入准备与实际模型执行；不把
CPU 准备或旧吞吐记为新测试成绩。

## 冻结来源

- [标准](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a3d96977005f7257502ded2938a051c70de60b02/website-publication/data/dataset-standard-v1/standard.json)：SHA-256
  `66c050b7c6d705853f10bc437e6ace1707a9b46d43b6bdb7159724045045b756`。
- [运行指南](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a3d96977005f7257502ded2938a051c70de60b02/website-publication/data/dataset-run-guide-v1.json)：SHA-256
  `5db88d61ceaa2e831ab808f77f58f8241aa279e68c66ac4bd1de94c9f3d22c08`。
- [参考执行器](https://github.com/vLLM-HUST/vllm-hust-engine/blob/448362c7565597c8963c970dcfea0b235fd5fcf7/one_click.py)
  与其 README 已按固定提交读取；私有工具包按外部固定 manifest 核验，六个公开协议附件全部匹配标准给出的大小与 SHA。

标准明确：其他引擎保持相同数据范围、使用各自执行器，并分别记录实际 baseline/candidate/model/hardware。 因此不能将 Engine 的 B0/B1 组合成绩贴为
BidKV 或 PegaFlow 的插件成绩，也不能把任一项目的历史结果追认为本标准通过。

## 结论与重做范围

| 核对项       | BidKV 现有证据                                    | PegaFlow 历史及已完成的 restart-02            | 新标准要求                                                               |
| ------------ | ------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------ |
| 数据集       | EvoScientist KV 压力负载，含 TP4/C12 单元与负结果 | SWE 多轮前缀复用，8 sessions/360 turns        | 固定五项：200/200/160/64/48 请求/臂                                      |
| EOS 与预算   | 历史 runner 使用 ignore-eos 和 trace 输出长度     | 固定输出预算及多轮实际输出续接                | 保留 EOS；各数据集固定上限，不补齐短输出                                 |
| 缓存         | 压力/抢占机制研究                                 | 有意测量前缀复用与外部恢复                    | trace 使用新 cache_salt、零 KV 复用，先核验全部 /tokenize                |
| 吞吐         | 历史基准持续时间口径                              | `[0,900)` chunk token 数 / 900                | 全 solver batch：首提交至末响应完成的墙钟时间                            |
| 质量         | 不能代替冻结五项评分                              | 不测 SWE 解题正确率；有独立恢复样本与数值限制 | MMLU 原评分；HLE 四层、Frontier 两轨、共同固定 judge；trace 无任务解决率 |
| 本标准完整性 | 未建立                                            | 未建立                                        | 本机同配置连续 4K/16K C8 资格，加完整两臂五项                            |

**两项都需要另行执行新五项配对。** 这不是撤销其原工况的历史结果。
[#664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664) 对 BidKV
历史固定成果“本轮无需重测”的归档豁免，不能代替用户新增指定标准下的实测。 原 restart-02
五档已全部完成并单独报告：10个900秒观察点、2702条正式请求，吞吐各点均负；不计入新标准五项或连续 C8
资格。见[完成报告](restart-02/completed-cohort/REPORT.md)。

## 资源与实际执行身份（2026-11-14更新）

- PegaFlow使用ascend-91两张910B2（物理3/4、逻辑0/1），容器64GiB/8CPU。原KV20GiB
  SWE五档完成后，按已公开条件执行独立KV3GiB小规模压力验证；新冻结五项采用另一独立KV16GiB/image-enabled配置，先取得本机C8资格，不能复用旧SWE成绩。见[91新执行源与计划](pegaflow91-v1/README.md)。
- BidKV使用用户重启后的yanbo-81（物理6/7、逻辑0/1，容器64GiB/8CPU）；此前初次申请时物理0/1的记录保留，但不套用为本次执行卡号。
- 官方0.23原镜像缺少BidKV需要的策略接口，已新增并通过CPU/CI校验的[隔离实验测量载体PR28](https://github.com/vLLM-HUST/vllm-hust-bidkv/pull/28)，固定执行源码53eed4dd65bc4ba12efa37f9f1e334fa1ae51076，非已发布通用兼容。两臂相同prepared
  scheduler，仅切换策略。
- 81已完成两臂原合成C8、逐项实际128 token
  ID校验，以及Native342/Candidate343条SSE独立复算，四phase零失败、内部/外部prefix命中增量0。usage缓存详情为null仍保留未知，靠新盐与计数器交叉佐证。两臂策略调用/决策均0，不能宣称BidKV机制收益。[数值记录与受控索引](../bidkv/c8-results-v1/README.md)。
- 全五项输入已在受控NAS恢复并核验，包括HLE23图像及两份完整原source对象、MMLU原五文件、Frontier160、SWE64及Terminal48。正式执行使用原任务与原grader，派生contract仅另记实际模型路径和传输身份。离线token
  IDs以冻结Transformers5.14.1/tokenizers0.22.2生成；实际官方Ascend0.23服务保留其精确SDK pin
  Transformers5.5.4，全部trace实际token IDs必须逐项相同，不静默重算。
- 81完整五项controller已按[预先公开并逐字节读回的源/计划](finite-driver-v1/README.md)启动。Native全五项→Candidate全五项→全新Native共同judge；尚不能把加载、CPU检查或C8记为正式完成。91采用相同固定范围和统计口径，在自己项目内比较Native/PegaFlow。无随机交错/独立重复的时间漂移限制保留，不跨机器宣称插件排名。

每个项目计划 672 solver 请求/臂，合计 1344；两个项目合计 2688，另计资格、图像接口预热和 judge。 HLE/Frontier 每项目另有至多 720 个计划 judge
输入，共同 judge 的流量不混入 solver 吞吐或费用。 12–16 小时是参考 Engine 的排期估计；本次插件承载不同，须用实际启动、资格和完整批次耗时重新估计。

## 交付边界

新标准原始题目、答案、对话、SSE 与评分原件保存在受控位置；公共仓库提交可公开的配置、哈希清单、聚合结果和受控位置索引。 不复制私有 Engine 工具或受限 trace
全文到公共仓库。保留全部失败准备、失败模型尝试和未执行阶段，不换题、复制、补齐或自动重试推理。 五项执行完整与四项目标达标分别判断；没有实测的质量、长稳、利用率、费用项目写未验证，不填零。

## 四目标的独立审计口径（仅同时申请项目验收时）

以[固定项目最终目标来源](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a3d96977005f7257502ded2938a051c70de60b02/website-publication/data/five-datasets-standard-20261113/project-final-goals-v1.json)为核对依据：3.4.1同工况独立参考效率至少90%；3.4.2吞吐至少2倍，平均TTFT/TPOT分别降低至少20%；3.4.3至少32768上下文的吞吐与P95/P99稳定；3.4.4单位token成本降低至少30%并保留高多租户利用率证据。既定来源没有数字化尾延迟或利用率容差，不自行添加。旧Engine中期50%算力门槛不替代本次最终口径。

该固定来源说明原项目文档二进制不可得、采用用户提供摘录及明确解释；本次保留这一来源边界。网页入口b0-b1.html工具无法访问且本机HTTP403，未假称重新读取网页正文。缺少同有效工作/卡数的独立参考、有效算力、真实32K长稳、适用报价或多租户证据时对应项标未验证；完成五项与原4K/16K
C8资格本身不会自动通过四目标。
