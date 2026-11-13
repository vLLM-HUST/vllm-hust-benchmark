# 新五项标准的适用性核对

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

| 核对项       | BidKV 现有证据                                    | PegaFlow 历史及正在执行的 restart-02          | 新标准要求                                                               |
| ------------ | ------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------ |
| 数据集       | EvoScientist KV 压力负载，含 TP4/C12 单元与负结果 | SWE 多轮前缀复用，8 sessions/360 turns        | 固定五项：200/200/160/64/48 请求/臂                                      |
| EOS 与预算   | 历史 runner 使用 ignore-eos 和 trace 输出长度     | 固定输出预算及多轮实际输出续接                | 保留 EOS；各数据集固定上限，不补齐短输出                                 |
| 缓存         | 压力/抢占机制研究                                 | 有意测量前缀复用与外部恢复                    | trace 使用新 cache_salt、零 KV 复用，先核验全部 /tokenize                |
| 吞吐         | 历史基准持续时间口径                              | `[0,900)` chunk token 数 / 900                | 全 solver batch：首提交至末响应完成的墙钟时间                            |
| 质量         | 不能代替冻结五项评分                              | 不测 SWE 解题正确率；有独立恢复样本与数值限制 | MMLU 原评分；HLE 四层、Frontier 两轨、共同固定 judge；trace 无任务解决率 |
| 本标准完整性 | 未建立                                            | 未建立                                        | 本机同配置连续 4K/16K C8 资格，加完整两臂五项                            |

**两项都需要另行执行新五项配对。** 这不是撤销其原工况的历史结果。
[#664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664) 对 BidKV
历史固定成果“本轮无需重测”的归档豁免，不能代替用户新增指定标准下的实测。 当前 restart-02 五档实验继续推进并单独报告，不计入新标准的五项或连续 C8 资格。

## 资源与当前实际阶段

- PegaFlow 使用 ascend-91 的两张 910B2（物理 3/4、逻辑 0/1）；当前继续原五档 900s 方案。
- 用户另启动 yanbo-81，分配给 BidKV。现场核验为两张空闲 910B2（物理/逻辑 0/1）、64 GiB 容器内存、8 CPU 配额。
- 五项输入已按固定来源恢复：MMLU 原五个输入/评分文件逐字节匹配；HLE 200 含 23 图像、四层选择核验；Frontier 160；SWE 64、Terminal 48。 使用冻结的四个
  tokenizer/config 文件、Transformers 5.14.1、tokenizers 0.22.2，离线生成 112 个 trace 输入 token ID
  哈希，长度均与冻结清单匹配。 这些仍是 **CPU 输入准备，模型请求数为零**；实际两臂 /tokenize 与模型资格尚须现场执行。
- 81 的现有官方 vLLM 0.23 镜像缺少 BidKV 当前要求的原生抢占策略接口。需要先完成并验证兼容承载，不能把安装成功当作 runtime effective。
  两臂须使用相同承载版本，仅切换插件，另记代码/patch/依赖身份，不能借用 Engine 固定 B1 名称。

每个项目计划 672 solver 请求/臂，合计 1344；两个项目合计 2688，另计资格、图像接口预热和 judge。 HLE/Frontier 每项目另有至多 720 个计划 judge
输入，共同 judge 的流量不混入 solver 吞吐或费用。 12–16 小时是参考 Engine 的排期估计；本次插件承载不同，须用实际启动、资格和完整批次耗时重新估计。

## 交付边界

新标准原始题目、答案、对话、SSE 与评分原件保存在受控位置；公共仓库提交可公开的配置、哈希清单、聚合结果和受控位置索引。 不复制私有 Engine 工具或受限 trace
全文到公共仓库。保留全部失败准备、失败模型尝试和未执行阶段，不换题、复制、补齐或自动重试推理。 五项执行完整与四项目标达标分别判断；没有实测的质量、长稳、利用率、费用项目写未验证，不填零。
