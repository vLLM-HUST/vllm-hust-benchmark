# q40 posterior32仍有新增错误；同源码外置对照复现全部32条完整输出

**内置和外置q40都为29/32，01741均答C而金答案为E，输出5551 tokens后自然stop。“不新增错误”失败，full200没有启动。**
07338均恢复为H/stop/3292；此前局部八题成功不能代表整体修复。

同a050源码、同posterior32执行历史下，主分析和独立复核确认：**全部32条完整token序列、文本、finish/stop、原评分、按真实admission映射的调度，以及四rank
collective几何全部相同。** 这支持本次外置prefix到内置prefix移植在该冻结实验中的可观测等价；不证明所有输入等价，也不证明quota是退化的唯一原因或某个算子故障。

## 32题完整结果

每格为“原canonical scorer提取答案 / 实际finish_reason /
输出token数”。历史四臂来自各自原full200，对应的前置批次与本轮posterior32不同；没有把历史结果当作新匹配进程。`length`仍是不完整和错误，不因文本含金答案改判。

| batch | case  | gold | native1 历史    | native2 历史    | raw 历史          | prefix q4 历史    | builtin q40     | external q40    |
| ----- | ----- | ---- | --------------- | --------------- | ----------------- | ----------------- | --------------- | --------------- |
| 3     | 00803 | B    | B / stop / 734  | B / stop / 734  | B / stop / 734    | B / stop / 734    | B / stop / 734  | B / stop / 734  |
| 3     | 00864 | C    | C / stop / 2234 | C / stop / 2234 | C / stop / 3169   | C / stop / 3515   | C / stop / 1769 | C / stop / 1769 |
| 3     | 03584 | H    | H / stop / 1060 | H / stop / 1060 | H / stop / 1058   | H / stop / 1285   | H / stop / 1075 | H / stop / 1075 |
| 3     | 03644 | E    | E / stop / 1178 | E / stop / 1178 | E / stop / 1163   | E / stop / 1210   | E / stop / 1134 | E / stop / 1134 |
| 3     | 03706 | B    | B / stop / 1911 | B / stop / 1911 | B / stop / 1762   | B / stop / 1263   | B / stop / 1234 | B / stop / 1234 |
| 3     | 03766 | B    | B / stop / 1018 | B / stop / 1018 | A / stop / 6396   | B / stop / 1053   | B / stop / 5412 | B / stop / 5412 |
| 3     | 03826 | C    | C / stop / 505  | C / stop / 505  | C / stop / 505    | C / stop / 505    | C / stop / 505  | C / stop / 505  |
| 3     | 03886 | A    | A / stop / 1351 | A / stop / 1351 | A / stop / 1239   | A / stop / 1336   | A / stop / 1274 | A / stop / 1274 |
| 6     | 10611 | B    | B / stop / 647  | B / stop / 647  | B / stop / 687    | B / stop / 669    | B / stop / 711  | B / stop / 711  |
| 6     | 10672 | A    | A / stop / 1198 | A / stop / 1198 | G / stop / 1046   | A / stop / 2613   | A / stop / 1198 | A / stop / 1198 |
| 6     | 10734 | I    | I / stop / 840  | I / stop / 840  | I / stop / 840    | I / stop / 840    | I / stop / 840  | I / stop / 840  |
| 6     | 06847 | D    | D / stop / 156  | D / stop / 156  | D / stop / 191    | D / stop / 156    | D / stop / 156  | D / stop / 156  |
| 6     | 06907 | I    | I / stop / 271  | I / stop / 271  | I / stop / 266    | I / stop / 272    | I / stop / 271  | I / stop / 271  |
| 6     | 06967 | C    | C / stop / 567  | C / stop / 567  | C / stop / 551    | C / stop / 519    | C / stop / 567  | C / stop / 567  |
| 6     | 07028 | B    | B / stop / 153  | B / stop / 153  | B / stop / 147    | B / stop / 153    | B / stop / 153  | B / stop / 153  |
| 6     | 07090 | J    | J / stop / 275  | J / stop / 275  | J / stop / 265    | J / stop / 256    | J / stop / 275  | J / stop / 275  |
| 7     | 07151 | H    | H / stop / 289  | H / stop / 289  | H / stop / 445    | H / stop / 445    | H / stop / 486  | H / stop / 486  |
| 7     | 07213 | J    | J / stop / 302  | J / stop / 302  | J / stop / 237    | J / stop / 302    | J / stop / 302  | J / stop / 302  |
| 7     | 07275 | E    | E / stop / 379  | E / stop / 379  | E / stop / 805    | E / stop / 447    | E / stop / 378  | E / stop / 378  |
| 7     | 07338 | H    | H / stop / 4723 | H / stop / 4723 | H / stop / 5133   | — / length / 8192 | H / stop / 3292 | H / stop / 3292 |
| 7     | 07401 | A    | A / stop / 575  | A / stop / 575  | A / stop / 702    | A / stop / 682    | A / stop / 598  | A / stop / 598  |
| 7     | 07461 | G    | G / stop / 468  | G / stop / 468  | G / stop / 481    | G / stop / 462    | G / stop / 468  | G / stop / 468  |
| 7     | 07524 | I    | I / stop / 467  | I / stop / 467  | I / stop / 302    | I / stop / 417    | I / stop / 477  | I / stop / 477  |
| 7     | 07585 | A    | A / stop / 532  | A / stop / 532  | A / stop / 544    | A / stop / 533    | A / stop / 533  | A / stop / 533  |
| 14    | 01557 | C    | C / stop / 396  | C / stop / 396  | C / stop / 397    | C / stop / 397    | C / stop / 431  | C / stop / 431  |
| 14    | 01619 | G    | G / stop / 1034 | G / stop / 1034 | G / stop / 473    | G / stop / 1165   | G / stop / 1034 | G / stop / 1034 |
| 14    | 01680 | B    | H / stop / 585  | H / stop / 585  | H / stop / 1049   | H / stop / 2294   | C / stop / 818  | C / stop / 818  |
| 14    | 01741 | E    | E / stop / 377  | E / stop / 377  | — / length / 8192 | E / stop / 590    | C / stop / 5551 | C / stop / 5551 |
| 14    | 01802 | C    | I / stop / 426  | I / stop / 426  | I / stop / 678    | I / stop / 655    | I / stop / 538  | I / stop / 538  |
| 14    | 01862 | D    | D / stop / 654  | D / stop / 654  | D / stop / 601    | D / stop / 605    | D / stop / 601  | D / stop / 601  |
| 14    | 01923 | C    | C / stop / 323  | C / stop / 323  | C / stop / 505    | C / stop / 456    | C / stop / 323  | C / stop / 323  |
| 14    | 01984 | A    | A / stop / 202  | A / stop / 202  | A / stop / 444    | A / stop / 413    | A / stop / 202  | A / stop / 202  |

两个历史native各30/32；raw27/32；旧prefix q4为29/32但07338达到8192
length。本轮两种q40均29/32且全部32题stop完成，错题为01680、01741、01802；**唯一两次native都正确却被q40新增答错的是01741。**
两种q40复现同一失败不构成质量通过；总分等于旧prefix也不代表逐题不退化。历史完整200题严格完整性门仍FAIL，当前32题的完整性不能替代它。

## 相同历史与唯一布局切换

内外置均使用clean `a05029484b2c9bc21e4e6fd705cc396d111dfaa5`、Qwen3.5-35B-A3B BF16、TP4/EP4、物理3/4/5/6、HCCL
deterministic、layers_per_slice40及相同graph/backend。原200计划字节保留，仅按原顺序选择batch3/6/7/14，每批seed输出一个token后加入原七peer，整批自然完成才进入下一批。temperature0、seed42、max_tokens8192、tokenizer和评分器不改，无额外request
warmup、数值observer或成功路径活请求取消。

外置对照把builtin
prefix明确设off，再安装冻结模块`6431c4c0abdd869ef17e7122f681c457ad2b4d5741f86c6a0653e9fca44e02fc`；全配置比较只允许该布局旗标变化，两者保留每批一次只读snapshot
RPC节奏。源码身份用于来源证明，并不把外置诊断冒充新质量候选。

两者实际各15457次serving（四批5413/1199/3293/5552），每rank D replay增量15453、P eager forward增量8；每批seed P一次及七peer
P一次，peer推进40层并收齐七ACK。8个cohort中4个D0 seed各81次bypass、4个peer各81次真实P
collective；四rank原prompt指纹、布局偏移、单次原collective、返回input别名、路由与3slot恢复均验证。请求/进程ID按原admission和角色规范化，不要求运行时Python对象ID相同，也未把aggregate计数当逐算子数值证据。

## 因果边界与后续

原旧q4
full200、此前后验诊断和本轮posterior32的前置历史不同，当前移植对照没有把这些历史差异消除。01741相对native的token14分歧早已存在；旧q4/currentq40目标前57tokens相同，同批peer01923在22、01984在34变化，目标在58变化。详情见保留的[CPU token定位](builtin-token-localization-v1.json.gz)。不能据这些边界按题号、金答案选择quota，也不能跳过实际完整轨迹复现直接归因某核。

后续预声明方案是在相同a050运行时，对原batch14做两个独立新进程的q4/q40自然完整八题诊断，只有quota不同；该方案不是本报告中已执行的结果。它仍需各自调度/完成证据和完整八条历史轨迹核对，不能将q4十轮P推进套用q40单轮门。**本报告不报告该未来实验、不报告吞吐或Goodput收益，也不授权失败q40候选的full200。**

## 终态、公开证据与草稿谱系

内置终态2026-10-30T13:48:19.319811+00:00；外置终态2026-10-30T14:28:00.889891+00:00，均exit0、自有进程为空、四卡空闲。root在外置终态及新空闲复查后才授权读取child/log。主分析和独立复核均PASS，表示采集与等价证据通过，答案相对门仍FAIL。

| 证据                 | SHA256                                                             |
| -------------------- | ------------------------------------------------------------------ |
| builtin child        | `c1dab8b9c7cca79850c69b1a63d3adc368cb241b864ab596aac8c7103f88189e` |
| builtin primary      | `8cf06b3396fc484baab24e2af710b74890f2b28269515c40fea1d692ef50910a` |
| builtin independent  | `99094b40a6cbf63287986713e44e56db5690b1198752ac99f022327c9d6fc09d` |
| external child       | `d6fdb9311a6a25d4b94716e091c0de7bcb4afef597a7d1e3312a9c9414233b59` |
| external controller  | `ff95526437d6275b2c75eac1dad6362e5252970d43b8cf84c0645919bcbee876` |
| external primary     | `f6ca2553f1b75bc978f9a29c420f6f4738415326eb034e054e8380370c1c285f` |
| external independent | `1dbef9dc360c636c97954434a32564d04cc813dcfccedc6253763dfc001b6923` |

完整tokens/文本run、调度JSONL、日志、每批四rank快照均为无损gzip副本；主/独立、root终态门和冻结清单同时保留。[发布映射](publication-map-v2.json)绑定每份原件与存储副本SHA。builtin源码包包含原22producer和25runtime源码，external
addendum仅补新增桥接文件，复用既有22项和原200输入，避免重复打包。没有大NPY、模型权重或完整旧200报告；服务器依赖未打包。不宣称原始短logits已上传，也不把本地校验说成匿名HTTPS读回。

[REPORT-v1.md](REPORT-v1.md)与v1表/status是外置运行期间形成的PENDING草稿，原字节保留；本v2才记录已终态对照。两次真实模型尝试各自保留，没有选择性重跑。独立复核保留静态审查前/后的脚本源码，最终结论只绑定通过的最终脚本和原终态数据。关联：[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)、[此前局部八题](../muxpd-q40-full-answer-20261030/REPORT-v1.md)。
