# q40 内置运行时 posterior32 仍有新增错误；外置同历史对照待完成

**实际内置 q40 候选完成了原 posterior32，采集与独立复核通过，但答案“不新增错误”失败。** 原评分器给出29/32，全部自然 `stop`
完成。01741金答案E，两个历史native和旧prefix q4均答E；本轮内置q40答C，输出5551
tokens。07338恢复为H/stop/3292，并不能抵消01741新增错误。**未启动full200；当前候选不能宣称答案质量验收通过。**

外置q40同源码、同32题历史对照为
**PENDING**。本草稿未读取其正在运行的控制器、子报告、日志、调度或token数据，也没有预填等价/不等价、评分或吞吐结论。此目录是本地staging，不表示已发布、已上传或通过匿名HTTPS读回。

## 32题与历史对照

保留原冻结200题计划字节，仅按原顺序执行batch3、6、7、14，共32题；每批仍先seed输出一个token，再加入原七条peer，并等待整批自然结束。温度0、seed42、max_tokens8192、自然停止和原canonical
scorer未改。历史各臂对应行来自其完整200题，之前执行的批次历史与本轮posterior32不同；它们不是新增匹配进程。当前对照的重点是同一a050源码和同一posterior32顺序下，内置与外置prefix是否产生相同实际结果。

每格为“原提取答案 / finish_reason / 输出token数”。`—`是原评分器不提取答案；`length`按原规则不完整且错误，不以文本含金答案改判。外置列尚无数据。

| batch | case  | gold | native1 历史    | native2 历史    | raw 历史          | prefix q4 历史    | builtin q40 本轮 | external q40 |
| ----- | ----- | ---- | --------------- | --------------- | ----------------- | ----------------- | ---------------- | ------------ |
| 3     | 00803 | B    | B / stop / 734  | B / stop / 734  | B / stop / 734    | B / stop / 734    | B / stop / 734   | PENDING      |
| 3     | 00864 | C    | C / stop / 2234 | C / stop / 2234 | C / stop / 3169   | C / stop / 3515   | C / stop / 1769  | PENDING      |
| 3     | 03584 | H    | H / stop / 1060 | H / stop / 1060 | H / stop / 1058   | H / stop / 1285   | H / stop / 1075  | PENDING      |
| 3     | 03644 | E    | E / stop / 1178 | E / stop / 1178 | E / stop / 1163   | E / stop / 1210   | E / stop / 1134  | PENDING      |
| 3     | 03706 | B    | B / stop / 1911 | B / stop / 1911 | B / stop / 1762   | B / stop / 1263   | B / stop / 1234  | PENDING      |
| 3     | 03766 | B    | B / stop / 1018 | B / stop / 1018 | A / stop / 6396   | B / stop / 1053   | B / stop / 5412  | PENDING      |
| 3     | 03826 | C    | C / stop / 505  | C / stop / 505  | C / stop / 505    | C / stop / 505    | C / stop / 505   | PENDING      |
| 3     | 03886 | A    | A / stop / 1351 | A / stop / 1351 | A / stop / 1239   | A / stop / 1336   | A / stop / 1274  | PENDING      |
| 6     | 10611 | B    | B / stop / 647  | B / stop / 647  | B / stop / 687    | B / stop / 669    | B / stop / 711   | PENDING      |
| 6     | 10672 | A    | A / stop / 1198 | A / stop / 1198 | G / stop / 1046   | A / stop / 2613   | A / stop / 1198  | PENDING      |
| 6     | 10734 | I    | I / stop / 840  | I / stop / 840  | I / stop / 840    | I / stop / 840    | I / stop / 840   | PENDING      |
| 6     | 06847 | D    | D / stop / 156  | D / stop / 156  | D / stop / 191    | D / stop / 156    | D / stop / 156   | PENDING      |
| 6     | 06907 | I    | I / stop / 271  | I / stop / 271  | I / stop / 266    | I / stop / 272    | I / stop / 271   | PENDING      |
| 6     | 06967 | C    | C / stop / 567  | C / stop / 567  | C / stop / 551    | C / stop / 519    | C / stop / 567   | PENDING      |
| 6     | 07028 | B    | B / stop / 153  | B / stop / 153  | B / stop / 147    | B / stop / 153    | B / stop / 153   | PENDING      |
| 6     | 07090 | J    | J / stop / 275  | J / stop / 275  | J / stop / 265    | J / stop / 256    | J / stop / 275   | PENDING      |
| 7     | 07151 | H    | H / stop / 289  | H / stop / 289  | H / stop / 445    | H / stop / 445    | H / stop / 486   | PENDING      |
| 7     | 07213 | J    | J / stop / 302  | J / stop / 302  | J / stop / 237    | J / stop / 302    | J / stop / 302   | PENDING      |
| 7     | 07275 | E    | E / stop / 379  | E / stop / 379  | E / stop / 805    | E / stop / 447    | E / stop / 378   | PENDING      |
| 7     | 07338 | H    | H / stop / 4723 | H / stop / 4723 | H / stop / 5133   | — / length / 8192 | H / stop / 3292  | PENDING      |
| 7     | 07401 | A    | A / stop / 575  | A / stop / 575  | A / stop / 702    | A / stop / 682    | A / stop / 598   | PENDING      |
| 7     | 07461 | G    | G / stop / 468  | G / stop / 468  | G / stop / 481    | G / stop / 462    | G / stop / 468   | PENDING      |
| 7     | 07524 | I    | I / stop / 467  | I / stop / 467  | I / stop / 302    | I / stop / 417    | I / stop / 477   | PENDING      |
| 7     | 07585 | A    | A / stop / 532  | A / stop / 532  | A / stop / 544    | A / stop / 533    | A / stop / 533   | PENDING      |
| 14    | 01557 | C    | C / stop / 396  | C / stop / 396  | C / stop / 397    | C / stop / 397    | C / stop / 431   | PENDING      |
| 14    | 01619 | G    | G / stop / 1034 | G / stop / 1034 | G / stop / 473    | G / stop / 1165   | G / stop / 1034  | PENDING      |
| 14    | 01680 | B    | H / stop / 585  | H / stop / 585  | H / stop / 1049   | H / stop / 2294   | C / stop / 818   | PENDING      |
| 14    | 01741 | E    | E / stop / 377  | E / stop / 377  | — / length / 8192 | E / stop / 590    | C / stop / 5551  | PENDING      |
| 14    | 01802 | C    | I / stop / 426  | I / stop / 426  | I / stop / 678    | I / stop / 655    | I / stop / 538   | PENDING      |
| 14    | 01862 | D    | D / stop / 654  | D / stop / 654  | D / stop / 601    | D / stop / 605    | D / stop / 601   | PENDING      |
| 14    | 01923 | C    | C / stop / 323  | C / stop / 323  | C / stop / 505    | C / stop / 456    | C / stop / 323   | PENDING      |
| 14    | 01984 | A    | A / stop / 202  | A / stop / 202  | A / stop / 444    | A / stop / 413    | A / stop / 202   | PENDING      |

历史native1/native2各30/32且这些选中案例均完整；raw为27/32、旧prefix q4为29/32且仍有不完整案例。当前builtin
q40为29/32、32题均完整；错误为01680、01741、01802，其中**只有01741是两次native都正确但候选错误的新增错题**。旧prefix07338的length退化已恢复。不能仅凭总分与旧prefix同为29/32宣称无退化，也不能凭8题局部成功宣称整体修复。历史完整200题的严格完整性门仍FAIL，不能把posterior32的完整性替代它。

## 实际内置候选与调度

候选源码commit为`a05029484b2c9bc21e4e6fd705cc396d111dfaa5`，模型Qwen3.5-35B-A3B BF16，TP4/EP4，物理卡3/4/5/6，HCCL
deterministic。实际内置prefix配置、显式layers_per_slice4→40、模型config构造前后、tokenizer、全worker
policy和源码收据均经主/独立校验。没有数值观察器或更换原评分器。

实际15457次serving，四批分别5413/1199/3293/5552次。各rank D图replay增量15453，P eager forward增量8；每批两个P
cohort（seedD0、七peer伴随seedD1），peer一次推进40层并收齐七条ACK，每个cohort保留原81次P reduction或D0
bypass。D覆盖根据实际调度随自然完成缩减计算，未用固定token截止数代替。

此前单独新进程external q40
full8的batch7全部正确；当前内置posterior32的batch7八条完整输出也逐token延续为同样的完整输出。这只支持batch7的一致性。batch14的01741相对native首次token差异14早已存在于旧prefix
q4；新旧prefix目标前57tokens相同，最早同批peer01923在22、01984在34变化，目标在58变化。完整差异记录见[终态CPU token定位](builtin-token-localization-v1.json.gz)。该定位不证明某算子根因，也不支持按题号、金答案或两例结果选择quota。

## 外置同历史桥接：PENDING

预先声明的唯一对照使用同a050源码、同batch3/6/7/14原32题顺序、同8192自然完整答案和每批一次快照RPC节奏；将builtin prefix明确设为off，安装冻结external
prefix模块`6431c4c0abdd869ef17e7122f681c457ad2b4d5741f86c6a0653e9fca44e02fc`，其余q40、环境、模型和采样保持。外置控制并非新候选通过声明。

root确认终态后，才补全部32条完整tokens/text/finish/stop与原评分、实际admission映射后的调度及各rank collective几何比较。双方自然输出长度允许不同，D
replay分别按真实调度重算；合法不等将报告为port-equivalence差异，不伪装成采集失败。无论该对照是否相等，当前内置01741C/5551失败均保留。没有full200授权、性能或Goodput收益结论。

## 已固定的内置终态证据

| 项目     | SHA256或结果                                                       |
| -------- | ------------------------------------------------------------------ |
| 终态     | 2026-10-30T13:48:19.319811+00:00，exit0，自有进程为空，四卡空闲    |
| 子报告   | `c1dab8b9c7cca79850c69b1a63d3adc368cb241b864ab596aac8c7103f88189e` |
| 控制器   | `6baaeb18ad8b70e819c8dcc8cae57fb1b27a05077d8da2af5be501af84793ffd` |
| 主分析   | `8cf06b3396fc484baab24e2af710b74890f2b28269515c40fea1d692ef50910a` |
| 独立复核 | `99094b40a6cbf63287986713e44e56db5690b1198752ac99f022327c9d6fc09d` |
| 源身份   | `8d04890973c369a5412e80cc08c6cb55ef405e9360bd38e2bfafa9b5d65fe4fe` |
| 冻结清单 | `507f4c76f999476e7a182fca9b2e7ba1655a8678d37454b4286d18664f3fdfcb` |
| 原评分器 | `157b52ffdbae6d7f0f990bb27379c463df1807daeebf7298fd396968e96b17fd` |

[发布映射](publication-map-v1.json)记录原件与无损gzip副本的SHA及字节数；[源码包](builtin-source-capsule-v1.tar.gz)包含原22文件producer、25个实际绑定runtime文件、独立脚本与本stager，逐成员SHA见包内MANIFEST。原200输入只保留一次，历史完整200报告及模型依赖未重复打包；没有大NPY或任何新logits张量。本地已逐字节复核gzip还原和全部源码包成员。

外置终态以后应新增明确版本的addendum、比较结果和报告，保留当前builtin固定证据及PENDING草稿谱系；不得覆盖失败样本或选择性重跑。关联：[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)、[此前局部8题完整答案](../muxpd-q40-full-answer-20261030/REPORT-v1.md)。
