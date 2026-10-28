# 原生分步执行复现 raw 数值轨迹，三项反事实与新对照尚未带来质量缓解

本报告接续[固定提交1c70f51a的质量复核](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/1c70f51a70e90943851e246a41fa13e9afc5332c/reports/muxpd-quality-review-20261028/REVIEW-v1.md)。四个新实验均完成采集、shutdown调用返回、请求/KV
回到各自空闲基线及自有进程清理PASS，但10672第17个输出仍为 raw 的ID2663。采集 PASS 与答案质量 PASS
是不同判据；[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)保持 **OPEN**。

## 原质量结果与三道丢失题不变

历史两次 fresh native 均165/200，完整 token/text/termination 一致；同步 raw/group
均166/200且彼此完整输出一致。候选在两次原生均正确的165题中新增三道错误，另有四道新增正确，不能用总分相抵。严格无截断门及补充预算内逐题答案不退化门均保持
**FAIL**，输入、gold、评分器、分母、预算和判据均未修改。

| 历史 case      | Gold | Native1 = Native2    | Raw = Group                     | 首次不同输出位置（从1起） |
| -------------- | ---- | -------------------- | ------------------------------- | ------------------------: |
| mmlu-pro-03766 | B    | B，stop，1018 tokens | A，stop，6396 tokens            |             66：ID77→ID16 |
| mmlu-pro-10672 | A    | A，stop，1198 tokens | G，stop，1046 tokens            |        17：ID19987→ID2663 |
| mmlu-pro-01741 | E    | E，stop，377 tokens  | 无提取答案，length，8192 tokens |         14：ID1865→ID3234 |

上述是历史完整答案事实。本轮只对10672所在的原 batch6 作短前缀定位，没有为03766、01741补做模型实验，也没有完成新的200题评分。前报告限定负载的历史 Goodput
收益仍按其原版本与工作负载引用，本轮没有新增或撤销性能结果。

## 固定输入、实际运行身份与干预

八请求按原顺序为10611（seed）、10672、10734、06847、06907、06967、07028、07090，均带`mmlu-pro-`前缀。seed实际输出恰好1个 token
后入队七个 peers；采样仍为 temperature0、seed42、max_tokens8192、ignore_eos=false、detokenize=false。目标生成第18个 token
后，控制器主动 abort 并排空；这不是自然完成的答案或把答案预算改成18。

本轮 fresh process 只重放原 batch6；未重放原200题进程此前五个
batch，因此进程内缓存和暖机历史不同。相应前缀与历史角色的一致性桥接已经记录，但不代表全部原200题运行状态相同。

四个新进程使用实测四张910B2、物理卡0/2/3/4、TP4/EP、BF16、同步调度、HCCL_DETERMINISTIC=true及原 fusion/图配置。observer
源码为`6c606f789450a45ef35afee398b0a3e516caa1a1`；其中历史运行时`f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`已有
src 文件逐一保持字节一致。宿主 core 为`d0f22d2bda562156e4dbf433ce645e1769b4f804`，Ascend
为`03766ac696fde5ab1980d80ca0b8543d3580c989`。这些证据不覆盖当前安装包、最新 main 或异步候选。

| 新尝试                 | 唯一预注册干预                                                                                           | 采集终态 | 保存长度              |
| ---------------------- | -------------------------------------------------------------------------------------------------------- | -------- | --------------------- |
| `cf-raw-q40-01`        | 原 raw q4 的层配额改为40，完整 P 与一次 seed D 在一轮完成                                                | PASS     | seed19，七 peers 各18 |
| `cf-native-split-01`   | 普通原生 worker；原生 eligibility allowlist 先执行一次 seed D，再执行七 peers 的完整 P，随后恢复原生调度 | PASS     | seed19，七 peers 各18 |
| `cf-raw-paged-01`      | 保持 raw q4；仅把第二次 P 调用的完整新七请求 cohort 从`PrefillNoCache`改为`ChunkedPrefill`一次           | PASS     | seed28，七 peers 各18 |
| `cf-raw-q4-control-01` | 新进程原 raw q4 对照，无调度或 attention-state 干预                                                      | PASS     | seed28，七 peers 各18 |

native-split 的两次 allowlist 均留下验证与恢复回执，实际调度量依次为1和9613 tokens；没有手工修改 sampled tokens、计算位置、KV 或 worker
行序。paged 的四个 rank 均确认安装一次、激活一次、第一次 seed P 原样、第二次七请求 P 改变 state，decode hook 未改。记录直接证明 attention-state
改变；分页 FIA 路径依据固定源码分支推断，没有独立的 kernel-dispatch trace。不得把这一层证据描述成已观测全部内部 FIA 张量或每个内核调用。

## 144个向量：新对照与三个反事实均未改变 raw 轨迹

每个 observed 进程保存八请求各前18个完整词表向量，共144个`.npy`；seed额外生成的 token 仍保留在前缀记录中，但超过18的位置不纳入此向量集合。比较按 case
和输出位置对应，SHA256 覆盖整个 NPY 文件字节。

完整分析包含四个新进程与三个既有有效对照（native01、native02、raw02），共21组配对。独立审计重新验证1008个向量文件的 SHA/大小及四 rank
的采样绑定，核对全部当前尝试、来源清单和配对计数，无遗漏：

| 配对集合                                             | 配对数 | 每对144个向量中字节相同数 |
| ---------------------------------------------------- | -----: | ------------------------: |
| 既有 native01 vs native02                            |      1 |                       144 |
| 既有 raw02、q40、native-split、paged、新 q4 五者之间 |     10 |                       144 |
| 两次既有 native 各自对上述五者                       |     10 |    1，仅 seed 首个 P 向量 |

新增 q4 对照与既有 raw02 的八条完整保存前缀也全部相同，确认反事实前后的原 raw 轨迹可复现。paged 与这两个 q4 角色的八条前缀相同；q40 与 native-split
的八条前缀相同。跨19/28 seed长度的两组只能说七 peers 前缀相同、seed共同19-token前缀相同，不能把不同长度称为完整前缀一致。

三个反事实的七个首 P 向量均与 raw02 逐字节相同。目标第1、2、17个向量也未变；paged 干预的结果属于“本次捕获的最终 logits 未改变”，不是“logits
改变但收益不足”。这不证明所有内部 attention 数值均未改变。

目标第17个输出仍有唯一最大值：native 的ID19987/ID2663为32.000/31.750，差值+0.250；raw及三个反事实为31.750/31.875，差值−0.125。实际
token 与各自 argmax 一致。窄 margin 不能作为接受原答案退化的理由。

## 129个当前上下文匹配不等于历史状态相同

q40与原native的144对向量中，分析器标记129对`strict_current_context_equal`。这个字段只检查当前目标 prompt/prefix、整个有序 batch 的
token与计算计数、graph descriptor；它不包含干预配置、attention state/layout，也不证明过去 KV/GDN 的数值内容相同。

q40把 seed 进度对齐后，目标第2和第17个输出确有相同的当前有序 batch/graph
元数据，向量仍不同；最大绝对差/RMS分别仍为0.734375/0.1375783818及0.857421875/0.1236860342。它们之前的 P 仍不同：原native计算seed
D加七P共9614 tokens，q40的P独立计算七P共9613 tokens。因此129不能解释成“129个完全相同计算状态下的错误”，也不能消除历史状态这一条件。

native-split在恢复decode后，原生行序为七个peers占row0..6、seed在row7；q40与原native为seed在row0。当前有序上下文不匹配，不能通过重排输入去制造匹配。按case单独核对，native-split/q40在目标第1、2、17个输出所用的物理block
IDs相同；有序KV列表的不相同来自行序，不能误读成已发现KV错配或污染。相同block IDs也不证明缓存数值相同。

## 结论与保留边界

q40消除了九个seed步骤的偏移，却没有改变此次捕获的144个向量；这条轨迹不需要十次slice推进才能出现。普通原生worker在D1后单独执行七P，复现了同样144个向量；这条分歧也不依赖MuxPD双runner、独立通信组或greenlet重叠才能出现。单次state改为`ChunkedPrefill`未改变这些最终logits，不能作为有效质量缓解采用。

这些结论仅限原batch6的有界观察，尚未区分batch形状、投影/MoE/GDN路径和其他算子选择的独立贡献，未建立唯一数值根因或可验收的质量修复。目标第18个输出之前已发生历史分叉，不能继续把不同前缀的向量距离解释成相同输入误差。

当前四个新尝试均shutdown调用返回且清理PASS；原生executor内部退出记录包含SIGTERM，这不是整个进程树完全无信号退出的承诺。控制器exit0、无自有进程残留及空闲基线恢复是这里的采集清理判据。上一轮`raw-observed-01`因空槽校验误判形成的原始FAIL仍保留，不重标成功。完整200题的原质量门保持FAIL；后续只有形成有证据支持的实际缓解，才另行版本化验证完整答案，沿用原两次fresh
native、评分、截断及逐题否决标准。本轮没有吞吐、Goodput、HTTP SLO或安装包资格结论。

## 原件与分析入口

原件Release为[evidence-muxpd-counterfactual-20261028](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-counterfactual-20261028)，归档名`muxpd-counterfactual-20261028-evidence-v1.tar.gz`。归档SHA256：`1ea0a4861112593f111ad9e67ca3edb66fe60f1ad5d2e91555c10167d7604ff8`。三个公开附件均已匿名
HTTP 200、大小与 SHA256 验证通过，全部 738
个原始成员逐项校验通过；见[下载回执](public-download-verification-v1.json)、[成员回执](public-member-verification-v1.json)及[API回执](release-api-v1.json)。

归档内`analysis-results-v1/analysis.json`为完整21组分析，60,955,830
bytes，SHA256为`41e90b3b0a44ca460fe276d0a6a599b59a15df4847a74f1504e6ce1961fb5e83`。`final-matrix-audit-v1/summary.json`及审计脚本/SHA256SUMS给出独立计数、全部配对与终态检查；三个`preliminary-*-v1`目录保留阶段性结论。冻结probe与attention-state
hook共有66项CPU测试通过；测试通过和归档完整不等于答案质量通过。

本轮归档的 `runs/` 包含全部四次新尝试。用于比较的三个既有有效对照与上一轮 `raw-observed-01`
原始失败保留在[上一轮不可变 Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-quality-review-20261028)中，不重复装入本轮归档。上一轮资产名为
`muxpd-quality-review-20261028-evidence-v1.tar.gz`，SHA256 为
`e14cdfdb3401b2bb7da5d52e1feb2865f1423a1a6adf923d88c2d88ad9f8ecbe`。重算需先校验两份归档与各自成员清单，将它们分别展开到两个目录，再将本轮目录传给分析器的
`--root`、上一轮目录传给 `--previous-root`；输出必须使用新的目录。清单保留生产时绝对路径作为 provenance，分析器按指定根目录读取原件。
