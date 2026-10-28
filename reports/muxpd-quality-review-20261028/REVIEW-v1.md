# MuxPD：保留场景收益，继续追踪答案退化

本报告回应
[2026-10-27 的复核留言](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23#issuecomment-6446722727)。结论保持：特定交互负载减少最坏停顿和改善每请求
Goodput 的历史结果成立；固定200题的答案不退化门槛仍失败。安装包、同步实现、历史 streaming 和异步候选分别绑定实测源码及配置，不能相互借用资格。调度修复 #24
的关闭不改变质量问题 #23 的状态。

\*\*当前状态：\*\*三道历史丢失题的原件审计及10672的八请求实机定位已完成。五个 fresh 尝试全部终态：四个有效采集，另一个保留为诊断清理校验器失败。新证据复现了第17个输出的
logits 排序翻转，但未定位唯一根因、未修改运行时代码、未重跑200题或改变质量 FAIL。原始证据已公开并完成下载及逐成员验证，见文末回执。

## 四组版本与配置边界

| 证据对象           | 实际运行身份                                                                                                                     | 配置与可引用范围                                                                                                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 安装包 `0.2.0a1`   | 插件 `658b4d0b4e8982bd281499005500ac4c3e0100f5`；wheel SHA256 `7cc6b1d9fc8a2afa90526495a7f33e4d08901001d7c7195fbac3bf9ad06ecd7f` | TP4、同步 raw-P q4、isolated 通信、独立 D graph、Manager 显式激活。安装、启用、取消与回滚等功能验收；非精确等价，无此安装包性能声明。                                                                  |
| 同步 raw/group     | `f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`                                                                                       | TP4、HCCL=true、同步调度、fusion 后端、npugraphEx/static=false。raw/group 的 P 分组不同，完整200题输出相同，但答案不退化均 FAIL；较短最大停顿与吞吐、TTFT、P99 代价并列。                              |
| 历史 TP4 streaming | `5342e8cd9f7407f4b7d43ea34517b59eff2b3253`                                                                                       | 同步 raw-P q4、isolated 通信、独立 D graph；对照原生 async/chunked 16K。固定到达率、固定每请求双 SLO 的历史 Goodput，不是当前包或后续候选的性能资格。                                                  |
| 异步 Decode 候选   | 实测 `880173922196f07e96b5ae43fee20335be02efa2`；合并 `eff31ee8d1ce1bacd6e9d30840fd709895a98fe8` 的 runtime/tests 相同           | TP4、HCCL=true、group-P q4、fusion 后端；纯 D 两批异步，P 前排空。有限 offline 负载吞吐恢复接近原生并保留短最坏停顿；TTFT/P99 仍较差，未重跑200题质量门，未建立新 HTTP Goodput、Manager 或安装包资格。 |

固定来源：[安装包发行资格](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/19081ff39a25c3c8ac821ae7885688ea24a03e90/reports/muxpd-ecpa-20261025/RELEASE-QUALIFICATION.md)、[同步质量与参考 v5](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ebddb585390c333ac1f8c6ea40cfeead3c4321af/reports/muxpd-followup-20261025/quality-reference-v5)、[历史 streaming 原件](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a76de7e466e5fabb98b1803a6d8a7037e17cb9a8/archive/mod-targeted-20261023/muxpd-documents/snapshot/docs/performance-tp4-goodput.md)、[异步调度报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/1da4ff27295870c697204d8e6e5ccb72dfc2b661/reports/muxpd-scheduling-20261026/)。

最终发行 Manager 与实机 Manager 的 wheel 不同，原发行报告只通过42个运行时文件建立字节关联，不能改写直接上卡的包身份。source98 的普通 worker/Manager
桥接也只覆盖其原功能范围。当前 main 或后续文档提交均不是以上实测版本的替代品。

历史 streaming 主 SLO 为 scheduled TTFT≤1 s，且整个响应的每个可观测单 token 间隔≤100 ms。三个到达
seed、交换卡组的六个窗口中，合格请求117/378→368/378，Goodput 0.162500→0.511111
requests/s（3.1453×），固定窗口吞吐约+0.4773%。TTFT 和 pooled gap P99 同时更差。每角色是两次服务启动、各启动顺序执行三个
seed，不是六次独立冷启动。该历史结果不能覆盖同步 v5 的负面参考，也不能赋予新安装包或异步候选相同收益。

## 既定质量结果保持 FAIL

两次 fresh native 均165/200，完整 token/text/termination 相同；raw/group
均166/200，彼此完整输出相同。候选在原生双对照均正确的165题中新增三道错误，同时另有四道新增正确题。冻结的逐题否决条件不能用总分相抵；严格无截断门与补充预算内答案不退化门均保持 FAIL。

候选与 native 的完整序列仅11/200相同，这与七项正确性变化是不同观察；序列差异比例不能直接解释成语义错误比例。本轮没有改变输入、gold、评分器、分母、截断规则或门槛。

## 三道丢失题：原件与首次分歧

| Case           | 原始 batch / batch 内位置（从0起） | Gold | Native1 = Native2      | Raw = Group                       | 首次不同输出（从1起） |
| -------------- | ---------------------------------- | ---- | ---------------------- | --------------------------------- | --------------------- |
| mmlu-pro-03766 | 3 / 5，peer                        | B    | B / stop / 1018 tokens | A / stop / 6396 tokens            | 66：ID77→ID16         |
| mmlu-pro-10672 | 6 / 1，peer                        | A    | A / stop / 1198 tokens | G / stop / 1046 tokens            | 17：ID19987→ID2663    |
| mmlu-pro-01741 | 14 / 3，peer                       | E    | E / stop / 377 tokens  | 无提取答案 / length / 8192 tokens | 14：ID1865→ID3234     |

两项是自然停止的错误答案，第三项是新增截断，不能全部归因于预算。相同冻结 tokenizer 对三个 token 对分别解码为
`n`→`1`、` intensity`→` values`、` real`→` property`。这些是保存的离散输出差异，旧质量运行没有保存相应 logits，尚不能判定数值机制。

新建只读审计绑定35个原始记录/producer 输入的 SHA256，逐项核对四个角色的 token/text 身份、入队映射、选中24题的最终 token 数及完整
scheduler-receipt 文件。三个 `batch-XX-cases.jsonl` 是原 case 行的原字节副本；它们是观察失败后选取的定位材料，不是新的独立准确率样本。

三组完整八请求如下，所有简写 ID 均有 `mmlu-pro-` 前缀：

- Batch3：00803（seed）、00864、03584、03644、03706、**03766**、03826、03886。
- Batch6：10611（seed）、**10672**、10734、06847、06907、06967、07028、07090。
- Batch14：01557（seed）、01619、01680、**01741**、01802、01862、01923、01984。

原协议在 seed 恰好生成第1个 token 后同时入队七个 peers，并等待整组结束后进入下一组，不能替换为性能试验的 token32 入队规则。原采样为
temperature0、seed42、max_tokens8192、ignore_eos=false、detokenize=false。

## 新定位到的调度上下文

从原同步 prefill-completion 与 D-retirement 回执重建的计数，与24题各自的完整保存长度一致。三道目标首次分歧时，原组八个 D 请求仍都在当前事务中，各调度一个
token；seed 的输出位置不同：

| 目标输出     | Native 全局 step / 退休后 seed 输出数 | Raw/group 全局 step / 退休后 seed 输出数 |
| ------------ | ------------------------------------- | ---------------------------------------- |
| 03766 第66个 | 11292 / 67                            | 10129 / 76                               |
| 10672 第17个 | 27295 / 18                            | 30212 / 27                               |
| 01741 第14个 | 69379 / 15                            | 60609 / 24                               |

统一的九 token 偏移与 q4 prefill 经十步推进、seed 同时继续 D 的路径一致；native peers 在单个 P 步完成。此处确认的是 host
调度差异，不能直接宣布它是答案错误的原因，也不能把不同完整 batch 上下文的 logits 当作同批数值等价比较。

10672 是 batch6 最早发生分歧的 peer；其余 peers 首次差异在第40个或更晚，或完整相同。seed 的首次差异为第35个，晚于这里的18/27位置。03766 同样是
batch3 最早不同的 peer，seed 完整相同。这两组适合优先观察首次分歧，仍须保留 seed 长度不同的事实。法律 batch14 的01802在第9个 token
已先行分歧，早于01741第14个，应作为关联上下文一起追踪。

## 10672 实机定位：四个有效采集与一个保留失败

本轮运行完整原 batch6 的八请求，保留8192-token采样上限与 seed-first-token
入队。目标10672到达第18个输出后，由控制器主动中止全部未完成请求并排空。最大步数160、启动后运行180秒、清理最多20步。这不是把答案预算改成18，也不是自然完成的回答；没有运行200题评分器或把主动中止结果当作质量成绩。

实际使用 observer 源码 `6c606f789450a45ef35afee398b0a3e516caa1a1`，producer 校验 f7 已有 src 文件逐一未变，诊断 observer
以 extension 加入普通 worker。所有有效采集保持原质量同步路径、TP4/EP4、BF16、HCCL=true、相同 fusion
后端及图模式，没有使用最新异步候选替代原失败对象。新控制器的实际设备记录为四张910B2，物理卡0、2、3、4；与旧轮次卡组不同。每个进程独立启动，当前设备身份只适用于本轮。

| Fresh 尝试           | Producer | 采集结论                               | 保存输出范围                               |
| -------------------- | -------- | -------------------------------------- | ------------------------------------------ |
| `native-observed-01` | v1       | PASS，原生观察器控制                   | seed19、七个 peers 各18 tokens             |
| `native-observed-02` | v1       | PASS，独立原生重复                     | seed19、七个 peers 各18 tokens             |
| `raw-observed-01`    | v1       | **FAIL，保留原件，不纳入有效数值比较** | 中止后的 worker 行槽校验误判；不是成功样本 |
| `raw-observed-02`    | v2       | PASS，原 raw 路径与观察器              | seed28、七个 peers 各18 tokens             |
| `raw-plain-01`       | v2       | PASS，同输入普通 worker 桥接           | seed28、七个 peers 各18 tokens             |

v1 的失败发生在主动中止后的检查：原生输入表保留的全 `None` 空槽列表非空，校验器把它误判为仍有请求。原始失败与非零退出保留，不修改为 PASS。v2
只修正诊断清理检查，保留空槽原始读回，同时要求每条 lane 的实际
`num_reqs=0`、请求映射为空及原有迭代状态清空；没有更改插件模型计算、采样或调度实现。独立终态审计确认全部五次已退出、shutdown
返回、无自有进程残留，设备恢复空闲；失败进程仍不作为有效样本。

四个有效采集的所有保存前缀均匹配对应角色的历史原件。Native N1/N2 的八条前缀相同，144对完整词表 `.npy` 文件字节相同，并有一致的当前 batch 上下文读回。Raw
observed/plain 的八条前缀也全部相同；plain 没有 observer 安装或 logits JSONL。该桥接只支持本次短前缀观察，不能证明 observer
对任意执行没有影响，更不能替代完整200题的答案验证。

## 第17个输出：排序已经翻转，并非采样并列最大值

目标10672前16个输出 token 相同，第17个输出再次出现 native ID19987、raw
ID2663，与历史角色前缀一致。对应捕获向量的248320个词表项全部有限，两者各有唯一最大值：

| 第17个输出的 logits      | Native N1 = N2 | Raw observed |
| ------------------------ | -------------: | -----------: |
| ID19987（` intensity`）  |         32.000 |       31.750 |
| ID2663（` values`）      |         31.750 |       31.875 |
| ID19987 − ID2663         |         +0.250 |       −0.125 |
| top1 − top2              |          0.250 |        0.125 |
| CPU argmax / 实际采样 ID |  19987 / 19987 |  2663 / 2663 |

因此这个输出点上，返回 token 符合各自 logits 的唯一 argmax，不能解释为并列最大值的随机选取。此结论只定位到采样前已有排序差异，没有证明底层数值变化的唯一来源，也不把窄
margin 当作可以接受答案退化的理由。

差异更早已存在：目标第1个输出（P 完成时）的 logits 字节已不同，虽然 argmax 相同；第2个输出进入 D 后仍不同。以下距离只作条件定位，不是同完整上下文的数值等价测试：

| 目标输出位置（从1起） | 最大绝对 logit 差 |       RMS 差 | 条件                                                 |
| --------------------- | ----------------: | -----------: | ---------------------------------------------------- |
| 1                     |          0.531250 | 0.1004021493 | 同目标 prompt；native P含seed D与7个P，raw P只含7个P |
| 2                     |          0.734375 | 0.1375783818 | 同目标已生成前缀；seed 的位置/输入不同               |
| 17                    |       0.857421875 | 0.1236860342 | 同目标前16个输出；seed 的位置/输入不同               |

首个目标 P 的 native batch 有9614个 tokens（seed D + 七个P），raw P batch 为9613个 tokens。第2和第17个输出时，两边当前 graph
descriptor 都是8 tokens/8 requests，记录的物理 KV block IDs 也相同；但采样前 seed
已生成前缀长度分别为2/11及17/26，完整输入、已计算位置和历史状态不匹配。物理块编号相同不代表 KV 或 GDN 数值内容相同。

Native/raw 的144对向量中，字节相同的仅1对，完整当前上下文匹配的也仅1对；不能把其余143对统称为“同上下文误差”。目标第18个输出的已生成前缀已经不同，分析未继续计算可比距离。完整逐项
top-k、上下文、物理块身份及比较判据保留在 `analysis-results-v1/analysis.json`，分析文件 SHA256 为
`04efe8abf94bef8840991cc49e040ff8c14056e19dded6b14a40afa3b481603a`。

本轮证据把问题推进到“P 的首个输出向量已经不同，后续同目标前缀下出现明确 argmax 翻转”。进一步定位仍需区分 P batch 形状、各请求位置、历史 KV/GDN
数值及逐层算子路径。当前证据尚不足以把差异归因于 seed 的九 token 偏移，或指定一个已验证的调度修复。

这个 fresh 单组 probe 没有重放原200题进程之前的
batches、缓存分配和暖机历史；虽然保存前缀匹配，也不等于精确重放整个原历史进程。本轮只上卡10672组，03766和01741仍只有原件追踪，没有新的 logits
实验。没有回答最终正确性结果，也没有解决01741的8192-token终止问题、获得新性能/Goodput数据或修复运行时代码。

## 后续质量门与公开归档

只有依据数值证据形成实际缓解后，才另行版本化重跑完整原200题、两次 fresh native 控制，并沿用原 strict 与 supplementary
评分/截断/逐题门槛。保留全部失败、自然停止与 length 输出。只修复某个选中前缀、原生共同正确题仍有新增错误、或总分提高，均不足以关闭 #23。

原输入副本、派生脚本、producer v1/v2、五次尝试（含原始失败）、完整 tokens/logits
与日志已发布到[本轮 benchmark Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-quality-review-20261028)。三个公开资产均通过[匿名 HTTP 200、大小及 SHA256 校验](public-download-verification-v1.json)，下载归档的全部746个原始成员通过[逐成员验证](public-member-verification-v1.json)。Git
仅保存本报告、小摘要、[manifest](MANIFEST-v1.json)、[校验和](SHA256SUMS-v1)及验证回执；原始大文件保存在 Release。

归档为 `muxpd-quality-review-20261028-evidence-v1.tar.gz`，259561638 bytes，SHA256
`e14cdfdb3401b2bb7da5d52e1feb2865f1423a1a6adf923d88c2d88ad9f8ecbe`。包含617389878 bytes原始成员，排除 compiler
cache；manifest 列出选中目录及所有五次终态。Release tag绑定 benchmark 基础提交作发布标识，不是本轮实际运行源码。

[终态审计](terminal-audit-v1.json)与归档中的
`analysis-results-v1/analysis.json`、`analysis-losses-v1/audit.json`
分别提供本轮采集和历史三题追踪。诊断校验器18项、分析器7项、归档器8项及匿名下载验证器7项CPU测试通过，测试输出保留在归档内。采集、归档和公开下载通过都不代表答案质量门通过。

异步调度 [REPORT-v2](../muxpd-scheduling-20261026/REPORT-v2.md) 曾残留 publication-pending
文字，现由[独立订正](../muxpd-scheduling-20261026/PUBLICATION-STATUS-CORRECTION-v1.md)指向当时已保存的匿名下载与212成员验证回执。原报告、实测数据与回执不改写；订正不代表新下载、新性能实验或质量通过。
