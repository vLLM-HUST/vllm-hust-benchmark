# vLLM-HUST 运行时 MOD 研究总报告

Fletcher · 共同阅读版 · 2026-10-27

## 1. 当前结论与证据地图

本报告汇集本仓库已有评审意见、源码研究与独立实验，不新增实验、不替作者发布结论。目录盘点覆盖24项；不是24项都完成了源码审计或设备验收。BetterScale与BidKV按Fletcher要求跳过。

**vSpec在匹配投机长度后未观察到明显性能优势。** 该MOD尝试通过自适应调整投机长度，在接受收益不足时减少草稿计算成本。最新35B/B16三轮56题对照匹配有效gamma4及对应target
FULL graph buckets，原生MTP与vSpec的输出吞吐中位数分别为811.05、795.60
tokens/s，当前证据不支持为该负载引入此MOD。被测配置实际没有发生gamma切换，不泛化为自适应思路本身无效。原280题得分下降保留为尚未可靠归因的历史观察，不再作为已证明算法损害精度的结论。

**DiffSpec性能收益不合格，需要重新评估项目可行性。**
有效数据是统一B16/C04的三臂56题首轮，无抢占；默认检索计数为0。无有效的完整三臂280题结果。Fletcher已决定停止追加实验，不能用坏掉的原生Eagle链路充当算法反证。

| 方向                                 | 当前判断                                         | 证据性质与边界                                            |
| ------------------------------------ | ------------------------------------------------ | --------------------------------------------------------- |
| vSpec / 35B MTP                      | 相同有效投机长度下无明显性能优势，当前不支持引入 | 新增gamma4三轮56题对照；实际无gamma切换，非完整自适应消融 |
| DiffSpec / 27B Eagle                 | 性能收益不合格，需重新评估项目可行性             | 我们的56题首轮；检索未生效，完整280题未完成               |
| OPset                                | 融合不合格：比ACLNN增时延48.37%–82.31%           | 两卡算子实测；去清零另作有用小补丁                        |
| Prefix Router                        | 有需求，值得做；优先联接分布式KV存储             | 方向判断与源码研究，不是生产验收                          |
| StateAxis异步KV传输                  | 原理正确，需适配35B-A3B及缓存压力验证            | 方向认可；35B适配与新workload尚未验证                     |
| StateAxis自动前缀缓存                | 有潜力；需高效异步recurrent state checkpointing  | 特定数据集前缀复用潜力，不能干扰计算                      |
| StateAxis流水权重加载                | 实用，建议进入系统主线并默认启用                 | 采用建议；历史收益不全归因于流水                          |
| 其余有损KV、同步搬运、重算与调度路线 | 按具体实现作不采用或低优先级判断                 | 见评审原文；不扩大成所有未来实现的否定                    |

### 阅读顺序

- 第2章以两列表格汇总各MOD的当前正式评价，历史对话记录不收入报告。
- 第3章仅保留vSpec匹配gamma4的关键实验与结论；历史对照、源码差异和排查过程留在仓库研究笔记。
- 第4章给出DiffSpec有效短对照与失败边界；第5章是OPset独立算子复现。
- 第6章合并StateAxis三条数据路径；第7章保留24项官网目录索引。

### 证据口径

**官网声明、源码观察、作者历史实验、我们的评审判断、我们的独立实测**是不同层级。本报告保留原始限定词，不把“measured /
validated”自动解释为收益或生产可用。章节中的原始日期、历史“未重测”等表述均对应当时状态；当前状态以本章及第3、4章为准。

完整源码调用链仍保留在各专题principles.md；本报告通过可点击链接连接它们，不将全部日志和逐题输出灌入正文。本地证据链接需与本仓库一起使用，外部来源链接可独立打开。

## 2. MOD正式评审意见

来源：[专题原文](comments.md)。

以下为截至2026-10-27的正式评价，由 Fletcher
代表。评价针对本轮审阅实现及当前部署需求；实测、历史证据与待验证方向的边界见各专题。BetterScale、BidKV按要求跳过，不作技术评价。

| MOD                                | 正式评价                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SimLLM                             | 匹配规则不足以可靠区分事实性差异，无法保障精确信息的保真性；不适用于要求准确性保证的任务。不将此判断扩展到严格相同前缀缓存。[依据](simllm/principles.md)                                                                                                                                                              |
| Pipeline Microbatch                | PP不是本轮目标部署的主流路线，专项微批调度缺乏优先投入依据，不继续研究或复现。DeepInfer的大规模部署采用EP/DP；其“不采用TP”的明确说明仅作架构背景，不等同于对PP的否证。[机制](pipeline-microbatch/principles.md)；见表后引用。                                                                                         |
| KV Materialization Arrival Control | 已有可复用缓存且存在在线流量或其他活跃任务时，主动重算的计算机会成本过高。应优先改良缓存容量、分层存储与复用路径，避免通过重算处理缓存问题。[SGLang HiCache说明](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)                                                                                               |
| KV Tiering                         | 当前Ascend同步设备搬运路径不可取。同步复制与device synchronization阻塞执行并干扰计算，不满足生产采用要求；应改为可与计算重叠的异步路径。此处为实现与采用判断，非本轮干扰幅度实测。[依据](kv-tiering/principles.md)                                                                                                    |
| Ascend KVCompress                  | 打分及数据搬运引入额外开销，同时牺牲精确full-attention语义；既有结果收益有限，当前投入产出比不足。[依据](kvcompress/principles.md)                                                                                                                                                                                    |
| MuxPD                              | 当前吞吐不合格。按层暂停增加中间激活保存及调度开销；优先采用mixed batching、chunked prefill，并在显存与延迟预算允许时扩大batch。[依据](muxpd/principles.md)                                                                                                                                                           |
| OPset                              | 融合不合格：相对普通ACLNN分开执行，时延增加48.37%–82.31%。删除多余清零有用，但宜作为主线小补丁，不宜独立作为MOD。[两卡实测](opset/reproduction/README.md)                                                                                                                                                             |
| Core Attention Boundary            | 缺少有效benchmark，尚无充分证据证明CPU元数据边界替换带来实质性能收益，优化价值未得到验证。[依据](core-attention-boundary/principles.md)                                                                                                                                                                               |
| GDN State Codec                    | 参考BetterScale既有LeapQuant实验，在Ascend上未获得稳定收益；AIC/AIV协作与数据搬运成本是主要限制，不重复投入同类研究。本轮未重新复核既有实验。[依据](gdn-state-codec/principles.md)                                                                                                                                    |
| Prefix Router                      | 缓存感知路由有明确需求，值得推进。应结合分布式KV存储，将缓存可达性、传输成本与节点负载纳入决策，优先复用而非重算；当前版本并未完成这一系统能力。[依据](prefix-router/principles.md)                                                                                                                                   |
| PyramidKV                          | 当前整体性价比不足。KV容量下降未转化为整体收益，额外打分、搬运及精度代价抵消其价值；已读配置长输入吞吐下降2.6%–10.3%。[依据](pyramidkv/principles.md)                                                                                                                                                                 |
| Quantized KV Cache                 | 不推荐依赖独立量化、反量化及稠密中间结果物化的路径，其额外GM往返可能抵消收益。优先考虑产出与消费保持量化状态的实现；融合online antiquant需另证净收益，不与显式还原混同。[依据](quantized-kv/principles.md)                                                                                                            |
| vSpec（35B）                       | 通过自适应调整投机解码长度，尝试在接受收益不足时减少草稿计算成本；但在35B/B16、相同有效投机长度gamma4及匹配target FULL graph buckets的三轮对照中，未观察到相对原生MTP的明显性能优势，当前证据不支持引入该MOD。被测配置未发生gamma切换，结论不泛化为自适应方法本身无效。[实测](speculative-mmlu-pro/gamma4/RESULTS.md) |
| DiffSpec                           | 性能收益不合格，需要重新评估项目可行性。结论限于B16/C04三臂56题首轮，检索未生效，完整三臂280题未完成；不构成完整方法的否证。[实测](speculative-mmlu-pro/results-diffspec.md)                                                                                                                                          |
| StateAxis Async KV Transfer        | 原理正确；需适配Qwen3.5-35B-A3B，并选择合适的缓存压力workload进一步验证。[依据](stateaxis-data-paths/principles.md)                                                                                                                                                                                                   |
| StateAxis Automatic Prefix Cache   | 有潜力，主要来自recurrent state checkpointing对特定数据集前缀匹配的支持。需高效异步实现，确保状态保存与搬运不干扰计算；目前为待验证方向。[依据](stateaxis-data-paths/principles.md)                                                                                                                                   |
| StateAxis Pipelined Weight Loading | 实用，建议推广到系统默认主线，做成default-in（默认启用）。历史收益包含校验实现等共同优化，不全部归因于传输流水。[依据](stateaxis-data-paths/principles.md)                                                                                                                                                            |

### DeepInfer部署引用

DeepInfer官方部署Q&A原文：“TP and CP are not used for now.” 其decode配置为EP64–128 /
DP64–128。此引用说明大规模部署可以采用EP/DP路线，但不直接证明PP在所有推理场景中无价值，也不构成对行业采用比例的统计。[官方原文](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)（核对日期：2026-10-27）

## 3. vSpec：35B关键实验

**评语：相同投机长度下无明显性能优势，不建议引入。**

vSpec通过自适应调整投机解码长度，尝试在接受收益不足时减少草稿计算成本。我们用固定相同投机长度的原生MTP作基线，检验它能否带来额外吞吐收益。

### 关键对照：原生MTP4 vs vSpec

同一Qwen3.5-35B-A3B模型，Ascend双卡TP2、BF16、B16；冻结56道MMLU-Pro题目，每臂独立运行三轮，输出上限4000
tokens。两臂有效投机长度均为gamma4，并匹配对应的target FULL graph buckets；不放宽接受条件。

| 轮次   | 原生MTP4（tokens/s） | vSpec（tokens/s） | vSpec相对原生 |
| ------ | -------------------: | ----------------: | ------------: |
| 1      |               825.71 |            797.27 |        -3.44% |
| 2      |               811.05 |            794.91 |        -1.99% |
| 3      |               796.26 |            795.60 |        -0.08% |
| 中位数 |               811.05 |            795.60 |        -1.91% |

### 结论

**匹配有效投机长度后，三轮均未测到vSpec的额外吞吐收益。** 先前相对较短投机长度基线的优势，不能作为该MOD优于匹配原生MTP的证据。这组35B/B16负载下，引入该MOD缺乏性能依据。

本轮vSpec实际未发生gamma切换；结论针对当前实现与该负载，不是否定自适应方法本身。其余runtime差异与共享宿主干扰仍存在，约1.9%的差距不作统计显著减速认定。

实验配置、原始结果与复算入口见[完整实验记录](speculative-mmlu-pro/gamma4/RESULTS.md)；历史实验及源码差异保留在[研究笔记](speculative-mmlu-pro/results-qwen35.md)。

来源：[专题原文](speculative-mmlu-pro/report-summary-qwen35.md)。

## 4. DiffSpec：有效证据与停止边界

来源：[专题原文](speculative-mmlu-pro/results-diffspec.md)。

2026-10-27，ascend87。**这不是完成的三臂280题实验，也不是对DiffSpec完整检索算法的否证。** 35B/vSpec的完整主实验另见
[results-qwen35.md](speculative-mmlu-pro/results-qwen35.md)。

### 评语

**性能收益不合格，需要重新评估项目可行性。**

此评语依据当前有效运行点的性能表现；保留下文样本量、检索未生效及未完成范围的限定，不将其扩大为对完整方法的否证，也不自动追加实验。

### 有效的短对照：B16、固定56题、每臂一轮

在原大缓存路径出现退化后，三臂统一使用memory0.4（C04）、TP4、B16、相同固定perf56、temperature0、max_tokens4000。原生Eagle与DiffSpec均使用gamma3。每臂新engine，按原生→Eagle→DiffSpec串行运行，计时排除加载/编译/预热。三臂均无连续token0退化、无解析失败、**抢占重算增量0**。

| 臂                   |  正确 | 截断 | 输出tokens | generation秒 | 输出tokens/s |
| -------------------- | ----: | ---: | ---------: | -----------: | -----------: |
| 原生 target-only     | 44/56 |   14 |     86,748 |      373.716 |        232.1 |
| 原生 Eagle3          | 46/56 |    9 |     73,219 |      766.069 |         95.6 |
| DiffSpec auto/linear | 46/56 |   12 |     84,884 |      929.285 |         91.3 |

原生Eagle接受24,661/145,623（约16.9%）；DiffSpec接受25,927/176,796（约14.7%）。单轮输出吞吐分别约为原生的41%和39%。DiffSpec相对Eagle的wall更长，但输出也更多；按输出吞吐仅低约4.4%，**没有重复实验，不将其声称为稳定的4.4%回退**。

此运行点没有显示加速，但不能由56题的44/46/46判定精度优劣；各臂截断和输出长度不同，样本小。也不泛化到其他target/draft配对、长上下文负载或充分优化的部署。当前目标为本地Qwen3.8-27B、草稿为作者指定的Qwen3.5-27B-EAGLE3-v2；不把目标权重的本地名称当作已核验的上游不可变revision。

### 实际运行的DiffSpec机制边界

四rank witness确认实际使用`AscendDiffSpecEagleProposer`及其cache，不是插件名注册了、实际仍跑plain
Eagle。可是当前默认auto解析为linear，线性路径没有proposed
tree，`proposed_token_ids is None`被传作`is_prefill=True`，使检索入口提前返回；未修改算法门禁以制造另一个方法。执行原源码方法体的CPU测试和设备计数共同支持此限制。

本轮warm/post witness的`retrieval_count=0`、`retrieval_cycle=0`、`selected_slots=0`；不声称完整attention-score
retrieval生效。滚动尾部cache路径仍可能工作，不能将“没有检索”扩大成“没有任何cache操作”。因此这组数据评价的是**当前实际执行的默认artifact**，不是完整DiffSpec检索思路的上限。

### 失败证据与修订，不能混进有效算法比较

1. **原memory0.7/B32**：native
   Eagle和DiffSpec短检查共享19/56题连续`!`（token0）退化。原生Eagle280题94/280、145解析失败的结果受污染，排除算法精度推断。原生target-only的235/280不能与这些坏输出组成有效对照。
1. eager、B4、async都未改变那19题。反转题序后坏题跟着位置变化；取消14题warmup后只坏最后5题，均从累计第52个请求开始。raw target
   logits检测到NaN；严格greedy采样审计未发现与参考不符。不要把warmup后的request编号直接当测试行号。
1. **memory0.4绕行**：65个单层cache
   group共享大buffer；较小预算的实际num_blocks=5508。56×16及56×128检查不再退化。大缓存寻址/状态边界是待定位方向，未证实具体算子、整数溢出或修复根因。共享vLLM/Ascend源码未改。
1. **C04/B32质量轮**：原生238/280；Eagle233/280、44截断、0解析失败，但Eagle最终计数器显示**1488次抢占重算**，generation3348.201秒。普通日志没出现preempt文字不代表没有抢占。这个耗时不能全归因于算法。随后DiffSpec
   B32质量轮在初始化期被停止，未产生有效280题结果。
1. 因此新短性能对照统一改B16，并加完成后抢占计数门。上表三臂均通过，不与B32质量轮混成同一配置。

所有停止仅作用于经argv/PGID核对的本任务进程组；错误日志、停止receipt、已完成及部分输出保留。诊断wrapper曾出现tuple接口和RPC序列化错误，已单独记录，不能作为被测算法故障。

### 投资与当前状态

- 已完成：上述无抢占的B16三臂56题首轮、兼容/容量诊断与默认检索门禁审计。
- 未完成：DiffSpec有效三臂280题、B16性能第2/3轮、全量12,032题。
- 完整B16三臂280题按本轮generation外推约2.9小时，另有加载/预热开销，实际可能变化。后两轮性能预计另约一至两小时。**Fletcher于2026-10-27明确决定停止追加实验、整理现有证据。**
  本轮到此收口，不再跑上述280题或第2/3轮；只有获得新的明确授权才恢复。

### 证据

- [协议及修订](speculative-mmlu-pro/plan.md)、[过程、错误和纠正](speculative-mmlu-pro/progress.md)、[结构化汇总](speculative-mmlu-pro/evidence/screen-analysis.json)。
- `evidence/screen-performance-c04b16-*-r1-{config,result,exit}.json`：上表原始三臂配置、逐题输出、完整指标及退出状态。
- `evidence/screen-performance-c04b16-diffspec-r1-*-witness.json`：运行机制与检索计数。
- `evidence/diag-native-eagle-*.json`、对应logs：容量/排序/预热/采样诊断；这些同步诊断耗时不作性能证据。
- `test_diffspec_linear_gate.py`：直接执行固定源码方法体，验证默认线性检索门禁。

## 5. OPset：两卡独立算子复现

来源：[专题原文](opset/reproduction/README.md)。

2026-10-26。验收基线为 **普通 ACLNN 分开执行（F.linear + npu_swiglu）**，不是原生 Triton。

**不合格：融合算子相对 ACLNN 时延增加 48.37%–82.31%。**

当前独立算子实测不支持以该融合实现替代普通 ACLNN 路径。没有复现端到端 +5.24%，也没有证明取消了 GM 往返。删除多余清零仍是有用的小补丁，与融合验收分开判断。

### 条件与结果

物理卡2、7分别独立测量；各卡三轮配对、每轮每项50次同步墙钟计时，10次预热排除JIT；BF16，K3584、N18944，seed20261026。使用未修改 OPset
kernel（source `7d3cf948da548220fb2e0a32bf2aff80b5647dc4`），验收基线为普通 ACLNN
`F.linear + npu_swiglu`。另加载宿主 `linear_persistent` 加 `npu_swiglu` 作为 Triton 背景对照，不作为采用判定基线。这是 eager
同步 microbenchmark，包含host launch开销，非Graph replay或HTTP性能。

每格为三轮均值平均，单位ms：

| 物理卡 |   M | ACLNN基线（ms） | 融合（ms） | 时延增加（不合格） |
| ------ | --: | --------------: | ---------: | -----------------: |
| 2      |   1 |          0.2333 |     0.4253 |            +82.31% |
| 2      |  32 |          0.2485 |     0.4432 |            +78.37% |
| 2      | 128 |          0.3246 |     0.4828 |            +48.71% |
| 7      |   1 |          0.2380 |     0.4233 |            +77.87% |
| 7      |  32 |          0.2700 |     0.4618 |            +71.01% |
| 7      | 128 |          0.3298 |     0.4893 |            +48.37% |

时延增加 = (融合耗时 / ACLNN耗时 - 1) ×
100%，按原始未舍入三轮均值计算，表内毫秒值仅用于展示。两卡、三个shape全部慢于ACLNN。卡7部分轮次有波动，原始50样本和SD完整保留，不筛掉慢轮。

背景对照：融合相对指定原生Triton路径降低约6.8%–14.0%时延，但这是较弱性能对照，不能作为相对ACLNN的验收收益；原始Triton数值保留在证据中。

所有基线/候选在三个shape、两卡上通过事前 FP32 reference 的 finite 与 `rtol=.02, atol=.125`
门槛；误差指标在JSON。这不证明任务质量，也没验证跨batch不变性。普通ACLNN不自动满足batch-invariant语义要求，不能以它更快就宣称可无条件替代该语义合同。

### 物理GM证据

实测生成的kernel元数据包含非零workspace（4096/131072/524288 bytes），源码单kernel显然不能当成“没有workspace”。

额外将M32的缓存ttadapter在同机CPU离线lowering，保存 `evidence/fused-m32-offline.hivm.mlir`：

- 195–207行：两份32×128 FP32 tile经 `fixpipe_nz2nd_float_to_float_4d_to_2d` 从 `cc` 写到 `gm` workspace。
- 257/270行：Vector执行 `sync_block_wait`。
- 264/277行：`load_gm_to_ubuf_1d_float` 从GM读入UB，随后执行激活与乘法。

**这份离线lowering明确仍经过GM，不是寄存器直连SwiGLU。**
未做设备时间线分析，因此没有证明有效Cube/Vector流水或收益来源。离线命令基于同一ttadapter和公开compiler选项；其object与实测缓存object二进制比较不相同，故不冒称此IR是实测binary的精确反汇编。

离线调用（无设备计算）：

```sh
bishengir-compile <M32-cache>/_linear_swiglu_kernel.ttadapter \
 --target=Ascend910B2 --enable-auto-multi-buffer=true \
 --enable-auto-bind-sub-block=true --enable-hfusion-compile=true \
 --enable-hivm-compile=true --enable-triton-kernel-compile=true \
 --save-temps=<task>/offline-ir -o <task>/offline-ir/kernel
```

### 资格、资源及失败披露

原始checker失败：Core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` / Ascend
`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8` 不匹配作者锁定版本。Torch2.10.0 / torch-npu2.10.0.post4 /
Triton-Ascend3.2.2，CANN目录9.1.0。没有修共享源码或全局包，没有绕过checker声称插件集成通过；独立算子迁移结果不替代端到端验收。

首次错误地把物理编号当runtime index，进程894046实际进入物理5，另一进程因非法device
index失败。主动终止894046，首次所有数据作废并保留日志。Fletcher强调4/5用于压力测试后，后续仅使用2/7。runtime映射为逻辑0/1/2/3→物理2/4/5/7；修正入口逐次查询并核验PID落卡，正式结果的placement文件提供证据。

正式进程均exit0，已释放物理2/7的三类协作锁；结束检查两卡无进程，4/5压力测试进程仍运行。没有停止其他人的任务。远端只保留 任务私有工作目录（精确路径见专题原文）
的源码、venv、cache和证据，无后台服务。

### 决策意义

- **普通 ACLNN 是本次性能验收基线；融合时延增加48.37%–82.31%，不合格。** 对较弱Triton对照的改善不能替代这一结论。
- 也不应说“融合所以省掉GM”：离线lowering反而清楚展示GM workspace及同步。
- 当前实现不是普通ACLNN路径的性能替代品；若确实要求batch-invariant Triton，局部优化仍可能有用。
- 去多余清零的机制判断不变，本轮没有另外重测其独立贡献。当前native Triton baseline仍含zeros，因此融合对比也不是剥离清零贡献后的纯epilogue消融。

[预注册及映射修正](opset/reproduction/contract.md) · [探针](opset/reproduction/probe.py) ·
[资源与映射门禁](opset/reproduction/launch.py) · [原始证据](opset/reproduction/evidence)

## 6. StateAxis：三条数据路径

来源：[专题原文](stateaxis-data-paths/principles.md)。

2026-10-26，只读独立仓库与历史报告；未运行设备。三个项目均依赖StateAxis宿主，不是普通vLLM安装即得完整数据面的独立实现。以下评语由 Fletcher
于2026-10-27给出；属于方向与采用建议，不代表已完成新增适配或验证。

### 源码身份

- async-kv-transfer：`03143f99a5e9718f93310af8228ea349e4f45897`
- automatic-prefix-cache：`92327b78ab2d8aa717544ede3621b0f109df31fd`
- pipelined-weight-loading：`c28890ec48ed69dad9e980be3e4e6d10422465d2`

### 1. Async KV Transfer

**评语：原理正确；需适配 Qwen3.5-35B-A3B，并选择合适的缓存压力 workload 进一步验证。**

意图是host-tier命中后异步H2D恢复KV，并让其他设备工作推进，不是跨节点分布式KV传输。

独立仓库的Python主要是固定配置与Manifest，Rust `TransferCoordinator`负责：

- 按state_id、generation、snapshot SHA身份合并重复恢复；相同state/generation却不同snapshot拒绝；
- unique work和correlation分别上限64；
- begin_next发lease，commit/abort校验身份；StateAxis激活合同max_retries=0；
- 不包含ACL stream、DMA或snapshot内存分配。其commit只是协调器完成记账，不自行证明设备已经完成。

真正device stream、snapshot内存、数据发布/rollback由宿主负责。不能仅凭该Rust队列就称已证明异步DMA，也不能因其没有DMA就否认历史宿主实现。

历史R5 Qwen2.5-14B BF16、单910B2，control/candidate各3 fresh processes：host-tier-hit p50
592.597→333.716ms（−43.686%）；并发pair
makespan607.648→355.473ms；3/3候选观测到设备overlap，HBM持平。是限定组件工作负载的正结果，非通用serving吞吐；当前拆分版未继承资格，新的deterministic-worker生命周期测试不执行NPU。

### 2. Automatic Prefix Cache

**评语：有潜力，主要来自 recurrent state checkpointing；需高效异步实现，不能干扰计算。**

在具备可复用前缀的特定数据集上，recurrent state checkpointing 有望支持前缀匹配与状态复用。这里指待实现和验证的潜力，不是当前索引已提供任意前缀 checkpoint
的结论；关键是高效的异步 checkpointing，避免状态保存与搬运干扰主计算。

以内容找到可复用的已有StateAxis resident prefix state，避免调用者显式携带state身份。不是Prefix Router那样挑实例，也不自带device cache搬运。

Rust实现
`PrefixCacheIndex<V>`：namespace/salt隔离，128-token块哈希后构建root；按已有条目的长度由长到短匹配请求前缀，命中后再核对完整token序列防碰撞；宿主is_resident确认state仍有效，更新LRU。索引返回泛型state值与reused_tokens，实际复用由宿主完成。

这是已登记前缀状态条目的索引，不是自动为任意共同token前缀创建可恢复checkpoint。容量满时只淘汰未leased条目，全被占用就拒绝插入。Rust库可参数化，但当前Python准入合同固定max_entries=2/block_tokens=128，是很窄的实验配置。

历史3+3 closure：token exact、lifecycle clean，吞吐开销0.0256%；8次插入、6次淘汰，**零cache
hits**。只证明该工作负载下插入/淘汰无明显伤害，没有验证命中路径的加速。独立版未获性能资格。

### 3. Pipelined Weight Loading

**评语：很实用的小技术，建议推广到系统默认主线，做成 default-in（默认启用）。**

优化worker启动到Ready，不是运行时逐层offload/换入权重，也不是decode加速。

宿主方案是两个64MiB pinned staging
slot：读文件/增量校验下一chunk时，上一chunk异步H2D；复用slot前等event，全部pack身份/数据验证通过才事务性Ready。固定plan digest、packed
root、geometry和execution artifact一致；无ECPA授权则保留serial_pageable_sync。独立仓库没有loader内核，主要发布配置/授权合同，真正
`ResolveWeightLoadPlan/PinnedWeightLoader` 内置于StateAxis host/worker。

历史Qwen2.5-14B、同一29.54GB 48+1pack，两种page-cache条件各3+3：

| process-to-Ready中位数 |  control |           candidate |
| ---------------------- | -------: | ------------------: |
| cold                   | 344.168s | 53.895s（−84.340%） |
| warm                   | 324.809s | 36.392s（−88.796%） |

**不是纯流水消融**：control是scalar SHA + pageable + sync H2D；candidate同时改成ARM SHA2/OpenSSL SHA256 +
pinned双buffer + async
H2D。不能将84–89%全部归因于DMA重叠。两边都校验实际载入全部字节；cold/warm以私有inode和mincore验证，无全局drop-caches。TPOT门槛通过，说明该测试启动优化未伤害稳态，不代表稳态加速。

另有MiniMax layer0 EP8 1+1 loader点估计−54.748%，不是整模型serving；更新Qwen3.5 TP2 pre-sharded
checkpoint诊断Ready337.70→301.62s（−10.68%），但没有ECPA激活、未控制cache状态，不能混为同一个机制/资格。

### 综合判断

三者分别处理“选中可复用状态”“将host状态恢复到设备”“启动时加载权重”，需求有区别。前两者可在宿主层组合，但此轮未证明独立包已打通组合。

异步KV恢复和权重加载有历史特定场景正向结果；自动前缀缓存历史验收零命中，收益证据缺失。都需区分历史宿主数据面、拆分库/合同与当前端到端资格。没有证据支持把这三个统称为已生产验证的性能MOD。

### 来源

- [async源码](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer/blob/03143f99a5e9718f93310af8228ea349e4f45897/native/src/lib.rs)
- [APC源码](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache/blob/92327b78ab2d8aa717544ede3621b0f109df31fd/native/src/lib.rs)
- [weight合同](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading/blob/c28890ec48ed69dad9e980be3e4e6d10422465d2/src/stateaxis_pipelined_weight_loading/__init__.py)
- [async历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-async-kv-transfer/snapshot/README.md)
- [APC历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-automatic-prefix-cache/snapshot/README.md)
- [weight详细历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/evidence/ISSUE15_WEIGHT_LOADER_RESULT.md)

## 7. 官网24项目录索引

来源：[专题原文](runtime-mods.md)。

观察日期：2026-10-26。以下是官网目录事实，不是我们的源码审计或复现实验结论。

### 范围与来源

- [工坊页面](https://vllm-hust.sage.org.ai/plugins.html)
- [组件 registry](https://vllm-hust.sage.org.ai/data/ecosystem.json?v=state-scheduler-archive-20261018)
  与
  [taxonomy](https://vllm-hust.sage.org.ai/data/mod-taxonomy.json?v=state-scheduler-archive-20261018)
- 按 `public_surface != false` 且 `taxonomy.kind == runtime_mod` 筛选，共 **24 个组件**；本次这些组件的项目 ID 互不重复。
- 不把 connector、外部 KV 服务、工具、控制面和平台 profile 自动纳入。运行时 MOD 是网站分类，真实部署边界仍须核对。
- [结构化摘录](catalog/2026-10-26-runtime-mods.json) 保留接入面、版本、平台和原始证据入口。URL 查询参数不是内容版本保证，摘录代表此次读取。

### 完整索引

| MOD / 源码入口                                                                                                  | 目录中的职责                  | 效果标签（原值）                | 证据入口                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ADM](https://github.com/vLLM-HUST/ascend-distributed-metadata)                                                 | `dp_metadata_sync`            | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-ascend-distributed-metadata/snapshot/qualifications/qwen3moe-dp4-int64-word-20260928.json)                   |
| [BidKV](https://github.com/vLLM-HUST/vllm-hust-bidkv)                                                           | `scheduler_policy`            | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-bidkv/snapshot/results/qwen35-evoscientist-20260929/README.md#high-concurrency-kv-pressure-result) |
| [DiffSpec](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec)                                              | `decoding_mechanism`          | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-ascend-hust-diffspec/snapshot/docs/evidence/sage-mate-20260904-tp4-graph.md)                            |
| [vSpec](https://github.com/vLLM-HUST/vllm-hust-vSpec)                                                           | `decoding_mechanism`          | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-vSpec/snapshot/docs/gsm8k_b128_recovery_and_arc_easy_20260913.md)                                  |
| [LatchMoE](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE)                                              | `operator_runtime`            | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/f57dbc2b1020c6f50b7a0a36db43fcd8a0bd0198/reports/vllm-hust-midterm-20261019/b1-mod-source-audit-20261020.json)                                                                                |
| [OpFuse](https://github.com/vLLM-HUST/vllm-hust-opfuse)                                                         | `operator_fusion`             | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-opfuse)                                                                                                                                                                                                      |
| [Ascend KV Compression](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust)                               | `kv_state_manager`            | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website/snapshot/docs/FRONTIER-QWEN35-KVCOMPRESS.md)                                               |
| [Prefix Router](https://github.com/vLLM-HUST/vllm-hust-prefix-router)                                           | `request_router`              | `beneficial`                    | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/f57dbc2b1020c6f50b7a0a36db43fcd8a0bd0198/website-publication/data/leaderboard_prefix_routing_mean_evidence.json)                                                                              |
| [KV Tiering](https://github.com/vLLM-HUST/vllm-hust-kv-tiering)                                                 | `kv_residency_provider`       | `inconclusive`                  | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website-extra/snapshot/docs/FRONTIER-KV-TIERING-20260926.md)                                       |
| [KNorm](https://github.com/vLLM-HUST/vllm-hust-knorm)                                                           | `kv_compression_method`       | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-knorm)                                                                                                                                                                                                       |
| [PyramidKV Ascend](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust)                                     | `kv_compression_provider`     | `validated`                     | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/0dd2d1293e8d0f2e23c1e3c5bfe0a38e588a890c/archive/mod-git-20261022/pyramidkv-website-source/snapshot/evidence/current/2026-10-09-paired-evaluation)                                            |
| [Quantized KV Cache](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust)                          | `kv_quantization_research`    | `validated`                     | [原始材料](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/tree/ae7b44bfee695d0ba580040dffd10eb802104485#%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%8E%AF%E5%A2%83)                                                                                       |
| [SimLLM](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust)                                                  | `kv_reuse_policy`             | `beneficial`                    | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-ascend-simllm-hust/snapshot/benchmarks/results/qwen35_latency_20261008/README.md)                       |
| [Split-Batch / Full-Graph Parallel](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust)                  | `graph_execution_policy`      | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust)                                                                                                                                                                                          |
| [Pipeline Microbatch](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch)                               | `pipeline_scheduling_policy`  | `inconclusive`                  | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website/snapshot/docs/FRONTIER-QWEN35-PIPELINE-K8S.md)                                             |
| [KV Materialization Arrival Control](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control) | `kv_materialization_policy`   | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website/snapshot/docs/FRONTIER-QWEN35-KV-MATERIALIZATION.md)                                       |
| [BetterScale](https://github.com/vLLM-HUST/BetterScale)                                                         | `graph_execution_policy`      | `measured`                      | [原始材料](https://vllm-hust.sage.org.ai/betterscale.html#service-throughput)                                                                                                                                                                                  |
| [StateAxis Automatic Prefix Cache](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache)               | `prefix_cache_policy`         | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-automatic-prefix-cache/snapshot/evidence/ISSUE95_BOUNDED_APC_CLOSURE_20260815.md)                                 |
| [StateAxis Async KV Transfer](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer)                         | `kv_transfer_overlap`         | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-async-kv-transfer/snapshot/evidence/issue76-async-transfer-20260815-r5/result.json)                               |
| [StateAxis Pipelined Weight Loading](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading)           | `model_weight_loading`        | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/evidence/qwen35-loader-sharded-qualification-20261012/RESULT.json)              |
| [GDN State Codec](https://github.com/vLLM-HUST/gdn-state-codec)                                                 | `recurrent_state_compression` | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/evidence/qwen35-agentx-v020-expanded-20261009/ROUND-RESULTS.md)                                    |
| [OPset](https://github.com/vLLM-HUST/vllm-hust-opset)                                                           | `operator_runtime`            | `beneficial`                    | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/linear-swiglu-graph.md)                                                         |
| [Core Attention Boundary](https://github.com/xmdhb/vllm-hust-core-attention-boundary)                           | `attention_execution_policy`  | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website-extra/snapshot/docs/FRONTIER-QWEN35-OP02-CORE-ATTENTION-BOUNDARY.md)                       |
| [MuxPD](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust)                                                    | `graph_execution_policy`      | `validated`                     | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/19081ff39a25c3c8ac821ae7885688ea24a03e90/reports/muxpd-ecpa-20261025/RELEASE-QUALIFICATION.md)                                                                                                |

### 读表规则

- `measured` 只表示有测量；不等于正收益。`validated` 也不能自行解释为性能验收。
- `beneficial` 和 `not-beneficial-in-tested-cell` 均须绑定具体模型、版本、硬件和 workload；不可外推到整个插件。
- `preview`、`inconclusive` 是重要边界，不应被兼容性标签覆盖。
- taxonomy evidence、compatibility status 与 public effect 是不同维度；有差异时保留原值，沿证据链接核实，不取最乐观者。
