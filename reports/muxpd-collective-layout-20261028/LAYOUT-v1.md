# MuxPD Issue #23：TP4 通信布局短程对照

日期：2026-10-28。范围：`muxpd-quality-layout-20261028/runs/` 中已终态的五次实验。

**在本条冻结轨迹上，给原有 P collective 的通信载荷前面补一行零，使目标的全部 120 个已捕获层内张量，以及七个 peer 的 126 个 logits 文件，与 native
对照逐文件字节相同。** 同形状复制 `copy_only` 保持旧 raw 结果；把同一行零放在末尾的 `suffix` 没有恢复 native，目标在第 11 个生成 token
首次分叉。这支持继续检查通信布局相关因素，但还不能确定唯一根因或某种 HCCL 算法，也不是完整答案质量恢复证明。

原 200 题严格不退化门仍为 **FAIL**：两次 native 对照各为 165/200，均有 9 次 `length`；raw 为 166/200、7 次 `length`，相对 native
丢 3 题、得 4 题。总分多一题不满足逐题不退化要求。本轮没有新的完整答案评分、200 题质量资格、吞吐或 Goodput 测量，也没有发布生产修复。

## 固定输入与停止边界

模型为 Qwen3.5-35B-A3B，BF16，TP=4、EP=4，使用四张 910B2，物理编号 `0、2、3、4`。五次实验顺序执行，分别启动新进程；运行时为同步 raw-q4、隔离 P
通信组，保留原 decode graph 路径。除观察器配置外，候选的运行时配置、冻结计划及布局实现相同。

输入来自冻结 quality-reference-v5 的原始 batch 6。seed 为 `mmlu-pro-10611`，七个 peer 为
`10672、10734、06847、06907、06967、07028、07090`；每个请求均保留完整原始 token 输入。seed 先生成一个 token，再按原 admission 加入七个
peer。目标 `10672` 的 prompt 为 1,657 tokens；seed prompt 为 1,546 tokens，七个 peer 合计 9,613 个 P token 行。

采样保持 `temperature=0`、`seed=42`、`max_tokens=8192`、`ignore_eos=false`。在目标生成第 18 个 token 后主动
abort/drain；18 是诊断停止点，原 8192 生成上限没有改成 18。所有保存的生成前缀都不是完整答案评分材料。实验从 batch 6 单独启动，没有复现完整 200
题进程前五批的缓存历史。

## 改动位于原有 TP 通信载荷

先前六次层内定位实验把 native/raw 的首个已观测差异收窄到第 0 层 GDN 输出投影的 TP collective：对应 rank 的目标局部输入相同，返回值在目标位置
`800、801、1601、1602` 不同。本轮沿用其有效 v3 native/raw 对照，测试通信长度、行偏移和复制分配的影响。

布局器在原完整 scheduler admission 上冻结实际调度的 P/D token 数，再由原 lane slicing 继续执行；未调度或被推迟的 D 请求不计数。冻结值跨 P
挂起保留，直到原 `_clear_prefill` 安全释放 cohort。实际 peer admission 为 `P=9613、D=1`；这个 D=1 来自调度回执，并非根据活动请求数量推测。

| 模式        | 实际通信载荷             | 载荷形状      | P 拷回起点 | 本轮用途                  |
| ----------- | ------------------------ | ------------- | ---------- | ------------------------- |
| 原 raw      | 原 P 部分和              | `[9613,2048]` | 原张量     | 原始对照                  |
| `copy_only` | 新 scratch 中的 P 部分和 | `[9613,2048]` | 0          | 复制与分配对照            |
| `prefix`    | `[1 行零, P 部分和]`     | `[9614,2048]` | 1          | 同时改变总长度及 P 行偏移 |
| `suffix`    | `[P 部分和, 1 行零]`     | `[9614,2048]` | 0          | 增加总长度，保留 P 行起点 |

三种模式都在原生产流上分配 BF16 scratch，复制实际 P 部分和，只调用原有 collective 一次，再将 P 结果写回原输入张量，并返回**同一个原输入对象**。不新增
collective、通信组或请求，不改变 dtype、KV/attention 元数据，也不执行虚构的 D 模型计算。零行只是通信占位，不能等同于真实 native D 行的部分和。

每个 rank 的实际记录均为：seed 的 D=0 cohort 有 81 次原样 bypass，不分配 scratch、不填充、不复制；七 peer cohort 有 81
次对应模式处理调用。每次调用的路由增量均为 `caller=0、decode=0、prefill=1`，仍由原 `LaneCollectiveRouter` 选择已有的隔离 P sibling
通信组。81 是该次前向中实际被包装的 TP 调用数，不是新增调用数，也不只是第 0 层的一次捕获。

原 P 张量仍为 `[9613,2048]`。v3 观察器捕获的是布局 API 的未填充入口/返回值；真正送入 collective 的 `[9614,2048]` 形状与 P 偏移由布局回执确认。该
API 边界包含 scratch、填充、复制、原 collective 和拷回，不能把本轮此边界的全部效果单独归于 collective 内部。

即使 D 数相同，前补 D 行通常也不能保证完整 native P 顺序。native 的局部重排可能改变 P 的相对位置，例如 `[P_A,P_B,D,P_C]` 可变为
`[D,P_B,P_A,P_C]`，独立 P lane 可能仍保留 `[P_A,P_B,P_C]`。本轮验证了实际 token、position、request 顺序及 query
offsets，但布局器明确记录 `native_equivalent_order_claimed=false`。真实 D 值、通信组、全批上下文和执行历史仍不同。

## 五次尝试全部列入

| Run                       | 模式与观察器               | 原 controller / child | 独立生命周期审计 | 作用             |
| ------------------------- | -------------------------- | --------------------- | ---------------- | ---------------- |
| `layout-prefix-layer-01`  | prefix；logits + v3 层捕获 | PASS / PASS           | PASS             | 层内比较         |
| `layout-prefix-logits-01` | prefix；仅 logits          | PASS / PASS           | PASS             | 控制层观察器影响 |
| `layout-prefix-plain-01`  | prefix；无两类观察器       | PASS / PASS           | PASS             | 控制整个观察路径 |
| `layout-copy-logits-01`   | copy_only；仅 logits       | PASS / PASS           | PASS             | 复制与分配对照   |
| `layout-suffix-logits-01` | suffix；仅 logits          | PASS / PASS           | PASS             | 末尾补零对照     |

五次均完成 collection、abort/drain、shutdown 和所属进程清理；原始日志未发现使接受结论失效的退出错误。布局器三个 hook
的所有权、正常卸载和精确原对象恢复均通过。full-layer 运行先卸载其 14 个观察 hook，再验证并卸载布局器三个 hook，最后进行正常 shutdown。原件及所有尝试均保留。

五次共 20 组 rank 前后快照中，`compilation_counter` 与 Dynamo counters 均未变化；decode graph 独立计数记录
`decode_replay_calls: 0→27`、`prefill_forced_eager_calls: 1→3`。这只描述记录窗口中的计数，不能扩大为全程绝无重新编译或零额外开销的保证。

## prefix：先通过观察器桥接，再解释层张量

prefix 三种 profile 之间的检查包括相同输入、采样、源码、运行时配置、实际布局及终止边界：

| 同模式比较               | 全词表 logits       | 八条完整保存前缀 | 结论 |
| ------------------------ | ------------------- | ---------------- | ---- |
| full-layer ↔ logits-only | 144/144 字节相同    | 8/8 相同         | PASS |
| logits-only ↔ plain      | plain 不保存 logits | 8/8 相同         | PASS |

这里的“144/144”证明 prefix 观察器桥接通过，不表示 prefix 与 native 的 144 个 logits 全相同。基线输出相等是实验结果，不是判定候选运行有效的条件。

通过桥接后，prefix 与 native 的目标层内捕获为 **120/120 张量对逐值、逐文件字节相同**：四个 rank 各 30 个目标局部张量，包括 GDN 0–2 层的
`mixed_qkv/b/a/core_out`，MoE 0–3 层的输入、router logits、shared/routed 输出，以及第 0 层 TP 入口/返回值。这是 120
个张量对，不能称为 120 层或全部模型中间状态。每个张量覆盖目标完整 1,657 个 prompt 位置；原 BF16 等值扩展为磁盘 FP32 进行比较。

相对旧 raw，prefix 的首个已捕获差异仍在第 0 层 TP API 返回值：每 rank 1,534 个元素不同，位置仍为 `800、801、1601、1602`，最大绝对差
`0.00048828125`、RMS `6.779813771266654e-07`；该边界入口保持相同。prefix、native、raw 各自内部的四 rank TP
返回值分别字节相同。因此，本轮没有观察到同一次 collective 内各 rank 结果不一致。

最终 logits 的边界同样需要按请求区分：

| prefix 对照项                             | 结果                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| 七个 peer 各前 18 个 logits，对 native    | **126/126 字节相同**                                         |
| 七个 peer 保存的 18-token 前缀，对 native | **7/7 相同**                                                 |
| seed 的前 18 个 logits，对 native         | 仅首个相同，1/18                                             |
| 全八请求 logits，对 native                | 127/144 相同                                                 |
| seed 保存前缀                             | prefix/raw 各 28 tokens；native 为 19；共同前 19 tokens 相同 |

目标第 17 个 token 在旧 raw 为 `2663`，prefix/native 为 `19987`；prefix/native 第 17、18 步 logits 也字节相同。seed
的保存长度和部分 logits 仍不同，不能称为全批 native 等价或全部 144 个 logits 恢复。保存长度差发生在主动停止的短程轨迹中，不应写成完整答案长度差。

## copy_only 与 suffix：保留阴性和不同输出结果

`copy_only` 对旧 raw 的 **144/144 logits、8/8 保存前缀全部字节或 token 相同**。这说明在当前 observed 轨迹上，同形状 scratch
分配与复制本身没有改变已观测结果。它没有同模式 plain 对照，也没有层捕获；结论限于已观测数值与前缀，不能扩大为所有复制行为均无影响。

`suffix` 与 prefix 的载荷总长度相同，但 P 起点保留为 0。它对 native 仅有 1/144 logits 字节相同，对 raw 为 18/144；后者均属于 seed，七个
peer 的 logits 没有逐文件相同项。六个非目标 peer 的 18-token 前缀仍与两基线相同；seed 与 raw 相同，对 native 则只有共同前 19 tokens
相同、保存长度不同。

目标的前 10 个生成 token 与 native/raw 一致；在第 11 个 token，suffix 选择 `567`，native/raw 均选择 `1042`。第 11
步是在相同目标生成历史下作出的不同选择；suffix 对 raw 的该步最大 logits 绝对差为 `0.5`、RMS 为 `0.09220515588908351`。这时当前
target/batch/graph 元数据匹配，不证明过去 KV/GDN 数值状态相同。对 native 的全批上下文仍不匹配。

**从第 12 步起目标历史已经不同，不再计算或解释跨向量距离为相同历史下的误差传播。** 全部 144 个文件的身份比较仍保留，但文件是否相同与同历史数值归因是两个问题。suffix 没有同模式
plain 桥接，也没有层捕获；当前仅能报告 observed 输出分叉，不能指出其首个内部差异位置或宣布质量变坏。

三种布局的结果说明：在本轨迹中，复制本身保持 raw，而总长度相同的 prefix/suffix 产生不同结果，单有“长度补到 native”并不足以恢复已观测 native 输出。prefix
组合了总长度和行偏移变化；缓冲区、真实通信实现、整批输入及历史尚未完全隔离。零行布局实验不能识别唯一 HCCL 规约算法，也不能推断硬件故障或证明普遍的 native 顺序等价。

## CPU 复核与证据身份

冻结主分析为 `COMPLETE`、`errors=[]`；该状态表示证据处理完成，不是质量门或所有模式观察器桥接均通过。

| 独立 CPU 复核目录             | 直接复核范围                                                                                             | 结果                         |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `prefix-complete-audit-v1`    | 三 profile 的 144+8 / plain8 桥；360 个不同层 NPY；重算对 native/raw 共 240 个张量对                     | PASS；对 native 120/120 相同 |
| `copy-only-terminal-audit-v1` | 具名终态 copy_only/raw；source/input、布局、生命周期、144 logits 与 8 前缀                               | PASS；全部保持 raw           |
| `suffix-terminal-audit-v1`    | 具名终态 suffix/native/raw；432 个 logits NPY 的身份/shape/dtype、布局、生命周期及前缀；重新计算目标比较 | PASS；与冻结分析完全一致     |

这些复核没有修改原 run 或冻结分析。suffix 审计单独调用经过 SHA 验证的冻结 CPU helpers，并非另一套独立实现的数值算法；未读取层张量，也未访问后续
`quality-runs/` 子产物。所有控制身份和读入文件 SHA 保留在各审计的 `source-identities.json` 与 `SHA256SUMS` 中。

运行时身份沿用前置层定位证据：vLLM-HUST 为 `d0f22d2bda562156e4dbf433ce645e1769b4f804`，vllm-ascend-HUST 为
`03766ac696fde5ab1980d80ca0b8543d3580c989`；原质量运行时为
`f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`，既有诊断扩展为 `6c606f789450a45ef35afee398b0a3e516caa1a1`。source
receipts 核对启动前后稳定及两基线一致，并保留共同的 Ascend `csrc/build_aclnn.sh` 本地 build 差异，不能将工作树写成完全无修改。

| 文件（相对 campaign 根目录）                | SHA256                                                             |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `analysis-results-v1/analysis.json`         | `beae9b1878593576fc61b116cdb30da8d688df046785ac8dc0e07675a7baaa53` |
| `prefix-complete-audit-v1/summary.json`     | `9f04c9bbb7ed0d8acbe270ca832f3e370187fa33a7efffccaadbde1db83afbf3` |
| `copy-only-terminal-audit-v1/summary.json`  | `993194a727836ecf4838df55e6b2e43da80a43bc625875d742c96b7f33f38029` |
| `suffix-terminal-audit-v1/summary.json`     | `d1c17ef31a0c66d523bdf1a04859434b6092a4481cfe0b705c4b832249762361` |
| `layout-v1/prefill_collective_layout_v1.py` | `6431c4c0abdd869ef17e7122f681c457ad2b4d5741f86c6a0653e9fca44e02fc` |
| `probe-v1/freeze-manifest-v1.json`          | `72da2fc9309ba027cee425d1d58a755fceb259ffda6d2f4daffe778d66193554` |

前置基线位于 `muxpd-quality-layer-localization-20261028/runs/layer-native-v3-01` 与
`layer-raw-v3-01`；其六次尝试、历史失败退出和观察器桥接在前置 `publication-draft/REPORT-v1.md`
中单独说明。本报告的五次布局实验不能替代或隐藏那些前置证据。

## 发布与后续资格边界

主归档包含 879 个原始成员及内嵌 manifest，736,535,964 字节，SHA256
`6b1ce3564093c0bcc4eb9f0d0a3147d8cb332032a19d42f3e72e42fb6d487e04`；完整本地成员校验通过。独立 suffix
审计完成于主包快照之后，作为独立补充包保留，48,767 字节，SHA256
`c1187a0c145d1ab7874deeeb96b8b4172d25f1c35c27b84943a0c42dcc48b567`。见[主清单](MANIFEST-v1.json)、[补充包清单](SUPPLEMENT-v1.json)、[本地校验](local-member-verification-v1.json)。

发布入口为
[evidence-muxpd-collective-layout-20261028](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-collective-layout-20261028)。`PUBLIC_READBACK_INCOMPLETE`：上传已完成，但匿名大文件下载未完成，不能把本地校验当作远端可用性证明。前置
native/raw v3
对照来自[层定位 Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-layer-localization-20261028)，与[层定位报告](../muxpd-layer-localization-20261028/LAYER-LOCALIZATION-v1.md)共同复核。

后续需要为仍缺失的模式补齐同模式 plain 桥接；若要解释 suffix 内部差异，还需要相应层捕获及其桥接。任何候选都必须独立验证完整答案、严格不退化门、生命周期与性能。布局器增加的
scratch、清零、双向 D2D 复制和通信载荷可能改变时延或重叠，本轮没有测量这些成本。后续完整答案实验的结果不在本报告证据范围内；Issue #23 的质量门继续为 **FAIL**。

公开匿名读回已保留为
**INCOMPLETE**，具体已验证的小附件/区段、失败和未完成范围见[公开读回状态](public-readback-status-v1.json)。本地完整校验、上传完成和公开完整下载核验是不同结论，未将前两者写成第三者通过。

## 后续完整质量结果（2026-10-28）

以上为各自冻结的短程诊断结果。其后完成的[原200题 prefix 复测](../muxpd-prefix-quality-20261028/FULL200-v1.md)为167/200，恢复旧三题，但新增07338截断且总截断11题，原逐题不退化门仍
**FAIL**；[后验24题](../muxpd-prefix-quality-20261028/POSTERIOR24-v1.md)的改善不替代这一失败。两项后续质量小证据包的公开读回状态由该集合单独记录，不改变本集合大证据包的
INCOMPLETE。
