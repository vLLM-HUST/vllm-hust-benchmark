# MuxPD Issue #23：层内数值定位与 TP 规约边界复核

日期：2026-10-28。范围：`muxpd-quality-layer-localization-20261028` 已终态的六次实验。

**当前进展是把一条已复现的数值差异定位到实际 TP collective 边界，尚未修复答案退化。** 在目标 `mmlu-pro-10672` 的第 0 层，四个 TP rank 各自对应的
native/raw 局部规约输入逐值、逐文件字节一致；原有规约返回的目标行却在位置 `800、801、1601、1602` 出现差异。每个执行方式内部，四 rank
的返回值彼此一致。这不是本次观测到的同一次 collective 内 rank 结果不一致，也不能唯一归因于某个 HCCL 或硬件算法缺陷。

Issue #23 原有答案不退化门槛继续为 **FAIL**。本轮没有重跑完整 200 题，没有新的准确率、吞吐或 Goodput 结果，也没有发布生产运行时修复。此前特定场景的 Goodput
证据并未被本轮重新验证或扩大适用范围。

## 实验范围与对照

模型为 Qwen3.5-35B-A3B，BF16，TP=4、EP=4，使用当前服务器的四张 910B2（物理编号 `0、2、3、4`），各次实验顺序运行、独立进程启动。

取冻结 quality-reference-v5 的原始 batch 6，保留八个请求及其完整 token 输入：seed 为 `mmlu-pro-10611`，七个 peer 为
`10672、10734、06847、06907、06967、07028、07090`。目标 `10672` 的 prompt 长度为 1,657 tokens。seed 先生成一个
token，再按原 admission 加入七个 peer。

采样参数保持 `temperature=0`、`seed=42`、`max_tokens=8192`、`ignore_eos=false`；在目标生成第 18 个 token 后主动
abort/drain。18 是诊断停止边界，不是把质量测试的生成预算缩短为 18。该流程从 batch 6 单独启动，前五批的缓存历史并未复现，因此不构成完整质量复跑。

在被采集的首次 peer-P forward：

| 项目                                  | native 对照                               | MuxPD raw-q4 对照                       |
| ------------------------------------- | ----------------------------------------- | --------------------------------------- |
| 实际 token 布局                       | 1 个 D token + 7 个完整 P 请求            | 7 个完整 P 请求                         |
| P 前向总 token 行数                   | 9,614                                     | 9,613                                   |
| 目标 prompt 在 packed tensor 中的起点 | 1                                         | 0                                       |
| P 执行                                | 原有 compiled 路径，`skip_compiled=false` | 独立 raw P，`skip_compiled=true`        |
| 观测时 P graph runtime mode           | `NONE`                                    | `NONE`                                  |
| TP collective 通信组                  | 原有 `tp:0`                               | 原有隔离 P sibling `tp:muxpd_prefill:0` |

所有差异按目标 prompt 的零基位置对齐。相同目标位置不等于完整批次、其他请求行、执行流或通信组相同。

## 六次尝试全部保留

| Run                  | 仪器版本 | 原 controller / child 声明 | 生命周期审计                                   | 用途                                                     |
| -------------------- | -------- | -------------------------- | ---------------------------------------------- | -------------------------------------------------------- |
| `layer-native-01`    | v1       | PASS / PASS                | PASS                                           | 有效的历史 native 记录；不与无效 raw-v1 配对作正式层比较 |
| `layer-raw-01`       | v1       | PASS / PASS                | **FAIL**：四个 rank 的 shutdown ownership 错误 | 原样保留，排除出正式数值归因                             |
| `layer-native-v2-01` | v2       | PASS / PASS                | PASS；13 个 hook 正常卸载                      | v2 正式 native 对照                                      |
| `layer-raw-v2-01`    | v2       | PASS / PASS                | PASS；13 个 hook 正常卸载                      | v2 正式 raw 对照                                         |
| `layer-native-v3-01` | v3       | PASS / PASS                | PASS；14 个 hook 正常卸载                      | v3 正式 native 对照                                      |
| `layer-raw-v3-01`    | v3       | PASS / PASS                | PASS；14 个 hook 正常卸载                      | v3 正式 raw 对照                                         |

v1 raw 的采样和张量收集已结束，但退出时，诊断器留在 P owner 的 `_model_forward` wrapper
没有恢复；`DecodeGraphController.close()` 的所有权检查因此报错。四个 Traceback 未传递为 controller/child 的失败状态，所以其原始 JSON
仍写着 PASS。独立 `lifecycle-audit-v1/audit.json` 明确覆盖该接受结论为 FAIL；没有重写原始 controller、child、日志或数组。

v2 增加独立卸载回执：先检查所有当前 slot 仍由诊断器持有，再逆序恢复安装前的精确对象；原本没有实例 slot 的方法用 `delattr` 恢复类方法。它恢复 decode graph
controller 原来的 wrapper 对象后，才进行正常 shutdown。v3 保留这一机制，并覆盖新增的 TP group 实例 hook。有效 v2/v3
均通过原始日志、shutdown 返回、空闲/所属进程清理和恢复回执检查。

## 观察器先通过同角色桥接，再比较内部张量

v2、v3 每个角色都与早先独立、未安装层观察器的同角色控制比较：native 使用 `native-observed-02`，raw 使用 `raw-observed-02`。

| 正式运行  | 完整 vocabulary logits | 八条保存的生成前缀 | 源码与输入核对 |
| --------- | ---------------------- | ------------------ | -------------- |
| native-v2 | 144/144 文件逐字节相同 | 8/8 相同           | PASS           |
| raw-v2    | 144/144 文件逐字节相同 | 8/8 相同           | PASS           |
| native-v3 | 144/144 文件逐字节相同 | 8/8 相同           | PASS           |
| raw-v3    | 144/144 文件逐字节相同 | 8/8 相同           | PASS           |

这表示观察到的是各自既有行为中的差异，不能读成 native 与 raw 的答案相同。每个角色仍复现其原有输出。桥接还检查采样输入、原始计划字段、完整 source receipts
以及启动前后源码一致性。

层观察器在 warmup/graph capture 之后安装，位于现有 opaque custom op 内动态解析的实例方法。它没有替换
`DecoderLayer.forward`、编译图或注册算子的数学实现。输入在原调用前克隆，输出在原调用后克隆；原函数调用一次、原结果对象不变。克隆在生产流中排队并记录事件，原 P 采样记账和既有
logits 观察器返回后，才同步事件并 D2H 写盘。

v2 每 rank 保存 28 个张量：GDN 0–2 层的 `mixed_qkv/b/a/core_out`，MoE 0–3 层的输入、真实 router logits、shared/routed
输出。MoE 两个输出尚未经过外层相加和最终 TP reduction，不是完整 decoder-layer 输出；本轮没有记录实际选中的 expert IDs。v3 额外保存第 0 层 TP
collective 的输入和输出，共 30 个张量。

每个张量保存目标完整 1,657 行的 rank-local shard、原 dtype/shape/offset/流及 SHA。磁盘数组使用 float32：BF16/FP16 是精确扩展，原
FP32 不降精度。记录没有增加跨 rank all-gather 或 reduction。

两个版本各自的八组 rank/role 前后快照中，`compilation_counter` 与 Dynamo counters 均没有记录变化。raw 的
`decode_replay_calls` 为 `0→27`、`prefill_forced_eager_calls` 为
`1→3`，这是独立执行计数。计数快照与输出桥接不能被扩大为“全程绝无重新编译”的证明。

## 从层间区间收窄到真实 TP 边界

v2 首先确认：第 0 层 GDN 的三个输入和 core 输出在四个 rank 的对应 native/raw 比较中全部字节相同；首次已观测差异是随后 MoE 输入。v3
在两者之间，进一步观测输出投影所用的现有 TP collective。

新增 hook 只作用于原全局 TP coordinator 的 `_all_reduce_out_place`，由现有 `vllm.all_reduce` opaque op 动态进入。窗口从
GDN0 core 返回后开始，在 MoE0 入口关闭，只接受一次对应 TP 调用。实际 communicator 原地修改输入，因此输入 clone 明确排在原调用之前，输出 clone
排在返回之后。raw 的现有 lane router 继续选择独立 P 通信组；四 rank 均记录调用增量 `caller=0、decode=0、prefill=1`，没有额外
collective。

| 已观测边界                             | 每个 rank 的 native/raw 比较        | 受影响的目标 prompt 行 | 最大绝对差    |
| -------------------------------------- | ----------------------------------- | ---------------------- | ------------- |
| GDN0 `mixed_qkv`、`b`、`a`、`core_out` | 4 个键 × 4 ranks 全部逐文件字节相同 | 无                     | 0             |
| GDN0 输出投影的 TP 输入                | 4/4 对逐值、逐文件字节相同          | 无                     | 0             |
| 同一次 TP 调用的返回输出               | 每 rank 1,534 个元素不同            | 800、801、1601、1602   | 0.00048828125 |
| MoE0 输入                              | 每 rank 1,190 个元素不同            | 同上四行               | 0.03125       |
| MoE0 router logits                     | 每 rank 51 个元素不同；其输入已不同 | 同上四行               | 见逐张量指标  |
| GDN1 core 输出；MoE1–3 输入            | 差异传播到后缀                      | 800–1656，共 857 行    | 见逐张量指标  |

TP 输出 RMS 差为 `6.779813771266654e-07`；MoE0 输入 RMS 差为 `4.396053435374927e-05`。全部已捕获张量在目标位置 `0–799`
没有数值差异。

TP 输入、输出的原始完整形状分别为 native `[9614,2048]`、raw `[9613,2048]`，dtype 均为 BF16，原 collective 均返回原输入对象。四个
rank 的**对应局部输入跨执行方式相同**，不表示各 rank 的局部输入互相相同。规约后则是 native 内四 rank 输出彼此字节相同，raw 内四 rank
也彼此字节相同；两个角色各自六种 rank 配对，合计 12 对，均直接验证。

四个位置对应 prompt token IDs `8、198、220、23`，tokenizer pieces 为右括号、换行、空格和 `8`。这些只用于定位输入，不是答案判分依据。

当前可公开的结论是：**在这条复现轨迹的目标行上，差异在既有 TP collective 的真实输入到返回值之间出现，并在后续层传播。** 完整张量形状、其他行、通信组和 stream
treatment 仍不同；尚未隔离总长度、行偏移、缓冲区、通信策略或执行历史中的哪项因素。四个相同差异位置可以提出布局假设，但不能直接证明某种分段算法有缺陷。后续 GDN
输入已经变化，后缀传播也不能反推 GDN kernel 损坏；MoE/router 输入已经变化，不能用输出差异反推路由实现错误。

## 两次独立 CPU 复核

正式分析 v2、v3 均为 `COMPLETE`，`errors=[]`。独立审计没有修改原件，并直接核对层数组的 SHA、shape、dtype、有限值、位置和差异：

| 审计                            | 直接核验 NPY 文件 | 重新计算 native/raw 张量对 | 与冻结分析的元素数、行、最大差、RMS |
| ------------------------------- | ----------------- | -------------------------- | ----------------------------------- |
| `final-layer-audit-v1`，对应 v2 | 224               | 112                        | 全部一致                            |
| `final-tp-audit-v1`，对应 v3    | 240               | 120                        | 全部一致                            |

独立审计还核对生命周期和恢复回执、source receipts、前后编译计数，并直接比较每角色八条保存前缀。144 个最终 logits 的逐字节相同来自冻结桥接
producer；上述独立脚本重算的是层 NPY，没有把它描述成又一次独立重算所有 logits。

## 版本与证据身份

| 项目             | 固定身份                                                                             |
| ---------------- | ------------------------------------------------------------------------------------ |
| vLLM-HUST        | `d0f22d2bda562156e4dbf433ce645e1769b4f804`                                           |
| vllm-ascend-HUST | `03766ac696fde5ab1980d80ca0b8543d3580c989`，加两角色共同记录的本地 build-script 差异 |
| 原质量运行时     | `f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`                                           |
| 既有数值诊断扩展 | `6c606f789450a45ef35afee398b0a3e516caa1a1`                                           |

source receipts 保留 Ascend 的 `csrc/build_aclnn.sh` 本地修改：本模型构建跳过未使用的 `compressor`
项。它在两角色及启动前后相同，不能把该工作树写成完全无修改。`6c` 对 `f7` 已有的 20 个 runtime source
文件逐字节保持一致，新增的是诊断代码、脚本、文档和测试。这些身份不自动赋予其他已安装 wheel、Manager 或后续异步分支相同的质量/性能资格。

下列 SHA256 对应当前冻结原件；路径相对本 campaign 证据根目录：

| 文件                                     | SHA256                                                             |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `analysis-results-v2/analysis.json`      | `0799a8c3d759e1c506c7d6892b631e8a629f46536f01bb7898050e75c22ed5f8` |
| `analysis-results-v3/analysis.json`      | `cb33bf0eb1cf86ba56fde704052a9e3e26ed1b72eddad3bdb06efb54b3a56f7d` |
| `final-layer-audit-v1/summary.json`      | `b32045b2e48ea2ff5698dbb8b97228a1613e006579fc8f8aec1a6f508b99d327` |
| `final-tp-audit-v1/summary.json`         | `ee830940b023763b08872ed1df7ec789c01771a4d9f60c4c6b40ba54f19bc3bc` |
| `instrumentation-v2/layer_capture_v2.py` | `fa1fa70c8b49d91e381005815a15d2d99c7c8f04a1672868372c9de8d74c3aa7` |
| `instrumentation-v3/layer_capture_v3.py` | `c234ad088f665778bac39581d2ccd1e86ca23cc4f33273965da59c12a32df6dc` |

复核入口为各版本冻结的 `analysis-command-*.json`、producer freeze manifest、独立审计脚本和 `SHA256SUMS`；应保留所有六次原始 run 的
controller、child、日志和数组。早先同角色控制来自 `muxpd-quality-review-20261028`，属于显式前置证据，不能因本轮报告只讨论六次 layer runs
而省略其来源。

## 发布状态与下一步边界

原件已在本地形成可完整重建的归档，包含 1,815 个原文件和五个发布元数据成员，全部 1,820 个成员的大小与 SHA 校验通过。完整 gzip 为 3,346,139,157 字节，SHA256
为 `60a0154a71e49f8ad8263d24c58e9fcdc396dd25293efd2602b0e583d92b583a`；按顺序分为 1,800,000,000 和
1,546,139,157 字节两片。清单及组装方法见 [归档回执](package-receipt-v1.json)、[原件清单](MANIFEST-v1.json)和
[本地成员校验](local-member-verification-v1.json)。

发布入口为
[evidence-muxpd-layer-localization-20261028](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-layer-localization-20261028)。`PUBLIC_READBACK_INCOMPLETE`：上传已完成，匿名大文件下载因网络故障未完成。旧同角色对照仍需
[前一轮质量复核 Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-quality-review-20261028)；它们的准确身份见
[外部依赖](external-dependencies-v1.json)。

后续可分别控制通信总长度、P 行偏移和复制分配的影响。任何候选若改变 raw 输出，应重新建立同一候选的 plain/observed 对照，再验证完整质量与生命周期；不能把“候选与旧 raw
不同”自动当成失败，也不能把单条轨迹接近 native 当成 200 题答案已恢复。本报告不包含任何后续布局实验结果，且保持 Issue #23 质量门 **FAIL**。

公开匿名读回已保留为
**INCOMPLETE**，具体已验证的小附件/区段、失败和未完成范围见[公开读回状态](public-readback-status-v1.json)。本地完整校验、上传完成和公开完整下载核验是不同结论，未将前两者写成第三者通过。

## 后续完整质量结果（2026-10-28）

以上为各自冻结的短程诊断结果。其后完成的[原200题 prefix 复测](../muxpd-prefix-quality-20261028/FULL200-v1.md)为167/200，恢复旧三题，但新增07338截断且总截断11题，原逐题不退化门仍
**FAIL**；[后验24题](../muxpd-prefix-quality-20261028/POSTERIOR24-v1.md)的改善不替代这一失败。两项后续质量小证据包的公开读回状态由该集合单独记录，不改变本集合大证据包的
INCOMPLETE。
