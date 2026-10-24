# TP8 持续 decode 与长 prefill 争用实验

本轮研究长 prompt 到达时，已经输出 token 的请求会停顿多久，以及缩短停顿
需要付出多少 prefill 延迟和吞吐代价。此前的
[TP8 有限批次比较](performance-tp8.md) 没有建立稳定吞吐收益；只比较所有
请求结束后的总 tokens/s，也不足以判断生成过程中的明显停顿。

**结果：guard v2 在三种长 P 争用负载中，将 D 请求最大事件间隔的 P95
从原生 16K 的 385–398 ms 降到 66–68 ms；全部六轮同时满足 D 100 ms /
P TTFT 1 s 的预登记观察目标。代价是吞吐比原生 16K 低 1.96%–3.54%，
P TTFT P95 增加 22.62%–28.49%。**
这是固定模型与负载下的延迟取舍，不是吞吐全面加速、硬 SLO 保证或完整 MuxWise 移植。

## 与原论文和公开实现的对应关系

本轮依据正式 [MuxWise 论文 v3，ASPLOS 2026](https://arxiv.org/html/2504.14489v3)，
不使用早期名为 Drift 的版本作为机制说明。论文结合片内计算资源分区、争用
时延预测和调度，优先满足 decode，并按 decode 执行时间匹配 prefill 的层数。
其 SLO goodput 实验与本轮固定请求队列的测量口径不同；论文收益不能直接迁移。
[论文 §3–4](https://arxiv.org/html/2504.14489v3#S3)

公开 SGLang 参考代码固定为 `slo_config` 分支提交
`eeac5148f456428a05823c330e1d40afb0ef2438`。其中
[multiplexing.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/multiplexing.py#L129)
按 `split_forward_token_budget / extend_num_tokens` 决定一次推进的 P 层数；
之后独立同步 D 并处理结果，通过 P event 和跨 rank 完成确认再合并 P。
这条实际代码路径与论文完整预测器应分别核对，不能因仓库有相应模块就认定
所有论文策略已经接入。本文对比的是上述固定版本，不声称覆盖作者全部分支。

| 机制 | 当前 Ascend 插件的实际范围 |
| --- | --- |
| P/D 独立调用模型 | 已有独立 runner、输入状态及 P 跨轮保留；完整 prompt 准入，不把 P token chunk 与 D 混在一次模型 batch 中 |
| D graph / P 执行 | 已有独立 D ACL graph；P 强制 graph mode `NONE`。满足完整新 P 条件时可选择 FX 编译路径；需要中途暂停的 P 使用逐层路径 |
| 空间资源分区 | 未移植 CUDA Green Context 对应的空间分区；本轮 graph 路径为 `fenced`，不能据此声称 P/D kernel 同时执行 |
| 时延预测 | 未移植论文的空间分区预测器和离线争用模型；现有主机观测只能用于本机调度启发式，不构成最坏时延证明 |
| 通信隔离 | 已有实验性 eager overlap 通信隔离路径，但不是本轮 fenced graph 候选的执行方式 |
| KV / 请求生命周期 | 共享模型与 KV 底层资源，P 未完成时保留所有权，取消需等待 worker ACK；此语义不能因时延优化而放松 |
| 多轮 prefix 复用 | 当前验证配置关闭 prefix caching；不能评价论文涉及的跨请求复用收益 |
| P 抢占 | 当前只有一个未完成 P cohort，未实现任意挂起 P 的多级抢占或层间批次重组 |

实现入口为 [decoupled scheduler](../src/vllm_muxpd/decoupled_scheduler.py)、
[decoupled worker](../src/vllm_muxpd/decoupled_worker.py) 和
[decode graph controller](../src/vllm_muxpd/decode_graph.py)。因此，本轮候选是
**面向 Ascend fenced 执行的时延预算适配**，不是完整 MuxWise 的等价移植。

## 修改动机与待验证假设

对照插件固定在 `7cb5088137a0d46a39408a9f034f5261a9307519`。其 work-token
配额和 TTFT 启发式可以增加一次执行的 P 层数，但原有
`decode_slo_feasible_without_prefill` 只报告单独 D 的估计可行性，没有把
`tpot_slo_ms` 转成 P 工作上限。完整新 P 也可能通过 P-first 暂缓已有 D。
这些选择有利于减少 P 的分轮开销，未必适合持续生成时的短停顿目标。

在 fenced 路径中，D forward、P 片段和采样存在同步依赖。较长 P 片段可能
推迟本轮 D 的客户端可见结果，即使多轮平均 TPOT 较小，单次输出间隔仍可能
超标。因此本轮分别测量完整请求的最大事件间隔、长 P 到达附近的完整相邻
间隔，以及 P TTFT；不把平均 TPOT 当作最大停顿的替代指标。

候选 `decode_budget_guard` 的审查目标是：有活跃 D 时，让 P 层数受 D 剩余
预算约束；允许该约束覆盖 work-token 下限和 TTFT 导致的扩大片段；未知或
不可行时保留最小进度，并显式记录预算冲突。P-first、完整 token budget
挤掉本轮 D、无 D 时尽快完成 P、取消/暂停及迟到 ACK，都必须使用一致的
活跃请求和完成定义。仅检查“本轮是否选中 D”不足以判定不存在需要保护的 D。

候选实现将 raw P 首片的一层固定成本与续片的每层成本分开，P 按原始
token 长度 multiset 区分，D 按 batch 数量、总 context 和最大 context
的桶区分。优先使用已有采样同步之后可确认完成的 lane span，取 host
submit 与 device elapsed 的较大者；另保留完整 host round trip 的回退
估计。两类来源不混为一个累计最大值，避免冷启动的整个 step 成本一直
污染后来的局部层成本。没有相应形状观测时从一层开始。

首片一次推进多层的观测单独保留，不直接当作首层固定成本；完整编译 P
或完整 raw forward 都不训练 raw 切片估计。完整 FX forward 的平均层
成本不能直接代表 raw 切片，40 层中 GDN 与 full attention 的成本也不必
相同。已有观测最大值不是未来最坏值，lane span 还不能覆盖所有跨 rank
残余等待、网络和客户端延迟，因此这些规则不承诺 100 ms 硬上限。

独立源码审查已核对 P-first/full-budget 的活跃 D 保护、decode debt 的
抢占/完成清理、控制轮零进度、取消后不训练估计，以及校验在生命周期
更新之前执行；未发现需要阻止后续实机验证的问题。trace 校验覆盖层进度、
完成回执与所报预算的算术一致性，不宣称完整重放了预测器的历史样本缓存。
实际冻结源码、版本区别与实机通过范围如下。

[独立源码来源审计 (archive member: `evidence/2026-10-11-tp8-contention/source-provenance-audit.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
逐文件核对旧插件 initial/clean 的 12 个运行文件与 `7cb5088` 快照，以及
guard v2 initial/confirm 的 13 个运行文件与 `258aef6`，同时检查原生
argv 选择器、请求计划和 producer；没有混用不同实现的计时轮次。

guard v1 固定在 `f673277`，其计时阶段为 `guard-initial`，另用
`guard-diagnostic` 采集生命周期和估计器证据。v1 的
[trace 校验 (archive member: `evidence/2026-10-11-tp8-contention/guard-diagnostic-check.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 通过
1046 条层进度回执，覆盖完整 16383-token 准入；
[独立生命周期审计 (archive member: `evidence/2026-10-11-tp8-contention/guard-diagnostic-lifecycle-proof.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
记录完成与取消 ACK，诊断自然语言对照也达到
[16/16 文本精确一致 (archive member: `evidence/2026-10-11-tp8-contention/guard-diagnostic-correctness.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。
这些是 v1 的通过范围，不代替 v2 的验证。

v1 的 [估计器审计 (archive member: `evidence/2026-10-11-tp8-contention/guard-v1-estimator-audit.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
发现首次独立 D 的约 67 ms 观测会持续影响该形状的最大值估计，而后续稳定
独立 D 约为 18–21 ms，合格续片的非 P 残差约为 18–26 ms；不能将这个
差额直接归因于上一轮 P 的设备尾部。v2 固定为
`258aef6ce52b7fd77cfe9a7f70d1ea29a13f7d41`，保留独立 D 观测，另学习
已完成 fenced raw P 续片的 `host_step_ms - device_elapsed_ms`。只使用
非首片、非末片、实际包含 D、计时有效且残差为正的样本，并按实际 D 形状
保存；pending P 优先使用匹配残差，新 P/P-first 仍使用独立 D 估计。
残差包含 D、RPC、其他 host 工作与跨 rank 尾部等待，不是纯 D kernel
时间或硬上界，也不通过删除首次较慢观测来改善统计。

v2 对应独立的 `guard-v2-initial` / `guard-v2-confirm` 阶段，不能与 v1
的三个 repeats 混成同一候选。上述修改由 v1 的完整结果和独立 trace
促成，属于探索后的新实现；下文最终比较仅合并 v2 自身的六轮测量。

预期取舍同样需要记录：更小 P 片段可能改善 D 停顿，却增加主机调度次数，
推迟 P 首 token，降低吞吐；如果 D 单独执行已超预算、最小一层 P 仍过慢，
或 CPU/通信出现额外抖动，保护逻辑也可能无法满足目标。结果必须同时给出
D 和 P 两侧指标，不能只挑选改善的一侧。

## 环境与预登记对照

本轮初始快照为
[environment-initial.json (archive member: `evidence/2026-10-11-tp8-contention/environment-initial.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，
记录时间为 2026-10-11 04:24:32 UTC：八张 Ascend 910B2 初始无 NPU 任务，
CPU 配额为 32 核，内存上限为 256 GiB。模型解析路径为
`/data/shared_upload/models/vspec/Qwen3.5-35B-A3B`。这是本轮启动前的观察，
不是测量期间设备空闲的声明。模型软链修复、metadata 对照与未完整计算
权重 SHA 的边界沿用 [TP8 模型来源说明](performance-tp8.md#本轮环境与模型来源)。
本轮只做同一 TP8 环境中的配置对照，不据此推导纯 TP4→TP8 扩展效率。

共用 Qwen3.5-35B-A3B BF16、TP8 + EP8、`max_model_len=16384`、
`max_num_seqs=8`、同步调度、关闭 prefix cache、graph capture sizes
`[1,2,4,8]`。实际启动参数、软件 pin、加载源码 SHA 和是否成功 ready 应由
每个服务的 manifest / 日志确认，不能由计划文件代替运行证据。

对照组与选择规则登记于
[experiment-plan-v2.json (archive member: `evidence/2026-10-11-tp8-contention/experiment-plan-v2.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。
[原计划 (archive member: `evidence/2026-10-11-tp8-contention/experiment-plan.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 保留；v2 在首次
正式计时前将 P 到达时间设为所有 D 首 token 之后 0.2 s，以便分开观察 D
初始准入与注入长 P 的影响。该修改不改变脚本默认值，所有对照通过 CLI
一致设置 `--prefill-start-delay 0.2`，不得把不同延迟的轮次混为同一配置。

| 对照角色 | Token budget | 与本轮问题有关的区别 |
| --- | ---: | --- |
| `native16k` | 16384 | 原生 chunked-prefill / 混合模型 batch 路径，较大的 chunk budget |
| `native2k` | 2048 | 相同最大模型长度与 graph 的较小 chunk 对照，允许原生策略保护 D 时延 |
| `old-work256` | 16384 | 已合并 `7cb5088` 的插件配置，固定快照运行 |
| `guard` | 16384 | 加入 D 预算保护，目标 D 100 ms、P TTFT 1000 ms；v1 与 v2 独立测量，最终四角色比较选择 v2 |

插件要求完整 prompt 准入，因此必须保留足够大的 token budget；原生可用
2048 chunk 处理相同长 prompt。这是显式实验变量，不把原生也强制使用大
chunk 后就声称战胜其全部延迟配置。若探索阶段修改 guard 参数，需使用
独立标签保留每一轮，最终候选仍需重复和前后原生对照确认。

候选第一次启动曾因安装绘图依赖导致 NumPy 1.26.4 被本地 2.5.4 暂时遮蔽，
在服务 ready 和任何客户端请求之前被发现并停止。没有该次启动的计时样本，
也没有删除较差测量；六份原始启动/停止材料已改名保存为
`guard-startup-aborted-*`，由
[中止回执 (archive member: `evidence/2026-10-11-tp8-contention/guard-startup-aborted-receipt.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
记录原名与 SHA。绘图依赖已移至独立 `/root/workspace/.plot-venv`，
[runtime-after-rollback.json (archive member: `evidence/2026-10-11-tp8-contention/runtime-after-rollback.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
记录原 NumPy 路径与 torch/torch-npu 恢复验证；候选随后从新进程重新启动。
这个协议偏离应与最终结果一起保留，不把未发请求的启动中止算作失败计时轮。

确认顺序也发生了有记录的修订：
[首次顺序调整 (archive member: `evidence/2026-10-11-tp8-contention/confirmation-order-amendment.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
把 v1 trace 诊断提前，随后
[并行工作调整 (archive member: `evidence/2026-10-11-tp8-contention/confirmation-parallel-work-amendment.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
安排固定旧版本服务与 v2 开发交错进行。实际 `old-work256-confirm` 的服务
于 05:22:57.419935 UTC ready，CPU 回归测试日志最后写入于
05:23:14.510878 UTC，与首个客户端阶段重叠。
[排除决定 (archive member: `evidence/2026-10-11-tp8-contention/confirmation-cpu-overlap.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
根据这些时间戳、在查看性能指标之前作出：保留原四种负载的全部结果，
待 CPU 测试结束后在同一个固定旧版本服务上完整重跑四种负载，命名为
`old-work256-confirm-clean`。最终旧插件对照使用 initial 与 clean 两阶段；
受影响阶段以明确原因排除，不从其中挑选未受影响或较好的单独轮次。

## 显式启用 TP8 decode 预算保护

以下示例在本仓库根目录执行，使用本机已经配置好的
`/root/workspace/runtime/env.sh`、模型路径及八卡 TP8 + EP8。它复用现有
[graph 启动脚本](../scripts/serve-qwen35-decode-graph.sh)，显式替换完整 policy
JSON；不能只传 `decode_budget_guard` 一个键并假定其余选项会与脚本合并。
本轮实际 producer 是
[guard-v2-initial-launch.sh (archive member: `evidence/2026-10-11-tp8-contention/guard-v2-initial-launch.sh`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
的直接 Python argv，下面的配置与之等价，并把默认 100 / 1000 ms 目标写明。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MUXPD_TP_SIZE=8
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/serve-qwen35-decode-graph.sh \
  --batch-admission-policy-config '{
    "execution":"decoupled",
    "stream_mode":"fenced",
    "layers_per_slice":8,
    "adaptive_layers":true,
    "prefill_first":true,
    "decode_graph":true,
    "compile_full_prefill":true,
    "prefill_layer_token_budget":262144,
    "decode_budget_guard":true,
    "tpot_slo_ms":100,
    "ttft_slo_ms":1000
  }'
```

guard 默认仍为 `false`，现有启动脚本的默认行为未改变。v2 估计器要求本轮
`258aef6` 或包含该实现的后续版本；旧版本不能因同名参数而视作同一实验。
此示例使用独立 D graph、完整 P 准入及 fenced 层调度，未启用设备资源分区
或 P/D kernel overlap。100 ms 是调度目标，不是部署后的硬时延保证。

## 请求负载和计时方式

[benchmark_contention.py](../scripts/benchmark_contention.py) 使用固定 seed 42
生成合成 token IDs；temperature 0、`ignore_eos=true`、
`return_token_ids=true`、`add_special_tokens=false`。所有 prompt IDs、参数、
payload/plan SHA 和 producer SHA 写入报告。它检验传输与 token 数量，
不是自然语言正确性或跨执行配置输出等价的证明。

当前冻结 producer SHA-256 为
`6f373ff7383ccc61579d4e9406ee0c9b9523ee8a4ef0d5330d76471e421f6712`。
若后续修改脚本，必须区分 producer 与计划版本，不能将已有结果静默归入
新实现。脚本的 34 项 CPU 测试覆盖失败保存、SSE 审计、到达门槛和间隔
边界；这只是客户端验证，不是候选 NPU 运行已通过。

| 工作负载 | 初始 D 请求 | 注入 P 请求 | P 到达方式 |
| --- | --- | --- | --- |
| `decode-control` | 4 × 128 输入 / 256 输出 | 0 | 无 P 对照 |
| `p8k` | 同上 | 1 × 8192 输入 / 8 输出 | 所有 D 首 token 后 0.2 s |
| `p12k` | 同上 | 1 × 12000 输入 / 8 输出 | 同上 |
| `p12k-burst` | 同上 | 3 × 12000 输入 / 8 输出 | 首次延迟同上，随后每 0.8 s 一个 |

“无 P”指没有额外注入长 P；四条 D 仍各自需要 128-token prefill，因而
该控制组包含它们的初始准入成本，不是跳过 prefill 的纯设备 decode 测量。

每组先完成一轮相同输入/输出长度、并发数和到达间隔的完整负载预热，再
独立计时三轮；预热原始回执保留，排除在计时汇总之外。P 调度以“所有 D
收到首个生成 token ID”为门槛，空文本 token 也计入；不能只依赖字符串
非空来触发。每个 P 发出时必须仍有 D 尚未收到全部输出 tokens，之后还须
观察到 D token 继续到达。仅等待 usage/DONE 的连接不算活跃 D。

P 首 token 前没有 D 事件可以是有效的停顿观测，不据此删除该轮。若 P 到达
时 D 已全部生成结束，则该轮不构成预定争用实验，报告失败并保留证据；
不能将这个退化负载计为成功。报告记录计划/实际 dispatch 时间和延迟、
活跃 D IDs，以及各 D 在 P dispatch 之后及 P 首 token 之前的事件数量。

这是有限数量的长时间生成请求加定时 P 到达；没有在 D 完成后补充新 D，
也不是无限请求流、Poisson 到达或稳定态饱和容量实验。结果只支持该负载
范围的比较，不把请求结束后的吞吐称为生产最大承载能力。

## 流审计与统计口径

每个请求保存完整 HTTP 解码后的 SSE body 字节（base64）、其 SHA、byte
chunk 大小及 monotonic 到达时间、解析事件、完整 token IDs、文本和 usage。
事件时间取组成完整 SSE frame 的最后一批字节到达时刻。严格要求 HTTP
200、SSE content type、稳定 response ID、完整一致的 prompt IDs、恰好
约定数量的生成 IDs、`finish_reason=length`、最终 usage 和唯一 `[DONE]`。
断流、超时或协议错误保留 partial 报告并以非零状态退出，后续轮次停止。

这里的 D/P 标签表示请求角色：D SLO 评价最初的四条持续生成请求，P SLO
评价后来注入请求的首 token；后者的 8 个输出 token 仍全部检查并计入吞吐。

| 指标 | 明确定义 |
| --- | --- |
| D event gap | 同一 D 请求中，相邻包含生成 token IDs 的 SSE 事件到达时间之差 |
| D 请求 SLO | 整个请求的最大 event gap 分别不超过 100 / 200 ms；保留 D 初始阶段的停顿 |
| D 平均 TPOT | 首末 token-event 时间差除以生成 token 数减一；同时标记首事件是否只有一个 token，不能代替最大 gap |
| P TTFT / SLO | 从该请求实际 dispatch 至首个含 token IDs 的事件，分别使用 1 / 2 s 阈值 |
| 分位数 | 按 `(n-1) × p/100` 线性插值，不使用旧 benchmark 的 nearest-rank 口径 |
| 输出吞吐 | 成功完整计时窗口内 D、P 或两者的生成 token 数 / 窗口 wall time |
| SLO request goodput | 满足 D gap 阈值的 D 请求数 + 满足 P TTFT 阈值的 P 请求数，再除以完整窗口时长；另列分角色值 |
| Joint-window goodput | 仅当所有 D 和 P 请求都满足相应阈值时计入窗口内全部请求，否则为 0 |

单 token 与多 token 消息分别统计。多 token 消息内部没有可测的逐 token
时刻，不能复制一个到达时间或插值来宣称逐 token SLO；该请求的逐 token
观察 SLO 标为未知，可观察的 event-gap SLO 仍单列。即使消息各含一个
token，也可能因网络读取而共享时间戳，因此所有上述 gap 都是客户端观察，
不是设备 kernel 时间或服务端采样时刻。

主口径保持完整请求 SLO，另列所有 D 首 token 门槛之后及各个
`P dispatch → P first token` 窗口的诊断。相邻时间戳构成的间隔始终保留
原始完整长度，分为窗口内、跨开始边界、跨结束边界和跨两个边界；不裁剪
跨界停顿，也不通过丢掉门槛之前的间隔改善主 SLO。多个 P 窗口可能重叠，
不能把这些窗口中重复出现的间隔合并成独立样本。

失败或无效争用轮次的 goodput / throughput 汇总标为未知，原始窗口与
已收到事件保留，不缩短分母或丢弃失败样本制造正结果。所有轮次都保留，
最终比较同时列出 D 停顿、P TTFT、双方完成情况、吞吐和 goodput。
设备 profiler / 生命周期 trace 使用独立诊断运行，不混入正式计时；它们
能支持什么结论须由本轮对应证据单独确定，客户端 SSE 不能证明设备重叠。

## 最终六轮比较

[独立 SSE 审计 (archive member: `evidence/2026-10-11-tp8-contention/contention-audit.json.gz`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 对全部
40 份计时报告通过，最终选择 32 份：四角色 × 四负载 × 两阶段，每份
含三个 measured repeats。原始 warmup 另行校验且不计时；v1 的四份报告
和 CPU 测试重叠的旧确认阶段四份报告完整保留。审计还通过 8 项原始事件
破坏检测、17 项汇总检查与 9 项 token 比较自检。

最终阶段严格为 native16k initial/confirm、native2k initial/confirm、
old-work256 initial/confirm-clean、guard-v2 initial/confirm。每个数字覆盖
全部六轮，不选择最快阶段。吞吐以合并生成 token 数除以六轮完整窗口总
时长；分位数对全部请求计算，不平均阶段分位数。D gap P95 是每请求最大
事件间隔的 P95，平均 TPOT P95 是每请求平均 TPOT 的 P95，两者不能互换。

| 负载 / 角色 | 输出 tokens/s | 阶段吞吐范围 | D 最大 gap P95，ms | D 平均 TPOT P95，ms | P TTFT P95，ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无 P / Native 16K | 173.421 | 172.50–174.35 | 371.38 | 21.77 | 无 P |
| 无 P / Native 2K | 172.481 | 172.09–172.87 | 394.21 | 21.87 | 无 P |
| 无 P / 旧 work256 | 170.758 | 170.10–171.42 | 375.99 | 22.18 | 无 P |
| 无 P / Guard v2 | 168.874 | 167.76–170.01 | 65.81 | 21.73 | 无 P |
| P8K / Native 16K | 166.476 | 166.35–166.61 | 391.58 | 22.92 | 411.43 |
| P8K / Native 2K | 133.776 | 133.59–133.96 | 402.63 | 28.91 | 1986.66 |
| P8K / 旧 work256 | 162.153 | 161.93–162.37 | 377.62 | 23.55 | 419.90 |
| P8K / Guard v2 | 162.621 | 162.16–163.08 | 65.84 | 23.04 | 504.48 |
| P12K / Native 16K | 168.166 | 167.90–168.44 | 385.26 | 22.68 | 406.14 |
| P12K / Native 2K | 129.856 | 127.90–131.88 | 400.32 | 30.12 | 2351.86 |
| P12K / 旧 work256 | 163.073 | 161.93–164.23 | 376.65 | 23.59 | 440.12 |
| P12K / Guard v2 | 162.215 | 161.69–162.74 | 67.47 | 23.03 | 521.85 |
| P12K burst / Native 16K | 152.310 | 151.82–152.80 | 397.86 | 25.52 | 415.52 |
| P12K burst / Native 2K | 86.723 | 84.84–88.69 | 413.55 | 47.33 | 5386.11 |
| P12K burst / 旧 work256 | 149.055 | 147.56–150.58 | 376.45 | 26.34 | 444.95 |
| P12K burst / Guard v2 | 149.329 | 146.94–151.79 | 68.19 | 25.82 | 522.30 |

相对原生 16K，guard v2 在 P8K/P12K/burst 的 D 最大 gap P95 分别下降
83.19% / 82.49% / 82.86%，P TTFT P95 则由 411.43 / 406.14 / 415.52 ms
增加到 504.48 / 521.85 / 522.30 ms。对应吞吐下降 2.32% / 3.54% / 1.96%；
D 平均 TPOT P95 也上升 0.52% / 1.55% / 1.16%。无注入 P 时，吞吐下降
2.62%，所以该保护不是无成本的通用默认策略，参数默认仍关闭。

相对旧插件，guard 的争用负载吞吐变化为 +0.29% / −0.53% / +0.18%，
主要收益仍是停顿减少。原生 2K chunk 是必要的公平对照：在这套 Qwen3.5 /
Ascend 路径中它仍产生约 400 ms 的请求最大 gap，并增加 P TTFT 和总轮数；
不能把这个局部结果概括为 chunked prefill 在其他平台或模型上无效。

各阶段范围同时展示顺序变化，例如 guard burst 为 146.94–151.79 tokens/s，
native2k burst 为 84.84–88.69；六轮合并不是置信区间，也不足以支持生产
容量或任意请求到达分布的结论。

## SLO 与 goodput

每个负载共有 24 条初始 D；P8K/P12K 各 6 条注入 P，burst 为 18 条。
下表固定 D 100 ms + P TTFT 1 s，列出请求级 goodput 和要求整轮全部请求
都通过的 joint-window goodput。两者都除以所有六轮的完整窗口时长。
原生的 joint-window 值为 0 不表示所有请求失败：无 P 时部分 D 通过，
有 P 时原生 16K 的 P 请求仍全部在 1 s 内返回首 token。

| 负载 / 角色 | D 100 / 200 ms | P 1 / 2 s | Joint 100 ms + 1 s | 请求 goodput，req/s | Joint goodput，req/s |
| --- | --- | --- | --- | ---: | ---: |
| 无 P / Native 16K | 18/24；18/24 | 无 P | 0/6 | 0.508 | 0.000 |
| 无 P / Native 2K | 18/24；18/24 | 无 P | 0/6 | 0.505 | 0.000 |
| 无 P / 旧 work256 | 18/24；18/24 | 无 P | 0/6 | 0.500 | 0.000 |
| 无 P / Guard v2 | 24/24；24/24 | 无 P | 6/6 | 0.660 | 0.660 |
| P8K / Native 16K | 0/24；0/24 | 6/6；6/6 | 0/6 | 0.161 | 0.000 |
| P8K / Native 2K | 0/24；0/24 | 0/6；6/6 | 0/6 | 0.000 | 0.000 |
| P8K / 旧 work256 | 0/24；0/24 | 6/6；6/6 | 0/6 | 0.157 | 0.000 |
| P8K / Guard v2 | 24/24；24/24 | 6/6；6/6 | 6/6 | 0.788 | 0.788 |
| P12K / Native 16K | 0/24；0/24 | 6/6；6/6 | 0/6 | 0.163 | 0.000 |
| P12K / Native 2K | 0/24；0/24 | 0/6；0/6 | 0/6 | 0.000 | 0.000 |
| P12K / 旧 work256 | 0/24；0/24 | 6/6；6/6 | 0/6 | 0.158 | 0.000 |
| P12K / Guard v2 | 24/24；24/24 | 6/6；6/6 | 6/6 | 0.786 | 0.786 |
| P12K burst / Native 16K | 0/24；0/24 | 18/18；18/18 | 0/6 | 0.436 | 0.000 |
| P12K burst / Native 2K | 0/24；0/24 | 0/18；0/18 | 0/6 | 0.000 | 0.000 |
| P12K burst / 旧 work256 | 0/24；3/24 | 18/18；18/18 | 0/6 | 0.427 | 0.000 |
| P12K burst / Guard v2 | 24/24；24/24 | 18/18；18/18 | 6/6 | 0.997 | 0.997 |

Guard 所有 measured D gap 都小于 100 ms，四种负载的观察最大值分别为
71.89 / 66.52 / 67.49 / 68.19 ms。全部注入 P TTFT 小于 1 s，因而四组
SLO 阈值组合均通过。该观察不等于未来最坏时延上界；采样数量、网络和
CPU 抖动边界仍然存在。详细四种阈值组合与分角色 goodput 保留在审计 JSON。

## 完整时间线与范围诊断

下图固定展示所有六轮 `p12k-burst`，每轮各 D 的全部事件和各 P 的等待
区间都保留；四列共用横轴，以所有 D 首 token 的门槛为零，负时间区域
保留初始准入停顿。红/黄线是超过 200 / 100 ms 的完整相邻事件间隔，
紫色粗线是 P dispatch 到首 token。没有选择最佳轮次或裁去跨界停顿。

[全部六轮 burst 的完整时间线 (archive member: `evidence/2026-10-11-tp8-contention/figures/contention-timeline.svg`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)

主分布图包含所有请求：无 P 对照每个角色 24 条 D，注入 P 样本数为
6 / 6 / 18。这里没有 multi-token 消息，仍不能把客户端到达时刻当作
设备执行或服务端采样时间。

[请求最大间隔与 P TTFT 的全部样本分布 (archive member: `evidence/2026-10-11-tp8-contention/figures/contention-ecdf.svg`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)

[post-gate 诊断图 (archive member: `evidence/2026-10-11-tp8-contention/figures/contention-scoped-gaps.svg`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
进一步区分包含边界穿越的 intact gap 与完全位于门槛后的 contained gap。
Native 与旧插件无 P 组每轮各有一条 D 发生约 0.37 s 初始停顿，因而总体
仅 18/24 通过；门槛后完全包含的最大 gap 约 30 ms。旧插件的跨门槛
间隔还会出现在 post-gate intact 口径，不能裁掉后当作整体 SLO 通过。
有 P 时，旧插件在长 P 期间仍有约 0.2–0.3 s 的停顿，而 guard v2 的
整体与门槛后间隔都保持在本次目标内。完整范围字段由独立审计复算。

[绘图脚本 (archive member: `evidence/2026-10-11-tp8-contention/plot_contention.py`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 使用独立
Matplotlib 3.10.7 环境，保存全部输入和
[图表 manifest (archive member: `evidence/2026-10-11-tp8-contention/figures/plot-manifest.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，
包括报告、计划、绘图脚本及审计 SHA。正式产物为 SVG；本地 PNG 预览已
实际检查全部 burst 与主比较图的轴、图例和裁切，预览目录不进入公开归档。

## 保留的 v1 结果与机制边界

v1 只有 initial 三轮，不能与 v2 六轮混池。其四负载的整体 D gap 也都
满足 100 ms，但所有注入 P 均超过 1 s（仍在 2 s 内）；小片段推进提高了
交互流畅度，却明显推迟 P。以下列出全部三个 repeats 的合并结果。

| v1 initial 负载 | 输出 tokens/s | D 最大 gap P95，ms | P TTFT P95，ms | D 100 ms | P 1 s |
| --- | ---: | ---: | ---: | --- | --- |
| 无 P | 157.909 | 36.31 | 无 P | 12/12 | 无 P |
| P8K | 154.819 | 49.11 | 1093.65 | 12/12 | 0/3 |
| P12K | 152.093 | 49.54 | 1167.88 | 12/12 | 0/3 |
| P12K burst | 141.636 | 60.10 | 1473.59 | 12/12 | 0/9 |

v2 优先使用合格续片的非 P 残差，减轻早期独立 D 最大值长期限制层数的
问题，同时保持未知形状保守首片和原有生命周期契约。这个改进来自实际
诊断，不能声称论文的空间争用预测器已经移植。

[v2 估计器独立审计 (archive member: `evidence/2026-10-11-tp8-contention/guard-v2-estimator-audit.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
核对 983 组 schedule/receipt 配对：50 个合格残差样本的 median / P95 /
最大值为 19.042 / 22.294 / 24.269 ms；55 步实际选择残差估计，其中
39 步推进多于一层且超过同一步只用独立 D 后备估计的算术配额。12K
完整 P 从 v1 的 40 个单层步骤变为 v2 的 9 步；3×128 与 16383-token
两组诊断也从 40/40 步变为 9/8 步。这是诊断机制对照，不是额外的
匹配性能试验；本次未出现“没有独立 D 样本却使用残差”的实机样本，
这一分支仅有 CPU 测试覆盖。

现有 profile 的 P128 编译 FX 与 P12000 raw 分层路径不同，不能把延迟
差全部归因于配额匹配。固定源码的原生 runner 以实际 scheduled tokens
切输入，graph capture 上限为 8；P128/8K/12K 不因 compile range 为
`[1,16384]` 就填充到 16K。动态 FX 缓存也保留符号 token 维度。
Ascend 的 npugraph_ex `force_eager=True` 仍调用 FX eager 算子路径，
并非完整 P 的设备 graph replay；profile 每个 P forward 有 40 层 MoE /
30 层 GDN 的 host custom-op 调用。
[紧凑 profile 审计 (archive member: `evidence/2026-10-11-tp8-contention/profile-hotspots.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
提供次数与 inclusive 时长，但没有逐算子输入形状，不能由它认定 kernel
内部 padding 或精确分摊全部固定成本。guard 同时改变部分 P 的 FX/raw
执行形态和切层配额；本轮未做这两个因素的完整析因实验。

## 文本正确性与合成 token 稳定性

原生 16K 与 v1/v2 的诊断自然语言对照均为 16/16 **文本精确一致**，
对应 arithmetic、counting、抽取与长文抽取，isolated/paired 各重复两次，
各端 oracle 无失败。v2 证据见
[自然语言回执 (archive member: `evidence/2026-10-11-tp8-contention/guard-v2-diagnostic-correctness.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。
该脚本比较文本及 oracle，不是 token IDs 精确一致证明。

争用基准使用合成 prompt。独立审计从原始 SSE 重建 840 组完整 token IDs
（包括保留阶段及 warmup），并分别统计重复稳定性和跨配置一致性。最终
每个角色有 21 个逻辑输入，各 6 次 measured 观测，共 126 组；warmup
另列，不把单次 warmup 的“稳定”算作重复证据。

| 最终角色 | 六轮全序列稳定输入 | 六轮首 token 稳定输入 | 本角色 initial/confirm 配对全序列一致 | 同阶段/同 repeat 对 native16k 全序列一致 |
| --- | --- | --- | --- | --- |
| Native 16K | 6/21 | 14/21 | 23/63 | 自身参考，不作为稳定性证据 |
| Native 2K | 5/21 | 15/21 | 22/63 | 44/126 |
| 旧 work256 | 2/21 | 13/21 | 17/63 | 40/126 |
| Guard v2 | 4/21 | 15/21 | 20/63 | 44/126 |

原生本身跨阶段只有 23/63 全序列一致；guard v2 对原生为 44/126，不能据
传输审计通过宣称任意输入数值等价，也不能把所有合成差异归因于插件。
这些计数只描述 token exactness，没有语义质量阈值或完整因果归因。
所有配置强制相同输出长度，因此本轮吞吐不会因提前 EOS 或较少输出而
获得优势，但不同输出可能影响 MoE 路由等执行工作，仍是比较限制。

## 独立生命周期与设备证据

v2 诊断的 progress、cancel、满 budget、自然语言和 trace checker 五个
命令均成功。校验通过 983 条进度回执、26 个 cohort，记录 29 个完成请求
与 1 个取消请求；最大完整 prompt 为 16383 tokens。来源为
[trace 校验 (archive member: `evidence/2026-10-11-tp8-contention/guard-v2-diagnostic-check.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 与
[独立生命周期证明 (archive member: `evidence/2026-10-11-tp8-contention/guard-v2-diagnostic-lifecycle-proof.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。

12K cohort 2 在 steps 3–11 以 `1,1,6,6,6,6,6,6,2` 层推进到 40 层，
完成前已有 8 轮 D 回执。取消 cohort 4 在 step 74 到达 37 层，step 75
返回取消 ACK 且层数仍为 37，不以伪造完成释放 KV。满预算诊断首次准入
16383-token P 时，本轮 D 未全部选中仍受活跃 D 保护；测量 cohort 10
首片为 1 层，step 418 后 step 419 偿还 D，原生计数保持连续。这些是
协议/进度证据，不是 kernel overlap 或未来 100 ms 上界证明。

设备证据来自单独 old/native profiler 运行，正式计时未混入 profiler
或详细生命周期 trace。两组均导出全部八 rank。profile 窗口的 compute
活跃覆盖率约 28.50%–29.02% / 26.74%–26.94%，所有 rank 观察到跨物理流
compute 交叠为 0，符合 fenced 路径；这不是 AI Core 占用率，也不是新的
v2 硬件并行证据。完整 profile trace/CSV 与独立分类脚本供离线复核。

## 复算与证据发布

[archive_evidence.py (archive member: `evidence/2026-10-11-tp8-contention/archive_evidence.py`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 已将
所有阶段（含失败/排除回执）及完整 16-rank trace/CSV 保持相对目录归档。
较大的 JSON/JSONL/CSV 使用确定性 gzip，索引分别记录源、存储、解压字节
的 SHA-256；八组小型临时文件测试通过，日志和
[回执 (archive member: `evidence/2026-10-11-tp8-contention/archive-self-test-receipt.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 也保留。

公开环境快照仅移除原始系统级 `processes` 字段，避免公开无关管理命令
和连接参数。原件留在本地，索引记录这一转换及原件/公开版本的不同 SHA；
其他选中原始证据逐字节保留。推理 `token_ids` / `max_tokens` 不作删改。
三份外部 CPU 测试日志映射到 `validation/`，包含早期 fixture 失败及后续
180 项 targeted、582 passed / 2 skipped 的整套验证，通过范围不外推。

先验证并解包到本仓库 `local-results/` 下新目录，再运行 profile auditor，
以保持其 `scripts/analyze_npu_trace.py` 导入和原相对目录布局。绘图使用
独立 `.plot-venv`，所有复算都应在推理计时之外进行。归档脚本不会执行任何
服务启动器。最终 manifest 记录源码、producer、审计和图表 SHA；实际
数据选择规则、版本调整、CPU 重叠重跑和依赖回滚全部保留。

从仓库根目录可按以下命令复算；输出目录必须尚不存在，详细规则见
[归档说明 (archive member: `evidence/2026-10-11-tp8-contention/ARCHIVE.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。

```bash
python3 scripts/archive_tp8_evidence.py verify \
  --archive "$BENCHMARK_COLLECTION_ROOT/evidence/2026-10-11-tp8-contention"
python3 scripts/archive_tp8_evidence.py unpack \
  --archive "$BENCHMARK_COLLECTION_ROOT/evidence/2026-10-11-tp8-contention" \
  --output local-results/tp8-contention-replay
```

上述命令只验证/解包历史输入，不执行归档内源码。历史重放使用的
`audit_contention.py` 和 `profile-hotspots-audit.py` 按原字节保留在归档中；
如需再次执行，须先按当前代码审查流程迁入受维护的工具目录并显式指定外部输入。
下述重放通过记录是历史结果，不是本次数据迁移重新运行了这些脚本。

归档后的 pack、独立 verify 与完整 unpack 均通过；随后直接从封存的 gzip
重放 40 份原始 SSE 审计，32 份入选结果、所有指标/SLO、840 组 token 数组
及 34 项审计自检均与封存结果一致。估计器与生命周期重放 JSON 逐字节相同；
16-rank profile 重放仅规范化来源目录，其他所有字段严格一致。完整命令、
输入/输出 SHA 及出版检查见归档目录之外的
[重放回执 (archive member: `evidence/2026-10-11-tp8-contention-replay-validation.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，
封存目录本身未因复算而修改。

[最终环境回执 (archive member: `evidence/2026-10-11-tp8-contention/environment-final.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 记录
全部实验服务已停止、八卡空闲，NumPy 仍为恢复后的 1.26.4；初始没有用户
服务，因此没有自动恢复其他服务。
