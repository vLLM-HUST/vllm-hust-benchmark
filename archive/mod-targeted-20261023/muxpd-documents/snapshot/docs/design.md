# MuxPD 接入设计与验证边界

本项目面向 HUST 的 vLLM / vllm-ascend 环境，以可选择的插件方式引入
MuxWise 的阶段分离机制。用户显式选择 MuxPD 调度器后，prefill 处理完整输入，
每次模型调用只包含 prefill 或 decode 请求，取消 token chunk 和 P/D 混合批次。

## 实现层次

| 层次 | 行为 | 当前状态 |
| --- | --- | --- |
| 同阶段批次调度 | 完整 prefill；decode 与 prefill 使用不同批次；共享原引擎内的缓存管理 | 第一版实现路径；端到端正确性和性能以测试回执为准 |
| 串行执行 | 每次运行一个同阶段批次 | 组件默认接入路径；没有实现空间并发 |
| `layered` 执行 | 双 runner 分别执行完整 P/D 批次，在层边界交错；一次 scheduler transaction 等待两个阶段均完成 | 已实现；单次完整 P 期间只产生一轮 D 结果 |
| `decoupled` 执行 | P 的完整输入和中间状态跨引擎迭代保留；每轮 D 独立采样、输出，P 使用固定、token-work 或 deadline 层数配额 | 已实现；TP2/TP4 的设备进度、取消与 API 回执见下文链接 |
| `stream_mode=fenced` | 每个层片段通过事件建立跨流依赖，保留阶段执行隔离 | 双 runner 默认路径；不宣称设备 kernel 同时执行 |
| `stream_mode=overlap` | 独立流与 workspace；`shared` 通信组使用事件门控，`isolated` 为 P 新建 TP/EP 通信组 | eager 实验路径；shared 有 TP2/TP4 证据，isolated 已验证指定 TP4+EP 配置；不支持 graph |
| `decode_graph=true` | 独立 D runner 捕获和重放原生 ACL graph；P 始终使用 graph mode `NONE` | 仅 decoupled fenced；指定 TP4+EP 配置已验证，完整 P 另有可选 FX 编译 |
| NPU 资源分区 | 分别控制 P/D 的 Cube / Vector 资源 | 尚未接入控核；待逐算子支持与 profiler 验证 |
| 层数自适应与资源预测 | 已有 token-work、EWMA/deadline、prefill-first，以及可选 decode budget guard | guard 使用本机片段观测；尚无论文的硬件分区预测器、在线争用修正或 SLO 保证 |

串行同阶段调度消除了 mixed batch 和 token chunk，但长 prefill 仍会阻塞
decode。它是接入与对照基础，不能称为完整 MuxWise，也不能据此声称改善了
TBT 或实现了算力隔离。实验性双 runner 不自动意味着并发安全或性能更好。

serial 对原生同阶段批次耗时更新 EWMA。layered 使用固定 `layers_per_slice`；
decoupled 默认同样固定，可显式启用下述工作量和 TTFT 层数配额。P 等待阈值也受已有
pending cohort 和可用 KV 容量约束，不是请求到达至首 token 的硬上界。

组件默认与调优脚本不同：`adaptive_layers`、`prefill_first`、`decode_graph`、
`compile_full_prefill`、`decode_budget_guard` 默认均为 `false`，
`prefill_layer_token_budget` 默认为 `0`。
[`serve-qwen35-decode-graph.sh`](../scripts/serve-qwen35-decode-graph.sh) 显式选择
fenced graph、前四个开关及 `262144` 的工作量预算，保持 guard 关闭，并捕获 D batch size
1/2/4/8。该组合的验证范围和延迟取舍见 [TP4 性能报告](performance-tp4.md)，
不能据此认为单独开启某个选项或任意负载都会改善性能。

## 固定参考版本

- HUST vLLM：`d0f22d2bda562156e4dbf433ce645e1769b4f804`。
- HUST vllm-ascend：`03766ac696fde5ab1980d80ca0b8543d3580c989`。
- MuxWise artifact：`ykcombat/sglang` 的 `slo_config` 分支，
  `eeac5148f456428a05823c330e1d40afb0ef2438`。
- 本机运行环境的 CANN 目录为 `cann-9.1.0`；兼容环境变量为
  `VLLM_VERSION=0.25.1`。完整运行版本应另外保存于每次设备验证回执。
- 默认模型为 `Qwen3.5-35B-A3B`。它包含混合 attention / Mamba 状态，不能将
  Llama 的 KV-only 正确性推论直接用于这个模型。

源码出处、论文正式版本和具体入口见 [references.md](references.md)。

## 请求和状态约束

完整 prefill 需要足够的 token budget。启动配置应满足
`max_num_batched_tokens >= max_model_len`，并显式关闭 chunked prefill。
超过配置上限的请求应由原有请求校验拒绝，不能通过静默截断或偷偷开启 chunk
来满足调度预算。等待队列、抢占恢复和请求输出仍应服从 vLLM 的状态协议。

普通 decode 不能误包含尚未完成 prefill 的请求。阶段判断必须考虑已计算
token 数、缓存命中及抢占后状态，不能只看请求是否已经进入 running 队列。
对于存在 KV/状态不足的情况，应继续使用原引擎的容量检查与回收机制。

按层执行需要为一个 prefill 保存位置、hidden states、residual、layer cursor、
attention metadata 和相关模型状态。仅首个片段执行 embedding，仅最后一个
片段执行最终 norm、logits 和采样。这里的片段指连续模型层，输入 token 不分块。
prefill 的输出在最后一层完成前不能进入 decode。

双 runner 若共用模型权重和 KV，仍必须隔离输入 staging buffers、attention
workspace、采样状态及 graph 的静态缓冲区；不能因为模型权重只读就假设其余
运行时状态也只读。Mamba state、slot mapping、缓存回收与抢占尤其需要检查。
取消请求后，必须先处理仍在运行的设备任务与事件，才能复用其缓存或状态槽位。

## 跨迭代 prefill 协议

`execution=decoupled` 使用
[`DecoupledMuxPDScheduler`](../src/vllm_muxpd/decoupled_scheduler.py) 与
[`DecoupledMuxPDNPUWorker`](../src/vllm_muxpd/decoupled_worker.py)。每次最多保留
一个 prefill cohort。首次使用原生 scheduler 为整个 prompt 分配 KV，并保存
P 子输出的完整 token 成本；worker 为 P 保存可继续执行的调用栈、输入和中间状态。
D 使用另一个 runner。若本轮同时有 P 和 D，fenced 后端先完成 D forward，
随后推进 P 的当轮层数配额。shared overlap 交替发射 D/P 的完整模型层，
避免共享 communicator 把整个 D 网络串在 P 前面；isolated overlap 先提交
P 的整个层片段，再提交 D，二者使用独立通信组。下述 prefill-first 可在调度
阶段选择一次 P-only 步骤。每轮只采样 D 和本轮完成的 P。
guard 关闭时，若当轮没有 D，scheduler 请求一次运行剩余 P 层。
guard 开启时还检查存活但未被本轮选中的 D；只有没有可运行 D 才直接完成剩余 P。
worker 只遵从显式完成标志，不能根据本轮 D batch 为空自行放宽配额。

原生调度在发射完整 P 时已经增加 `num_computed_tokens`，但此时尚未完成
forward 或产生首 token。本插件显式记录 pending cohort，后续迭代暂时从
原生 running 选择集移除这些请求，并保留其 KV 引用和 worker 槽位。P 完成前
不能进入 decode、被重复发射或被原生 KV 抢占；其他 D 仍可按原生容量规则抢占。
取消但尚未 ACK 的 P 也占用槽位。现有 cohort 结束前不接纳下一组 P。

| 跨进程字段 | 语义 |
| --- | --- |
| `muxpd_prefill_request_ids` | 本轮新发射的 P；后续继续执行时为空 |
| `muxpd_pending_prefill_ids` | 尚未完成或取消 ACK 的全部 P，包含 host 已取消的请求 |
| `muxpd_cancel_prefill_request_ids` | 请求 worker 安全停止或过滤的 P |
| `muxpd_prefill_cohort_id` | 单调递增的 cohort 标识，用于拒绝错序回执 |
| `muxpd_finish_prefill` / `muxpd_pause_prefill` | 无 D 或层数配额要求时完成剩余 P / 防御性暂停控制 |
| `muxpd_completed_prefill_ids` / `muxpd_cancelled_prefill_ids` | worker 的完成及取消 ACK；二者不相交 |
| `muxpd_prefill_layers_completed` / `muxpd_prefill_layers_total` | worker 已提交的层数及模型总层数；不单独表示设备任务已完成 |
| `muxpd_prefill_progress_source` | `none` 表示无 P；`layer_hooks` 表示逐层 hook；`full_forward_return` 仅用于获准编译的新 P 完整 forward 成功返回 |
| `muxpd_prefill_slice_timing` | guard 启用时的可选首片/续片计时；包含层增量、完整 P token 数、host 发射跨度和已有同步确认完成后的 driver-rank 设备跨度 |

worker 回执只包含当轮 D 和本轮完成 P 的样本。scheduler 先校验 cohort、
样本行及 ACK，再分别调用原生 `update_from_output()` 回收 D 子输出和已保存
的 P 子输出，按客户端合并结果。P 未完成时不执行原生 P retirement；因此 D
可在多次引擎迭代中持续输出，P 的首 token 只在其全部层和最终采样完成后出现。

此模式要求同步 `EngineCore.step()`，显式关闭 async scheduling，且仅允许
一个引擎 batch 在执行。即使 `total_num_scheduled_tokens=0`，也必须执行
worker 和 sampler：该步骤可能继续最后一个 P，或传递取消 ACK。批次队列路径
可能跳过零 token worker 调用，因此拒绝使用 `schedule_batch()`。

## 可选层数配额与 prefill-first

`prefill_layer_token_budget` 是非负整数，默认为 `0`；非零值只允许用于
`execution=decoupled`。它对应 artifact 按 token 工作量决定层数的机制。
基础配额为：

```text
budget = 0: base_layers = layers_per_slice
budget > 0: base_layers = max(layers_per_slice,
                             min(model_layers, budget // cohort_full_prefill_tokens))
```

实际配额再限制到剩余层数。分母是原始 cohort 的完整 P token 总数，包含抢占
重算时需要重放的已生成 token；不会随着层数推进或部分取消而缩小。调度预览
使用候选请求总量，KV 分配后按实际接纳的 cohort 重新计算。该预算不改变完整
prompt 的 token admission，也不是硬工作量上限：配置层数下限和 deadline
都可能要求执行更多层。可选 guard 在此基础上限制实际层数；没有存活 D 时仍
一次完成剩余 P，暂停和整组取消不推进层数。

`adaptive_layers` 默认为 `false`，只能与 `execution=decoupled` 同时使用。
启用后先观测独立完整 P 和独立 D 的耗时；未开启 guard 时，两类有效样本未齐备使用上述基础
配额，只有工作量预算为 `0` 时才保持固定 `layers_per_slice`。
采样窗口是 host 从 schedule 到 worker 回执的墙钟时间，
包含调度、通信和采样开销，不是 profiler 的逐层 kernel 时长。混合 P/D、
悬挂 P 的后续步骤、取消及晚到 abort 不用于这两类独立延迟校准。

P 样本按 token 数的二次幂范围分桶，同桶不同长度先归一到共同 token 数再
更新 EWMA，避免把短输入的耗时直接绑定到长输入。预测优先使用覆盖输入长度的
已有样本，仅在超出最大实测长度时按比例外推。D 使用独立步骤耗时的 EWMA。
`ewma_alpha` 控制更新，`estimate_safety_factor` 放大估计；这些是近似预算，
没有逐层区分 full attention / Mamba，也没有建模当前 batch 大小的争用。

调度器用 cohort 中最早请求的到达时间和 `ttft_slo_ms` 计算剩余 TTFT 预算，
扣除估计剩余 P 耗时，再计算还容得下几轮 D。配额取上述基础配额与 deadline
所需层数的较大值，再限制到剩余层数；预算越紧，必要时完成全部剩余层。
该策略优先缩短 P 被多轮 D 延后的时间，可能增加 fenced 模式的 D 输出间隔，
因此它不是 TPOT 上界，也不是论文的 contention guard 或资源分区预测器。

`prefill_first` 默认 `false`，要求 decoupled、fenced 和 `adaptive_layers=true`。
若新 P 的预览配额足以完成整个模型，可以在原生 token 记账前选择 P-only，
避免它先等待一轮完整 D。非零工作量预算可在 EWMA 尚未校准时触发这个选择；
否则需要已校准的 deadline 估计。开启 guard 后，预览还必须通过下述片段预算，
未知样本不能直接触发整 P 优先。被延后的已有 D 记为债务，下一轮有可运行
D 时先偿还，再接纳新 P。完整 prompt 占满 token 容量而被选为 P-only 时
同样记债，即使它的首 token 立即结束也不能让后续 P 连续越过 D。取消、
抢占和请求 ID 复用会清理相应债务；P 的 KV 分配失败时回退以保持 D 推进。
trace 区分 `token_budget_prefill_first`、`deadline_prefill_first` 和
`full_budget_prefill_first`，不能把所有 P-only 步骤都解释成 deadline 决策。

trace 中 `decode_slo_feasible_without_prefill` 仅检查独立 D 估计是否已经超过
`tpot_slo_ms`，不包含 P 片段成本。若单独 D 已超过目标，改变 P 层配额不能
使该目标自动可达。应保留固定配额对照并验证真实 TTFT、TPOT 和 goodput。

scheduler 要求回执的总层数与模型一致、累计提交层数与本轮配额一致，且完成 P
必须到达最终层。`full_forward_return` 还要求本轮新 P、完整配额、编译选项
开启以及整个 cohort 完成；悬挂 P 不能中途改用这个来源。这个计数用于检测
错序和执行进度；缓存释放仍只能依据完成或
取消 ACK，不能因看见最后一个层 hook 就提前复用设备内存。

### 可选 decode budget guard

`decode_budget_guard` 是布尔值，默认 `false`。它要求 `execution="decoupled"`、
`adaptive_layers=true`、`stream_mode="fenced"`，不接受 serial、layered 或 overlap。
可与独立 decode graph、完整 P 编译、工作量预算和 prefill-first 同时使用；这些
选项各自原有的校验条件仍成立。已选择 decoupled scheduler/worker 时，可使用以下
完整 policy JSON。命令行提供的 JSON 会替换原 policy 配置，不能只假定合并一个字段。

```json
{
  "execution": "decoupled",
  "stream_mode": "fenced",
  "adaptive_layers": true,
  "decode_budget_guard": true,
  "layers_per_slice": 8,
  "ttft_slo_ms": 1000,
  "tpot_slo_ms": 100,
  "trace_path": "local-results/guard-{pid}.jsonl"
}
```

guard 先保留工作量/TTFT 算出的期望配额，再用 `tpot_slo_ms` 减去距最近 D
结果退休的时间和匹配的非 P 耗时估计，给 P 片段分配剩余预算。实际配额取
期望配额、剩余层数和预算可容纳层数的较小值，允许低于 `layers_per_slice`。
至少推进一层；若这一层也放不进预算，记录 `minimum_progress_exceeds_budget`，
而不是永远暂停 P。TTFT 已超期也不能强制越过 D 预算，估计冲突记为
`ttft_feasible=false`；缺少有效估计时可行性为未知，已经超期则明确为不可行。

学习使用以下不同观测，不把整段 P 的 EWMA 当作逐层成本：

- P 按原始 cohort 的完整 token 长度组合精确匹配，包括重算的输出历史；部分
  取消不缩小这个形状。首片只用实际推进一层的观测作为启动成本，续片单独保存
  每层成本。首次推进多层的跨度按其实际层数保留，不冒充一层启动成本。
- 独立 D 使用无 pending P 的 D 步骤，按 batch 请求数、总上下文长度与最大
  上下文长度的二次幂区间匹配。另一个缓存保存已完成 fenced P 续片的非 P
  残差：`host_step_ms - device_elapsed_ms`，不能减去 host/device 的较大值。
  只接纳非首片、非末片、raw 层执行且本轮实际选中 D 的有效完成样本；零或
  负残差记录拒绝原因，不截断成零。样本按实际选中的 D 形状保存。
  pending P 优先使用匹配的残差，没有时使用独立 D 观测；新 P 与 P-first
  预判仍只用独立 D。残差包含 D、RPC、其他 host 工作和跨 rank 尾部等待，
  不是纯 D 时间或最坏情况上界，也不删除独立 D 的首次较慢观测。
  形状不匹配、尚无首层/续层样本或退休时间未知，仍回到一层冷启动；无需
  专门暂停 P 增加一次 D 校准步骤。
- 设备计时不增加同步。只有现有采样已证明片段完成时，才可使用
  `max(host_submit_ms, device_elapsed_ms)`；它是 driver rank 的 lane 跨度，含
  host 发射空隙，不是纯 kernel 时间或全部 TP rank 的最大值。未完成的 P-only
  片段不提供可用完成耗时。缺少有效 lane 计时时，使用已有采样完成的整步 host
  耗时作为后备，其中包含 D；再加独立 D 成本会重复计入 D。
- 同一类观测保留已见最大值，并应用 `estimate_safety_factor`；lane 与 host
  后备历史分开保存，有有效 lane 样本时优先使用。取消、暂停、晚到 abort 和
  `full_forward_return` 不训练 raw 逐层成本。这里的 guard 不使用 `ewma_alpha`，
  该参数仍控制上文的独立阶段 deadline EWMA。

P、独立 D 和 fenced 续片残差的估计映射各最多保存 256 个条目，命中或更新时移动至最近使用位置，
插入超限时逐出最久未使用的条目。P 的首片/续片及不同首片层数各占独立条目，
不是保证缓存 256 个完整形状。逐出只影响估计，使用可用后备或回到一层冷启动，不
修改正在运行的 cohort、D 债务或 KV 生命周期。trace 的 `estimate_cache` 记录
固定上限、当前条目数与各缓存累计逐出次数。预算协议 v2 的
`decode_estimate_source` 区分 `solo_decode_round_trip` 与
`fenced_continuation_residual`；`estimated_decode_cost_ms` 是实际采用的估计，
`estimated_solo_decode_ms` 始终保留独立 D 的含义。checker 兼容 v1，校验
v2 来源选择、残差观测算术和配额；它不重放整个缓存历史，也不证明硬 SLO。

对于新 P，预算模型为“一层启动跨度 + 其余层数 × 续层成本”；继续运行的 P
只使用续层成本。完整 token 准入与 KV 所有权不变。原生分配只接纳部分候选 P
时，按实际 cohort 重新匹配样本并计算配额。即使完整 prompt 占满 token 预算、
本轮不能选 D，也不能据此当作 D 已消失并完成全部 P；下一轮 pending P 可以
与独立 D 批次继续推进。P-only 延后的 D 保留执行债务，即使 `prefill_first=false`
也适用；取消和原生抢占会清理对应债务。真正没有可运行 D 时直接完成剩余 P。

`quota.decode_budget` 记录 active/selected D、原 P 形状、匹配样本数、计时来源、
期望/实际层数、预算上限及可行性。`decode_budget_sample` 和
`prefill_slice_timing` 保存学习与设备计时依据；取消 ACK 和严格层进度仍按原协议验证。
`check_decoupled_trace.py` 验证回执和配额算术，并兼容没有 guard 字段的旧 trace；
它没有完整重放预测器的全部历史，不能独立证明估计准确。

这是尽量减少 D 停顿的本机启发式，不是硬性 SLO 保证。冷启动、算子差异、
跨 rank 尾部等待和单层不可再分等都会造成估计偏差；首层过慢也可能立即超预算。
开启 guard 可能改善 D 间隔，同时增加 P 的 TTFT 或降低吞吐。需保留关闭 guard
的匹配对照，分别测量服务端/客户端间隔、TTFT、吞吐和 goodput；不能把 SSE
多 token 事件间隔称为精确逐 token TBT，也不能据此声称复现论文的硬件资源分区。

## 取消、暂停与缓存生命周期

host 取消 P 时，原生请求对象可以立即离开等待/运行队列；插件通过
`pop_blocks_for_free()` 分离块所有权但保留引用计数。在 worker ACK 之前，
这些块不回到空闲池，原生 `finished_req_ids` 通知也会被屏蔽，防止 worker
提前删除悬挂 P 的输入行。收到 ACK 后才释放缓存并清理 worker 元数据。

部分取消无法直接从已经执行若干层的 hidden-state 批次中删行。worker 记住
取消 ID，继续执行其余 cohort，并在完成后过滤取消样本、统一 ACK。整组取消
则关闭悬挂调用栈、同步 P stream，再 ACK。若取消发生在 worker 完成和原生
结果处理之间，允许晚到的 completed 回执，释放持有的块并忽略已取消请求的样本。
完整取消控制必须获得整组 ACK，禁止静默留存无法推进的 cohort。

存在未 ACK 的 P cohort 时，`PAUSED_ALL`（keep pause）在修改暂停状态前
直接拒绝，避免原生 keep 流程随后清空缓存。`PAUSED_NEW` 可继续 drain 或
abort，等待取消 ACK。`reset_prefix_cache(reset_running_requests=True)` 同样
拒绝悬挂 cohort；不能通过 reset 抢占持有中间状态的 P。worker 配置另行拒绝
sleep mode。原生 deferred frees、KV/EC connector、结构化输出、LoRA、返回
routed experts 和有状态输入续写不在 decoupled 支持范围内。
此外拒绝 `enable_mfu_metrics`：原生 MFU 核算依赖单轮新请求/缓存请求 metadata，
不能直接解释跨迭代 P 的分阶段 retirement；不能以空 metadata 的零 FLOPs 充作测量。

## Ascend 执行路径

原 artifact 用 CUDA GreenContext 将 stream 绑定到 SM 资源。Ascend 的候选
替代接口为 `torch_npu.set_stream_limit(stream, cube_num, vector_num)`，
配合 `torch_npu.npu.config.allow_internal_format = False`。该接口只保证其
支持的 Ascend C 算子按相应资源限制运行；两个 NPU stream 本身不能证明
实现了独立算力分区。

设备验证应检查各算子的 `BlockDim` 与流资源设置是否一致，并确认设备时间线
存在 P/D 执行重叠。Cube 与 Vector 核数要分别记录；不能将 CUDA SM 数字
直接转换成同名配置。普通流优先级也不能替代资源限制。

TP/EP 的不同阶段可能同时发起集合通信。独立计算流不足以保证 collective
顺序正确；需要验证 HCCL communicator、通信流与各 rank 的完成一致性。
参考 artifact 为 P/D 复制 TP communicator。实验实现的 TP/EP 支持范围
必须由实际多卡验证确定。

双 runner worker 的白名单为 Ascend A2、Qwen3.5 BF16、V1 同步 runner、text-only，
禁用 MC2、FlashComm、动态 EPLB、shared-expert 多流、原生 DBO/ubatching；
overlap 还要求关闭 fine-grained TP。未启用 `decode_graph` 时要求 eager；
图模式使用下面的独立校验。PP、DP、CP、speculative decoding、prefix caching、
LoRA 和量化等不在当前支持范围内。这些限制界定已经审计的算子与通信路径。
当前没有调用 `set_stream_limit()`，也没有接入论文的在线争用修正与硬件资源预测器。

### 独立 decode ACL graph

[`decode_graph.py`](../src/vllm_muxpd/decode_graph.py) 复用原生
`ACLGraphWrapper`、capture 和 attention metadata 更新路径。`decode_graph=true`
只允许 decoupled fenced，要求 `enforce_eager=false`、`VLLM_COMPILE` 和
`FULL_DECODE_ONLY`；capture sizes 必须为有序、不重复的正整数，且不超过
`max_num_seqs`，最大 capture size 必须与列表一致。ENPU、xlite、模型 offload、
动态权重传输及 graph + overlap 组合均拒绝。

P/D 共享原始权重和 KV 所有者，D 则有自己的 graph wrapper、输入静态缓冲区、
metadata builder、workspace 和更新 stream。启动时把 owner 的 capture 临时
重定向到 D，同时绑定原生 capture context 和 `torch.npu.graph` 的 capture
stream；capture 和 replay 使用同一个 D stream 与 workspace。原生 metadata
更新及 graph replay 同步保留。capture 一旦开始后失败，worker 必须重启，
不能回退使用部分捕获的图。

P 的 dispatcher 始终强制 eager，forward context 强制 graph mode `NONE`、
`capturing=false`，默认还设置 `skip_compiled=true`。这覆盖 profile、dummy run
和真实请求；新请求即使只有一个输入 token，也不会被当作 D graph 重放。
不能简单移除 eager 校验并让两个 runner 共用原有 graph wrapper。

`compile_full_prefill=true` 是独立 opt-in，要求上述图模式。它在启动的原生
`profile_run()` 内、D capture 之前允许 P 的 FX 编译和预热，P 的 graph mode
仍为 `NONE`。运行时只允许本轮新建、无已知取消或暂停、配额覆盖整个模型的
P 进入编译上下文；已悬挂的 P 始终继续逐层 raw 执行。编译完整 P 成功返回、
原生 `execute_model_state` 就绪后才能记录 `full_forward_return`；若原生回退
到带 hook 的执行，hook 数必须精确等于总层数。失败不补记层数，完成 ACK 仍
等待最终采样同步。该路径允许 FX 执行，并未捕获或重放 P 的 ACL graph。

controller 的 `snapshot()` 按实际 descriptor 对应的已捕获 graph entry 区分
D capture、replay 与 eager fallback，仅看到 `FULL` 不计作 replay。
`prefill_compiled_calls` 表示允许编译执行的调用数，不能单独证明实际 FX 命中；
本轮另外保存了 [四 rank 的编译执行 profile 证据](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/afd47a8e9964b71844dec75a8e908f279ae98202/archive/mod-history-20261023/muxpd/manifest.json)（归档成员 `evidence/2026-10-10-tp4-optimization/graph-compiled-profile-proof.md`）。

纯 D 步骤直接复用原生 runner/sampler，省去逐层 greenlet 编排。仅在 graph
fenced、连续两轮纯 D、前轮已成功采样且未配置 profiler 时，省去本轮冗余的
caller fence；仍保留最新 backend tail 依赖、D forward/sample 末尾 fence
和 caller 恢复。P/D 转换、首次 D、异常或配置 profiler 时走完整依赖路径。
自定义 collective RPC 若在 caller stream 发射设备工作，必须遵守 backend
事件发布协议；绕过该协议的任意外部设备调用不在支持范围内。

关闭时先同步 P/D 和 metadata 更新 stream，再释放独立 D graph、恢复 instance
包装，最后由原生 owner 清理共享缓存；不对 D runner 再调用原生 shutdown。
流同步失败时保留资源供 shutdown 重试。原生模块的全局 `GraphParams` 不会由
本插件清空，新 controller 发现已有 owner 会拒绝初始化，因此同一进程内
close 后重建 graph worker 不受支持，必须重启进程。

### Shared 与 isolated overlap 通信

[`overlap.py`](../src/vllm_muxpd/overlap.py) 为 P/D 分别维护 stream、forward
context 和 workspace，以事件发布共享初始化工作。默认
`overlap_communication=shared` 对同一 device process group 的 `all_reduce`、
`all_gather` 和 `reduce_scatter` 按主机发射顺序门控。它依赖原生同步 collective
API 将内部 HCCL 工作连接回调用 stream，且要求所有 rank 发射顺序一致。

`overlap_communication=isolated` 只允许 decoupled overlap eager。
worker 为 P 建立 rank membership 相同的 TP/EP sibling groups，D 和 caller
继续使用原生组；[`LaneCollectiveRouter`](../src/vllm_muxpd/communicators.py)
按 active lane 路由上述三个 device collective，不替换全局 TP/EP coordinator。
所有 rank 必须按相同顺序建组，并在内存 profile 前预热。关闭时先同步 lane，
再恢复路由并销毁本 worker 自建的组，不销毁原生组。该路由支持当前白名单的
通信调用，不能推广成任意 MoE backend 或 collective 的隔离保证。

isolated8 已完成 Qwen3.5-35B-A3B BF16、TP4+EP、eager 的自然语言对照、
长 P 进度、取消和四 rank profile；事件数下降，但该 capture 的计算交叠没有
增加，实测性能也未超过原生。shared 的 TP4 结果和 graph fenced 的后续优化
同见 [TP4 性能报告](performance-tp4.md)。这些是不同执行组合，不应把
isolated 通信组与 decode graph 的验证结果合并成未测试的 graph overlap 配置。

shared overlap 的早期设备证据见 [2026-10-10 回执](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/2144d58de108cddaff3800fdcbf12dc9d3ab1d18/archive/mod-consumers-20261023/muxpd/manifest.json)。
在指定 Qwen3.5-35B-A3B BF16 / Ascend 910B2 / TP2+EP / 固定 8 层配额的
capture 中，两 rank 的 P/D compute 区间交集分别约占活跃并集的 10.78% 和
9.16%。计算使用已映射到 P/D 的 CANN physical stream，排除 CPU、拷贝、
event wait 和通信区间。它证明该 capture 存在设备 compute 任务重叠，不能
证明同时的每核占用、空间资源隔离、完整工作负载利用率或吞吐提升。profile
有记录窗口结束时的停止告警；导出区间已检查，但不能据此保证未记录任务的完整性。

## 验证与测量

`scripts/smoke_serving.py` 只提交固定合成数据，保存响应、usage 和时间回执：

```bash
python scripts/smoke_serving.py \
  --base-url http://127.0.0.1:8002 \
  --output artifacts/smoke-muxpd.json
```

检查内容包括 greedy 算术、并发中文 SSE、2305-token 完整输入，以及主动关闭
长生成连接后服务 health 和后续请求是否正常。SSE 回执要求 `[DONE]`、
finish reason 和 usage；长输入用显式 token IDs 校验实际 prompt token 数，
避免字符长度被误认为 token 长度。取消用例验证客户端断开后的服务可用性，
并不单独证明设备任务已立即停止或缓存已全部释放。

设备正确性还应比较固定模型、固定输入、相同采样参数下的 baseline 与插件
输出，覆盖缓存压力、抢占恢复、取消、混合状态模型和不同请求长度。
greedy 输出差异需区分正常数值扰动与状态/顺序错误，不能仅用 HTTP 200 验收。

`scripts/compare_outputs.py` 可对两个配置分别运行同一组自然语言契约：
先逐个请求，再双并发；第二轮交换单请求顺序，检查前一请求状态是否泄漏。
长输入在计数请求已输出首个内容事件、且仍未完成时发射。脚本保留完整合成文本、
SSE、usage、功能 oracle、重复稳定性及文本哈希；`--require-exact` 可额外要求
两个服务逐文本一致。客户端请求重叠仍需 scheduler trace 确认 P 实际跨迭代。
`--baseline-report` 可复用已完成回执中的 baseline 端点，只请求新 candidate；
复用前验证模型、重复次数、case 内容及功能结果、覆盖范围和文本哈希。
`--baseline-only` 可先采集这份完整 baseline，供停服换配置后的对照复用。
该脚本用于正确性比较，复用历史 baseline 的时间数据不构成同条件性能比较。

同一 prompt 的单请求结果在相同配置下应先具有可重复性，再比较两个模式的
单请求和并发结果。并发 batch 形状可改变 BF16 数值路径，文本不一致本身既
不能证明状态损坏，也不能证明只是正常数值误差。需要定位首个分歧，必要时在
相同 batch 形状和强制相同历史 token 的条件下比较 logits。少量契约全部通过
只说明这些用例通过，不能替代长期随机、取消与内存压力测试。

图模式还使用 token-ID 状态复用探针，覆盖单 token P、双 token 对照和长 P
前后历史。当前 [三路调查](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/afd47a8e9964b71844dec75a8e908f279ae98202/archive/mod-history-20261023/muxpd/manifest.json)（归档成员 `evidence/2026-10-10-tp4-optimization/one-token-prefill-investigation.md`）
记录了原生 graph 的单 token P 首输出不稳定，而原生 eager 与插件 graph 在
同一探针中稳定；双 token P 的后续差异在三路均存在。证据支持继续检查原生
GDN graph 路由与状态更新，尚未证明缓存泄漏，也不能据此归因所有生成差异。

性能测试应固定相同模型、并行方式、缓存预算、请求到达序列和输出长度，
分别报告 TTFT、逐 token TBT、请求级 TPOT、吞吐量及给定 SLO 下的 goodput。
一个 SSE content event 可能携带多个 token，因此 inter-event gap 只能作为
流式响应间隔，不能直接冒充逐 token TBT。所有性能结论都需要保存原始回执，
未经测量不报告收益倍数。

[`test_decoupled_scheduler.py`](../tests/test_decoupled_scheduler.py) 使用真实
vLLM scheduler 和 KV allocator，覆盖 D 跨轮输出、完整 P 成本、槽位预留、
KV 抢占、部分/全部取消、晚到取消、首 token 即结束、错序回执、keep pause
提前拒绝以及原生 `EngineCore.step()` 的零 token continuation/ACK 路径。
这些主机侧回归不能替代真实 worker 的模型输出、设备访存和多卡通信验证。
trace 的 `decoupled_schedule` / `decoupled_complete` 记录 cohort、pending P、
本轮 D、配额来源、prefill-first 债务、层进度来源和 ACK；后续 P 层执行不会
再次记入 prompt token admission。trace 可关闭以测量常规执行性能；关闭 trace
的运行不能作为逐轮层进度证据，需保留独立协议验证回执。


TP4 optimization links open the benchmark member manifest. Retrieve the named original through EVIDENCE.md; omitted full captures are not recovered.
