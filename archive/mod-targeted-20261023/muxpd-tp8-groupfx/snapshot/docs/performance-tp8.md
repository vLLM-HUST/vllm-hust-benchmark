# TP8 验证与性能比较

本报告记录 2026-10-10 将已有 MuxPD 配置迁移到八张 Ascend 910B2、
Qwen3.5-35B-A3B BF16 的验证。本轮比较原生 graph 与 MuxPD work256 graph，
沿用已合并实现，不把 TP4 的通过结果或性能收益直接当作 TP8 结果。

**六轮汇总的吞吐基本持平，尚无稳定加速证据。** 混合负载为原生 177.302 →
插件 177.314 tokens/s（+0.007%），decode-heavy 为 243.153 → 244.002
tokens/s（+0.349%）。原生自身前后两阶段分别变化 +1.91% / +4.33%，大于
上述差值；候选相对后测原生阶段反而低 0.94% / 1.78%。不能把小幅正差值
写成稳定收益，或只选择较慢的原生 initial 作为对照。

自然语言 16/16 对照文本一致，两侧合成请求传输均 56/56 通过，但合成文本
只有 33/56 跨端点一致，且双方均有重复不稳定。三路各 30 条状态探针和
插件严格 trace 生命周期检查已完成，观察范围与未确定根因见下文；不能将
这些有限用例概括为任意输入的输出等价或全部正确性问题已解决。

## 本轮环境与模型来源

容器于 2026-10-10 05:24:16 UTC 启动。初始记录未发现 vLLM API/engine/worker
进程，八张 NPU 无运行任务；这是启动服务之前的观察，不是后续采集时的状态。
完整动态记录位于 [environment.json (archive member: `evidence/2026-10-10-tp8/environment.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，
保留了同轮初始快照、`npu-smi`、模型修复记录的路径和散列。该快照只记录
启动配置；后续服务可用性由独立请求回执记录。完整证据目录包含环境、
启动日志、运行清单、原始结果及离线审计脚本。

| 项目 | TP8 本轮配置或观察 |
| --- | --- |
| 设备 | 8 × Ascend 910B2；每卡 HBM 容量 65536 MiB |
| CPU / 内存 | cgroup `cpu.max=3200000 100000`，即 32 核配额；内存上限 256 GiB |
| CPU affinity | 允许 CPU 0–191；不等于可占满 192 核 |
| 软件 | CANN 9.1.0；driver 26.0.rc1；torch 2.10.0+cpu；torch-npu 2.10.0.post4 |
| 运行方式 | V1 同步 runner，TP8 + EP，DP/PP/PCP/DCP 均为 1，language-model-only |
| HUST vLLM | `d0f22d2bda562156e4dbf433ce645e1769b4f804` |
| HUST vllm-ascend | `03766ac696fde5ab1980d80ca0b8543d3580c989` |
| 插件运行基线 | 已合并 main `3eb6f2ea0dea5b95109435182d106ff1ce15e8cd` |

Ascend checkout 的既有 `csrc/build_aclnn.sh` 本地修改及 SHA 已单列于环境
记录；插件基线采集时没有 tracked 修改。每次实际启动仍应记录源码清单，不能
让后续修改自动归入此前测量。环境采集仅使用系统、git、包 metadata 和文件
读取，没有导入 torch/torch_npu、创建设备上下文或向推理 API 发请求。

与前次 TP4 相比，CPU 配额由 **16 核变为 32 核**，容器和可见设备集合也变更。
因此本轮可比较同一 TP8 环境中的原生与插件；不能把历史 TP4 → 本轮 TP8 的
吞吐比直接写成纯 TP 扩展效率，也不能由卡数翻倍推断吞吐或延迟变化。

模型入口仍为 `/root/workspace/models/Qwen3.5-35B-A3B`。原软链指向已经不可用
的 `/data/home/vllm-hust-mhy-21rc/workspace/models/Qwen3.5-35B-A3B`，本轮修复
为 `/data/shared_upload/models/vspec/Qwen3.5-35B-A3B`；只修改本地软链，没有
改写共享模型文件。[model-resolution.json (archive member: `evidence/2026-10-10-tp8/model-resolution.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
保存修复前后路径和逐文件比较。

新位置的 10 项现有模型/tokenizer metadata SHA-256 与 TP4 库存匹配，14 个
safetensors 分片的文件大小也匹配。旧位置的 ModelScope `configuration.json`
在新位置缺失。**未计算全部权重分片 SHA-256，也未建立不可变上游模型 revision**；
metadata 与大小一致不证明两处权重逐字节相同。两种 TP8 服务使用同一个新
解析路径，历史 TP4 对照须保留这项限制。

## TP8 的每 rank 形状

模型共有 40 层，其中 30 层 GDN/linear attention、10 层 full attention。
以下由本轮读取的 `config.json` 和固定源码的分片公式推导，不是设备 tensor
dump 或 kernel 正确性测试结果。

| 项目 | 模型总量 | TP8 / EP8 下每 rank |
| --- | ---: | ---: |
| Full attention Q heads | 16 | 2 |
| Full attention KV heads | 2 | 1；每个全局 KV head 复制到 4 个 rank |
| Full attention head dimension | 256 | 256 |
| GDN key heads | 16 | 2 |
| GDN value heads | 32 | 4 |
| GDN key / value head dimension | 128 / 128 | 128 / 128 |
| Routed experts，无 redundant experts | 256 | 32；shared expert 另计 |

公式来源为 [Qwen3NextAttention](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/model_executor/models/qwen3_next.py#L237)、
[EP 专家划分](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/model_executor/models/qwen3_next.py#L109)
和 [Ascend GDN 分片](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L113)。
这些形状区别于 TP4，需要在本轮重新检查输出、状态复用和 graph replay。

## 匹配的 graph 对照

两种服务使用相同模型解析路径、TP8 + EP、BF16、同步调度、16384 的模型长度
和 token budget、`max_num_seqs=8`、`gpu_memory_utilization=0.85`、prefix cache
关闭、`mamba_cache_mode=none`、无 speculative decoding。两者均使用
`VLLM_COMPILE` + `FULL_DECODE_ONLY`，capture sizes 为 `[1,2,4,8]`，关闭
`enforce_eager`。有效参数见
[native initial (archive member: `evidence/2026-10-10-tp8/native-initial-manifest.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)、
[work256 (archive member: `evidence/2026-10-10-tp8/work256-manifest.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 和
[native bracket (archive member: `evidence/2026-10-10-tp8/native-bracket-manifest.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 的进程清单。

| 项目 | 原生 graph | MuxPD work256 graph |
| --- | --- | --- |
| Scheduler / worker | 原生 vLLM / Ascend | DecoupledMuxPDScheduler / DecoupledMuxPDNPUWorker |
| Prefill 输入 | 保留原生 chunked-prefill 调度 | 完整 prompt，不做 token chunk |
| P/D model batch | 保留原生混合批次路径 | P 与 D 分别调用模型 |
| 图执行 | 原生 `FULL_DECODE_ONLY` 判定 | 仅独立 D runner capture/replay；P 强制 graph mode `NONE` |
| 跨轮 P | 原生调度方式 | 按模型层保留 P 中间状态，D 独立采样 |
| Stream mode | 原生 | `fenced`；不以此证明 P/D kernel 并发 |

候选沿用 [TP4 性能报告](performance-tp4.md) 的 work256 联合配置，实际由
归档的 [work256-launch.sh (archive member: `evidence/2026-10-10-tp8/work256-launch.sh`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 直接
调用 Python API server，显式设置 TP 为 8。其 policy/graph 参数等价于
[`serve-qwen35-decode-graph.sh`](../scripts/serve-qwen35-decode-graph.sh) 的配置：

```json
{
  "execution": "decoupled",
  "stream_mode": "fenced",
  "layers_per_slice": 8,
  "adaptive_layers": true,
  "prefill_first": true,
  "decode_graph": true,
  "compile_full_prefill": true,
  "prefill_layer_token_budget": 262144
}
```

预算按完整 cohort 的 P token 总量计算基础层配额，再与 deadline 所需配额
取较大值；它不是 token chunk 大小，也不是硬时间上限。prefill-first 在新 P
能整网完成时允许先执行 P，随后偿还被延后的已有 D 债务。只有新 P 的完整
forward 才可允许 FX 编译，P 的 ACL graph mode 仍是 `NONE`。未完成的 P
继续逐层执行，其取消与 KV 回收仍需 worker ACK。详细协议见
[设计文档](design.md)。

组件默认 `compile_full_prefill=false`、`prefill_layer_token_budget=0` 等保持
不变；上述是显式选择的实验配置。TP4 的联合配置收益不证明该配置在 TP8
也更快，更不能把它拆解为某个开关的独立收益。本轮计时关闭 trace/profiler；
协议诊断另行采集并记录配置，不混入计时重复。本轮没有采集设备 profiler，
不以 CPU 事件或日志证明 device 执行路径或 kernel 重叠。

## 阶段分离与资源隔离的边界

本轮候选保持完整 P 和独立 P/D model batch，在一个引擎中共享权重及 KV
所有权；D 有独立 graph、静态输入和 workspace。这不等于两份模型驻留，也
不等于将 P/D 分发到不同机器。

fenced 模式通过事件建立 P/D 执行依赖，目标是正确地跨轮保留 P 并降低主机
编排开销。本轮没有使用 CUDA GreenContext，也没有调用 Ascend
`set_stream_limit()` 分配 Cube/Vector 核，不应宣称完成硬件核分区、空间隔离
或论文中的 SLO contention predictor。即使测得吞吐提高，也不能由此推出
设备 kernel 并发。旧 TP4/TP2 overlap 或 isolated 通信组的证据不属于本轮
fenced graph 配置；graph + overlap 仍不支持。

## 测量与验收口径

性能使用 [`benchmark_serving.py`](../scripts/benchmark_serving.py) 的合成
token-ID 请求。两类负载均为每轮 8 个请求、每请求输出 64 tokens，
temperature 0、seed 42、`ignore_eos=true`，先对每个输入长度预热 8 个输出
tokens。原生 initial/bracket、候选 stage1/confirm 各为每负载三轮，合计每种
配置每负载六轮、48 个请求。逐文件参数和复算结果见
[benchmark-summary.json (archive member: `evidence/2026-10-10-tp8/benchmark-summary.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。

| 负载 | 输入长度 | 请求间隔 |
| --- | --- | --- |
| 混合 | 循环 `128,2048,512,4096` | 0.12 s |
| Decode-heavy | 全部 128 | 同时发起 |

聚合吞吐按所有有效重复的输出 token 总数除以 wall time 总和计算，包含请求
间隔、prefill 和尾部完成时间。TTFT 是客户端首个非空文本 SSE 的时间；TPOT
是每请求首末文本事件间隔除以 `completion_tokens-1`，其 P95 是请求平均
TPOT 的 P95，不是逐 token 延迟分位数。SSE 内容事件可能携带多个 tokens，
最大内容事件间隔不等于逐 token TBT。分位数使用 nearest rank：48 请求
汇总的 P95 取排序后的第 46 项，不平均各轮 P95。

保留全部有效重复和失败/不完整记录，不能挑选较慢的原生阶段作分母；少量
重复不构成统计显著性证明。预热、自适应历史及非随机测试顺序应随运行清单
保留。未固定请求形状、图模式或 CPU 配额的旧运行不纳入本轮匹配汇总。

正确性使用本轮 TP8 原生报告作为 baseline，不能复用 TP4 输出代替 TP8
对照。自然语言单请求与 paired-active、合成 graph shape、严格串行 token-ID
状态复用、长 P 期间 D 输出、取消及层进度 ACK 分别保存回执。HTTP 成功不
代表输出等价；文本差异须报告首分歧及各侧重复稳定性，不能自动归为 MoE/BF16
误差，也不能在没有内部状态证据时认定缓存污染。

## 六轮性能汇总

以下均合并各自全部六轮；百分比为 `100 × (插件 / 原生 - 1)`，延迟列负值
表示降低。八份原始性能 JSON 均通过 usage、内容事件时间顺序、时间范围及
存储指标复算，输入散列和逐轮数据保留在
[完整汇总 (archive member: `evidence/2026-10-10-tp8/benchmark-summary.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。

| 负载 | 原生 → 插件 tokens/s | 吞吐变化 | 原生 → 插件 TTFT P95 / s | 原生 → 插件平均 TPOT P95 / s |
| --- | --- | --- | --- | --- |
| 混合 | 177.301681 → 177.313688 | +0.00677% | 0.777201 → 0.742212（−4.50%） | 0.039226 → 0.039567（+0.87%） |
| Decode-heavy | 243.152892 → 244.001916 | +0.34917% | 0.761966 → 0.739808（−2.91%） | 0.027746 → 0.027491（−0.92%） |

两种配置、两种负载各有 **48/48** 请求满足本脚本阈值 `TTFT ≤ 2 s` 且
`请求平均 TPOT ≤ 0.1 s`。这是有限合成负载上的观测，不是任意到达率下的
SLO 保证。混合负载的最大内容事件间隔 P95 为 0.400017 → 0.413623 s
（+3.40%）；decode-heavy 为 0.384968 → 0.379985 s（−1.29%）。因此连同
混合 TPOT 的上升，不能把 TTFT 降低描述成所有延迟指标改善。

为保留阶段波动，以下列出每个三轮阶段的聚合吞吐，而不是只展示最优一次。

| 按执行顺序的阶段 | 混合 tokens/s | Decode-heavy tokens/s |
| --- | ---: | ---: |
| Native initial | 175.642782 | 238.103726 |
| Work256 stage1 | 179.655585 | 246.134515 |
| Work256 confirm | 175.032061 | 241.905954 |
| Native bracket | 178.992215 | 248.420840 |

原生 initial → bracket 自身变化 **+1.91% / +4.33%**，超过候选对原生六轮
汇总的差值。候选全部六轮若与 native bracket 比较，混合为 **−0.938%**、
decode-heavy 为 **−1.779%**。阶段顺序、预热、缓存或其他运行波动尚未通过
随机交错及更多独立启动隔离；当前证据不支持稳定吞吐增益。候选 stage1 与
confirm 属于同一服务生命周期，不能当成两个独立冷启动。

Native bracket 的 [启动日志 (archive member: `evidence/2026-10-10-tp8/native-bracket-server.log`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
记录 AOT cache 加载 Triton 失败后自动重编译的 warning。
计时在服务健康就绪、脚本 shape warmup 完成之后开始，没有将启动重编译耗时
计入吞吐。计时结束后的
[initial/bracket 自然语言复验 (archive member: `evidence/2026-10-10-tp8/native-initial-vs-bracket-correctness.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
为 **16/16 文本一致**，两侧 oracle、重复稳定性及 isolated/paired 对照通过。
它支持保留 bracket 作为已完成的本轮对照，但不能证明两次启动的所有内部
执行路径完全相同，也不能把启动 warning 本身归因为推理退化。

## 已完成的输出对照

本节依据 [correctness-summary.json (archive member: `evidence/2026-10-10-tp8/correctness-summary.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
及其原始回执。自然语言和合成 SSE 比较的是解码文本，状态复用探针比较的
是服务返回的 token IDs；这两种“精确一致”不能混用。汇总已复核服务 manifest
中的成功客户端回执及输入 SHA，复用的 baseline 也与最初报告逐条核对。

| 验证项目 | 原生 graph TP8 | MuxPD work256 graph TP8 |
| --- | --- | --- |
| 自然语言 oracle / 完成回执 | 16/16 | 16/16；与原生 16/16 文本一致 |
| 自然语言重复与 isolated/paired | 均通过本组检查 | 均通过本组检查 |
| 合成 SSE transport | 56/56 | 56/56；与原生 33/56 文本一致 |
| 合成重复不稳定组数 | 10/28 | 11/28 |
| 严格串行状态探针 transport | 30/30 | 30/30 |

自然语言原始报告为 [native-correctness.json (archive member: `evidence/2026-10-10-tp8/native-correctness.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
和 [native-vs-work256.json (archive member: `evidence/2026-10-10-tp8/native-vs-work256.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)；合成报告为
[native-serving.json (archive member: `evidence/2026-10-10-tp8/native-serving.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 和
[native-vs-work256-serving.json (archive member: `evidence/2026-10-10-tp8/native-vs-work256-serving.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。
合成 transport 审计逐条重放保存的 SSE body，核对 HTTP 200、响应标识、
`[DONE]`、finish reason、usage 和 body SHA；自然语言报告保存事件与完成
标志，没有原始 HTTP body/status 字段，证据范围较窄。客户端并发也不单独
证明设备上的实际 decode batch size、graph replay 或 P/D 重叠。

原生与候选都有合成输入重复不稳定，不能把全部 23 条跨端点文本差异都归为
插件引入，也不能据此排除插件特有错误。性能 JSON 自身亦保留逐请求文本
hash 变化；其 synthetic token IDs 不构成自然语言质量测试。

状态复用使用 `[100]`、`Hello=[9419]`、`HelloHi=[9419,12675]`，分别生成
1/16 tokens。每组为两次 before、一次 4096-token P + 8-token D 背景请求、
两次 after；六组共 30 请求，时间戳核对为严格串行。`before` 不是新进程或
清空缓存后的观察。原始数据为
[native-state-reuse.json (archive member: `evidence/2026-10-10-tp8/native-state-reuse.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 和
[work256-state-reuse.json (archive member: `evidence/2026-10-10-tp8/work256-state-reuse.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，另补
[native-eager-state-reuse.json (archive member: `evidence/2026-10-10-tp8/native-eager-state-reuse.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
作为原生 eager 对照；三路各 30/30 完成传输、usage、token-ID 与严格串行
顺序检查。eager 启动配置见
[native-eager-manifest.json (archive member: `evidence/2026-10-10-tp8/native-eager-manifest.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。

| Prompt / 输出上限 | 原生 graph | 原生 eager | 插件 graph |
| --- | --- | --- | --- |
| `[100]` / 1 | 从首个生成 token 分歧 | 四次完整序列稳定，首 token 为 8 | 四次完整序列稳定，首 token 为 8 |
| `[100]` / 16 | 从首个生成 token 分歧 | 首 token 为 8；before 的后段有差异 | 四次完整序列稳定，首 token 为 8 |
| `Hello` / 1 或 16 | 两组均从首个生成 token 分歧 | 两组四次完整序列稳定，首 token 为 11 | 两组四次完整序列稳定，首 token 为 11 |
| `HelloHi` / 1 | 四次稳定，首 token 为 0 | 四次稳定，首 token 为 0 | 四次稳定，首 token 为 0 |
| `HelloHi` / 16 | 首 token 均为 0，后段不稳定 | 首 token 均为 0，后段不稳定 | 首 token 均为 0，后段不稳定 |

四个单 token 分组中，首生成 token 稳定的组数分别为原生 graph **0/4**、
eager **4/4**、插件 **4/4**；完整序列稳定的组数分别为 **0/4、3/4、4/4**。
eager 的 `[100] / 16` 在 before 两次请求的生成 index 7 出现 `23 ↔ 22`
差异，不能写成 eager 所有单 token 长生成都稳定，也不能声称其全部完整
序列与插件相同。

双 token/max16 的 before、after、跨历史比较首差异索引，原生 graph 为
`2 / 1 / 1`，eager 为 `相同 / 1 / 1`，插件为 `相同 / 1 / 6`；索引从首个
生成 token 的 0 开始。双 token 对照只在首 token 和 max1 上稳定。
这将 native graph 的单 token 首步现象与三路共有的后段差异分开，尚不足以
确认错误 cache 索引、缓存污染或唯一根因。TP4 的三路调查只作为背景，
没有替代本轮 TP8 对照。

## 跨轮 prefill、取消与 decode 债务

这部分使用独立启动的 [work256 traced 服务 (archive member: `evidence/2026-10-10-tp8/work256-traced-manifest.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，
在 [纯 P 校准请求 (archive member: `evidence/2026-10-10-tp8/traced-calibration.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 后执行长 P、取消
和自然语言对照。它们没有计入前述吞吐。实际配额由本轮校准与 trace 读取，
不能把校准后的行为当成冷启动时的固定保证。

[严格 checker (archive member: `evidence/2026-10-10-tp8/traced-check.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 通过：22 个 cohort，
21 个完成、1 个取消，466 个层进度回执。最大完整 P 为 12000 tokens，最大
P 完成前 D retirement 步数为 1。回执来源为 20 次 `full_forward_return`、
4 次 `layer_hooks`、442 次无 P 的 `none`；这些是 host 协议计数，不是
device kernel 或 FX 命中次数。

[lifecycle-proof.json (archive member: `evidence/2026-10-10-tp8/lifecycle-proof.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 结合客户端时间
窗口定位 cohort，并保存关键原始 trace 行：

| 用例 | 完整 P 准入 | 实际层进度 | ACK 与随后状态 |
| --- | --- | --- | --- |
| 正常长 P，cohort 4 | step 21 一次准入 12000 tokens | step 21：0→21；step 22：21→40，新增 19 层 | step 22 完成 ACK；step 23 清除 pending cohort |
| 取消长 P，cohort 6 | step 86 一次准入 12000 tokens | step 86：0→21；step 87：21→21，新增 0 层 | step 87 取消 ACK；step 88 清除 pending cohort |

两次 12K P 的后续步骤都没有重新准入 prompt token。正常用例的首轮有 D
完成回执；[客户端进度报告 (archive member: `evidence/2026-10-10-tp8/traced-progress.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 记录 P
请求开始后、P 首内容事件之前有 **1 个 D 内容事件**，支持 P 悬挂期间 D
确实推进。它不是多个独立 token 的精确计时，也不证明 kernel 同时执行。
[取消报告 (archive member: `evidence/2026-10-10-tp8/traced-cancel.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 与上述 ACK 联合验证 host
生命周期；没有读取 KV tensor 内容，不能把 cohort 清除写成设备内存已逐字节
归零的证明。

本次还观察到 4 组 `token_budget_prefill_first`，发生在步骤
`236→237`、`301→302`、`395→396`、`460→461`。每组被延后的 D 都在下一步
获调度、完成并清债，computed-token 计数没有因延后而提前增加。该证据覆盖
这 4 组请求，不等于任意负载下的公平性上界。

traced 服务的 [自然语言复验 (archive member: `evidence/2026-10-10-tp8/native-vs-work256-traced.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
另有 16/16 文本与原生一致。完整
[1398 行 trace (archive member: `evidence/2026-10-10-tp8/work256-traced-86528.jsonl`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 为 876277 bytes，
SHA-256 `935052d0ab64e8a53b9021706c6d2abbc53534f0846f2e5109ccd11636967740`。
生命周期审计选取的 54 行与该原始 JSONL 对应，未用 TP4 的预设层数替代实测。

## 证据复算与结束状态

原始性能、输出对照、状态探针、启动日志及 source/producer SHA 均保留于
[本轮证据目录 (archive member: `evidence/2026-10-10-tp8/README.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。三个离线入口分别为
[性能汇总 (archive member: `evidence/2026-10-10-tp8/summarize_tp8.py`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)、
[正确性审计 (archive member: `evidence/2026-10-10-tp8/correctness-summary-audit.py`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 和
[生命周期审计 (archive member: `evidence/2026-10-10-tp8/audit_lifecycle.py`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。这些脚本读取已有
请求回执和 trace 生成离线摘要，不发起推理请求；性能 JSON 没有保存的原始 SSE、finish reason 等字段不能由汇总
补推，应查独立 serving 回执。

[final-runtime-validation.json (archive member: `evidence/2026-10-10-tp8/final-runtime-validation.json`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
确认本轮 12 个 runtime Python 文件 SHA 未变，运行基线仍为上述已合并 commit。
本轮进行了 TP8 硬件验证与离线审计，没有将 TP4 历史单元测试计数算作本轮
重跑结果。初始无服务；结束时 native initial、work256、native bracket、
native eager、work256 traced 五组实验服务及其子进程均已停止，八张 NPU
报告无运行进程。进程消失不单独证明每个 worker 都完成了正常 allocator
清理；停止回执保留了这一限制。

本轮没有 TP8 device profile。因此不能从性能、graph 模式配置、host 层回执
或客户端事件推断硬件空间并发、逐核占用和真实利用率；也不能将 TP4 的
device trace 迁移为 TP8 证据。
