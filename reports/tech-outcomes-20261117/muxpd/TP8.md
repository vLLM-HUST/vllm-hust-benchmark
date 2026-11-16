# TP8 历史争用证据：Guard v2 的延迟取舍

在 Qwen3.5-35B-A3B BF16、8×910B2、TP8+EP8 的三种固定合成长 prefill 争用负载中，历史 Guard v2 将持续生成请求的最大客户端 token 事件间隔
P95 从 **385–398 ms 降至 66–68 ms**，代价为注入 P 请求 **TTFT P95 增加 22.62%–28.49%**、输出吞吐下降
**1.96%–3.54%**。每种负载六轮均满足本次 D 100 ms / P 1 s 观察目标。这是特定模型、版本、到达计划下的延迟取舍， 不能据此声称硬 SLO、全面吞吐加速或完整
MuxWise 移植。

本节使用 2026-10-11 实测历史原件，于本轮重新计算；没有新增设备运行。 候选是 `258aef6ce52b7fd77cfe9a7f70d1ea29a13f7d41` 的 **Guard v2
fenced 执行路径**，不是当前默认模式或当前版本性能的测量。其 `decode_budget_guard` 为显式启用选项，历史默认仍为关闭。TP2 的工作负载、候选和 SLO 见本目录
[总报告](README.md)，两者不合并为同一结果。

## 固定原件与保管关系

- [冻结的 TP8 争用报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b354f203cabb4cfeba72a74e4eefb29dee85776b/archive/mod-targeted-20261023/muxpd-tp8-groupfx/snapshot/docs/performance-tp8-contention.md)
  包含四角色、无 P 控制、版本变更、排除决定、失败记录、质量和生命周期证据。
- [固定集合 manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
  记录来源仓库提交 `935ce04ab2b52e8b7f8bc43b53cc74475bc19f60`、全部成员 路径/字节数/SHA-256 与 Release
  包摘要。保管迁移本身不构成新增性能验收。
- [Release 原始包 muxpd.tar.gz](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-mod-collections-20261023-v1/muxpd.tar.gz)：
  **120,091,343 bytes**；SHA-256 `937ae9c56f777efe2fe5b7499a29de9b52a0dbd68e37a0cf4e7565bc8612afbe`。
  Release URL 与固定 manifest 中的摘要共同确定读取对象，不重复上传原始包。

包内成员前缀统一为 `snapshot/evidence/2026-10-11-tp8-contention/`。 本轮直接读取以下 12 份原始请求报告：

```text
native16k-{initial,confirm}-{p8k,p12k,p12k-burst}.json.gz
guard-v2-{initial,confirm}-{p8k,p12k,p12k-burst}.json.gz
```

这里花括号表示逐项展开，实际完整成员名、压缩字节数、压缩/解压 SHA-256 逐一列于 [tp8-recomputed.json](tp8-recomputed.json) 的
`sources`；压缩字节数与 摘要也嵌入 [recompute_tp8.py](recompute_tp8.py)，来源为上述固定 manifest。 `results`
保留每个阶段、每轮及每个请求的指标，未复制原始 SSE 或 token 数组。

同包内可复核的上下文成员为 `environment-initial.json`、 `source-provenance-audit.json`、`experiment-plan-v2.json`、
`native16k-{initial,confirm}-manifest.json`、
`guard-v2-{initial,confirm}-manifest.json`、`contention-audit.json.gz`、
`guard-v2-diagnostic-correctness.json`、`guard-v2-diagnostic-check.json` 和
`guard-v2-diagnostic-lifecycle-proof.json`。所有成员均沿固定集合 manifest 找到其摘要；本轮新脚本只重新审计上述 12 份计时报告。

## 环境、负载与命令

环境由归档运行回执记录，未从目录名推断硬件：

| 项目                   | 历史运行值                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 模型                   | Qwen3.5-35B-A3B，BF16；解析路径 `/data/shared_upload/models/vspec/Qwen3.5-35B-A3B`                                 |
| vLLM                   | `d0f22d2bda562156e4dbf433ce645e1769b4f804`                                                                         |
| vLLM-Ascend            | `03766ac696fde5ab1980d80ca0b8543d3580c989`                                                                         |
| Guard v2 插件          | `258aef6ce52b7fd77cfe9a7f70d1ea29a13f7d41`                                                                         |
| 原四角色比较中的旧插件 | `7cb5088137a0d46a39408a9f034f5261a9307519`；不参与本轮两角色复算                                                   |
| Producer SHA-256       | `6f373ff7383ccc61579d4e9406ee0c9b9523ee8a4ef0d5330d76471e421f6712`                                                 |
| 设备与并行             | 8×Ascend 910B2；TP8+EP8；32 核 CPU 配额，256 GiB 内存上限                                                          |
| 服务配置               | `max_model_len=16384`、`max_num_seqs=8`、显存利用率 0.85；同步调度，关闭 prefix cache；D graph capture `[1,2,4,8]` |
| 基线                   | Native 16K：token budget 16384，启用原生 chunked prefill；无 MuxPD worker/scheduler 选择参数                       |
| 候选                   | Guard v2：token budget 16384，关闭 chunked prefill，独立 D graph / fenced P 层调度，显式开启预算保护               |

Native manifest 中出现插件提交/源码哈希不等于运行使用插件：其实际 `process_argv` 没有 MuxPD worker/scheduler 选择器。共享原生 runtime
提交由初始环境和历史来源审计 关联，未对每个原生执行步骤独立封存全部 runtime 源码。模型 metadata 与来源说明 见冻结报告关联的
`performance-tp8.md`；未完整计算模型权重 SHA，不能表述成权重 逐字节认证。

每轮四个 D 请求均为 128 输入 / 256 输出。P8K 注入 1 个 8192 输入 / 8 输出 P； P12K 注入 1 个 12000 / 8 P；P12K burst 注入 3 个
12000 / 8 P，间隔 0.8 s。 首次注入在全部 D 首 token 后 0.2 s；合成 token IDs 使用 seed 42、temperature 0、
`ignore_eos=true`、`return_token_ids=true`、`add_special_tokens=false`。

精确服务器与客户端 argv 保留在四个选中阶段 `*-manifest.json` 的 `argv`、
`process_argv`、`client_runs[].argv`。公共服务器参数相同，角色选择参数分别为：

```text
Native 16K:
  --max-num-batched-tokens 16384 --enable-chunked-prefill

Guard v2:
  --max-num-batched-tokens 16384 --no-enable-chunked-prefill
  --scheduler-cls vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler
  --worker-cls vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker
  --batch-admission-policy vllm_muxpd.policy.MuxPDPolicy
  --batch-admission-policy-config '{"execution":"decoupled","stream_mode":"fenced","layers_per_slice":8,"adaptive_layers":true,"prefill_first":true,"decode_graph":true,"compile_full_prefill":true,"prefill_layer_token_budget":262144,"decode_budget_guard":true}'
```

以下归档 producer 的调用模式展开为每阶段三个负载，服务 URL 和输出路径取相应 阶段 manifest 的实际值；它说明历史命令，不由本轮审计启动服务：

```sh
python scripts/benchmark_contention.py --base-url "$BASE_URL" \
  --prefill-start-delay 0.2 --repeats 3 --label "$STAGE-p8k" --output "$OUT/p8k.json"
python scripts/benchmark_contention.py --base-url "$BASE_URL" \
  --prefill-start-delay 0.2 --repeats 3 --label "$STAGE-p12k" --output "$OUT/p12k.json" \
  --prefill-input-tokens 12000
python scripts/benchmark_contention.py --base-url "$BASE_URL" \
  --prefill-start-delay 0.2 --repeats 3 --label "$STAGE-p12k-burst" --output "$OUT/p12k-burst.json" \
  --prefill-input-tokens 12000 --prefill-requests 3
```

## 计算口径与六轮结果

D 指最初持续生成的四个请求，P 指后来注入的长输入请求。D 请求先计算**整个请求内** 相邻含 token IDs 的 SSE 事件到达间隔的最大值，再对六轮全部 24 个请求最大值求 P95。
包含初始准入和跨注入边界的完整间隔，不裁剪成较短窗口。P TTFT 从该请求实际 dispatch 到首个含 token IDs 的 SSE 事件；每角色 P8K/P12K 为 6 个 P
样本，burst 为 18 个。

P95 为排序后位置 `(n−1)×0.95` 的线性插值。六轮合并请求后计算，不能平均阶段或 单轮 P95。吞吐为六轮 D+P 总输出 tokens 除以六轮完整窗口总时长；预热不计入。 TTFT
增幅为 `100×(Guard/Native−1)`，D gap 降幅为 `100×(1−Guard/Native)`。

| 负载       | D 最大 gap P95，Native→Guard ms | P TTFT P95，Native→Guard ms | P TTFT 增幅 | 吞吐，Native→Guard tokens/s |   吞吐变化 |
| ---------- | ------------------------------: | --------------------------: | ----------: | --------------------------: | ---------: |
| P8K        |            391.581372→65.841721 |       411.433912→504.481980 | +22.615556% |       166.475509→162.621070 | −2.315319% |
| P12K       |            385.259117→67.467672 |       406.141451→521.854193 | +28.490749% |       168.166156→162.215196 | −3.538738% |
| P12K burst |            397.861825→68.192134 |       415.519091→522.303360 | +25.699004% |       152.309587→149.328564 | −1.957213% |

D gap P95 降幅依次为 83.185686%、82.487716%、82.860348%。Guard 每负载 24/24 D 满足最大 gap≤100 ms，P 为 6/6、6/6、18/18
满足 TTFT≤1 s，整轮联合通过均为 6/6。 Native D 为 0/24，P 全部通过，整轮联合为 0/6；这不表示 Native 所有请求都失败。 每轮和每请求值见 JSON
`results[role/workload].rounds`，共 36 个 measured rounds、 204 个请求；原始 warmup 仍在包中，本轮未把 warmup 当作计时样本。

## 复算方式和本轮验证

从仓库根目录执行，只有标准库依赖；`/path/to/muxpd.tar.gz` 是上面 Release 包的 本地副本。脚本先检查整个包字节数/SHA-256，再逐一检查 12
个成员的字节数/SHA-256， 只在内存读取 tar/gzip/JSON，不解包到文件系统，也不执行归档代码：

```sh
python3 reports/tech-outcomes-20261117/muxpd/recompute_tp8.py \
  /path/to/muxpd.tar.gz --check
```

`--check` 默认与脚本同目录的 `tp8-recomputed.json` 作完整字节比较，不写文件。 如需另存本轮复算结果并比较：

```sh
python3 reports/tech-outcomes-20261117/muxpd/recompute_tp8.py \
  /path/to/muxpd.tar.gz --output /tmp/tp8-recomputed.json
python3 reports/tech-outcomes-20261117/muxpd/recompute_tp8.py \
  /path/to/muxpd.tar.gz --output /tmp/tp8-recomputed.json --check
```

输出不含本次时间戳或审计工作区路径。脚本从 raw SSE 校验字节流哈希/长度与保存事件的 payload 顺序，重新解析 token IDs，确认 prompt
IDs、完整输出数量、usage、HTTP/transport 与 `finish_reason=length`；再从原始事件到达时间计算 gap/TTFT。本次选中事件均为单 token， 不存在为
multi-token 消息虚构内部 token 时刻的情况。本轮生成、默认 `--check` 和独立输出 比较均通过；更改输出文件后 `--check` 非零退出，错误的归档输入亦非零退出。

## 保留的失败、质量与适用边界

- 原始最终四角色比较还包含 Native 2K 和旧插件。Native 2K 是公平的小 chunk 对照； 原报告已完整保留，不能用本节两角色表宣称胜过所有原生策略或其他平台的 chunked
  prefill。 原四角色最终选择为 native16k/native2k initial+confirm、旧插件 initial+confirm-clean、 Guard v2
  initial+confirm。CPU 测试重叠的旧 confirm 和 Guard v1 的全部结果保留并单列， 不与本次 v2 混池。依赖遮蔽造成的服务 ready
  前启动中止也保留，未伪装成计时轮。
- 这是客户端事件到达时延，非平均 TPOT、服务端采样或 kernel 时间。100 ms/1 s 是有限样本 观察目标，网络/CPU/跨 rank
  等待的未来抖动没有硬上界；六轮范围不是置信区间或生产容量。
- 原报告 Guard v2 诊断自然语言对照为 16/16 文本精确一致，但不等于 token IDs 精确一致或 通用质量验收。合成负载对 Native 全序列一致仅为 44/126；Native
  自身重复亦不稳定， 不能把所有差异归因于插件，也不能宣称任意输入数值等价。
- v2 独立生命周期诊断校验通过 983 条进度回执、26 个 cohort，包含完成、取消 ACK 和 16383-token 完整准入；该历史诊断不是本轮新的设备验证，也不替代当前版本检查。
- fenced 路径没有移植论文 CUDA Green Context 空间资源分区及完整争用预测器。独立 old/native profile 的跨流 compute 交叠为 0，不能用客户端
  SSE 或旧 profile 声称 Guard v2 硬件并行。 Guard 同时改变部分 P 的 FX/raw 执行形态和切层配额，本实验未完整隔离两因素因果贡献。
- 公开初始环境快照移除了无关系统 `processes` 字段，原/公开摘要区别由归档索引记录；其余 选中的原始计时材料按 manifest 保留。此次增补没有修改历史原件或撤下负结果。
