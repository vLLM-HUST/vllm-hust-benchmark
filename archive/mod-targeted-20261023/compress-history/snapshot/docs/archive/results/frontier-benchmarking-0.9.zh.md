# Frontier 刷榜规程（0.9.0）

> 0.9 历史过程与发布验证；新测量请使用
> [测试规程索引](../../benchmark/README.zh.md)。

[English](frontier-benchmarking-0.9.md) | 简体中文

在 0.9 测试过程中，本规程替代旧 V4.6 A2/A3 作为刷榜目标；旧结果保留为历史证据，不能与这组工作负载合并。0.9.0 发布验证沿用本规程，但固定了新的模型 revision 和宿主栈；下文 2026-09-27 的 AgentX 正收益与 SWE 退化是合并源码提交 `17ffdc7` 上的历史结果。本地结果不等于官网正式提交。

SWE 的测试资格及将来向官网提交成绩，按[Frontier 成绩提交指南](https://github.com/vLLM-HUST/vllm-hust-website/blob/codex/frontier-submission-example/data/examples/frontier-submission/README.md)执行：先做独立的 C4／60 秒协议检查，再测 900 秒；保留原始 `summary.json`、`config.json`，将 `output_tokens_per_second` 填为总吞吐 `metrics.output_tps`（官网再按卡数相除）。本地测试或插件 HTML 页面上的点不会自动登上官网。SWE 使用 MTP 时必须保留真实生成 token 和实测接受率，不能套用历史 AgentX 的 synthetic sampler 或固定接受长度。

## 冻结工作负载

| Cohort | 来源 | 测量窗口 | 上下文容量 | 主要指标 |
| --- | --- | ---: | ---: | --- |
| SWE prefix reuse | [vLLM-HUST/swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse) | 900 秒 | 262,144 | 窗口内每卡输出 token/s、请求级 P90 解码速度 |
| AgentX 256k | [vLLM-HUST/agentx-bench](https://github.com/vLLM-HUST/agentx-bench) | 900 秒，另有原版预热 | 262,144 | 官方 replay 的吞吐、交互性、时延和有效性 |

当前模型为 Qwen3.5-35B-A3B BF16、Ascend 910B2 TP=2。Qwen3.8-27B BF16 属于后续独立模型 cohort。冻结精确模型与 tokenizer revision、准备文件 SHA-256、数据集 revision、客户端提交、宿主提交、插件提交与 wheel hash。SWE 的两次对照必须使用**同一个**准备文件；准备文件哈希或 token 增量不同就不是直接可比较的点。AgentX 不得筛选会话、裁剪输出、缩短原版预热或重写 DAG 与延迟。

2026-09-27 历史运行的模型与 tokenizer revision 为 `59d61f3ce65a6d9863b86d2e96597125219dc754`。0.9.0 发布验证采用 `712cf74392b05026a6db2bf213d343747d1f6d45`；官网早期 BetterScale 样例也使用过该 revision，但本次测量独立，不得复用其模型或 cohort ID。

### 仅 CPU 的工作负载准备记录（2026-09-27）

本地 SWE 客户端提交 `6861242dbd9f17b707003191e4200b7752911d7c` 使用实际 Qwen3.5 tokenizer、`transformers==5.17.0`、开启 thinking 模板以及 `--max-context 262144` 编译随仓 Open-SWE 样本。来源 gzip 的 SHA-256 为 `a31abfc7dea176a8b4bf339ff8b104194040f29244577eab9ba4ca951a01739b`；tokenizer 后端及模板指纹为 `3f9ca78537850303ee04bfa6640c020be89723c62f37121c0f27a4c0babc53e0`。准备文件位于 `../.benchmarks/swe-qwen35-262144.json`（已忽略，不提交），SHA-256 为 `8105957b4001e7fd21373150fb7b844a9931abe9f88c699b3db191b84a8596d3`。其中 8 条会话、360 轮全部通过，无淘汰；最大 prompt 为 140,423 token，最大 prompt 加输出历史为 141,269 token。B0/B1 必须读取同一文件并在运行前复核哈希。配置 262,144 容量**不代表**这个 SWE 样本实际达到 256K。

本地 AgentX 客户端提交 `e0c34de525246c98d14df590f9864d7be7a25075` 的隔离缓存已备有固定的 `semianalysisai/cc-traces-weka-062126-256k` 数据集 revision `8fecd2fc56694469f758f0afbbb6335ad3043740`；准备记录为 393 条会话、指纹 `0d8fdac271289f80`，且已重新离线核对。其 `smoke` 档在官方 primers 与每 lane 额外 10 次预热请求**之后**测量 900 秒。实际运行时仍须再次校验客户端离线数据身份。两个本地客户端测试均通过（SWE 24 项、AgentX 18 项）；这些仅是输入准备与客户端检查，不是服务端协议资格或性能结果。详见[机器可读准备记录](../../results/evidence/frontier-workload-preparation-20260927.json)。

固定的 AgentX 包装器要求投机解码具备匹配的 SPEED-Bench 接受率证据与服务端强制接受设置；本次尚无 Qwen3.5/CANN 9.1/MTP2 对应证据和强制接受适配器。因此 AgentX 的 B0/B1 都必须关闭 MTP 并声明该配置。SWE 是另一 cohort，以真实 MTP2 生成和接受率测试，不使用 synthetic sampler；不能直接比较两个工作负载的绝对成绩，也不能把 AgentX 运行当作 MTP2 验收。

## 配对与判读

1. 在授权且空闲的 NPU 上以相同硬件、模型、客户端、并发数和服务参数分别冷启动 B0（禁用压缩）与 B1（启用插件）。保存原样启动命令和硬件占用信息；不改动宿主源码。
2. 先运行短协议/缓存资格检查，再分别运行两个 900 秒窗口。SWE 需要精确 token-ID 回显、固定输出预算、流式 usage、稳定的会话缓存路由；AgentX 需要保留官方原始结果及无效原因。
3. 两组都记录实际达到的最大 prompt 长度、请求错误、服务端 prefix-cache 命中、预抢占、事务 commit/worker ack、NPU/HBM 采样以及退出和资源释放。仅设置 262,144 容量不证明已跑到 256K。
4. 同一 cohort、同一并发数下比较；报告吞吐与 P90 解码速度的绝对值及 B1/B0 差值。SWE P90 是各完整请求 `(输出 token 数 - 1) / (最后 token 时间 - 首 token 时间)` 的第 90 百分位，不能换算为 `1/P90(TPOT)`。不同模型、样本哈希或并发数不混排为单一加速比。
5. 只有协议有效、无静默失败、压缩事务与各 TP rank 回执对齐且核心指标出现正优化时，才能标记对应点的工程正结果；其他指标的退化也必须原样展示。单次 15 分钟结果不是统计显著性或正式一小时 AgentX 结果。

SWE 的未合并 B0/B1 配对可运行 `python scripts/kvcompress_frontier_swe_pair.py --baseline B0结果目录 --candidate B1结果目录`。工具只读官方 `summary.json`/`config.json`，拒绝无效或配置不匹配的两侧，输出官方总吞吐、每卡吞吐、P90 解码速度、TTFT P95 及差值。仍须人工检查原始报告与服务日志；该工具不能认证硬件身份、真实 rank 回执或公开证据的可访问性。

AgentX 配对可运行 `python scripts/kvcompress_frontier_agentx_pair.py --baseline B0运行目录/run.json --candidate B1运行目录/run.json`。工具只读原版包装器记录和 AIPerf 导出，检查两侧有效性与配置一致性，并将 `output_token_throughput.avg` 映射为总输出吞吐、`output_token_throughput_per_user.p90` 映射为官方逐请求反向 ITL 解码速度 P90；每卡吞吐按两张实际分配的卡计算。首 token 延迟、ITL、完成请求数、实际最长输入、错误与有效性也必须一并保留；900 秒冒烟不能冒充一小时正式结果。

SWE 测的是固定长会话的服务形状，不是 SWE 任务求解正确率。AgentX 的合成内容也不证明语义质量或真实工具执行成功。榜单数据与历史负结果见 [HTML 页面](../../results/benchmark-leaderboard.html)；每条新记录必须指向可审计的原始证据。

插件 HTML 页面现按官方 [Frontier 视图](https://vllm-hust.sage.org.ai/leaderboard-runs.html#frontier) 将**实测点**与**历史记录**分开，固定坐标为 P90 解码速度和每卡输出 token/s。`evidence/frontier-results.json` 已收入两组 AgentX 900 秒正收益配对的四个 B0/B1 点，含合并后复测；SWE cohort 仍无正收益点。每组点共用 `pair_id`、cohort、并发和设备数，并保留实际最长输入、原始运行 ID、报告哈希、证据链接和测量范围。这是插件本地榜单，不等于官网提交或 AgentX 一小时正式成绩。运行 `python scripts/kvcompress_leaderboard.py` 生成两个浏览器数据包，并在发布前运行 `python scripts/kvcompress_leaderboard.py --check`。不得从旧记录的百分比增量推算 Frontier 点。

旧 [V4.6 测试要求](../benchmark/kv-compress-test-requirements.zh.md)、[A2/A3 规程](../benchmark/benchmarking.zh.md)及[公开长上下文实验](public-long-context-benchmarks.zh.md)仅作为历史归档，不再是当前发布门槛。

## 2026-10-08 本地 0.9.0 发布验证

发布候选 wheel 在 910B2 设备 2、7 上以 BF16/TP=2 运行 Qwen3.5-35B-A3B，
模型 revision 为 `712cf74392b05026a6db2bf213d343747d1f6d45`。匹配的 vLLM
与 Ascend 提交分别为 `ebfcfba`、`7c8ec86`，wheel SHA-256 为 `c8de9f51...`。
服务配置 262,144 上下文、APC、真实 MTP2、异步调度、align 和
FULL_AND_PIECEWISE 图。SWE 两组各自的 C4／60 秒协议检查及 900 秒测量均有效，
请求失败数为零，准备文件 SHA-256 相同。

| SWE C4／900 秒指标 | B0，关闭压缩 | B1，启用压缩 | B1 相对 B0 |
| --- | ---: | ---: | ---: |
| 总输出 token/s | 153.297 | 149.143 | −2.71% |
| 每卡输出 token/s | 76.648 | 74.572 | −2.71% |
| P90 解码 token/s | 47.920 | 48.424 | +1.05% |
| 首 token P95，毫秒 | 1,472.57 | 1,610.33 | +9.36%（变差） |
| 窗口内完成请求 | 221 | 218 | — |
| 实际最长 prompt | 52,505 | 52,505 | — |

该配对的协议有效，**解码 P90 提升**，但总吞吐与首 token P95 退化。
实际最长 prompt 为 52,505 token，未证明真实 256K prompt 的性能。B1
服务全生命周期的 60 次调度压缩提交均有 TP0、TP1 回执；服务端还记录了真实
MTP 接受率和前缀命中。原始报告与日志保留在本机，[机器可读发布记录](../../results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)
列出其哈希。这是单组本地工程测试，不是官网成绩或可重复性证明。

同一候选随后完成 AgentX C4／900 秒配对。由于没有匹配的强制接受率证据，两组均按
规程关闭 MTP。两份 AIPerf 原始报告均标记有效、无无效原因，每组完成 65 个请求，
实际最长输入约 91.6K。

| AgentX C4／900 秒指标 | B0，关闭压缩 | B1，启用压缩 | B1 相对 B0 |
| --- | ---: | ---: | ---: |
| 总输出 token/s | 33.86587 | 33.86584 | −0.00009%（持平） |
| 每卡输出 token/s | 16.93294 | 16.93292 | −0.00009%（持平） |
| P90 解码 token/s | 27.709 | 28.024 | +1.14% |
| 首 token P95，毫秒 | 1,819.22 | 1,793.15 | −1.43%（改善） |
| token 间延迟 P90，毫秒 | 45.779 | 43.226 | −5.58%（改善） |

B1 服务记录 31 次调度压缩提交和 62 次 worker 回执，无服务错误或抢占。测量结束后
的清理阶段记录过一次超时，但全部 66 个 credit 已归还，原始报告仍有效。该结果的
**解码 P90 和延迟指标改善**，总吞吐基本不变。详见
[机器可读 AgentX 发布记录](../../results/evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json)。

## 2026-09-27 工程冒烟记录（非榜单结果）

在空闲 910B2 设备 2/3 上，以 Qwen3.5-35B-A3B BF16、TP=2、16,384 上下文、
`mamba_cache_mode=none`、无 APC/MTP/async 启动候选插件。插件内 Triton GDN
输出算子与 BF16 参考实现对照的最大绝对误差为 0.000244；8-token 短输入得到
HTTP 200 和 16-token 输出。单条 10,000-token 检索输入也得到 HTTP 200，服务
报告 10,000 个输入 token、128 个输出 token；但模型在输出上限内一直生成思考
文本，未回答目标代码，故该质量项为 **0/1，失败**。输入 token 的 SHA-256 为
`e3e11a53dbe68cf57aa6c3ebbbb773ad7a92573f3ee5370406fb959272938502`。
首轮服务日志级别为 ERROR，无法验证压缩是否实际执行；随后在 INFO 日志下重放
相同输入，记录到调度端 1 次 commit、TP0/TP1 各 1 次回执，语义 10,000 token
压至物理 8,192 token，质量仍为 0/1。详见[机器可读记录](../../results/evidence/kvcompress-qwen35-20260927-compat-smoke.json)。
该记录不包含 262,144 上下文、APC、MTP2、async、`FULL_AND_PIECEWISE`
或 900 秒官方工作负载，也没有配对基线，不计入榜单。

随后以 APC 开启、`mamba_cache_mode=align` 再次启动同一模型。两条相同的
10,012-token 非思考模式检索请求均返回正确代码 `10001337`（各 9 输出 token），
且各有 1 次调度端压缩提交和 TP0/TP1 各 1 次回执，压至物理 8,192 token。
第二条请求使服务端前缀命中累计数从 8,192 增至 16,384，即本次新增
8,192 命中 token；两次输出哈希一致。详见
[APC/align 工程记录](../../results/evidence/kvcompress-qwen35-20260927-apc-align-smoke.json)。
这是功能正确性的短冒烟，不代表 MTP2、async、图模式或 256K 官方工作负载已通过。

再启用 async 后，重复该非思考检索：相同输入串行两次均答对，第二次新增
8,192 个前缀命中 token；另外两条不同检索请求由两个客户端并发发送，分别
答对 `10001337` 与 `10009256`。四条请求均有调度端压缩提交和两个 TP rank
回执，未观察到请求串扰。详见
[async/APC/align 工程记录](../../results/evidence/kvcompress-qwen35-20260927-async-apc-align-smoke.json)。
并发客户端不等于已证明设备批次重叠；MTP2、图执行、256K 与官方 900 秒
工作负载仍待验证。

之后的显式 MTP2 诊断中，插件把额外的 MTP 注意力缓存层与 10 个已校准目标层分开，
服务因此能够启动；但首条 10,012-token 请求返回 HTTP 500，两个 TP rank 均报告
`aclnnCausalConv1d` 内部错误，当时根因尚未隔离。该次 MTP2 诊断**失败**，默认
启动拒绝；不完整路径只能用 `VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2=1` 显式
开启以便诊断。
详见 [MTP2 诊断记录](../../results/evidence/kvcompress-qwen35-20260927-mtp2-diagnostic.json)。

随后将失败定位为旧算子 ABI 的两个细节：非推测 prefill 必须只取二维 MTP 缓存表的
目标状态首列；推测卷积必须从扩展滚动状态的“已接受 token 偏移”读取历史，并计算
全部草稿位置。两处修复均留在插件内部。修复后 TP=2/eager/APC/align/async/MTP2
冒烟中，一条 7,012-token 未压缩对照和三条 10,012-token 压缩请求均答对；三条压缩
请求各有一次调度端提交与两个 TP 回执，物理预算为 8,192 token。重复请求及另一条
独立事实也正确。详见 [MTP2 工程冒烟证据](../../results/evidence/kvcompress-qwen35-20260927-mtp2-apc-align-smoke.json)。
这不构成 MTP2 默认启用资格，也不是 256K/900 秒性能结果；早先的失败记录继续保留。
原开发环境的 PyTorch 2.10/CANN 9.0.1 曾被同步后的 Ascend 原生源码的 ABI
检查拦下。现已在工作区隔离安装 PyTorch 2.13、torch-npu 2.13 与 CANN 9.1，
并在原始仓库中重编译、验证了宿主原生扩展。详见
[环境安装说明](../../development/environment-installation.zh.md)。

## 2026-09-27 官方 SWE C4／900 秒配对

匹配环境在同一 2/3 号 910B2 上运行两侧：Qwen3.5 BF16、TP=2、配置
262,144 上下文、APC、真实 MTP2、异步调度、align 与 FULL_AND_PIECEWISE 图。
两侧各自的 C4／60 秒协议检查和 900 秒窗口均正常退出，`valid=true`、无中止或
失败请求，且 prepared 文件 SHA-256 完全相同。B0 保留插件宿主兼容层，但把
压缩阈值设为 262,272，高于服务上限；B1 阈值 9,216，物理预算 8,192 token。

| 官方 900 秒指标 | B0，关闭压缩 | B1，开启压缩 | B1 相对 B0 |
| --- | ---: | ---: | ---: |
| 总输出 token/s | 220.322 | 202.954 | −7.88% |
| 每卡输出 token/s | 110.161 | 101.477 | −7.88% |
| P90 解码 token/s | 71.674 | 70.443 | −1.72% |
| TTFT P95，ms | 1,382.17 | 1,631.72 | +18.05%（变差） |
| 窗口内完成请求 | 318 | 295 | — |
| 最大实测输入 | 74,706 | 70,940 | — |

这是**协议有效但性能退化**的配对，不是 Frontier 正收益点。最大实测输入不到
75K；262,144 是配置容量，不是工作负载达到的长度。两侧均无抢占，C4 的缓存
压力较低，因此此点没有证明内存压力下的收益。固定窗口到达的轮次混合不同，
单次配对也不能证明可重复性。[机器可读配对记录](../../results/evidence/kvcompress-qwen35-20260927-swe-c4-pair.json)
保留运行 ID、原始 summary/config 哈希与限制；完整客户端输出保存在本地已忽略
结果目录中。

## 2026-09-27 官方 AgentX C4／900 秒配对

原版 AgentX 包装器与 AIPerf 回放完成两侧各自的 900 秒冒烟，均有
`submission_valid=true`、无无效原因，使用同一固定版本的 393 会话数据集。
两侧均为同一 Qwen3.5 BF16／TP2 部署，使用 2/3 号 910B2、配置 262,144
上下文、APC、异步调度、align 与 FULL_AND_PIECEWISE 图。AgentX 对推测解码
要求匹配的强制接受率证据，本轮缺少该证据，因此**两侧都关闭 MTP**；SWE 已
单独验证真实 MTP2。B0 压缩阈值高于服务上下文上限，B1 物理预算为 8,192 token。

| 原版 AIPerf 指标 | B0，关闭压缩 | B1，开启压缩 | B1 相对 B0 |
| --- | ---: | ---: | ---: |
| 总输出 token/s | 57.492 | 52.627 | −8.46% |
| 每卡输出 token/s | 28.746 | 26.313 | −8.46% |
| 逐请求反向 ITL 解码速度 P90，token/s | 63.687 | 61.701 | −3.12% |
| TTFT P95，ms | 1,585.69 | 1,873.90 | +18.18%（变差） |
| 完成请求数 | 86 | 82 | — |
| 最大实际输入 token 数 | 169,166 | 169,165 | — |

这也是**协议有效但性能退化**，不能列为 Frontier 正收益点。最长实际输入约
169K，未达到 256K；两侧均无服务端抢占。AIPerf 提示流式 usage 中缺少逐请求
缓存命中 token 明细，但两侧服务端前缀命中计数均增长。
[机器可读配对记录](../../results/evidence/kvcompress-qwen35-20260927-agentx-c4-pair.json)
保留原始报告哈希及限制；完整原版输出仍在本地已忽略目录。单次配对不能证明
可重复性，也不能证明语义回答质量。

两个官方工作负载均暴露初版候选的性能退化。**在这些配对运行之后**，插件内部
新增了部分 RoPE 融合评分路径：真实 NPU 数值冒烟通过，Qwen3.5 形状的单层
32,768-token 评分微基准从 8.06 ms 降至 0.80 ms（10.08 倍）。

随后用同一融合评分源码、1,024-token 重评分间隔重新做 SWE C4／900 秒配对，
两侧仍均协议有效、0 失败。B0 总输出 219.376 token/s、解码 P90 70.781
token/s；B1 分别为 213.334、69.832，即**吞吐下降 2.75%、P90 下降
1.34%**。TTFT P95 为 1,353.46 对 1,354.44 ms。单算子加速显著缩小
退化，但**仍没有正收益点**；两侧最长实测 prompt 均为 73,616 token。
详见[融合评分配对记录](../../results/evidence/kvcompress-qwen35-20260927-swe-fused-rw1024-pair.json)。
另一个 4,096-token 重评分间隔的 B1 运行同样协议有效、0 失败，但相对同一
融合评分 B0 的总输出为 212.788 token/s，解码 P90 为 68.253 token/s，
分别下降 **3.00%** 和 **3.57%**；TTFT P95 变差 6.58%。更长间隔
未带来正收益，详见其[独立负结果记录](../../results/evidence/kvcompress-qwen35-20260927-swe-fused-rw4096-pair.json)。
另以新的同源码配对评估了批量化 V3 选词优化。首次批量选词草案的 B0 窗口
曾在运行中被主动中止：正确性审查
发现各段配额不等时需按评分排序后再截取。其原始客户端 `error.json` 在本地
已忽略结果目录中保留，标记为 `KeyboardInterrupt`、`valid=false`，不参加
任何成绩比较。
修正后的 Qwen3.5 五层评分候选在另一组 32K 确定性样例上与 B0 同为
4/4 正确；B1 有 4 次压缩提交和 8 个 TP-rank 回执。这仅是
[有限的工程质量冒烟](../../results/evidence/kvcompress-qwen35-20260927-batched-stride2-quality.json)，
不是 SWE 任务准确率，也不能替代 900 秒配对。

修正后的批量选词／五层评分候选又完成新的 SWE C4／900 秒同源码配对，两侧
均 0 失败。B0 总输出 221.500 token/s、解码 P90 72.123 token/s、
TTFT P95 1,381.69 ms；B1 分别为 214.959、70.517 和 1,610.66 ms。
因此吞吐下降 **2.95%**、解码 P90 下降 **2.23%**、TTFT P95 增加
**16.57%**；两侧最长实测 prompt 均为 73,616 token。详见
[批量选词／stride-2 配对记录](../../results/evidence/kvcompress-qwen35-20260927-swe-batched-stride2-pair.json)。
这仍不是 Frontier 正收益点。逐请求输出预算分布提示很多短生成承担了压缩
成本。因此另做同源码候选，将触发压缩的预计最短输出从 64 提至 512 token。
其 B1 正式 900 秒运行有效、0 失败，总输出 212.142 token/s、解码 P90
70.689 token/s、TTFT P95 1,474.54 ms；相对同一 B0 分别为
**−4.22%**、**−1.99%**、**+6.72%**。60 秒预检和
[独立 4/4 质量冒烟](../../results/evidence/kvcompress-qwen35-20260927-batched-min512-quality.json)
也通过，但[512 门槛配对记录](../../results/evidence/kvcompress-qwen35-20260927-swe-batched-min512-pair.json)
表明未恢复 Frontier 正收益。这些单次比较不能证明结果可重复。

同一批量 V3／最短输出 512 候选随后在设备 2/3 上完成新的 AgentX 同源码
配对。两侧原版包装器报告均为 `submission_valid=true`、无无效原因；393 条
固定数据集、原版预热、C4 与各自 900 秒窗口均未缩减。B0 总输出
52.953 token/s（每卡 26.477）、解码 P90 60.925 token/s、TTFT P95
1,618.93 ms；B1 分别为 59.226（每卡 29.613）、64.162 与
1,952.38 ms。总吞吐提升 **11.85%**、解码 P90 提升 **5.31%**，但
TTFT P95 变差 **20.60%**。两侧最长实测输入分别为 169,167 和
169,165 token，不是 256K。B1 服务日志含 57 次压缩提交及 114 次
TP-rank 回执（包括预热）。详见[AgentX 正收益配对记录](../../results/evidence/kvcompress-qwen35-20260927-agentx-batched-min512-positive-pair.json)。
这是一组工程正收益的 900 秒 smoke，不证明可重复性、语义答案质量，
也不是一小时正式成绩或官网提交。AgentX 两侧按强制接受率规则关闭 MTP；
SWE 的真实 MTP 配对仍为负结果。

## 合并源码 `17ffdc7` 后的完整复测

候选分支先与当时最新远端主分支 `ed058fa` 合并，再在同一源码提交上分别运行
AgentX B0/B1。两侧使用原版 393 会话数据集及预热、C4、两张 910B2 和各自
900 秒窗口；原版报告均为 `submission_valid=true`，无无效原因或请求错误。
按强制接受率规则，AgentX 两侧均关闭 MTP。

| AgentX 官方指标 | B0 | B1 | B1 相对 B0 |
| --- | ---: | ---: | ---: |
| 总输出 token/s | 52.703 | 58.593 | **+11.18%** |
| 每卡输出 token/s | 26.351 | 29.297 | **+11.18%** |
| 逐请求反向 ITL 解码 P90，token/s | 60.381 | 63.562 | **+5.27%** |
| TTFT P95，ms | 1,601.49 | 1,858.66 | +16.06%（变差） |
| 完成请求数 | 83 | 89 | 闭环轮次混合不同 |
| 最长实际输入 token | 169,166 | 169,163 | 未达到 256K |

B1 日志包含 55 次调度提交和 110 次 TP-rank 回执（含预热）。详见
[合并后 AgentX 配对证据](../../results/evidence/kvcompress-qwen35-20260927-agentx-postmerge-positive-pair.json)。
正吞吐与解码收益的方向在合并后复现，但前后两组配对使用不同源码提交，
每组仅一次窗口，不能据此声称统计显著性。这不是 AgentX 一小时正式成绩，
合成回放也不评价语义答案质量。

SWE B0/B1 各自先通过 60 秒协议短测，再分别完成 900 秒窗口。两侧均
`valid=true`、未中止、0 失败请求，8 条 prepared 会话全部完成。两侧均实际
启用 MTP2、APC、异步调度、`mamba_cache_mode=align` 与
FULL_AND_PIECEWISE；仅压缩配置不同。

| SWE 官方指标 | B0 | B1 | B1 相对 B0 |
| --- | ---: | ---: | ---: |
| 总输出 token/s | 222.581 | 216.008 | **−2.95%** |
| 每卡输出 token/s | 111.291 | 108.004 | **−2.95%** |
| 解码 P90，token/s | 72.341 | 71.368 | **−1.34%** |
| TTFT P95，ms | 1,283.61 | 1,477.01 | +15.07%（变差） |
| 窗口内完成请求 | 329 | 315 | 轮次混合不同 |
| 最长实际 prompt token | 74,706 | 73,616 | 未达到 256K |

B1 的 60 秒短测与 900 秒窗口合计有 99 次调度提交、198 次 TP-rank 回执。
同源码严格配对及原始报告哈希见
[合并后 SWE 负结果证据](../../results/evidence/kvcompress-qwen35-20260927-swe-postmerge-negative-pair.json)。
协议有效和真实 MTP 接受率不代表正收益，也不测 SWE 任务成功率。既有
Qwen2.5 标准发布测试保留为历史记录；这些本地测试不等于官网正式提交。
