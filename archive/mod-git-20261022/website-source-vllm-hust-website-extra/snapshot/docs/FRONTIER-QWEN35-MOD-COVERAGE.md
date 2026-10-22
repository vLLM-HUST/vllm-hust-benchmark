# Qwen3.5-35B-A3B MOD 曲线覆盖进度

**截至 2026-09-28：仍有 MOD 未取得合格曲线。** 本报告区分已发布观测与固定版本的运行阻碍，不将资格失败、源码检查或导入成功当作性能结果。

Mooncake 最新复测已完成五档并发及匹配 Native，对照吞吐低 1.02%–2.66%。使用插件管理器启动 AscendStoreConnector + Mooncake；26
项检索和前缀复用通过，退出需清理遗留进程。此前资格失败记录保留为历史；最新结果见 [完整报告](FRONTIER-MOONCAKE-20260926.md)。

使用同一编译后的 SWE 多轮工作负载、900 秒测量窗口和匹配基线；保留 BF16、256K 上下文容量、MTP2、APC、async 和图执行。每点保留原始请求、窗口内
token、质量检查和服务释放回执。收尾时间不计入吞吐。部署环境只作为来源信息，不能单独命名为 MOD。

该协议是闭环生成：实际输出 token 会拼入下一轮输入，固定的是编译后的输入长度和输出预算，不保证各轮 token
内容跨运行完全相同。每组服务启动后先做资格检查，再串行测各档并发，档位之间不重置缓存。共同请求的逐 token 比较只是补充核验，不能推广为一般质量结论。

覆盖来源是网站目录的 46 个组件记录，其中 13 个为基础设施。下表的 33 个记录包括外部系统/连接器和工具/描述器配对，**不是 33
个独立优化算法**。缺少运行入口的记录不填零分、不连假曲线；适配中的项目也不视为完成。固定版本和清单哈希见
[完整覆盖账本](https://github.com/vLLM-HUST/vllm-hust-dev-hub/blob/4b962ea46df6ed9d62bac904efd4733b29e5be8f/scripts/frontier_curves/catalog-coverage.json)。

| 组件                               | 当前证据与下一步                                                                                                                                                                                                                                 | 固定来源                                                                                                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mooncake HUST                      | 插件管理器启动的 AscendStoreConnector + Mooncake 已完成五档及匹配 Native，吞吐低 1.02%–2.66%；检索26/26、前缀复用及真实设备释放通过。退出需清理遗留进程；详见[新报告](FRONTIER-MOONCAKE-20260926.md)。                                           | [8b8c7ae7](https://github.com/vLLM-HUST/mooncake-hust/tree/8b8c7ae705bdaf918f5a8fbc7a06cb7eb1d5f3ca)                                                                                                                                                  |
| Mooncake vLLM Connectors           | 插件管理器启动的 AscendStoreConnector + Mooncake 已完成五档及匹配 Native，吞吐低 1.02%–2.66%；检索26/26、前缀复用及真实设备释放通过。退出需清理遗留进程；详见[新报告](FRONTIER-MOONCAKE-20260926.md)。                                           | [d0f22d2b](https://github.com/vLLM-HUST/vllm-hust/tree/d0f22d2bda562156e4dbf433ce645e1769b4f804)                                                                                                                                                      |
| PegaFlow                           | PegaFlow 与 PegaKVConnector 已在完整 Frontier 合同下完成 C1/C2/C4/C8/C16；五点吞吐几何平均相对统一 Native 为 +6.34%，C1 回退而 C2-C16 提升。五点均真实执行 save/load、MTP2 与前缀复用，零请求失败。详见[实测报告](FRONTIER-QWEN35-PEGAFLOW.md)。 | [cd64ecc2](https://github.com/vLLM-HUST/pegaflow-hust/tree/cd64ecc283ff856a44437a9a25659929ef3a0653)                                                                                                                                                  |
| PegaFlow vLLM Connectors           | Extension Manager 已完成 inspect/check/plan/configure/enable/status，连接器以 read_write 模式连接外部服务；3,386 次 load 与 3,050 次 save 成功，失败为零。性能只在此连接器 MOD ID 下计一次，不与核心服务重复计分。                               | [PR #30 与完整证据](https://github.com/vLLM-HUST/pegaflow-hust/pull/30)                                                                                                                                                                               |
| BidKV                              | 新共同运行时下 C1/C2/C4/C8/C16 与 Native 配对测试全部完成，原始记录校验与设备释放通过，已由 PR #283 发布。未触发抢占，不能宣称抢占收益。                                                                                                         | [a0cba97d](https://github.com/vLLM-HUST/vllm-hust-bidkv/tree/a0cba97d9abdc99908e46616db622f0e0099127f)                                                                                                                                                |
| DiffSpec                           | 当前载体要求 EAGLE3/TP4 且关闭 APC、async，不能直接替代本轮 MTP2/APC/async 配置。                                                                                                                                                                | [42e5909f](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/tree/42e5909fc6fe276ba0defe1901257a523653aefb)                                                                                                                                      |
| vSpec                              | 发现针对 Qwen3.5-35B-A3B 的 EAGLE3 草稿模型；实际 EAGLE3 导入/属性 ABI 检查 13/13 通过；草稿权重下载仍遇 TLS 失败，补丁激活与硬件正确性尚未验证。                                                                                                | [d4c4f659](https://github.com/vLLM-HUST/vllm-hust-vSpec/tree/d4c4f659495826e64802eedb195de52019282b47)                                                                                                                                                |
| LatchMoE                           | 当前启动器拒绝 APC；已执行对应拒绝路径测试。                                                                                                                                                                                                     | [9b2d4acd](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/tree/9b2d4acdbfbe6463a22dd0bb8e6ca5bfda47e2c1)                                                                                                                                      |
| Adaptive Quantized KV              | 负责人已推进 [Host C8 继续预填充接口 PR #35](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/35)，但插件仍为 import_only；Qwen3.5 混合缓存、TP2、APC、MTP2、async 与图模式的完整组合尚未验证，没有性能曲线。                                  | [8587845f](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/tree/8587845f29d70256103990ed6492837ddaf5a800)、[负责人最新进度](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/1#issuecomment-5843295865) |
| Ascend Quant Toolkit               | 工具是离线检查点量化；运行时描述器仅校验元数据，不加载模型或设备。缺少量化产物加载和算子选择的宿主接入，不能作为本轮 BF16 在线优化开关。                                                                                                         | [f161daab](https://github.com/vLLM-HUST/vllm-ascend-quant-hust/tree/f161daab91b2b558bc8ee65d2b1bc76cd78e00b6)                                                                                                                                         |
| Ascend Quant Runtime Descriptor    | 工具是离线检查点量化；运行时描述器仅校验元数据，不加载模型或设备。缺少量化产物加载和算子选择的宿主接入，不能作为本轮 BF16 在线优化开关。                                                                                                         | [f161daab](https://github.com/vLLM-HUST/vllm-ascend-quant-hust/tree/f161daab91b2b558bc8ee65d2b1bc76cd78e00b6)                                                                                                                                         |
| Ascend KV Compression              | 已通过 ECPA 在完整 Frontier 配置下完成 C1/C2/C4/C8/C16；五点吞吐几何平均相对统一 Native 为 -3.55%，APC、异步提交、MTP2、混合缓存组、双 rank 压缩确认与设备释放均通过。详见[实测报告](FRONTIER-QWEN35-KVCOMPRESS.md)。                            | [2ca0f933](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/tree/2ca0f9335399a698342285870df1514e9c519bd4)                                                                                                                                    |
| Prefix Router                      | 固定版本没有可运行发布；不能以安装或导入代替实测。                                                                                                                                                                                               | [4e007c4f](https://github.com/vLLM-HUST/vllm-hust-prefix-router/tree/4e007c4fc1bd376a6dccfefbc1fd851019c8ceb6)                                                                                                                                        |
| KV Tiering                         | 通过插件管理器完成 C1/2/4/8/16 与 Native 配对，共 10 点，零失败；每点 900 秒，双方均通过 26 项检索检查。本次吞吐低 2.33%–5.23%，所有窗口仅写出、无回读；不是复用加速证据。依赖显式 Host 修复。                                                   | [完整对照及原始记录](FRONTIER-KV-TIERING-20260926.md)、[PR #3](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/pull/3)、[Host #40](https://github.com/vLLM-HUST/vllm-hust/issues/40)                                                                |
| KNorm                              | 固定版本完整 Git 树仅含四个文档/维护文件，没有运行实现。                                                                                                                                                                                         | [e0e872ab](https://github.com/vLLM-HUST/vllm-hust-knorm/tree/e0e872abfc9fa88659b3e83c1c8b8b2b3de88fc0)                                                                                                                                                |
| PyramidKV Ascend                   | 除 import_only 和缺少宿主接入外，固定实现只接受 Llama/Qwen2 全注意力单缓存组，拒绝 MTP、TP2 和 FULL_AND_PIECEWISE；本轮配置被多项独立检查拒绝（源码核查）。                                                                                      | [77b0862c](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/tree/77b0862c1e5be8c883fda934cdb383c57cf7ad0d)                                                                                                                                     |
| SliceGPT                           | 固定版本完整 Git 树仅含四个文档/维护文件，没有运行实现。                                                                                                                                                                                         | [6acf19d9](https://github.com/vLLM-HUST/vllm-hust-slicegpt/tree/6acf19d9cbb3ed6caa3f5e6341b1da41941f9f2e)                                                                                                                                             |
| Quantized KV Cache                 | 已有真实 INT8 attention backend 与管理器入口；需要显式 int8 KV dtype，当前 Frontier 固定 auto。冻结 Host 尚不接受该 dtype，插件与冻结 Ascend 的 FULL graph 反量化 scale 接口也不匹配；原组合尚不能补测。                                         | [ae7b44bf](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/tree/ae7b44bfee695d0ba580040dffd10eb802104485)                                                                                                                            |
| SimLLM                             | 完整源码树仅含来源校验和非激活描述器；该模块明确不含 SimLLM 实现，没有可测的相似度索引、嵌入或请求改写执行路径。                                                                                                                                 | [dcdc6edf](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/tree/dcdc6edf7bdcc68bdf35058888ebfd9752ae3566)                                                                                                                                        |
| Unified Communication              | 策略/注册表存在，但尚缺宿主 collective 接入。                                                                                                                                                                                                    | [f00d1ef4](https://github.com/vLLM-HUST/vllm-hust-unified-comm/tree/f00d1ef4c19a992d67ef8012952a9405d52dd447)                                                                                                                                         |
| Split-Batch / Full-Graph Parallel  | 已修复默认关闭路径及多 query 保护，32 项容器内测试通过；仍缺少完整宿主执行接入，MTP2 兼容检查仍拒绝本轮配置。未降低 Frontier 配置，不填性能点。                                                                                                  | [适配修复 PR #5](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/pull/5)                                                                                                                                                                    |
| KV Transfer Observability          | 新增插件侧 Host adapter 与配置、注册/注销处理；仍需宿主提供 vllm.kv-transfer.observer.v1 binding，共同运行时没有该接入。属于诊断设施，尚无原配置下开销测量。                                                                                     | [53e69013](https://github.com/vLLM-HUST/vllm-hust-kv-transfer-observability/tree/53e69013dcf27fd08a93d142288c93c701edabd3)                                                                                                                            |
| Layered Prefill                    | 固定版本的扩展清单声明 import_only，activation.entry_points 为空；尚缺可测运行入口。                                                                                                                                                             | [a45e4170](https://github.com/vLLM-HUST/vllm-ascend-layered-prefill-hust/tree/a45e41709ccacc3d7c736910b93c1b5985d9ee94)                                                                                                                               |
| Activation Sparsity                | 当前包仅含配置验证和哈希；实际 CPU 兼容检查拒绝 TP2（TP1 控制通过），且缺少投影变换/稀疏算子的宿主接入。                                                                                                                                         | [0e4d0628](https://github.com/vLLM-HUST/vllm-hust-activation-sparsity/tree/0e4d0628c1972d5086a217b0007576c1fd8998a3)                                                                                                                                  |
| Pipeline Microbatch                | C1/C2/C4/C8/C16 的 Native 配对观测已发布；每点仍为单次观测，不能据此宣称稳定加速。                                                                                                                                                               | [a15a2296](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/tree/a15a22961a0e4858da74a0ab806575c82cb254e6)                                                                                                                                  |
| QoS Scheduler                      | 已有请求期限模型和排序函数，但缺少 QoS API 元数据、排序注册和输出观察接入；本轮请求未携带 SLO，第一阶段契约尚未覆盖 MTP。                                                                                                                        | [13d376a7](https://github.com/vLLM-HUST/vllm-hust-qos-scheduler/tree/13d376a7d8990c4dcf5c0903cb6fbf2398ef0fb0)                                                                                                                                        |
| Scheduler Policy Lab               | 已有独立策略函数，尚缺策略注册、KV 预算/请求快照、完成事件和调度后观察接入；共同核心中未找到所需的四个协议标识。                                                                                                                                 | 内部来源，详见覆盖账本                                                                                                                                                                                                                                |
| Request Lifecycle Causal Profiler  | 插件依赖当前共同核心缺少的 kv_recovery_profile 观察接口；诊断功能不等于优化收益。                                                                                                                                                                | [e32a0e91](https://github.com/vLLM-HUST/vllm-hust-request-lifecycle-profiler/tree/e32a0e91027ae7a2b96bf48d2dcb7db1b3c42c87)                                                                                                                           |
| KV Materialization Arrival Control | 需要特定请求元数据和分段复用宿主接口；无元数据的软件探针只选择重新计算，不能据此生成优化成绩。                                                                                                                                                   | [10428b81](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control/tree/10428b81e2b383cdcb183d4548f38a98929fd0e4)                                                                                                                   |
| BetterScale                        | 网站已有实测；本轮不冒充新增结果。                                                                                                                                                                                                               | [仓库](https://github.com/vLLM-HUST/BetterScale)；账本已记录网页 40 点的四个逐点来源版本                                                                                                                                                              |
| DLA                                | 已知输出预算版本的五档并发全部完成并通过原始记录校验，已由 PR #283 发布。准入检查已执行，延后与抢占为零；不是学习型长度预测或已证实收益。                                                                                                        | [dc20d0f8](https://github.com/vLLM-HUST/vllm-hust-dla/tree/dc20d0f8ea8d09106f77571e1947b9a2f8702545)                                                                                                                                                  |
| TraceLoom                          | 已有可选 Scheduler/AsyncScheduler 运行时观察入口与离线分析器；现有硬件证据为 Qwen3-0.6B TP1/eager。原 Frontier 图模式、TP2 与记录开销尚未验证，不作为已测优化曲线。                                                                              | [ab8b5131](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer/tree/ab8b5131191c6d5aeee2dd8566c34411f49ceab0)                                                                                                                                        |

BidKV 另有 2026-09-29 的 TP4 EvoScientist 高 KV 压力单档探索性测试，输出吞吐提升 +13.88%（两轮配对提升率的算术平均），两轮均触发主动选择。该结果与原
TP2/SWE 五档指标口径不同；逐轮结果和负结果见[原始证据](evidence/qwen35-bidkv-tp4-20260929/README.md)。

已发布的新增对照点见网站 PR [#279](https://github.com/vLLM-HUST/vllm-hust-website/pull/279)
、[#280](https://github.com/vLLM-HUST/vllm-hust-website/pull/280) 与补齐 Pipeline 五档并发的
[#282](https://github.com/vLLM-HUST/vllm-hust-website/pull/282)。环境归组修正见
[#281](https://github.com/vLLM-HUST/vllm-hust-website/pull/281)。单次观测不能证明稳定加速；未触发的优化机制须明确标注。

Native/BidKV/DLA 的 15 个共同运行时观测已由 [#283](https://github.com/vLLM-HUST/vllm-hust-website/pull/283)
发布。Mooncake 原安装版导入退出 SIGABRT 的失败证据已保留；固定源码新构建已通过导入、四类传输和原始 Store 读写。实际服务在 131K 冷请求答对、复用缓存后连续输出 48 个
`!`，同时观察到 Store 容量不足和退出时堆损坏；尚未证明根因。原始失败归档与 SHA256 见覆盖账本。缺少 hccn.conf 未阻止已通过的传输测试，不能再单独据此判定不可运行。

16 GiB/rank 对照已完成：Native 26/26 通过，Mooncake 在第 10 项 warm-262080 再次返回 48 个 `!`，而对应冷请求通过。此次没有记录到 Store
put/get 失败，失败请求日志的外部 `need_to_load=0`，所以不能直接归因为外部读取错误。退出时两个 worker 被 SIGKILL，仍有堆损坏日志；最终设备均释放。2744
个源文件哈希复核一致，完整归档 SHA256 为
`637405d49011869855793b8aa200c864ff736716e32e332bd704041f708cbfab`。扩大容量不足以解决资格失败，未进行 Mooncake 性能测量。

已针对运行中 Mamba 跟踪器未传入已计算 token 数的问题制作并验证修复（Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`，相关调度测试 68
项通过）。独立硬件复测中 Native 26/26 通过，但 Mooncake 仍在 warm-262080 失败，退出堆损坏也仍存在，因此该修复不足以解决服务资格问题。修复版完整回执归档
SHA256：`557cbf246b158b037f27896e276657dd4c32a7f0ff6dd11cb722fd021c7647f4`；3895 个来源哈希一致，设备已释放，无性能测量。

来源：[固定实验覆盖账本](https://github.com/vLLM-HUST/vllm-hust-dev-hub/blob/4b962ea46df6ed9d62bac904efd4733b29e5be8f/scripts/frontier_curves/catalog-coverage.json)。其中的本地原始归档路径和
SHA256 用于追溯，并不表示归档已可公开下载。

## 2026-09-27 增量复核

对照昨日快照检查了 30 个仓库的默认分支及 17 个跟进线程，并读取发生变化的兼容性检查。没有发现新增、已能保持原 Frontier 配置直接补曲线的项目；这不表示所有未合并分支都经过完整审计。

- KVCompression 此后已由 [PR #8](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/8) 补齐
  APC、async、MTP2 和 align 混合缓存组路径；2026-09-28 的完整 Frontier 曲线已通过并发布，见下方增量。
- SplitBatch 更新到
  [8a36101](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/8a36101a796ee4915ed9ba563864258b405ed02e/src/vllm_ascend_split_batch/cascade_plugin.py#L235)。多
  query 和 speculative/MTP 仍退回原生注意力；不能把该回退当成优化路径已生效。
- AQK 的 [插件 PR #4](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/pull/4) 与
  [Host PR #35](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/35) 继续推进配置合同；仍缺校准 profile、BF16→C8
  选择/加载和 cache-write 所有权，插件保持 import_only，尚无原配置硬件通过证据。
- 截至该快照，QuantizedKV、PegaFlow、LatchMoE、LayeredPrefill、vSpec、DiffSpec 默认头未变化，
  跟进线程中未见已解决前述阻断的新证据；PegaFlow 的后续修复与实测见 2026-09-29 增量。

## 2026-09-28 KVCompress 增量

KVCompress 的 APC、async、MTP2 与 `mamba_cache_mode=align` 兼容修复已合并。使用 ECPA 启动后，完整 Frontier 配置的 五个 900
秒窗口均为零失败；434 次调度压缩提交均获得 TP0、TP1 确认，且每个窗口都观测到 APC 与原生 MTP 正增量。 C1/C2/C4/C8/C16 相对同一组统一 Native 分别为
+3.31%、-10.49%、-5.00%、-1.72%、-3.33%，几何平均为 -3.55%。负结果照常进入插件页排序和 Frontier 曲线；完整原始证据已随 KVCompress PR
#11 合并。

## 2026-09-29 PegaFlow 增量

PegaFlow 已补齐 Qwen3.5 混合缓存组与 Ascend IPC save/load 路径，并通过 Extension Manager 完成完整生命周期验收。C1/C2/C4/C8/C16
五个 900 秒窗口均为零请求失败，3,386 次 load 与 3,050 次 save 全部成功；相对统一 Native 的逐点吞吐变化为 -28.43%、+9.44%、+9.42%、
+24.16%、+27.79%，五点几何平均为 +6.34%。C1 回退与其余四点提升均保留，完整证据已随
[PegaFlow PR #30](https://github.com/vLLM-HUST/pegaflow-hust/pull/30) 合并。
