# Qwen3.5-35B-A3B：280题与三轮性能对照

2026-10-26 UTC，ascend87。此页是用户要求的35B主实验；此前14B结果仅作补充，不混为同一实验。

## 结论

**不合格：35B 实测质量下滑；现有整体配置对照未分离自适应策略、refill与graph/runtime支持的独立贡献。**

1. **质量下滑。** 280 题准确率相对原生下降 3.57 个百分点、相对 MTP2 下降 3.21 个百分点，当前质量表现不予接受。
1. **性能比较未控制变量。** 两臂均启用graph，但gamma预算、refill调度、capture
   buckets、动态宽度runtime及兼容层不同。整体配置虽然测得提速，但未做逐项消融，不能据此认定自适应策略相对宿主原生MTP2的独立收益，也未证明额外graph/runtime支持本身贡献了提速。[发布前证据审计](publication-audit.md)

保留速度实测与统计边界：准确率下降尚未得到可靠因果归因，不等于已证明算法必然损失精度；但这也不构成精度无损或当前方案合格的证据。

这不是全量MMLU-Pro成绩，也不是所有vSpec proposer的泛化结论。未自动扩到12,032题。

## Candidate 与 baseline + MTP 的系统实现差异

此表的baseline指本次宿主原生MTP2，而非未经修改的官方发行版。Core提交为 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，Ascend提交为
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`；发布审计时两宿主checkout均clean。vSpec固定提交为
`4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae`。下表vSpec相对路径统一位于
`src/vllm_hust_vspec/`；“配置开启/源码安装”不等于已测得该项性能贡献。

| 系统项                  | baseline + MTP2                           | vSpec candidate                                                                                          | 原仓库代码路径与定位                                                                                                                                                                                                                                                                                                                                                |
| ----------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模型与proposer接入      | 目标内置MTP权重；仅加载ascend插件         | 同一目标与内置MTP权重；增加vspec插件及宿主proposer钩子                                                   | [__init__.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/__init__.py)的register；[patches.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/patches.py)的apply_patches；backends/mtp.py的apply_mtp_patches                     |
| 投机预算与反馈          | 固定gamma=2；无vSpec控制器                | online控制，候选2/4/6；控制间隔4，窗口32                                                                 | [adaptive/runtime.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/runtime.py)的apply_adaptive_patches、scheduler_init/schedule；[adaptive/online.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/online.py) |
| 请求准入/refill         | 宿主原生调度，无vSpec refill钩子          | adaptive_refill_batch=8；MTP_COHORT_REFILL=1                                                             | [backends/mtp.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/backends/mtp.py)的_patch_mtp_cohort_refill；[adaptive/refill.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/refill.py)的install_online_refill_patch   |
| Graph模式（相同）       | FULL_AND_PIECEWISE，draft非eager          | 同为FULL_AND_PIECEWISE，draft非eager                                                                     | 宿主vllm_ascend/spec_decode/llm_base_proposer.py的load_model已有ACLGraphWrapper。不能将baseline描述为没有graph                                                                                                                                                                                                                                                      |
| Capture buckets（不同） | 7项：[3,6,9,12,18,24,48]                  | 48项：1至112的多宽度集合，不是连续1..112                                                                 | [cli.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/cli.py)的generate_capture_sizes；adaptive/runtime.py的restore_adaptive_capture_sizes；实际列表见q35-engine-configs.json                                                                                                                     |
| 动态宽度graph与dispatch | 宿主固定gamma路径，不安装vSpec钩子        | 增加多query-width graph keys、按gamma组织draft FULL runnables及dispatch/query-width适配                  | adaptive/runtime.py的_add_adaptive_decode_graph_keys、\_initialize_full_graph_runnables、adaptive_dispatch、\_runner_query_width；无逐hook消融或收益计数                                                                                                                                                                                                            |
| Graph严格检查           | 无vSpec额外guard                          | mtp_strict_graph=1；检查FULL draft并拒绝特定eager回退                                                    | backends/mtp.py的_patch_mtp_drafter_graph_guard、apply_mtp_patches。此项主要是安全检查，不能单列为提速优化                                                                                                                                                                                                                                                          |
| Hybrid状态与兼容层      | 使用宿主KV分组与MambaBuffers API          | vSpec增加Qwen MTP分组注解；本次私有适配保留宿主新版逐plane状态搬运，并修复动态draft padding/缓存阈值接口 | backends/mtp.py的_annotate_qwen35_mtp_kv_groups；私有vspec_compat.py与vspec-stats-compat.patch不属于作者原仓库，必须单列披露                                                                                                                                                                                                                                        |
| 本次未启用的优化        | local argmax reduction=false；无vSpec优化 | local argmax=false、device counts默认0、merged full=0、draft parallel graph updates=0                    | backends/mtp.py的_patch_mtp_local_argmax_reduction、\_patch_async_mtp_device_counts及config.py开关。不可将这些写成本轮提速来源                                                                                                                                                                                                                                      |
| 接受条件（未放宽）      | 原生MTP接受路径，无synthetic配置          | confidence margin为空并解析为None；无synthetic配置                                                       | [config.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/config.py)的_read_optional_float / PluginSettings；cli.py的synthetic配置生成分支未启用                                                                                                                                                   |

两臂共同开启async scheduling、APC、chunked prefill，TP2+EP、BF16、B16及4000输出上限相同。candidate从原仓库
`configs/qwen35-35b-a3b-frontier-mtp2.toml` 生成；baseline由本次 `prepare35.py`
保留共同参数并移除vSpec、固定gamma2。逐轮command/config与冻结JSON核验一致。

因此，现有结果支持**两套配置的性能对照**，不支持**自适应策略的单因素收益归因**。亦未证明graph/runtime差异本身导致提速，或全部可独立移植到baseline。详细开关、源文件回收比对及未补足的证据见[发布前审计](publication-audit.md)。

## 质量：固定280题，4000输出token上限

| 配置                                 |    正确 |   得分 | 截断 | 无法解析 |
| ------------------------------------ | ------: | -----: | ---: | -------: |
| 原生 target-only                     | 203/280 | 72.50% |   92 |        0 |
| 原生 MTP，固定深度2                  | 202/280 | 72.14% |   94 |        0 |
| vSpec adaptive MTP（2/4/6 + refill） | 193/280 | 68.93% |   93 |        0 |

每学科20题，固定哈希排名、排除已查看的14题pilot；分层宏平均，不按全量学科占比加权。全部结果无>90% token0退化。

| 配对（前者相对后者） | 得/失 |    分差 | 配对分层bootstrap 95%区间 |
| -------------------- | ----: | ------: | ------------------------: |
| 原生MTP2 / 原生      | 27/28 | -0.36pp |          [-5.36, +4.64]pp |
| vSpec / 原生         | 19/29 | -3.57pp |          [-8.21, +1.43]pp |
| vSpec / 原生MTP2     | 27/36 | -3.21pp |          [-8.57, +2.14]pp |

不能把“区间跨0”解释成证明没有精度损失；区间仍允许明显下降。但也不能把点估计当作已证实掉精度：

- 各臂约三分之一题触及输出上限，成绩反映固定预算效用，不是充分推理后的准确率。
- vSpec与原生MTP2共同未截断的172题中，vSpec有12得10失；总分差主要来自至少一侧被截断的题。这是事后选择的描述，不能作无偏总体推断。
- 原生自己在三次56题重复中为40、40、37题正确；前两次序列全部相同，第三次56条序列全部与第一次不同。配置、安全环境（除run
  label）、题序与输入长度一致，权重文件大小/mtime未变；根因未定位，不能把所有差异归因于插件。

## 性能：同一56题，交错次序，独立engine，三轮

| 配置                                 | 三次generation秒数     | 中位秒数 | 输出tokens/s中位数 |
| ------------------------------------ | ---------------------- | -------: | -----------------: |
| 原生 target-only                     | 297.89, 293.76, 292.24 |   293.76 |             512.39 |
| 原生 MTP，固定深度2                  | 227.54, 222.32, 224.86 |   224.86 |             689.11 |
| vSpec adaptive MTP（2/4/6 + refill） | 191.79, 196.27, 189.97 |   191.79 |             804.52 |

整套adaptive配置相对target-only的观测wall比为 **1.53×**，相对原生MTP2为 **1.17×**；按输出tokens/s中位数分别为 **1.57×** 与
**1.17×**。输出长度并非完全相同，因此不把wall比叫作等输出kernel加速。

三次MTP2实际接受率约82.0–82.5%；vSpec约66.4–67.2%。深度不同，接受率分母不同；较低比例并不自动意味着较低吞吐。本负载不支持“接受率太低所以无收益”。

计时排除模型加载、编译、graph捕获与预热；包括generate和逐完成请求JSONL保存。物理0,1上串行、trace关闭，未并行本任务的其它NPU实验。不是生产流量SLA：所有56请求集中提交，TTFT与queued-to-last包含离线队列等待。原始逐请求延迟、每请求平均TPOT及汇总保留在JSON中，不把TPOT均值分布冒充逐token抖动分布。

## 合同与实现边界

- Target：本机 `/models/Qwen3.5-35B-A3B`，内置单层MTP权重，运行时展开多个draft
  token；TP2+EP、BF16、B16、context16384、max_batched_tokens8192、memory0.9。不是14B+0.5B草稿，也不是Eagle3路径。
- 依据pinned vSpec的35B frontier MTP协议：APC、async、chunked
  prefill、FULL_AND_PIECEWISE；显式关闭近似confidence接受与synthetic
  acceptance；adaptive候选2/4/6，refill8。generation_config=vllm，temp0/top_p1/max4000。
- APC配置开启，但本批任务记录的prefix cache命中均为0；不声称观察到了APC收益。三臂全部质量轮与九次性能轮的 `num_preemptions` 增量均为0，无抢占重算计数。
- 对照是原生target-only、原生MTP2与vSpec推荐策略整体。vSpec同时含refill、动态宽度graph/runtime支持及私有兼容层；未进行逐项消融，不能把全部增量单归gamma选择，也不能反向认定这些支持本身造成了提速；也未声称原生已穷尽部署调优。
- 兼容补丁只在任务私有层：新增group-builder生命周期适配，保留宿主新版MambaBuffers逐plane拷贝，沿用此前API/统计/动态padding修正；不改target数学、采样或接受阈值。原生MTP不加载vSpec。
- 目标权重的证据是本地config、索引、文件大小和mtime；没有声称已确认不可变上游权重revision。其它source/dataset/draft版本见plan与progress。

## 证据入口

- [冻结协议](plan.md)、[过程/失败与修复](progress.md)
- [选题清单](evidence/screen-manifest.json)
- [35B实际engine配置](q35-engine-configs.json)、[环境](q35-environments.json)
- [结构化汇总与配对变化](evidence/screen-analysis.json)
- [原生重复配置审计](evidence/q35-native-repeat-audit.json)
- `evidence/screen-quality-q35-*-result.json`：280题原始输出、分数、延迟与计数。
- `evidence/screen-performance-q35-*-result.json`：三轮56题原始结果。

建议不直接扩全量。若继续研究精度，应先解决/界定生成预算截断与原生重复不一致，再另行决定投入；本轮没有自动追加精度实验。
