# MOD 逐项处置

本表由同目录[机器台账](exploration-ledger-v1.json)生成，范围为固定benchmark目录的全部69个分类条目。可追溯的原结果文本和证据等级保留在机器台账中；该表不是新增成果计数或性能认证。未测/资格不足不写成“已测失败”。

| MOD                          | ID                                    | 当前处置              | 原因                                                                                |
| ---------------------------- | ------------------------------------- | --------------------- | ----------------------------------------------------------------------------------- |
| 激活稀疏推理                 | `activation-sparsity-migration`       | 退役归档              | 退役/归档项，不构造新加速候选。                                                     |
| 昇腾分布式元数据             | `adm`                                 | 拓扑或模型不同        | 已有模型/精度或多芯片拓扑不同，不能直接叠加当前双卡BF16引擎。                       |
| 昇腾自适应 KV 量化           | `ascend-adaptive-quantized-kv`        | 混合cache资格待补     | 当前TP2、Qwen3.5混合cache、MTP2/APC/FULL尚未验收。                                  |
| 昇腾注意力计算边界调优       | `ascend-attention-boundary`           | 归因/身份资格不足     | 既有观测未形成可归因、稳定的组件性能增益。                                          |
| Ascend 注意力优化集          | `ascend-attention-optimizations`      | 没有接纳的可执行候选  | 没有当前可接纳的可执行候选，不能沿用旧指标。                                        |
| 昇腾映射主机内存 KV 卸载     | `ascend-mapped-kv-offload`            | 混合cache资格待补     | 当前TP2、Qwen3.5混合cache、MTP2/APC/FULL尚未验收。                                  |
| 昇腾模型量化工具             | `ascend-quant-toolkit`                | 模型/精度或离线变换   | 离线转换或模型/精度改变需独立验收，不是保持当前BF16身份的直接执行增量。             |
| BetterScale 执行与状态调度   | `betterscale`                         | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| BidKV 缓存抢占策略           | `bidkv`                               | 先查瓶颈              | 仅当当前r8实际存在KV压力及有效抢占机会才继续接入；TP4低KV正例不外推TP2/20GiB。      |
| 执行与常驻席位调优           | `capacity-tuning`                     | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| CLM 请求生命周期观测         | `clm-lifecycle`                       | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 核心注意力计算边界调优       | `core-attention-boundary`             | 归因/身份资格不足     | 既有观测未形成可归因、稳定的组件性能增益。                                          |
| CUDA 多词元预测（MTP）       | `cuda-mtp`                            | 不同设备              | CUDA路径与当前Ascend设备不同。                                                      |
| DiffSpec 投机解码            | `diffspec`                            | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| DLA 输出长度感知准入         | `dla`                                 | 原配置无动作          | 已测配置未触发实际控制动作，不能把准入检查当作调度收益。                            |
| GDN 执行优化集               | `gdn-optimizations`                   | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| GDN 状态压缩与恢复           | `gdn-state-codec`                     | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| 计算图捕获范围配置           | `graph-coverage`                      | 与已包含机制重叠      | 旧小模型正例保留；r8已有FULL图、预热封锁及短尾/预算调度，先证明有未覆盖瓶颈。       |
| 受保护的异步进度复制         | `guarded-progress-async`              | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| 锁页主机状态合并             | `hust-joint-packed`                   | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| 请求状态校正                 | `hust-m01-correction`                 | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| Hybrid Spec 混合状态投机解码 | `hybrid-spec`                         | 历史未合格            | 保留历史，不作为当前r8合格的投机增量。                                              |
| 实例扩缩容控制器             | `instance-controller`                 | 独立多副本方向        | 属于多副本服务控制，不能当作双卡TP2单引擎执行增量。                                 |
| KNorm KV 压缩                | `knorm-migration`                     | 缺当前模型证据        | 尚缺当前宿主完整模型性能与质量证据。                                                |
| 下一轮 KV 准备控制器         | `kv-controller`                       | 独立多副本方向        | 属于多副本服务控制，不能当作双卡TP2单引擎执行增量。                                 |
| KV 物化与重算                | `kv-materialization-arrival-control`  | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| KV 缓存分层管理              | `kv-tiering-migration`                | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| KV 传输追踪                  | `kv-transfer-observability-migration` | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 昇腾 KV 缓存压缩             | `kvcompress-ascend`                   | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| LatchMoE 专家权重卸载        | `latchmoe`                            | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| 分层预填充                   | `layered-prefill-migration`           | 退役归档              | 退役/归档项，不构造新加速候选。                                                     |
| 推理服务成本估算             | `llm-serving-cost-pricing-model`      | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 昇腾计算图显存跟踪器         | `memory-tracker`                      | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| Mooncake HUST                | `mooncake`                            | 外部系统              | 本轮未查到当前r8的质量合格新增收益，暂不列入优先叠加。                              |
| Mooncake KV 存储与传输       | `mooncake-vllm-connectors`            | 缺当前组合增量        | 官方连接器功能证据不等于当前r8的混合状态复用与预算增益。                            |
| MuxPD 分离预填充与解码       | `muxpd`                               | 质量与执行冲突待解    | 历史配方有性能正例及质量负例；P/D worker与Microflow/CorrectedWorker需重新协调。     |
| OpFuse 算子融合              | `opfuse`                              | 机制未实现            | 脚手架或差分测试不等于已有融合机制实现。                                            |
| PegaFlow                     | `pegaflow`                            | 外部系统              | 本轮未查到当前r8的质量合格新增收益，暂不列入优先叠加。                              |
| PegaFlow KV 传输             | `pegaflow-vllm-connectors`            | 外部缓存与慢尾待解    | 保留旧正例及恢复慢尾、额外锁页池和协议差异；不直接叠加当前逐请求重算。              |
| 流水线微批处理               | `pipeline-microbatch-migration`       | 拓扑或模型不同        | 已有模型/精度或多芯片拓扑不同，不能直接叠加当前双卡BF16引擎。                       |
| 短请求预填充准入             | `prefill-admission`                   | 与已包含机制重叠      | 旧小模型正例保留；r8已有FULL图、预热封锁及短尾/预算调度，先证明有未覆盖瓶颈。       |
| Microflow 预填充流水线       | `prefill-microflow`                   | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| Prefix Router 前缀感知路由   | `prefix-router-migration`             | 独立多副本方向        | 已测两副本路由，有历史吞吐正例和来源归因缺口；不是双卡TP2单引擎直接执行增量。       |
| PyramidKV 缓存压缩           | `pyramidkv-ascend-migration`          | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| QoS 服务质量调度             | `qos-scheduler-migration`             | 退役归档              | 退役/归档项，不构造新加速候选。                                                     |
| 质量约束推理策略             | `quality-bounded-inference`           | 运行时资格待补        | CPU宿主契约/生命周期已验证；没有当前NPU、模型质量或性能资格。                       |
| KV 缓存量化                  | `quantized-kv-cache-migration`        | 混合cache资格待补     | 当前TP2、Qwen3.5混合cache、MTP2/APC/FULL尚未验收。                                  |
| 请求生命周期因果分析器       | `request-lifecycle-profiler`          | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 请求限流控制器               | `request-throttling-controller`       | 复用既有场景结论      | 已有r8多轮缓存竞争正例与固定MMLU/Frontier负例；复用结论，不重复推荐固定负载原配方。 |
| RIDE 跨实例服务控制面        | `ride-control-plane`                  | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| RIDE 远程运行时边车          | `ride-runtime-bridge`                 | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 推理运行时开销优化           | `runtime-micro-optimizations`         | 原实现不重测          | 已有原实现负收益、接近零收益或未通过稳定性能筛选；不重复推荐原配置。                |
| 调度策略研究与测试           | `scheduler-policy-lab`                | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 服务预热与编译封锁           | `sealed-serving-warmup`               | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| SGLang 注意力调用预封装      | `sglang-prepared-dispatch`            | 不同宿主，仅移植研究  | 现有实现属于SGLang；当前r8 FULL路径是否仍有同类开销尚需证明。                       |
| 短预填充尾段优先调度         | `short-prefill-tail`                  | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| SimLLM 近似前缀复用          | `simllm-migration`                    | 质量风险，原实现排除  | 现有性能观察不能覆盖质量负例；没有实质质量修正不重新推荐。                          |
| SliceGPT 模型剪枝            | `slicegpt-migration`                  | 模型/精度或离线变换   | 离线转换或模型/精度改变需独立验收，不是保持当前BF16身份的直接执行增量。             |
| 分批次与全图并行             | `split-batch-full-graph-migration`    | 运行时资格待补        | 已有规划/双流和图执行预览，缺当前r8匹配完整模型性能证据；不能只因名称含全图就推荐。 |
| StateAxis 异步 KV 传输       | `stateaxis-async-kv-transfer`         | 先证实恢复瓶颈        | 旧组件恢复正例保留，当前独立版本及r8真实restore路径未验收。                         |
| StateAxis 自动前缀缓存       | `stateaxis-automatic-prefix-cache`    | 零命中，不重复原配置  | 历史closure没有缓存命中，只证明无害边界。                                           |
| StateAxis 流水化权重加载     | `stateaxis-pipelined-weight-loading`  | 仅冷启动收益          | 启动耗时收益独立保留，不列稳态吞吐加速项。                                          |
| TraceLoom 推理时间线分析     | `traceloom`                           | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| Tricard 生命周期观测         | `tricard-clm-lifecycle`               | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| 统一通信                     | `unified-communication-migration`     | 功能API，不是实测加速 | 统一通信是功能接入，实际MC2融合另有C07未通过记录。                                  |
| 扩展发现与启用管理           | `vllm-hust-extension-manager`         | r8已包含              | 已在冻结r8包含，不重计为新增叠加。                                                  |
| OPset 算子优化集             | `vllm-hust-opset`                     | 先查调用路径          | 先核对MoE底座是否实际命中受支持Linear–SwiGLU路径和ABI；没有命中即移除。             |
| vLLM 本地控制接口            | `vllm-local-control-host`             | 工具或控制面          | 保留工具、观测和服务控制作用，不把功能交付当作单实例吞吐提升。                      |
| vSpec 投机解码               | `vspec`                               | 暂停泛化推荐          | 局部argmax已测无收益；动态vSpec专题另有质量和后端归因问题，不能笼统列为首选。       |

仅restore/RMS既有接入、BidKV瓶颈检查、OpSet调用路径检查保留在[本轮优先级](README.md#%E5%85%88%E7%9C%8B%E7%BB%93%E8%AE%BA)。其余条目保留研究或工具价值；没有明确新机制、当前瓶颈和资格证据时，不作为当前r8直接叠加的加速推荐。
