# MMLU-Pro 0.18 Native 子集质量测试

## 对象与方法

Qwen3.5-35B-A3B BF16，Ascend 910B2，TP2/PP1/DP1。Native采用官方0.18源码及既有Qwen3.5 text-only兼容补丁，模块版本vLLM
0.18.0，发行包元数据0.18.0+empty，vLLM-Ascend 0.18.0；镜像CANN 8.5.1，宿主驱动26.0.rc1。

Native配置为32K上下文、EP开启、APC关闭、无MTP、max_num_seqs=16、max_num_batched_tokens=8192、FULL_DECODE_ONLY。混合状态适配将请求的attention
block size 128调整为2048；独立服务及新建空编译缓存。模型权重与tokenizer只读共享，不复用前轮推理状态或答案。

采用既有固定200题、原请求内容及原评分器，temperature=0、seed=0、输出上限8192、无限到达率、保留默认thinking与EOS。200题服务端分词计数均与既有输入一致。三个已测优化方案的请求内容与评分器一致，直接复用，其开发运行栈和配置不改名为0.18。

## 结果

| 方案                            | 正确/总数 | 传输失败 | 输出截断 | 相对0.18 Native判对率差 |
| ------------------------------- | --------: | -------: | -------: | ----------------------: |
| Native 0.18/0.18                |   151/200 |        0 |       18 |             0.0个百分点 |
| vLLM-HUST KV materialization    |   152/200 |        0 |       23 |            +0.5个百分点 |
| vLLM-HUST Microflow             |   149/200 |        0 |       25 |            -1.0个百分点 |
| vLLM-HUST恢复组合 E24/R36+KV+FA |   146/200 |        0 |       22 |            -2.5个百分点 |

KV
materialization是这三套已测方案中本子集判对数最高的方案，不表示其吞吐最高，也不将单次子集差异解释为全集质量或统计非劣。原生请求窗口1269.203014612秒，仅为质量执行时长，不作为限长性能排名。

## 证据

- `quality018-completed-audit-r1.json`：200条SSE独立重建、逐题评分、三方案配对变化及原始哈希；不含完整问答。
- `runtime-identity.json`、`cache-policy.json`：实际导入、启动参数和缓存初始化。
- `quality-input-identity-r1.json`：7个输入/客户端/评分器文件与原冻结计划一致。
- `receipt.json`：原控制器回执，客户端/API均退出0，但初次资源观察未释放，控制器退出1。
- `quality018-release-followup-r1.json`：worker自然退出后的进程组及四卡释放证据，无额外终止信号。
- `quality018-finalization-r1.json`：原清单保持不变，服务日志仅追加302字节，原151555字节前缀SHA一致；其余原件未变。
- `archive.json`：完整私有原件的保管信息。原始回答不随公开投影发布。

上述收尾观测限制保留原貌，不改写为原控制器退出0，不影响已独立复核的200条任务结果。
