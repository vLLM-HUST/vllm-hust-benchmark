# KVCompress 实测结果

对应任务：[KVCompress #32](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/32) ·
[技术成果补证 #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)

本报告整理已冻结的 14B LongBench-v2 结果，补充测试子集、准确率和时延开销，并将 35B 的适配与性能结果单独列出。本轮沿用已有归档，不重跑 14B 实验。

**14B 的主要结果是：物理 KV 块减少 61.96%，请求吞吐提高 7.78%；相应代价是准确率下降 0.86 个百分点，平均首 token 时延（TTFT）增加 4.00%。**
这说明压缩在该长输入子集上带来了容量和吞吐收益，同时也增加了首 token 等待时间。

## 14B LongBench-v2 实验

### 测试子集与配置

测试模型为 **Qwen/Qwen2.5-14B-Instruct，FP16，单张 Ascend 910B2，TP=1**。测试采用 LongBench-v2 中符合 32K 服务范围的 **116
个短子集样本**，输入长度为 **10,171–31,869 tokens**。其余 387 个 medium/long 样本超出本次服务范围，未截断纳入。子集准备文件的 SHA-256 见
[evidence-map.json](evidence-map.json)。

两组实验使用相同的插件、宿主和服务参数，通过 KV 预算区分基线与压缩组：

| 配置         | B0：不压缩基线              | B1：压缩组               |
| ------------ | --------------------------- | ------------------------ |
| KV 预算      | 32,768 tokens               | 8,192 tokens             |
| 压缩触发阈值 | 32,896 tokens，高于服务上限 | 9,216 tokens             |
| 运行方式     | 插件加载，保留不压缩路径    | 执行打分、选择及 KV 压缩 |

公共服务参数为：上下文上限 32,768 tokens，block size 128，并发 4，无限到达率，最大批 token 数 16,384，max sequences 4，开启 chunked
prefill，关闭 APC 和 async scheduling。B1 使用 1,024-token 重计算窗口、512-token 近期保护窗口、8,192-token 打分分块、stride
8，以及 64-token 最少请求输出门槛。

### 结果与分析

| 指标                          | B0 → B1 或相对变化                    |
| ----------------------------- | ------------------------------------- |
| 压缩事务的物理 KV 块          | 减少 **61.96%**                       |
| 请求吞吐                      | **+7.78%**                            |
| 总 token 吞吐                 | +7.77%                                |
| 多选准确率                    | **40.52% → 39.66%（−0.86 个百分点）** |
| 平均 TTFT                     | **+4.00%**                            |
| 平均每输出 token 时延（TPOT） | −4.34%                                |
| 平均端到端时延（E2E）         | −6.99%                                |
| 压缩提交 / worker 确认        | 116 / 116                             |
| 失败请求 / 静默截断           | 0 / 0                                 |

物理块减少反映了压缩事务释放的 KV 空间；吞吐提高、TPOT 和 E2E 下降则体现了该子集上的服务收益。与此同时，平均 TTFT 增加 4.00%，因此采用时需要在首 token
响应速度与后续生成效率之间取舍。这里报告的时延均为均值。

准确率下降 0.86 个百分点，处于实验预设的 1 个百分点容差内。配对 bootstrap 的准确率差值 95% 区间为 **[−4.31, +1.72]
个百分点**，该规模的样本仍有较大的质量估计不确定性。

每组进行了一轮冷启动测试，同机其他 NPU 同时存在测试负载，因此这里保留单轮观测值，不估计重复运行的性能分布。物理块减少比例按 `(source − destination) / source`
计算，反映压缩事务的块占用变化；整卡 HBM 还受到预分配 KV 池等因素影响。吞吐和时延变化按 `B1 / B0 − 1` 计算，准确率变化使用百分点。

## 35B 适配与后续实验

### 同期 LongBench-v2

Qwen3.5-35B-A3B 使用 BF16、两张 Ascend 910B2、TP=2、eager 执行和非 thinking 模板，并通过 legacy GDN ABI bridge
适配当时的宿主。该模型对应的可用短子集为 110 例，结果如下：

| 指标                    | B0 → B1 或相对变化       |
| ----------------------- | ------------------------ |
| 压缩事务的物理 KV 块    | 减少 63.76%              |
| 请求吞吐                | −5.83%                   |
| 多选准确率              | 47.27% → 47.27%          |
| 平均 TTFT / TPOT / E2E  | +5.66% / +4.44% / +4.97% |
| 压缩提交 / 逐 rank 确认 | 110 / 220                |

这组实验完成了混合状态模型和 TP 压缩事务的验证，但容量节省没有转化为吞吐收益，三个平均时延指标也均有上升。14B 与 35B 的样本、精度和执行配置不同，因此分别报告。压缩作用于
full-attention KV，GDN 状态由宿主管理。

### v0.10 SWE 实验

后续在 Qwen3.5-35B-A3B 上进行了 SWE C4、900 秒配对测试，使用 APC、MTP2、async scheduling 和
FULL_AND_PIECEWISE。三组候选保留如下：

| KV 预算 / 最少请求输出 | 每卡输出吞吐变化 | 请求 decode P90 变化 | TTFT P95 变化 |
| ---------------------- | ---------------- | -------------------- | ------------- |
| 8064 / 512             | −1.71%           | +1.50%               | +17.15%       |
| 4608 / 1024            | −0.44%           | +1.22%               | +17.06%       |
| 4608 / 2048，最终保留  | +1.33%           | +1.62%               | +8.65%        |

保留配置取得了小幅吞吐收益，但 TTFT P95 仍增加 8.65%。2,048-token 输出门槛使较短请求绕过压缩，效果因请求分布而异。基线同样是插件加载、大预算不压缩的服务。

该配置的服务日志累计记录了 20 次压缩提交和 40 个 rank 确认，统计范围为服务生命周期。最长实测输入为 52,505 tokens，配置上下文上限为 262,144 tokens。另有
32K 检索检查 4/4 通过，用于检查检索路径；SWE 任务语义质量未在这组测试中评估。此处的 P95 时延和每卡输出吞吐与前面的 LongBench-v2 均值及请求吞吐分别统计。

## 证据与复现出处

实验结果沿用固定归档提交 **`aa041953616fa170014a0c7f43fb06bf1c59b846`**。本目录的
[evidence-map.json](evidence-map.json) 列出原文件链接、SHA-256、字节数和指标字段位置。

| 材料                                                                                                                                                                                                                                                              | 内容                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [公开长上下文原报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/public-long-context-benchmarks.md)                               | 子集筛选、B0/B1 配置、启动及评分命令、14B 与 35B 结果 |
| [14B LongBench-v2 摘要](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-working-tree-20260920-qwen25-public-summary.json) | `/records/0`：本报告 14B 指标及原始结果哈希           |
| [35B LongBench-v2 摘要](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json) | `/records/0`：同期 35B 指标                           |
| [35B v0.10 SWE 摘要](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)                  | 三组候选的配对结果、配置和日志哈希                    |
| [同期环境与验证记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/validation.md)                                                   | 宿主版本、校准产物与工作树说明                        |

同期环境记录为 vLLM-HUST `6cdc0304a8ba`、vLLM-Ascend-HUST `5901bedbb718`、Python 3.11.16、torch 2.10 /
torch-npu 2.10.post2。公开摘要将插件标为 `0.6.0 working tree`，同期验证页记为
`0.7.0 release candidate based on de038e7`；本报告保留这两处历史版本记录。

本次整理已按[归档清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/manifest.json)校验所引用文件的哈希和长度。归档摘要包含逐请求结果及日志的哈希，其对应文件的公开下载位置仍需补齐，后续按哈希关联到
Release 或受控原件索引。
