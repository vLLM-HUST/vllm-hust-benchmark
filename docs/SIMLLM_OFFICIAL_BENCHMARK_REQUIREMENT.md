# SimLLM 官方 Warm-Cache 吞吐测试需求

## 背景与问题

当前通用 `random-online` 官方规格使用：

- `request_rate=1`
- 1024 input tokens
- 256 output tokens

它适合在线延迟基线，但不适合评估 SimLLM 的 warm-cache 吞吐收益，原因如下：

1. 1 req/s 的客户端到达率可能低于 Baseline 和 SimLLM 的服务能力，最终测到的是请求发生器上限，而不是引擎吞吐上限。
2. SimLLM 主要减少可复用长 prompt 的 prefill 工作量。短输入、长输出会弱化其目标负载特征。
3. SimLLM 的 measured pass 必须复用 warmup pass 建立的缓存；重启服务或变更 prompt、seed、token budget 都会破坏可比性。
4. 只记录最终 req/s 不足以证明收益来自 SimLLM 命中，需要同时提供 cache hit 和 scheduler rewrite 证据。

## 参考脚本

本需求以仓库中的以下脚本作为工程参考：

```text
scripts/run_simllm_saturated_throughput_warm_cache.sh
```

参考脚本调用现有 warm-cache runner：

```text
scripts/run_simllm_random_online_warm_cache.sh
```

工程师在编写官方脚本前应阅读并验证这两个脚本。官方实现应继承以下已经过 Ascend device 5 实机验证的逻辑：

1. 从现有 official random-online baseline spec 派生 saturated same-spec。
2. Baseline 和 SimLLM measured pass 使用相同 seed、prompt、模型、server 参数和 client 参数。
3. measured pass 设置 `request_rate=inf` 和有界 `max_concurrency`，解除 1 req/s 的客户端限速。
4. SimLLM 组先执行低速 warmup，且 warmup 与 measured pass 使用相同 seed 和 prompt 数；两者之间不重启服务。
5. 显式设置 `temperature=0`，避免模型 generation config 自动启用不适合该测试的采样路径。
6. 强制 `max_num_batched_tokens >= input_len`，防止 prompt 分块造成 warmup hash 与 measured hash 不一致。
7. SimLLM KV cache 容量默认与 prompt 数量一致，避免为未使用的 cache entries 占用 HBM。
8. 分别保存 Baseline 和 SimLLM 原始 benchmark JSON，并自动生成 JSON/Markdown 对比结果。
9. 检查两组 `completed == num_prompts`；结果不完整时非零退出，不允许使用成功请求子集计算收益。

参考脚本的默认实测入口为：

```bash
cd /workspace/vllm-hust-benchmark
ASCEND_RT_VISIBLE_DEVICES=5 \
CURRENT_MODEL_PATH=/data/shared_models/Qwen2.5-14B-Instruct \
bash scripts/run_simllm_saturated_throughput_warm_cache.sh
```

参考脚本当前产生的主要文件包括：

```text
.benchmarks/simllm-saturated-throughput-warm-cache/
├── saturated-same-spec.json
├── baseline-disabled/raw_benchmark_result.json
├── enabled-warm-cache/raw_benchmark_result.json
├── throughput_comparison.json
└── throughput_comparison.md
```


## 3. 测试目标


在同一硬件、模型、代码版本和请求集合下，对比以下两组：

- Baseline：SimLLM 关闭，不执行 SimLLM warmup。
- SimLLM warm cache：SimLLM 开启，先暖缓存，再在不重启服务的情况下执行正式测量。

主要指标为成功请求吞吐量 `request_throughput`，辅助指标包括 total token throughput、TTFT 和 TPOT。



## 4. 官方主测试规格

### 4.1 固定环境

| 项目 | 要求 |
| --- | --- |
| 硬件 | 单张 Huawei Ascend 910B2 |
| 模型 | `Qwen/Qwen2.5-14B-Instruct` |
| 精度 | FP16 |
| Tensor Parallel | 1 |
| 服务端 | vLLM-HUST + vLLM-Ascend-HUST，版本和 commit 必须写入 artifact |
| SimLLM | 使用被测 commit，不得在 Baseline 与 SimLLM 组之间切换其他代码 |
| 设备选择 | 由官方 runner 分配；必须记录实际 device id 和芯片型号，不能从文件名推断 |

### 4.2 固定请求规格

| 参数 | 值 |
| --- | ---: |
| scenario | `simllm-random-online-warm-cache-throughput` |
| endpoint | `/v1/completions` |
| dataset | deterministic random |
| seed | `0` |
| num_prompts | `32` |
| input_len | `4096` |
| output_len | `32` |
| temperature | `0` |
| measured request_rate | `inf` |
| measured max_concurrency | `16` |
| warmup request_rate | `1` |
| warmup passes | `1` |
| warmup seed | 与 measured seed 相同 |
| warmup num_prompts | 与 measured num_prompts 相同 |
| max_num_batched_tokens | 不小于 `4096`，主规格固定为 `4096` |
| SimLLM KV cache entries | 不小于 `32`，主规格固定为 `32` |

选择 4096-token 输入是为了让负载以 prefill 为主；选择 32 个请求和 16 并发，是当前实现已验证能够稳定完成 warm-cache 饱和测试的主规格。该规格不是通用扩展性结论，因此还必须执行第 8 节的稳定性测试。

## 5. A/B 执行流程

每次 repetition 必须执行完整、相互隔离的 A/B 流程。

### 5.1 Baseline

1. 确认目标 device 空闲并记录初始 HBM。
2. 启动 SimLLM disabled 服务。
3. 等待 `/health` ready，并执行一个不计入结果的健康检查请求。
4. 使用第 4.2 节的 measured 参数执行饱和压测。
5. 保存原始结果、服务日志、resolved same-spec 和运行元数据。
6. 停止完整服务进程树，并确认端口和 NPU 内存释放。

### 5.2 SimLLM warm cache

1. 使用相同代码、模型和 server 参数启动 SimLLM enabled 服务。
2. 等待服务 ready。
3. 使用与 measured pass 完全相同的 prompt 集合、顺序和 seed 执行一次低速 warmup。
4. warmup 完成后不得重启服务、清空缓存或改变请求集合。
5. 执行 `request_rate=inf`、`max_concurrency=16` 的 measured pass。
6. 保存 warmup 结果、measured 原始结果、cache hit/rewrite 统计、服务日志和运行元数据。
7. 停止完整服务进程树，并确认资源释放。

### 5.3 重复次数

- 正式采集至少执行 3 个有效 repetitions，建议 5 个。
- 每个 repetition 都必须重新启动 Baseline 和 SimLLM 服务，不能复用上一轮缓存。
- 官方汇总采用各 repetition 的中位数，同时报告最小值、最大值和变异系数。
- 若主指标变异系数大于 5%，不得直接发布单一官方结果；应继续运行或定位环境抖动。

## 6. 公平性与 same-spec 要求

Baseline 与 SimLLM measured pass 之间，以下字段必须一致：

- model、dtype、tensor parallel、block size 和服务端内存参数
- dataset、prompt token ids、seed 和请求顺序
- input/output token length
- temperature、endpoint、request rate 和 max concurrency
- max_num_batched_tokens
- vLLM-HUST 与 vLLM-Ascend-HUST commits
- CANN、torch、torch-npu 和 Python 环境

唯一允许的功能性差异是 SimLLM enable/config 环境变量，以及 SimLLM 组在 measured pass 前执行 warmup。

脚本必须生成 resolved same-spec 文件及稳定的 spec hash。汇总程序在发现 A/B setting signature 不一致时必须失败，不能只打印 warning。

## 7. 指标和证据

- successful requests / failed requests
- benchmark duration
- request throughput，req/s，主指标
- input、output、total token throughput
- mean、median、P99 TTFT
- mean、median、P99 TPOT 或 ITL
- Peak concurrent requests
- 每轮和汇总后的提升百分比

提升计算方式：

```text
improvement_percent = (simllm_median - baseline_median) / baseline_median * 100
```


## 本地可行性结果，仅供参考

以下结果来自非官方本地脚本，说明能够体现 SimLLM 吞吐差异：

| 指标 | Baseline | SimLLM warm cache | 差异 |
| --- | ---: | ---: | ---: |
| Requests/s | 1.4055 | 4.7897 | +240.78% |
| Total tokens/s | 5801.86 | 19771.68 | +240.78% |
| Mean TTFT | 3616.75 ms | 547.20 ms | -84.87% |
| 成功请求 | 32/32 | 32/32 | 均无失败 |



