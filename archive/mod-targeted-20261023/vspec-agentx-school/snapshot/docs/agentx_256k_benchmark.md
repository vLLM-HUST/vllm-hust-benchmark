# AgentX 256K 评测接入

## 结论

AgentX 不是一个新的单值评分。它使用 393 个 Agent 会话、68,266 个模型请求组成的
256K 长上下文闭环回放，同时报告吞吐、首 token 延迟、输出交互延迟和运行有效性。
它适合验证 vSpec 在长历史、多轮、子 Agent 并行和前缀复用负载下的服务收益，但不能
替代 GSM8K 正确性评测，也不能从合成文本判断模型答案质量。

本仓库提供 Qwen3.5 Frontier target-only 和固定 MTP2 的 target 模板，并为启动器增加
`--synthetic-acceptance-length`。同模型、MTP2、draft length 2 的本地
SPEED-Bench coding 验收长度现已测得：thinking on 为 `2.655840687164804`，
off 为 `2.6160458452722066`。测量口径与原始计数见
[`qwen35_speedbench_coding_al.md`](qwen35_speedbench_coding_al.md)。GSM8K 的
`2.8118` 对应不同负载，不用于 AgentX 合成验收。

## 指标含义

| 项目 | 含义 | 使用方式 |
|---|---|---|
| Output throughput | 测量窗口及官方 drain 口径下的输出 token/s | 主要容量指标，但不能单独使用 |
| Request throughput | 每秒完成请求数 | 请求长度不同，必须结合请求构成解释 |
| TTFT | 从请求发出到首 token 的延迟分布，越低越好 | 判断长 prefill 和排队代价 |
| ITL | 单个请求在首 token 后的平均 token 间隔，越低越好 | 判断 decode 是否流畅 |
| Output interactivity | 对每个请求计算 `1 / ITL`（tok/s/user）后再统计分位数，越高越好 | 判断聚合吞吐提升是否牺牲单个活跃用户的输出速度 |
| Errors / context overflow | 请求错误、上下文溢出和跳过数量 | 任一侧异常会破坏公平性 |
| `submission_valid` | 官方 harness 对协议锁和运行阈值的检查结果 | 必须为 `true`，但不等于官方认证 |

`--concurrency 4` 表示同时维持 4 棵 Agent 会话树，不是固定 4 个 HTTP 请求。子 Agent
展开后在途请求可以超过 4，工具等待阶段也可能低于 4。回放是闭环的，因此更快的服务
会在同一测量窗口内推进得更远，实际执行的请求组合可能不同。报告必须同时保留吞吐、
延迟、完成请求数、token 数和请求长度分布，不能只比较一个 token/s。特别要注意，
output interactivity 不是聚合 output throughput 的别名：前者是逐请求 decode 速度的
分布，后者是整段测量窗口内所有请求的总输出带宽。

`smoke` 的有效测量窗口为 900 秒，`formal` 为 3,600 秒。两者使用相同数据、固定种子、
25% 到 75% 的初始轨迹位置、prefix primer 和每条 lane 额外 10 个预热请求。首次数据
重建、预热、最多 30 秒 drain 和导出时间不计入上述窗口，因此实际墙钟时间更长。

## 公平对照

Target-only 与 MTP2 必须固定以下条件：

- 同一个 Qwen3.5-35B-A3B 权重和 tokenizer revision；
- TP2、EP2、两张 Ascend 910B2、BF16、KV dtype auto；
- 262,144 最大上下文、APC、chunked prefill、async scheduling；
- `FULL_AND_PIECEWISE` target graph、相同 host DRAM/KV offload 预算；
- 相同 AgentX commit、dataset revision、profile、seed 和 concurrency；
- 每个方案独立重启服务，不能让上一轮的 prefix cache 污染下一轮。

先用 `smoke` 在 concurrency 2 和 4 做容量检查，再冻结一个不会出现 context overflow、
OOM 或服务错误的并发点做成对复测。需要容量曲线时分别运行多个并发点，不要在一次
结果里混合并发。

## 服务启动

以下命令与 Frontier 固定协议一致。两侧都加载 vSpec 的 Qwen3.5 host 兼容补丁；
target-only 不传 `--speculative-config`，所以没有投机解码。

### Target-only

```bash
source /opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh
export PATH=/opt/vllm-hust-cann91/bin:$PATH
export PYTHONPATH=/root/data/vllm-hust-vSpec-github/src:/root/data/vllm-hust-latest:/root/data/vllm-ascend-hust-latest
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_PLUGINS=ascend,vspec
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_AOT_COMPILE=0
export HUST_VSPEC_ENABLED=1
export HUST_VSPEC_METHOD=mtp
export HUST_VSPEC_ADAPTIVE_SPECULATION=0
export HUST_VSPEC_MTP_STRICT_GRAPH=1
export HUST_VSPEC_MAX_NUM_SEQS=16

vllm serve /workspace/models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b-frontier-baseline \
  --host 127.0.0.1 --port 18185 \
  --dtype bfloat16 --kv-cache-dtype auto --block-size 128 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --seed 0 --scheduling-policy fcfs --distributed-executor-backend mp \
  --disable-custom-all-reduce --no-trust-remote-code --load-format auto \
  --generation-config vllm --no-enable-log-requests --uvicorn-log-level info \
  --enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --async-scheduling --no-enforce-eager \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --enable-expert-parallel \
  --additional-config '{"enable_cpu_binding":true}'
```

### MTP2

先把下列变量设为匹配 thinking mode 的本地 SPEED-Bench 实测值。MTP2 的合法区间是 `[1, 3]`，因为平均
验收长度包含 verification step 的 bonus token。

```bash
export SPEED_BENCH_ACCEPTANCE_LENGTH=2.655840687164804  # thinking on

./manage.sh run -- vllm-hust-vspec \
  --protocol qwen35-frontier-mtp2 \
  --target-model /workspace/models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b-frontier-mtp2 \
  --device 0,1 \
  --synthetic-acceptance-length "$SPEED_BENCH_ACCEPTANCE_LENGTH"
```

thinking off 改用 `2.6160458452722066`，同时让 AgentX target 配置的 thinking
mode 保持一致。该参数只用于合成负载性能评测。它会在宿主 speculative config 中设置
`rejection_sample_method=synthetic` 和 `synthetic_acceptance_length=<实测值>`。
vSpec 会拒绝将它与 Adaptive gamma 同时使用，避免固定验收假设和动态宽度互相污染。

## 运行 AgentX

使用独立的官方回放仓库，不复制或修改它的调度器：

```bash
git clone git@github.com:CubeLander/agentx-bench.git
cd agentx-bench
uv sync --locked
uv run --frozen python prepare.py
uv run --frozen pytest -q
```

Target-only：

```bash
mkdir -p .cache
cp /root/data/vllm-hust-vSpec-github/configs/agentx-qwen35-frontier-target-only.example.json \
  .cache/qwen35-target-only.json
# 替换所有 REPLACE_WITH_*，并记录真实 revision 和内存分配。
uv run --frozen python bench.py plan \
  --target .cache/qwen35-target-only.json --profile smoke --concurrency 4
uv run --frozen python bench.py run \
  --target .cache/qwen35-target-only.json --profile smoke --concurrency 4
```

MTP2：

```bash
cp /root/data/vllm-hust-vSpec-github/configs/agentx-qwen35-frontier-mtp2.example.json \
  .cache/qwen35-mtp2.json
# JSON 中的 forced_acceptance_length 必须与服务启动参数完全一致。
# 同时替换 SPEED-Bench 证据、thinking mode 和所有 revision 占位符。
uv run --frozen python bench.py plan \
  --target .cache/qwen35-mtp2.json --profile smoke --concurrency 4
uv run --frozen python bench.py run \
  --target .cache/qwen35-mtp2.json --profile smoke --concurrency 4
```

结果位于 AgentX 仓库的 `artifacts/<run>/`。至少保存 `run.json`、`harness.log` 和完整
`aiperf/` 目录。只有两侧均为 `submission_valid=true`、无异常错误，并且服务配置与
强制验收长度证据匹配时，才计算 MTP2 相对 target-only 的吞吐提升。

## 本地配对结果（2026-09-30）

以下结果使用 AgentX commit `59ff14f7f934cc462f3cbc02ddafee024c8820d1`、`smoke`
profile、concurrency 4 和完整 900 秒正式窗口。两侧使用相同模型、硬件和执行树；执行树
SHA256 为 `2e37ad547a6b9ee75a9ac7fc714e0166eea13d27c363ddc14f3fdb40f5f1006d`。

| 方案 | 有效 | 请求数 | 输出 token | 输出吞吐 (tok/s) | 请求吞吐 (req/s) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target-only Graph | 是 | 82 | 49,469 | 52.6266 | 0.08723 | 917.11 | 18.93 |
| 固定 MTP2 Graph | 是 | 113 | 79,791 | 84.8839 | 0.12021 | 1,262.21 | 14.64 |

MTP2 的输出吞吐为 target-only 的 **1.6129x**，请求吞吐为 **1.3780x**；平均 ITL
改善 **1.2928x**。代价是平均 TTFT 增加 `345.09 ms`（`1.376x`）。两侧
`submission_valid` 均为 `true`，`error_summary` 均为空，服务日志未发现 graph
fallback、OOM、HCCL、context overflow、`RuntimeError` 或 traceback。

这是闭环回放：更快的 MTP2 在同一窗口内推进了更多请求，因此请求数、prompt token
总量和输出 token 总量不同。以上加速比使用官方 AIPerf 的聚合吞吐字段，不用总墙钟或
请求数自行重算。MTP2 性能运行使用 thinking-on SPEED-Bench coding 的固定 AL
`2.655840687164804` 和宿主 synthetic sampler；它证明 serving 性能，不证明生成内容
正确性。原生 sampler 的 256K 正确性由独立的 26-case 检索测试覆盖，见
[`qwen35_frontier_mtp2.md`](qwen35_frontier_mtp2.md#256k-正确性)。

本地原始数据保存在：

- `benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/`
- `benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/`

## 当前边界

- vSpec 的自然 GSM8K MTP2 结果为 `1.3730x`，这不是 AgentX 结果。
- `--synthetic-acceptance-length` 已接入宿主已有的 synthetic sampler，并完成 15 分钟
  AgentX NPU 成对运行；该本地 AL 尚未进入上游 golden AL 列表，发布时必须同时附带
  SPEED-Bench 证据和 synthetic 限制。
- AgentX 合成内容不评估准确率、代码正确性或真实 Agent 任务完成率。

## 上游来源

- [AgentX bench 固定回放封装](https://github.com/CubeLander/agentx-bench)
- [AgentX 官方方法](https://inferencex.semianalysis.com/agentx/methodology)
- [官方 AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness)
- [AgentX 256K 数据集](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
