# Leaderboard Benchmark 场景附录（Same-Spec 约束）

## 文档目的

official baseline 与 current benchmark 做正式对照时，要求两者满足 same-spec 约束。因此，这份附录不再按“registry 默认值”直接罗列表，而是按下面两层来组织：

1. benchmark 仓当前显式支持哪些场景。
2. 当这些场景用于 official baseline 对照时，same-spec 真正要求哪些语义参数对齐。

这样做的原因是：随着版本演进、wrapper 抽象和向后兼容需求增长，`official baseline spec` 与 `current benchmark registry defaults` 可能不再逐字段一致。真正决定是否可比较的，是 same-spec 归一化后的语义参数，而不是原始输入长相是否完全一样。

本附录对应的代码快照以当前分支为准：`ws/leaderboard-handoff-scenarios-20260531`。

## Source Of Truth

当前 same-spec 语义由四层共同定义：

| 层次 | 文件 | 作用 |
| --- | --- | --- |
| 场景支持面 | `src/vllm_hust_benchmark/data/official_scenarios.json` | 定义 benchmark 仓显式支持的场景名、benchmark type、默认参数和 leaderboard 映射 |
| 场景加载与 override | `src/vllm_hust_benchmark/registry.py`, `src/vllm_hust_benchmark/models.py` | 把场景载入为 `ScenarioDefinition`，并做 `defaults + overrides` 合并与少量 alias 规范化 |
| official compare 合同 | `docs/official-baselines/*.json` | 定义 official baseline 对照所用的 canonical spec、model、precision、hardware、server/client 参数 |
| same-spec 归一化与 hash | `src/vllm_hust_benchmark/same_spec.py` | 将 official spec 解析成 `resolved_server_parameters` / `resolved_client_parameters`，去除非语义字段，并计算 `resolved_spec_hash` |

因此，判断“当前支持哪些 benchmark 场景”时看 registry；判断“official baseline 与 current benchmark 是否可比较”时看 official spec + same-spec 归一化结果。

## Same-Spec 定义

same-spec 不是“原始命令行完全一致”，而是“语义参数归一化后保持一致”。当前实现的 compare 锚点如下：

| 维度 | 约束 |
| --- | --- |
| spec 身份 | `spec_id`, `scenario` |
| 模型身份 | `model`, `model_parameters`, `model_precision` |
| 硬件身份 | `hardware_vendor`, `hardware_chip_model`, `chip_count`, `node_count` |
| server 语义参数 | `resolved_server_parameters` 去掉 `host`, `port`, `model` 之后的字段 |
| client 语义参数 | `resolved_client_parameters` 去掉 `host`, `port`, `model` 之后的字段 |
| 比较锚点 | `resolved_spec_hash` |

### 允许不同但不影响 same-spec 的字段

| 字段 | 为什么可不同 |
| --- | --- |
| `host`, `port` | 运行环境相关，不属于工作负载语义 |
| runtime `model` 路径 | current benchmark 可使用本地模型目录，official spec 仍以 canonical model id 为锚点 |
| 部分历史 artifact 的 component version 回填 | `submission_artifacts.py` 里的 historical overrides 只影响版本元数据，不改变 same-spec 参数 hash |

## 当前 official baseline 的公共 same-spec profile

### 在线 `serve` 公共 profile

适用于 `sharegpt-online`, `random-online`, `prefix-repetition-online`, `instructcoder-online`，`visionarena-online` 在此基础上额外增加 multimodal 限制。

| 维度 | same-spec 语义参数 |
| --- | --- |
| server | `tensor_parallel_size=1`, `dtype=float16`, `enforce_eager=""`, `trust_remote_code=""`, `disable_log_stats=""`, `disable_log_requests=""` |
| client 公共项 | `request_rate=1` |
| 模型与硬件 | 单机场景全部固定为 `Huawei 910B3`, `chip_count=1`, `node_count=1`；模型由各场景单独定义 |

### `throughput` 公共 profile

适用于 `sharegpt-throughput`, `sonnet-throughput`。

| 维度 | same-spec 语义参数 |
| --- | --- |
| server | `tensor_parallel_size=1`, `dtype=float16`, `enforce_eager=""`, `trust_remote_code=""`, `disable_log_stats=""` |
| client 公共项 | `backend=vllm`, `num_warmups=0` |
| 模型与硬件 | 当前 official baseline 固定为 `Qwen/Qwen2.5-14B-Instruct`, `FP16`, `Huawei 910B3`, `1 x 1` |

### `latency` 公共 profile

适用于 `random-latency`。

| 维度 | same-spec 语义参数 |
| --- | --- |
| server | `tensor_parallel_size=1`, `dtype=float16`, `enforce_eager=""`, `trust_remote_code=""`, `disable_log_stats=""` |
| client 公共项 | `input_len=1024`, `output_len=128`, `batch_size=8`, `num_iters_warmup=10`, `num_iters=30` |
| 模型与硬件 | 当前 official baseline 固定为 `Qwen/Qwen2.5-14B-Instruct`, `FP16`, `Huawei 910B3`, `1 x 1` |

## 按 same-spec 整理的场景与参数

### `serve`

| 场景 | official spec 锚点 | 当前 registry 默认值 | official same-spec 语义参数 | 差异说明 |
| --- | --- | --- | --- | --- |
| `sharegpt-online` | `official-ascend-jan-2026-v0.11.0-sharegpt-online-qwen25-14b-910b3`<br>`model=Qwen/Qwen2.5-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `backend=vllm`<br>`endpoint=/v1/completions`<br>`dataset_name=sharegpt`<br>`dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json`<br>`num_prompts=200` | server 使用在线 `serve` 公共 profile<br>client: `backend=vllm`, `endpoint=/v1/completions`, `dataset_name=sharegpt`, `dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json`, `num_prompts=200`, `request_rate=1` | registry 默认值缺少 `request_rate=1`，而 official compare 需要它；如果 current benchmark 只按 registry 默认值跑，需要显式补齐 |
| `random-online` | `official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3`<br>`model=Qwen/Qwen2.5-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `backend=vllm`<br>`endpoint=/v1/completions`<br>`dataset_name=random`<br>`num_prompts=200`<br>`input_len=1024`<br>`output_len=256` | server 使用在线 `serve` 公共 profile<br>client: `backend=vllm`, `endpoint=/v1/completions`, `dataset_name=random`, `num_prompts=200`, `random_input_len=1024`, `random_output_len=256`, `request_rate=1` | same-spec 会把 legacy `input_len` / `output_len` 归一化成 `random_input_len` / `random_output_len`。因此 raw 参数长相可以不同，但归一化后的语义必须一致 |
| `prefix-repetition-online` | `official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3`<br>`model=Qwen/Qwen2.5-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `backend=vllm`<br>`endpoint=/v1/completions`<br>`dataset_name=prefix_repetition`<br>`num_prompts=200`<br>`input_len=4096`<br>`output_len=256` | server 使用在线 `serve` 公共 profile<br>client: `backend=vllm`, `endpoint=/v1/completions`, `dataset_name=prefix_repetition`, `num_prompts=200`, `prefix_repetition_num_prefixes=10`, `prefix_repetition_prefix_len=3840`, `prefix_repetition_suffix_len=256`, `prefix_repetition_output_len=256`, `request_rate=1` | 这是最典型的“版本演进导致参数长相不同”场景：registry 仍保留 `input_len` / `output_len` 的旧表达，而 same-spec 会展开成 prefix/suffix/output/num_prefixes 四个语义字段；对比时必须以归一化结果为准 |
| `instructcoder-online` | `official-ascend-jan-2026-v0.11.0-instructcoder-online-qwen25-coder-14b-910b3`<br>`model=Qwen/Qwen2.5-Coder-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `backend=vllm`<br>`endpoint=/v1/completions`<br>`dataset_name=hf`<br>`dataset_path=likaixin/InstructCoder`<br>`num_prompts=2048` | server 使用在线 `serve` 公共 profile<br>client: `backend=vllm`, `endpoint=/v1/completions`, `dataset_name=hf`, `dataset_path=likaixin/InstructCoder`, `num_prompts=2048`, `request_rate=1` | 与 `sharegpt-online` 一样，registry 默认值没有显式带出 `request_rate=1`；official compare 时必须按 spec 补齐 |
| `visionarena-online` | `official-ascend-jan-2026-v0.11.0-visionarena-online-qwen25-vl-7b-910b3`<br>`model=Qwen/Qwen2.5-VL-7B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `backend=openai-chat`<br>`endpoint=/v1/chat/completions`<br>`dataset_name=hf`<br>`dataset_path=lmarena-ai/VisionArena-Chat`<br>`hf_split=train`<br>`num_prompts=1000` | server: 在线 `serve` 公共 profile + `limit_mm_per_prompt={"image": 1}`<br>client: `backend=openai-chat`, `endpoint=/v1/chat/completions`, `dataset_name=hf`, `dataset_path=lmarena-ai/VisionArena-Chat`, `hf_split=train`, `num_prompts=1000`, `request_rate=1` | 这是第二类特殊差异：multimodal official baseline 在 server 侧额外固定了 `limit_mm_per_prompt`。current benchmark 如果只照 registry 默认值跑，还不满足 official same-spec |

### `throughput`

| 场景 | official spec 锚点 | 当前 registry 默认值 | official same-spec 语义参数 | 差异说明 |
| --- | --- | --- | --- | --- |
| `sharegpt-throughput` | `official-ascend-jan-2026-v0.11.0-sharegpt-throughput-qwen25-14b-910b3`<br>`model=Qwen/Qwen2.5-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `dataset_name=sharegpt`<br>`dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json`<br>`num_prompts=200` | server 使用 `throughput` 公共 profile<br>client: `backend=vllm`, `dataset_name=sharegpt`, `dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json`, `num_prompts=200`, `num_warmups=0` | registry 默认值更偏“场景入口”，official spec 则固定了 compare 所需的 `backend=vllm` 和 `num_warmups=0`；current benchmark 做 official compare 时必须补齐 |
| `sonnet-throughput` | `official-ascend-jan-2026-v0.11.0-sonnet-throughput-qwen25-14b-910b3`<br>`model=Qwen/Qwen2.5-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `dataset_name=sonnet`<br>`dataset_path=benchmarks/sonnet.txt`<br>`num_prompts=200` | server 使用 `throughput` 公共 profile<br>client: `backend=vllm`, `dataset_name=sonnet`, `dataset_path=benchmarks/sonnet.txt`, `num_prompts=200`, `num_warmups=0` | 与 `sharegpt-throughput` 一样，raw registry defaults 不足以直接表达 official same-spec，需要显式补 `backend` 与 warmup 约束 |

### `latency`

| 场景 | official spec 锚点 | 当前 registry 默认值 | official same-spec 语义参数 | 差异说明 |
| --- | --- | --- | --- | --- |
| `random-latency` | `official-ascend-jan-2026-v0.11.0-random-latency-qwen25-14b-910b3`<br>`model=Qwen/Qwen2.5-14B-Instruct`<br>`precision=FP16`<br>`hardware=Huawei 910B3, 1 x 1` | `input_len=1024`<br>`output_len=128` | server 使用 `latency` 公共 profile<br>client: `input_len=1024`, `output_len=128`, `batch_size=8`, `num_iters_warmup=10`, `num_iters=30` | 这是第三类特殊差异：registry 默认值只保留 workload 身份，official spec 额外固定了 latency 实验的 batch 和迭代设置。对照 official baseline 时，current benchmark 不能只用 registry 默认值 |

## 参数差异与版本演进说明

下面这些差异是当前实现里最容易误判的部分：

| 类型 | 适用场景 | 说明 |
| --- | --- | --- |
| registry 比 official spec 更“薄” | 几乎所有场景 | registry 主要表达“这个 benchmark 场景是什么”，而 official spec 还要表达“如何让 current benchmark 与 official baseline 可比较”。因此 official spec 往往补充 server profile、`request_rate`、warmup 或 iteration 参数 |
| 长度参数归一化 | `random-online` | official same-spec 用 `random_input_len` / `random_output_len` 表示最终语义；current benchmark 仍可输入 legacy `input_len` / `output_len`，但归一化结果必须一致 |
| prefix-repetition 结构化展开 | `prefix-repetition-online` | official compare 不直接按 `input_len=4096, output_len=256` 比较，而是展开成 `prefix_len=3840`, `suffix_len=256`, `output_len=256`, `num_prefixes=10`。这是 same-spec 归一化最关键的特殊例子 |
| multimodal server 约束 | `visionarena-online` | `limit_mm_per_prompt={"image": 1}` 是 official compare 的一部分，不是仅仅由 dataset 决定的隐式行为 |
| throughput / latency 实验控制参数 | `sharegpt-throughput`, `sonnet-throughput`, `random-latency` | `num_warmups`, `batch_size`, `num_iters_warmup`, `num_iters` 这类参数在 official baseline 中是 compare 合同的一部分，不能因为 registry 没写就忽略 |
| host / port / runtime model path 差异 | 所有场景 | 这些字段不进入 same-spec hash。official baseline 与 current benchmark 可以使用不同端口、不同本地模型路径，只要 canonical model 与归一化语义参数一致即可 |
| 历史版本元数据回填 | 历史 same-spec artifact | `submission_artifacts.py` 里的 `HISTORICAL_SAME_SPEC_COMPONENT_OVERRIDES` 只用于回填 component versions，解决老数据源版本信息缺失问题，不属于 same-spec 参数差异 |

## Wrapper 覆盖范围差异

场景 registry 的支持面不等于所有 wrapper 都原生暴露了这些场景。

### 通用 wrapper

下面这些入口面向 registry，原则上可覆盖全部 8 个已注册场景：

| 命令 | 作用 |
| --- | --- |
| `list-scenarios` | 列出已注册场景 |
| `list-leaderboard-map` | 列出场景到 workload / business scenario 的映射 |
| `build-command` | 构建场景对应的 benchmark 命令 |
| `run` | 运行单个场景 |
| `run-both` | 同时对 `vllm-hust` 和 `vllm` 运行同一场景 |

### `run-ascend-ci` 特化 wrapper

当前 `run-ascend-ci` 并没有开放全部 registry 场景，而是只显式支持：

| wrapper | 显式支持场景 | 参数范围 |
| --- | --- | --- |
| `run-ascend-ci` | `random-online`, `sharegpt-online` | 运行元数据：`--run-id`, `--result-root`, `--raw-result-file`, `--submissions-root`, `--submission-dir`, `--aggregate-output-dir`, `--server-log`, `--runtime-root`<br>数据与 workload：`--dataset-path`, `--constraints-file`, `--num-prompts`, `--random-input-len`, `--random-output-len`, `--random-batch-size`, `--request-rate`, `--max-concurrency`, `--input-len`, `--output-len`<br>服务与模型：`--model`, `--model-parameters`, `--model-precision`, `--host`, `--port`, `--dtype`, `--max-model-len`, `--max-num-seqs`<br>硬件与发布：`--hardware-vendor`, `--hardware-chip-model`, `--chip-count`, `--node-count`, `--publish-to-hf`, `--hf-repo-id`, `--allow-random-hf-publish` |

这意味着新增 benchmark 场景时，要先判断你的目标是：

1. 扩展 registry 支持面。
2. 扩展某个特化 wrapper 的显式暴露范围。

很多变更只需要做第 1 类，不一定要同步改第 2 类。

## 维护建议

| 场景 | 建议 |
| --- | --- |
| 做 official baseline 对照 | 先看 official spec 和 same-spec 归一化结果，不要只看 registry 默认值 |
| 新增场景 | 先改 `official_scenarios.json`，再判断是否需要新增 official baseline spec |
| 改默认参数 | 视为合同变更，应同步评估对 same-spec compare 和 official spec 的影响 |
| 改 alias / 归一化规则 | 统一在 `same_spec.py` 或 `models.py` 收口，不要把兼容逻辑散落到 wrapper 层 |
| 对外交接 | 优先引用本附录，而不是从 CLI 帮助或 raw spec 文件手工拼参数表 |