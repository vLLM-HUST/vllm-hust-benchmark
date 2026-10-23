# 公开长上下文 Benchmark 记录

[English](public-long-context-benchmarks.md) | 简体中文

> **历史归档。** 2026-09-20 的公开实验（含负性能结果）继续保留，但不属于
> 0.8 Frontier cohort。新测量请见[测试规程索引](../../benchmark/README.zh.md)。

更新日期：2026-09-20。本记录在合成 commissioning 测试之外，覆盖三个公开长上下文/
KV 场景。它是同宿主工程对照，不是 V4.6 官方基线正式验收，也不替代 30 分钟 A3
稳定性测试。

标准发布测试按验收要求冻结为 Qwen2.5-14B-Instruct；Qwen3.5-35B-A3B 仅作为
面向相近 hybrid 架构的补充适配证据。旧 Qwen2.5-Coder 运行仍保存在
[HTML 测试榜单](../../results/benchmark-leaderboard.html)的历史记录中。

## 场景与冻结输入

| 场景 | 用途 | 实测范围 | 输入 token | 预处理 SHA-256 |
| --- | --- | ---: | ---: | --- |
| LongBench-v2 | 长上下文多选推理 | short 档中 116 条可容纳样本 | 10,171–31,869 | `06255143...b538be6c` |
| LongBench `passage_retrieval_en` | 段落检索 | 200/200 条 | 10,412–15,670 | `cc015530...98a2cb6` |
| LongBench `qasper` | 科研论文问答 | 200 条中达到 5,120-token 压力下限的 93 条 | 5,168–22,074 | `20b04612...b8a6da94` |

LongBench-v2 固定 Hugging Face revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`，下载文件 `data.json` 的
SHA-256 为 `15d61c22...604c7fe2`。LongBench 固定 revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，`data.zip` 的 SHA-256 为
`cb45b11a...857f7f64`。

所有接纳的 prompt 均未截断。服务上限为 32,768 token，因此超出该范围的 387 条
LongBench-v2 medium/long 档样本会登记到 unsupported 文件，不会缩短后混入测试。
Qasper 中低于 5,120 token 的样本无法触发原 4K 压缩阈值，因此不纳入本次压力集。
`--limit 0` 表示全部可接纳样本，而非抽样。

下载脚本在解压前校验固定 revision 与哈希。LongBench-v2 数据卡声明 Apache-2.0；
LongBench 汇集的源数据各有自身条款，因此脚本只从上游下载，本仓库和发行包均不
再分发 benchmark 正文。使用者须遵守各源数据集的许可条件。

## 复现步骤

用被测模型的准确 tokenizer 下载并预处理：

```bash
MODEL=/path/to/Qwen2.5-14B-Instruct
python scripts/kvcompress_benchmark_data.py download \
  --source all --root .benchmarks/datasets

python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-v2 --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-v2.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-passage-retrieval-en --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-passage-retrieval-en.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-qasper --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-qasper.jsonl
```

Qwen3.5 等默认开启思考模式的模型必须在三项 `prepare` 命令中追加
`--disable-thinking`。该选项把 `enable_thinking=false` 传给 chat template，并在
request-set manifest 中冻结 `chat_template_enable_thinking: false`；不得用默认思考
模板生成的 request set 与非思考 B0/B1 结果混合比较。

冻结 B0 在相同宿主与插件下设 32,768-token 预算和 32,896-token 阈值，所有请求
均不发生压缩。0.6 候选复用该 B0，因为宿主、模型、tokenizer、数据集和服务参数均
未变化。B1 使用[示例配置](../../../examples/triattention.json)：
8,192-token 预算、1,024-token 重算窗口、512-token 最近保护、8,192-token 评分
分块、每八层评分一次，并要求请求至少生成 64 token 才值得进入压缩路径。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve "$MODEL" \
  --served-model-name qwen2.5-14b --dtype float16 \
  --block-size 128 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 4 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm
```

针对两组服务分别运行每个预处理文件，按实际任务修改路径与标签：

```bash
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-v2.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-14b \
  --result .benchmarks/results/b1-longbench-v2.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 0 \
  --server-log .benchmarks/logs/b1.log

# passage_retrieval_en 的 max_output_tokens=32，必须验证绕过路径：
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-passage-retrieval-en.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-14b \
  --result .benchmarks/results/b1-passage-retrieval-en.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 0 \
  --server-log .benchmarks/logs/b1.log \
  --compression-expectation forbidden

python scripts/kvcompress_benchmark_score.py \
  --baseline .benchmarks/results/b0-longbench-v2.json \
  --plugin .benchmarks/results/b1-longbench-v2.json \
  --quality-tolerance-pp 1 \
  --output .benchmarks/results/paired-longbench-v2.json
```

服务必须使用 `--generation-config vllm`。运行器固定 `temperature=0`、
`top_p=1`、`top_k=-1`、`min_p=0`、presence/frequency penalty 0、
`repetition_penalty=1`、`n=1`、禁用 beam、空 stop、`seed=0`、
`add_special_tokens=true`，
保存逐请求输出、TTFT/TPOT/E2E、包版本、设备采样、服务端 token 计数和 scheduler/
worker 压缩证据。评分器实现 LongBench-v2 官方答案抽取、LongBench 检索得分及 QA
F1 归一化。如果请求失败、服务端 token 数显示截断、实际行为不符合声明的压缩预期、
worker 回执缺失或质量下降超过 1 个百分点，B1 会被判失败。压缩预期默认是
`required`；短输出检索显式指定 `forbidden`，用于证明绕过路径，而非默许一次没有压缩
的运行。

## 测试结果

### 标准模型：Qwen2.5-14B-Instruct

模型：`Qwen/Qwen2.5-14B-Instruct`，FP16。设备：每个服务使用一张 Ascend 910B2。
每组运行一个冷服务生命周期，不限请求到达速率，并发 4；服务端与客户端均使用冻结
采样契约。B0 定义如上。

| 场景 | B0 → B1 质量 | 请求数 / 压缩数 | 请求吞吐 | 平均 TTFT | 平均 TPOT | 平均 E2E | 物理 block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 40.52 → 39.66 accuracy（−0.86 pp） | 116 / 116 | **+7.78%** | +4.00% | **−4.34%** | **−6.99%** | −61.96% |
| 段落检索 | 98.75 → 98.75（0.00 pp） | 200 / 0（绕过） | **+2.41%** | +0.36% | **−4.13%** | **−2.33%** | 不适用 |
| Qasper | 43.60 → 43.36 F1（−0.24 pp） | 93 / 11 | **+1.35%** | **−2.94%** | **−1.08%** | **−1.41%** | 触发样本 −38.73% |

两组共 818 个请求全部完成，失败和静默截断均为 0。B1 有 127 次 scheduler 提交和
127 次 worker 回执。实际触发压缩的样本合计从 20,666 个源 block 降至 8,128 个
目标 block，减少 60.67%。三项质量下降均在冻结的 1 pp 门槛内。

工作负载边界很明确：10K–16K 输入的检索任务最多请求 32 个输出 token，因此
64-token 门槛按设计绕过评分和 copy，200 条均无提交/回执。Qasper 大部分请求低于
B1 的 9,216-token 阈值，仅 11 条压缩。每组只有一次运行，且同机其他 NPU 同期在跑
测试，因此性能差值只作探索性工程证据；质量、请求完整性和压缩事务证据是主要结论。

机器可读证据见
[Qwen2.5 公开基准摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen25-public-summary.json)。

### 补充模型：Qwen3.5-35B-A3B

Qwen3.5 使用 BF16、TP=2、eager 模式与非思考 chat template；每个服务使用两张
Ascend 910B2。它不是标准验收模型。

| 场景 | B0 → B1 质量 | 请求数 / 压缩数 | 请求吞吐 | 平均 TTFT | 平均 TPOT | 平均 E2E | 物理 block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 47.27 → 47.27 accuracy（0.00 pp） | 110 / 110 | −5.83% | +5.66% | +4.44% | +4.97% | −63.76% |
| 段落检索 | 100.00 → 100.00（0.00 pp） | 200 / 0（绕过） | −2.39% | +1.65% | +3.10% | +2.54% | 不适用 |
| Qasper | 51.07 → 50.64 F1（−0.43 pp） | 93 / 11 | −1.28% | +5.43% | +9.69% | +2.26% | 触发样本 −42.11% |

两组共 806 个请求全部完成，失败和静默截断均为 0。B1 有 121 次 scheduler 提交，
TP=2 下得到 242 次逐 rank worker 回执。三项质量门槛均通过，但本机宿主要求 eager
执行及显式旧 GDN ABI 兼容桥，性能没有优于 B0。该结果证明当前 hybrid 状态保持、
TP 同步和完整事务链路可运行；段落检索仍是未压缩的绕过验证，不证明压缩后的 needle
保持能力。机器可读证据见
[Qwen3.5 公开基准摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)。

## 尚存限制

- 性能对比每组只有一次运行，不是三轮冷启动统计，且同机其他 NPU 同期有测试负载。
- B0 是同宿主无压缩工程对照，不是 V4.6 指定的官方宿主基线。
- 超过 32K 的 LongBench-v2 medium/long 档需换用更长上下文服务后测试。
- Qwen3.5 是补充适配证据；后续标准发布测试仍使用 Qwen2.5-14B-Instruct。
- 本记录不是 30 分钟、六窗口 A3 稳定性测试；A3 结果单独记录。
