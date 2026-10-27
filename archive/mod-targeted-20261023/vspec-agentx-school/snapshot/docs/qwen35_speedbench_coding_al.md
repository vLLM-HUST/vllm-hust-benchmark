# Qwen3.5-35B-A3B MTP2 SPEED-Bench coding acceptance length

## 测量结果

2026-09-28 在 Ascend 910B2 上完成本地测量。两个模式各运行 SPEED-Bench
`qualitative` 的全部 80 条 coding 请求，`max-concurrency=1`，输出上限 4096；
服务均采用 Frontier 固定 MTP2 协议，80/80 成功，无图回退。

| Thinking | 已接受 draft tokens | Drafts | AL = 1 + accepted/drafts | 输出 tokens |
|---|---:|---:|---:|---:|
| on | 77,495 | 46,801 | **2.655840687164804** | 124,304 |
| off | 45,684 | 28,269 | **2.6160458452722066** | 73,949 |

这两个数是同模型的本地 SPEED-Bench 参考值，可在相同 thinking mode 的 AgentX
**合成验收**实验中分别设置为 `synthetic_acceptance_length`。它们尚未进入上游
golden AL 列表，也不是 AgentX 吞吐或收益结果。

## 固定口径

- 模型：`/workspace/models/Qwen3.5-35B-A3B`，BF16，checkpoint 内置 MTP head，
  `num_speculative_tokens=2`；target TP2 + EP2，on 使用 NPU `0,1`，off 使用 `2,3`。
- 服务：`max_model_len=262144`，APC、chunked prefill、async scheduling，target
  `FULL_AND_PIECEWISE`，proposer `FULL`，严格图保护开启。
- 数据：[`nvidia/SPEED-Bench`](https://huggingface.co/datasets/nvidia/SPEED-Bench)
  `qualitative` 的 coding 类别，revision
  `454f88454792dfa3ccfd7ef15fff248efde44cd1`。从 880 条中保持原顺序选出
  80 条，并调用官方 `prepare.py` 的 `_resolve_external_data` 补全题面。
  本地 `qualitative.jsonl` SHA256 为
  `f73950a606bd3d2f3a27a6d3cf0606a2d077aab8ecb5d71e56a33c8d1cc6c2ce`。
  下载的官方 `prepare.py` SHA256 为
  `a551be4df541474e54e21b480022b0cbb66c2da068fda61b2a64bd3223bbbed2`。
- 请求：vLLM `bench serve`，OpenAI completions，`--num-prompts -1`，
  `--speed-bench-output-len 4096`，`--max-concurrency 1`，`--seed 0`。
- 采样：thinking on 使用 temperature `0.6`、top-p `0.95`、top-k `20`、
  presence penalty `0`；off 使用 `0.7`、`0.8`、`20`、`1.5`。客户端模板分别传入
  `enable_thinking=true/false`。

模型 chat template 在 on 模式的提示词末尾已经写入 `<think>`，在 off 模式写入
`<think>\n\n</think>`。因此 on 的 `generated_texts` 不必再次包含开头标记。
off 的 80 条生成中有 1 条自行输出了空的 `<think></think>` 标记，原始文本已保存在
结果 JSON；模板参数仍为 `enable_thinking=false`。

## 复现与证据

数据准备脚本：[`scripts/prepare_speedbench_coding.py`](../scripts/prepare_speedbench_coding.py)。
下载官方 `prepare.py` 后，以包含 `datasets`、`tiktoken`、`pandas`、`numpy` 的
Python 环境执行：

```bash
python scripts/prepare_speedbench_coding.py \
  --official-prepare /tmp/vspec-speedbench-prepare.py \
  --output-dir /tmp/vspec-speedbench-data
```

分别运行：

```bash
DATASET_DIR=/tmp/vspec-speedbench-data NPU_IDS=0,1 PORT=18185 \
  bash scripts/measure_qwen35_speedbench_al.sh on
DATASET_DIR=/tmp/vspec-speedbench-data NPU_IDS=2,3 PORT=18186 \
  bash scripts/measure_qwen35_speedbench_al.sh off
```

原始结果与服务日志：

- on：`benchmark_results/qwen35_speedbench_al/20260928-044132-on/`
- off：`benchmark_results/qwen35_speedbench_al/20260928-045302-off/`

每个目录包含 `result.json`、`al_summary.json`、`bench.log`、`server.log`、
`serve_cmd.txt`、`bench_cmd.txt` 和 `dataset.sha256`。`result.json` 的
`spec_decode_num_drafts` 与 `spec_decode_accepted_tokens` 可直接复算上述 AL。
两个服务日志均记录 target PIECEWISE/FULL 和 proposer FULL graph 捕获，严格图
保护没有触发 eager fallback。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
