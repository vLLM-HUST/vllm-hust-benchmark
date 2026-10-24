# Draft B16 优化检查点（更新至 2026-09-23 UTC）

本文档记录暂停优化时的可复现状态，后续继续工作应以此文档为准。
它取代 `draft_b16_recovery_handoff_20260914.md` 中较早的性能判断和
resume order，但旧文档仍保留历史排查过程。

## 当前目标和统一口径

目标是在工程师提供的 GSM8K 在线压测口径下，使串行 Draft 达到或超过
EAGLE 的 `611.27 tok/s`。

- Target：`/data/shared-models/Qwen2.5-14B-Instruct`
- Draft：`/data/shared-models/Qwen2.5-0.5B-Instruct`
- 数据集：`/root/data/vllm-ascend-hust/benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl`
- Prompt：最终结果 200 条；N20/N50 只用于筛选
- Batch / `max-num-seqs`：16
- 每条最大输出：256 token
- FP16，`max-model-len=32768`
- `max-num-batched-tokens=8192`，`block-size=128`
- Prefix caching 关闭，chunked prefill 开启
- Graph：`FULL_DECODE_ONLY`
- `generation-config=auto`，因此保留模型中的 `repetition_penalty=1.05`
- `temperature=0`，`request-rate=inf`，`seed=0`
- 当前 Draft 使用 async scheduling；已有 EAGLE 参考值来自同步调度，二者并非完全公平，
  但本轮仍把 `611.27 tok/s` 作为工程目标。

## 当前可报告结果

| 方案 | Prompt | 输出 token | 耗时 | 输出吞吐 | 相对 Target-only | 相对 EAGLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target-only 参考 | 200 | 42,318 | 93.1919 s | 454.10 tok/s | 1.000x | 0.743x |
| EAGLE 参考 | 200 | 42,265 | 69.1432 s | 611.27 tok/s | 1.346x | 1.000x |
| 旧 Draft exact gamma=2 | 200 | 42,282 | 77.3557 s | 546.5919 tok/s | 1.204x | 0.894x |
| Draft strict adaptive 1..2，第 1 次 | 200 | 42,281 | 65.00 s | **650.49 tok/s** | **1.433x** | **1.064x** |
| Draft strict adaptive 1..2，第 2 次 | 200 | 42,308 | 62.83 s | **673.37 tok/s** | **1.483x** | **1.102x** |

当前正式方案没有启用 confidence margin，压测请求显式使用 `temperature=0`，保持
严格 top-1 验收。两次 N200 平均吞吐为 **661.93 tok/s**，相对 Target-only 为
**1.458x**，相对 EAGLE 为 **1.083x**；N50 筛选结果为 `623.75 tok/s`。两轮
均为 200/200 请求成功，输出 token 数没有通过提前终止人为减少。

当前严格结果文件：

```text
benchmark_results/draft_strict_temp0_20260923/historical669_repro_n200/GSM8K-n200.json
benchmark_results/draft_strict_temp0_20260923/historical669_repro_n200_r2/GSM8K-n200.json
```

服务退出时的控制器汇总显示 367 次选择中 gamma=2 为 365 次、gamma=1 为 2 次，
token-level acceptance rate 为 `83.653%`。提升来自在线选择、refill=8、gamma2
串行快路径以及 Target 图更新流水，不依赖 relaxed acceptance。

### 2026-09-22 恢复结果

本轮把可选 confidence-margin 验收直接融合进 repetition-aware Triton argmax。
Draft token 的 repetition-adjusted logit、严格 argmax 和 margin mask 在同一 finalize
内核中完成，不再为 margin 路径物化整张 FP32 processed logits。严格模式和 margin
模式使用两个独立 finalize 内核；未传 `--confidence-accept-margin` 时仍走原始轻量
严格内核，不分配 margin mask，也不增加额外 kernel 参数。

| 方案 | Prompt | 输出 token | 耗时 | 输出吞吐 | GSM8K | 相对 Target-only | 相对 EAGLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Draft exact 历史保留值 | 200 | 42,259 | 73.6672 s | 573.6473 tok/s | 未保存逐请求输出 | 1.263x | 0.938x |
| Draft margin=4.0，第 1 次 | 200 | 43,943 | 69.6391 s | 631.0108 tok/s | 114/200 (57.0%) | 1.390x | 1.032x |
| Draft margin=4.0，第 2 次 | 200 | 44,337 | 70.1241 s | 632.2646 tok/s | 107/200 (53.5%) | 1.392x | 1.034x |
| Draft margin=5.25，第 1 次 | 200 | 44,065 | 69.1745 s | 637.0126 tok/s | 109/200 (54.5%) | 1.403x | 1.042x |
| Draft margin=5.25，第 2 次 | 200 | 43,956 | **67.0770 s** | **655.3065 tok/s** | 110/200 (55.0%) | **1.443x** | **1.072x** |

这些数据仅保留为 relaxed acceptance 的历史实验，不再作为正式性能方案。margin
会改变 greedy token 验收并显著降低该批样本的 GSM8K 正确率；当前发布配置和性能
回归均不得设置 `--confidence-accept-margin`。逐请求正确率使用
`/data/datasets/gsm8k/test.parquet` 的标准答案和
`benchmarks/offline_ab.py::evaluate_gsm8k` 计算。

margin=5.25 两轮均值为 `646.1596 tok/s`、`68.1257 s`：相对 EAGLE 平均吞吐
提高 `5.71%`，平均墙钟时间降低 `1.47%`；相对 Target-only 平均吞吐为 `1.423x`。
这两轮输出 token 均多于 EAGLE，因此耗时优势不是由少生成 token 得到的。

历史 margin 结果文件：

```text
benchmark_results/draft_b16_resume_20260922/p5_split_margin5p25_n200_detailed/GSM8K-n200.json
benchmark_results/draft_b16_resume_20260922/p5_split_margin5p25_n200_detailed_r2/GSM8K-n200.json
benchmark_results/draft_b16_resume_20260922/p5_split_margin4_n200_detailed/GSM8K-n200.json
benchmark_results/draft_b16_resume_20260922/p5_split_margin4_n200_detailed_r2/GSM8K-n200.json
```

## 已保留的有效优化

### 1. Gamma=2 专用串行路径

位置：`src/vllm_hust_vspec/backends/draft.py`

开关：

```bash
export VSPEC_DRAFT_GAMMA2_FAST_PATH=1
```

`run_serial_gamma2_without_hidden_copies` 保留两次 Draft forward 和两次采样，
但移除了 Qwen2.5 Draft 不会消费的 continuation hidden-state 写入、切片和 padding。
它只在以下条件全部满足时启用，否则回退上游实现：

- `method=draft_model` 且固定 `gamma=2`
- `pass_hidden_states_to_model=false`
- 无多模态输入、无 MRoPE、无 DCP
- Full graph
- LM-head TP 关闭
- Ascend `enable_reduce_sample=true`
- 全部请求为 greedy sampling；非 greedy 自动回退

第一步产生的 token ID 必须 `clone()`。Graph/reduce-sample 会在第二次采样时复用
输出 buffer；曾尝试直接复用 `self.input_ids`，N50 降至 `545.06 tok/s`，且输出
token 数由 10,610 变成 10,587，已经回滚。若 draft probabilities 存在也必须 clone。

### 2. 精确 repetition-aware greedy

位置：`src/vllm_hust_vspec/backends/draft_repetition.py`、
`src/vllm_hust_vspec/backends/eagle_rejection.py`

开关：

```bash
export VSPEC_DRAFT_FUSED_REPETITION=1
```

当前实现使用持久化 `_PackedRepetitionState`，通过 Triton 对原始 logits 做精确的
repetition-penalty argmax，避免通用采样路径中的整张 logits FP32 临时拷贝和 mask。
这不是早期已失败的全词表 seen-mask materialization 方案；旧交接文档中“该开关应关闭”
的结论已经过期。

显式启用 `--confidence-accept-margin` 时，同一套 partial argmax 由独立 margin
finalize 内核计算 Draft token 的 repetition-adjusted logit 和 relaxed mask；默认严格
路径仍使用原始 finalize 内核，避免可选近似功能拖慢严格解码。

### 3. Draft FIA graph update 并行化

位置：`src/vllm_hust_vspec/backends/draft_parallel_update.py`

当前配置：

```text
--draft-parallel-graph-updates 4
```

实现缓存 graph descriptors/plans，以 4 个 worker 更新图任务，主线程负责 chunk 0。
配合 `--graph-event-ordering`、merged graph wrapper 复用、Draft LM head W8A16 和
`enable_reduce_sample=true` 使用。

### 4. 可选深度 profiler

位置：`src/vllm_hust_vspec/backends/draft.py`

设置 `VSPEC_DRAFT_PROFILE_PATH=/tmp/<name>.json` 后，记录 proposal、Target/Draft
graph update、graph replay、sample、metadata update 和 bookkeeping 等主机阶段。
关闭时不安装 profiler monkey patch。

Gamma=2 fast path 之前的代表性均值：

| 阶段 | 均值 |
| --- | ---: |
| Draft proposal | 15.681 ms |
| Target graph update | 11.821 ms |
| Draft graph update | 7.593 ms |
| Target sampling | 3.966 ms |
| Target graph replay call | 2.211 ms |
| Draft set inputs | 1.795 ms |
| Draft attention metadata update | 1.772 ms |
| Draft metadata build | 0.196 ms |
| Draft sampling | 0.227 ms |
| Bookkeeping | 0.111 ms |

原始 profile：`/tmp/vspec_draft_direct_rep_deep_profile_v3.json`。
该数据早于 gamma=2 fast path，下一轮第一件事应重新 profile，而不是直接按旧占比优化。

## 本轮已否决并回滚的方向

| 实验 | 结果 | 结论 |
| --- | ---: | --- |
| 延后 Target graph update，并设 Target workers=3 | N20 497.14 tok/s | 低于 N20 参考 520.01，否决 |
| 第一版通用 skip-unused-hidden-copy | N20 489.69 tok/s | 负收益，代码已删除 |
| Gamma=2 第一个 token 直接复用 `self.input_ids` | N50 545.06 tok/s | buffer alias 改变输出，必须 clone |
| repetition argmax partials 从 16 改为 8 | warm N50 568.62 tok/s | 低于默认值，恢复 16 |
| 自定义 dense Draft metadata builder | N20 476.32 tok/s | 负收益，代码已删除 |
| Draft graph-update workers=5/6 严格筛选 | N50 最高约 580.85 / 570.61 tok/s | workers=5 仅配合 margin 性能配置保留；严格配置继续用 4 |
| 复用 graph gate Event | N200 568.6569 tok/s | N50 偶有高值但 N200 回退，代码已回滚 |
| 延后 Draft graph update | N50 无稳定提升 | 代码已回滚 |
| repetition partials=32 | N50 约 368 tok/s | 严重回退，恢复 16 |
| bit-packed repetition seen mask | N50 292.97 tok/s | 严重回退，代码已回滚 |
| repetition vocab tile=4096 | N50 432.07 tok/s | 回退，恢复 2048 |
| compact continuation graph 与 gamma2 fast path 叠加 | N50 546.56-558.08 tok/s | 变慢且输出总量改变，代码已回滚 |
| margin=5.25 并保护 EOS 151645 | N50 638.30 tok/s，前 50 题 46% | 未改善吞吐且质量下降，不保留保护项 |

Dense metadata 初版还把 `actual_seq_lengths_q` 错写为全 1，触发 FIA `561002`
shape 错误；改成累计 offset 后虽正确运行，但仍变慢。不要原样重做。

相关失败/对照结果均保留在：

```text
benchmark_results/draft_b16_recovery_20260914/
benchmark_results/draft_b16_resume_20260922/
```

重点目录：

```text
online_direct_rep_targetp3_deferred_g2_n20/
online_skip_hidden_g2_n20/
online_gamma2_fastpath_v4_n50/
online_gamma2_fastpath_partial8_n50_repeat/
online_gamma2_fast_metadata_n20/
online_gamma2_fast_metadata_v2_n20/
```

## 当前复现命令

启动服务：

```bash
source /opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH=/root/data/vllm-hust-vSpec-github/src:/root/data/vllm-hust-latest:/root/data/vllm-ascend-hust-latest:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VSPEC_DRAFT_FUSED_REPETITION=1
export VSPEC_DRAFT_GAMMA2_FAST_PATH=1
export VSPEC_DRAFT_GAMMA2_UNIFIED_COMPACT_SECOND=1
export VSPEC_DRAFT_PIPELINED_TARGET_GRAPH_UPDATES=1
export VSPEC_DRAFT_TARGET_UPDATE_PREFIX=6
unset TORCHDYNAMO_DISABLE
unset VSPEC_DRAFT_PROFILE_PATH
unset VSPEC_DRAFT_FAST_DENSE_METADATA
unset VSPEC_DRAFT_TARGET_TENSOR_FIA
unset VSPEC_DRAFT_TENSOR_FIA
unset VSPEC_DRAFT_TARGET_PARALLEL_GRAPH_UPDATES
unset VSPEC_DRAFT_DEFER_TARGET_GRAPH_UPDATES
unset VSPEC_DRAFT_DEFER_GRAPH_UPDATES
unset HUST_VSPEC_CONFIDENCE_ACCEPT_MARGIN
unset VSPEC_CONFIDENCE_ACCEPT_MARGIN
unset VSPEC_EAGLE_CONFIDENCE_ACCEPT_MARGIN

/opt/vllm-hust-cann91/bin/vllm-hust-vspec \
  --method draft \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen2.5-14b-draft-vspec-fp16 \
  --gamma 2 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --max-model-len 32768 \
  --dtype float16 \
  --block-size 128 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 18183 \
  --graph-mode full-decode-only \
  --generation-config auto \
  --async-scheduling \
  --no-prefix-caching \
  --chunked-prefill \
  --adaptive-speculation \
  --adaptive-policy online \
  --adaptive-min-gamma 1 \
  --adaptive-control-interval 4 \
  --adaptive-online-exploration 0.01 \
  --adaptive-online-warmup-samples 1 \
  --adaptive-refill-batch 8 \
  --capture-sizes 1 2 3 4 6 8 12 16 24 48 \
  --draft-lm-head-quantization w8a16 \
  --draft-parallel-graph-updates 4 \
  --draft-target-parallel-graph-updates 2 \
  --graph-event-ordering \
  -- \
  --additional-config '{"enable_reduce_sample":true}'
```

这是当前通过两轮 N200 的严格动态配置。`--gamma 2` 是在线控制器的搜索上界，
实际在 gamma 1 和 2 之间选择；它不是固定 gamma=2。服务命令不设置采样温度，
必须由下面的 benchmark 请求显式传入 `--temperature 0`。

N50 筛选压测（确认稳定提升后把 `50` 改为 `200`）：

```bash
/root/.venvs/vllm-hust-latest/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18183 \
  --endpoint /v1/chat/completions \
  --model qwen2.5-14b-draft-vspec-fp16 \
  --tokenizer /data/shared-models/Qwen2.5-14B-Instruct \
  --dataset-name custom \
  --dataset-path /root/data/vllm-ascend-hust/benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl \
  --disable-shuffle \
  --custom-output-len 256 \
  --num-prompts 50 \
  --request-rate inf \
  --temperature 0 \
  --seed 0 \
  --save-result \
  --result-dir /root/data/vllm-hust-vSpec-github/benchmark_results/draft_b16_recovery_20260914/resume_n50 \
  --result-filename GSM8K-n50.json
```

## 下一轮优化顺序

1. 以当前严格 adaptive 1..2 配置采集新 profile，确认 Target/Draft graph update 和
   Draft proposal 的剩余占比，不再把 confidence margin 纳入正式优化路径。
2. 在保持严格 top-1 验收的前提下扩展并验证 gamma 1..4；必须先解决高 gamma 的图更新
   和 bucket 状态迁移开销，不能用降低准确率换吞吐。
3. 在不替换整个 metadata builder 的前提下，单独分析
   `attn_update_stack_num_spec_norm`：优先验证 positions clone/update 是否可省，以及累计
   query offsets 是否可缓存；必须保持 FIA metadata ABI。
4. 继续减少 Target/Draft graph task rebinding 的结构性主机开销。workers=6、Event 复用、
   延后 update 和 compact continuation 已失败，不要重复相同组合。
5. Gamma=3 只有在解决 graph update 随 speculative step 增长的问题后再测试；旧 direct
   gamma=3 仅 `430.54 tok/s`，当前不值得优先投入。
6. 每个候选先检查失败请求和输出 token，再看吞吐。压测必须显式使用
   `temperature=0` 且保持 margin 关闭；N50 需多次稳定接近或超过
   `611.27 tok/s` 才运行 N200；采样/状态修改必须增加逐请求输出一致性检查。
7. 达到性能门槛后，再用同一 async/sync 设置重测 Target-only 和 EAGLE，形成严格公平的
   最终报告。

## 代码和验证状态

- 本检查点已删除负收益的 `VSPEC_DRAFT_FAST_DENSE_METADATA` 实现。
- 当前没有 vLLM 服务或 benchmark 进程运行，NPU 未被占用。
- 代码和 benchmark 结果仍是 intentionally uncommitted，尚未 commit、push 或发布。
- 最近一次相关定向测试：169 passed，14 warnings。
- 更早的 full pytest：202 passed、3 failed；3 个失败均由当前环境安装的 extension
  discovery 中存在重复 Bundle v1 manifest 引起，不是 Draft 算法失败。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
