# 长文本性能（C1 单请求，8K/32K/128K/260K）

状态：**实测完成（2026-11-16）**
制品：Qwen3.5-35B-A3B W8A8 / W4A8+GDN（`Qwen3.5-35B-A3B-W8A8-NoRot`）

## 1. 口径

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B3 ×1（卡 7） |
| 拓扑 | TP1 |
| 模型 | W8A8：`/data/jxd/Qwen3.5-35B-A3B-W8A8`；W4A8+GDN：`/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot` |
| KV | 18.85 GiB（`20242376704` B，两臂相同） |
| 上下文 | `max-model-len 262144` |
| 前缀缓存 | 关 |
| 调度 | `max-num-seqs 1`、`max-num-batched-tokens 4096` |
| 客户端 | `vllm bench serve`，OpenAI backend，random dataset |
| 输入 / 输出 | 8,192 / 32,768 / 131,072 / 260,000；输出 256 token |
| 采样 | temperature 默认、`--ignore-eos`、seed 0 |
| 次数 | 每长度 1 次 warmup + 3 次正式；C1 单请求 |
| 指标 | TTFT、TPOT、ITL、output tok/s、total tok/s、完成/失败 |

## 2. 结果（3 次的中位数）

| 输入 | 臂 | TTFT p50 (ms) | TTFT p90 (ms) | TPOT p50 (ms) | out tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 8K | W8A8 | 699.3 | 699.3 | 16.4 | 52.7 | 1739.4 |
| 8K | W4A8+GDN | **656.9** | 656.9 | 16.5 | 52.5 | 1732.2 |
| 32K | W8A8 | 2728.1 | 2728.1 | 16.7 | 36.6 | 4727.6 |
| 32K | W4A8+GDN | 2744.3 | 2744.3 | 16.9 | 36.3 | 4680.1 |
| 128K | W8A8 | 18093.2 | 18093.2 | 18.0 | 11.3 | 5790.7 |
| 128K | W4A8+GDN | 18237.2 | 18237.2 | 18.4 | 11.2 | 5735.8 |
| 260K | W8A8 | 57060.2 | 57060.2 | 19.5 | 4.1 | 4193.8 |
| 260K | W4A8+GDN | 57624.2 | 57624.2 | 19.8 | 4.1 | 4152.3 |

> 因为每次 `vllm bench serve` 只发 1 个请求，单次请求内部的 p90 与 p50 相同；
> 表内数值是 3 次重复的中位数。

## 3. 对比

| 输入 | TTFT Δ (W4A8 vs W8A8) | TPOT Δ | out tok/s Δ | total tok/s Δ |
|---|---:|---:|---:|---:|
| 8K | −6.1% | +0.6% | −0.4% | −0.4% |
| 32K | +0.6% | +1.2% | −0.8% | −1.0% |
| 128K | +0.8% | +2.2% | −0.9% | −0.9% |
| 260K | +1.0% | +1.5% | 0% | −1.0% |

## 4. 结论

- TTFT 随输入近似线性增长：8K 约 0.66–0.70 s，32K 约 2.7 s，128K 约 18 s，
  260K 约 57 s。
- W4A8 与 W8A8 的 TTFT 基本一致（除 8K 的 −6% 可能是编译/测量噪声外，32K–260K 差 0.6–1.0%）。
- W4A8 的 TPOT 略高（+0.6%～+2.2%），output tok/s 低 0–0.9%，total tok/s 低约 1%。
  **没有看到 W4A8 在长文本上更快**；差异在单次观测的噪声量级内，不能写成加速。
- 本结果只覆盖 **C1 单请求**，不是并发吞吐；也没有采集 HBM。
- 260K 是 260,000-token 输入 + 256 输出，不是 262,144 满上下文。

## 5. 原始记录

- `long-context-perf/run_20261116T075647Z/out/w8a8_L*_r*.json`
- `long-context-perf/run_20261116T075647Z/out/w4a8_L*_r*.json`
- `long-context-perf/run_20261116T075647Z/out/longperf_runs.csv`
- `long-context-perf/run_20261116T075647Z/out/longperf_summary.csv`
