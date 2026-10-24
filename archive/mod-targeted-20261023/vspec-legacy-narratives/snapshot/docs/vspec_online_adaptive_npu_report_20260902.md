# vSpec Online Adaptive NPU 报告（2026-09-02）

## 结论

旧的 A-B-A/下一轮 entropy cap 已被替换为两层在线控制：

1. Online-UCB 在活动 batch bucket 内选择 batch-uniform 最大 Draft 预算。
2. 当前轮置信停止在该预算内逐请求截断低置信 token 后缀。

Qwen2.5 Draft 的纯 UCB 最佳完整采样结果为 `29.723s`，相对固定
`gamma=3` 的 `30.962s` 加速 `1.0417x`。启用当前轮置信停止并只在
`gamma>=3` 的图中计算置信度后为 `30.464s`，仍有 `1.0164x` 正向收益。
固定 `gamma=2` 的 oracle 结果为 `29.011s`，说明在线探索仍有约 5% 的有限请求
摊销成本。

EAGLE 在线组为 `25.556s`，同轮固定 `gamma=2` 为 `25.142s`，回退 1.65%。
因此当前默认建议是 Draft 使用 Online-UCB，EAGLE B128 使用固定 gamma2；不能声称
在线策略已经稳定超过 EAGLE。

## 测试配置

| 项目 | 配置 |
|---|---|
| NPU | Ascend，单卡单进程 |
| 数据集 | GSM8K，相同 manifest 前 200 条 |
| Batch | 128 |
| 最大输出 | 512 tokens |
| Target | Qwen2.5-14B-Instruct，BF16 |
| Draft | Qwen2.5-0.5B-Instruct |
| EAGLE | Eagle-Qwen2.5-14B-Instruct |
| 执行模式 | FULL Graph + async scheduling |
| Prefix cache | 开启 |
| `max_model_len` | 1024 |
| `max_num_batched_tokens` | Draft 33280；EAGLE 8192 |
| 采样 | greedy，seed 0 |

`elapsed_s` 不包含模型加载、编译和 Graph capture。比较组使用相同 prompt、Target、
batch、token budget 和采样参数；耗时是主门槛，正确率用于排查明显质量异常。

## 固定 Draft 扫描

| gamma | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | GSM8K |
|---:|---:|---:|---:|---:|
| 1 | 31.076 | 60,204 | 1,937.31 | 187/200 (93.5%) |
| 2 | **29.011** | 59,999 | **2,068.14** | 185/200 (92.5%) |
| 3 | 30.962 | 60,361 | 1,949.52 | 186/200 (93.0%) |
| 4 | 32.412 | 60,362 | 1,862.35 | 188/200 (94.0%) |

该扫描只用于验证在线控制是否识别正确臂，不作为 UCB 的运行时输入。

## 在线 Draft 结果

| 方式 | 范围 | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | GSM8K | 相对 gamma3 |
|---|---:|---:|---:|---:|---:|---:|
| 固定 Draft | 3 | 30.962 | 60,361 | 1,949.52 | 93.0% | 1.0000x |
| Online-UCB | 1..4 | **29.723** | 60,010 | **2,018.98** | 93.0% | **1.0417x** |
| Online-UCB + 当前轮置信停止 | 1..4 | 30.464 | 59,759 | 1,961.63 | 93.5% | 1.0164x |

纯 UCB 结果采齐 B128 的 1/2/3/4 四个臂并最终选择 gamma2。组合结果对 gamma1/2
复用原始图，只在 gamma3/4 图内计算 top-2 entropy；相比未 gating 的组合实现
`36.001s`，图级 gating 消除了大部分置信计算回退。

组合结果比固定 gamma3 快 1.64%，正确率高 0.5 个百分点，但仍比提前知道最佳臂的
固定 gamma2 慢 5.0%。这是有限 200 请求内探索成本，不能表述为超过 oracle。

## EAGLE 结果

| 方式 | 范围 | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | GSM8K | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| 固定 EAGLE | 2 | **25.142** | 60,171 | **2,393.24** | 93.0% | 1.0000x |
| Online-UCB | 1..2 | 25.556 | 59,959 | 2,346.16 | 93.0% | 0.9838x |

两臂探索成本没有在 N200 内摊平。EAGLE 默认不应开启在线动态 gamma，除非连续服务
时间足够长，或后续加入可跨请求持久化的 arm 状态。

## 实现要点

1. Reward 使用 scheduler 消费输出时的非重叠完成间隔：
   `committed_tokens / completion_interval`。异步重叠的 model execution 时间不再累加。
2. 每个 gamma 先按 `max(control_interval, warmup_samples)` 调度一个窗口；async 额外
   填充一步，确保切换帧之后至少有稳定反馈。
3. 初始探索按相邻档位逐级下降，`4 -> 3 -> 2 -> 1`，每次最多改变一级。
4. 反馈尚未返回时在区间中点等待，不再让异步队列无限堆积最低 gamma。
5. Reward 滑动窗口长度与累计观测数分离；累计数超过 32 后仍可继续控制。
6. 新 batch bucket 继承相邻 bucket 的在线先验，避免尾段重新完整探索四个臂。
7. 当前轮置信停止使用逐请求 token entropy；命中阈值后保留该 token，并把同轮后缀
   标记为 `-1`，同步 scheduler 可见长度和 rejection sampler 输入。
8. FULL Graph 不能撤销已经捕获的 Draft forward。当前实现只对 gamma3/4 捕获置信
   计算，gamma1/2 没有额外 logits 操作；该功能默认关闭。

## 启动

```bash
./run.sh draft \
  --gamma 4 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 33280 \
  --max-model-len 1024 \
  --prefix-caching \
  --graph-mode full \
  --adaptive-speculation \
  --adaptive-policy online \
  --adaptive-min-gamma 1 \
  --adaptive-control-interval 4 \
  --adaptive-online-warmup-samples 1 \
  --adaptive-online-exploration 0.01 \
  --adaptive-hysteresis 0.05 \
  --adaptive-full-graph \
  --adaptive-async
```

加上当前轮置信停止：

```bash
--adaptive-entropy-stop \
--adaptive-entropy-topk 2 \
--adaptive-entropy-threshold 0.3 \
--adaptive-entropy-scale 0.15
```

`--gamma` 是静态 buffer、Graph capture 和 UCB 搜索上界，不是固定运行长度。
Online 模式不读取 profile。正式吞吐测试关闭 `--adaptive-trace`。

## 结果文件

- `benchmark_results/draft_fixed_g3_contemporary_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/draft_fixed_g{1,2,4}_contemporary2_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/draft_online_ucb_valid_c4_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/draft_online_ucb_confidence_highgamma_final_c4_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/eagle_online_ucb_final_c2_full_b128_gsm8k_n200_max512.json`

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
