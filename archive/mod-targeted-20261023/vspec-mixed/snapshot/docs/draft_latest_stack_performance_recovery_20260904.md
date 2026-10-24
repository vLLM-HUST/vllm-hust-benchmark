# Draft 最新栈性能恢复报告

测试时间：2026-09-04 UTC<br>
插件版本：0.12.1

## 结论

Draft Online Adaptive 已恢复并超过原有高点。安装 wheel 后仅通过公开 CLI 参数
执行的最终复测为 `2357.96 output tok/s`、`25.7774s`，相对原高点吞吐提高
`1.99%`、耗时降低 `0.67%`。相对当前 Target-only Graph 为 `1.4989x`。

性能预设连续三次结果为 `2371.97`、`2367.97`、`2357.96 tok/s`，三次输出
token 数、逐条输出哈希和 GSM8K 结果完全一致。

## 回退原因

1. 新测试遗漏了原实验的 waiting refill group `8`，运行时退回 `4`。
2. 自适应 Draft 的 `capture_policy=auto` 只生成稀疏图桶，动态 gamma 需要的实际
   request/verification 尺寸没有全部精确捕获。原配置使用 55 个精确桶。
3. 修复前两项后，最新 vLLM-HUST/Ascend 栈的严格 Draft 路径仍低于旧栈；使用
   显式 logit margin 恢复剩余差距。这是性能/质量折中，不是严格等价优化。

插件现在将 Draft 自适应 refill 默认值设为 `8`，并让动态 gamma 的 `auto` 策略
生成完整精确图桶。`margin=5.25` 保持为显式参数，默认严格模式不会自动启用。

## 测试结果

| 配置 | 耗时 (s) | 输出 token | 输出吞吐 (tok/s) | GSM8K |
|---|---:|---:|---:|---:|
| 原历史高点 | 25.9509 | 59997 | 2311.94 | 185/200 (92.5%) |
| 最新栈回退现场 | 35.0335 | 60107 | 1715.70 | 187/200 (93.5%) |
| refill=8 + 55 精确桶，严格验收 | 27.3246 | 60688 | 2221.00 | 188/200 (94.0%) |
| margin=5.25，第 1 次 | 25.6251 | 60782 | 2371.97 | 182/200 (91.0%) |
| margin=5.25，第 2 次 | 25.6684 | 60782 | 2367.97 | 182/200 (91.0%) |
| margin=5.25，安装版正式 CLI | 25.7774 | 60782 | 2357.96 | 182/200 (91.0%) |
| 当前 Target-only Graph | 38.1468 | 60010 | 1573.13 | 188/200 (94.0%) |

最终结果相对原高点准确率下降 `1.5pp`，在既定“耗时为主门槛、允许小幅准确率
波动”的口径内。若要求严格 greedy 语义，应移除 `confidence_accept_margin`，此时
当前实测为 `2221.00 tok/s`。

## 固化内容

- `--adaptive-refill-batch 8`：公开控制 waiting request 的成组补入。
- `--confidence-accept-margin 5.25`：公开控制 Draft/Target logit margin 验收。
- 自适应 Draft 的 `capture_policy=auto` 现在等价于动态宽度精确图桶；B128、
  gamma=4 共生成 55 个尺寸，最大为 640。
- 新增 `configs/qwen25-14b-05b-adaptive-b128.toml` 和
  `./run.sh draft-adaptive` 预设。
- 未传 margin 时会显式清除父进程遗留值，避免严格测试被隐藏变量污染。

## 测试口径

- GSM8K 固定 manifest，200 prompts，Batch=128，`max_tokens=512`。
- Qwen2.5-14B-Instruct + Qwen2.5-0.5B-Instruct，BF16，TP1 + Draft TP1。
- `max_model_len=1024`，`max_num_batched_tokens=33280`，greedy，seed=0。
- FULL Graph、`enforce_eager=false`、async scheduling、prefix caching。
- Online gamma 1～4，control interval=2，每次最多改变一级。
- vLLM-HUST `762f85b311fbab0bcf8921dd216f5093cd58b9b8`。
- vLLM-Ascend-HUST `4e57439e58ed3d78e675f9fd7b4614fb183c5394`。

正式结果文件：

`benchmark_results/latest_stack_draft_online_formal_cli_refill8_exact_margin5p25_g1to4_full_b128_gsm8k_n200_max512.json`

同名 `.log` 保留完整配置、编译、55 个 Graph capture 和控制器汇总。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
