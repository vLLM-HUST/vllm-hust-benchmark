# EAGLE 动态 gamma 0～4 修复与实测

## 测试口径

- GSM8K，200 prompts，batch=128，max_tokens=512
- Qwen2.5-14B-Instruct + Eagle-Qwen2.5-14B-Instruct，TP=1
- FULL Graph，async scheduling，prefix caching
- `max_num_batched_tokens=8192`，refill threshold=4
- Online-UCB，候选 `gamma=0..4`，每次最多改变一级
- 最终性能轮启用 confidence margin=4

## 最终结果

| 配置 | Elapsed | Output token/s | GSM8K | 相对固定 g2 | gamma 选择次数 |
|---|---:|---:|---:|---:|---|
| 固定 g2 FULL | 25.1421 s | 2,393.24 | 0.930 | 1.0000x | 固定 g2 |
| 旧动态 1～2 | 22.8410 s | 2,674.80 | 0.920 | 1.1007x | g1=8, g2=326 |
| **修复后动态 0～4** | **22.7681 s** | **2,605.53** | **0.935** | **1.1043x** | g2=333 |

修复后动态 0～4 的端到端耗时比固定 g2 降低 9.44%，比旧动态 1～2 的
22.8410 s 门槛再降低 0.32%，GSM8K 从 0.920 提高到 0.935。吞吐不能直接代替
耗时比较，因为各轮生成的输出 token 总数不同；本项目仍以同口径端到端 elapsed
作为性能门槛。

## 修复内容

1. 修复 EAGLE 降档状态连续性。正 gamma 从宽档切到窄档时，过渡帧仍按旧宽度执行
   proposal，再把返回候选截断到新宽度，避免 `4 -> 3` 时 Draft KV 和 hidden
   state 与异步 optimistic state 失步。
2. Target-only 判断改为读取当前 Target query width，不再把 async 下一帧的
   proposal gamma 当成本帧状态。
3. 动态 EAGLE 与上游固定宽度 uniform-state 快路径隔离。Graph 初始化完成后，
   Target 状态宽度锚定在候选中点；只有当前 query width、运行时 gamma 和状态宽度
   一致的稳定帧才临时启用原生快路径。
4. 新 batch bucket 继承相邻 bucket 的在线先验和已训练 gamma，避免 batch 尾段
   重新执行完整冷启动扫描。
5. UCB hysteresis 改为升档、降档对称；近似同收益时优先选择靠近候选中点的档位，
   避免无收益地扩宽 proposal。
6. 对包含 0 且跨度至少为 4 的宽候选范围增加安全 burn-in：先在中点累计
   `8 * online_window` 个稳定观测，再按 UCB 从相邻档位逐级探索。
7. EAGLE confidence margin 只在当前 `gamma <= 2` 时生效；高 gamma 使用严格
   verifier，避免宽 proposal 放大置信接受误差。

## 全候选验证

最终 N=200 性能轮在 B128 bucket 得到 172 个稳定观测，尚未达到 256 个安全
burn-in 观测，因此本轮保持 gamma=2。这里的“0～4”表示控制器候选范围，不表示
这次短任务已经采样了所有 arm；长驻服务达到阈值后才会按 UCB 逐级探测。

修复过程另外保留了两组实际访问全部候选的运行：

| 验证配置 | Elapsed | GSM8K | gamma 选择次数 | 结论 |
|---|---:|---:|---|---|
| strict acceptance | 25.7250 s | 0.940 | g0=3, g1=194, g2=173, g3=32, g4=4 | 0～4 切换正确，质量高于固定 g2 |
| margin=4，中点控制 | 23.5562 s | 0.915 | g0=3, g1=5, g2=321, g3=3, g4=4 | 性能接近门槛，质量少 1 题 |

这两组结果证明修复后的 `0/1/2/3/4` 路径可以真实执行；最终安全 burn-in
配置则解决短任务为探索付出过高启动成本的问题。

## 原始文件

- `benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/eagle_online_g0to4_all_arms_strict_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/eagle_online_g0to4_all_arms_margin4_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/eagle_online_g0to4_safe_burnin_full_b128_gsm8k_n200_max512.json`
- 每个 JSON 都有同名 `.log`，包含完整 Online-UCB summary。

## 验证

- `python -m pytest -q`：123 passed
- `ruff` 未安装，因此本轮未执行 lint

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
