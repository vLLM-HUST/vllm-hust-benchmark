# W3.1 gelu e2e A/B：预注册判据（跑前写定，2026-09-10）

被测：插件 `51f5818`（fi_gelu triton 融合 + VLLM_HUST_FI_GELU 接线），工作树干净。
对照：ON=`VLLM_HUST_FI_GELU=1`，OFF=不设（default-off）。

## 统一配置（两腿对称）

conda hust、卡 7（已查空闲）、`cd /tmp`、`HF_HUB_OFFLINE=1`、
`VLLM_DISABLE_COMPILE_CACHE=1`、`--gpu-memory-utilization 0.85`、
`--max-model-len 4096`、`--max-num-seqs 64`、
`--compilation-config {"cudagraph_mode":"FULL"}`（**必须**：默认 FULL_AND_PIECEWISE 在
torch 2.13 基线启动即崩，`fusion_pass_compile AssertionError`——宿主既有缺陷，与插件无关，
OFF 腿已复证 plugin inert 时同样崩；两腿对称施加并记录）。

## 工作量

离线 `LLM.generate`：64 条同长 prompt（~256 tok）× `ignore_eos=True` × `max_tokens=128` ×
`temperature=0`，墙时/腿；每腿独立进程。腿序交替：off_a, on_a, off_b, on_b, off_c, on_c
（抗机器慢漂移，cascade rerun §7.3 教训）。输出 token 总量校验 = 64×128=8192/腿。

## 预注册投影与判定

- 投影：图 replay 实测 B=64 省 10.5µs/层 × 48 层 = 0.50ms/步；TPOT ≈112ms/步 →
  **|Δ| ≈ 0.2–0.6%**（decode num_tokens 在 32–128 间变动，仅低档有收益，实际偏下沿）。
- 判定（跑前定死）：Δmedian = (median_off − median_on)/median_off；spread = max−min 同腿内。
  - |Δmedian| < max(spread_off, spread_on) → **噪声带内不可判**（W2b 先例口径），如实记录；
  - 否则按方向出结论，并核对与投影 0.2–0.6% 的一致性。
- 无论结果如何：**禁止调参重跑凑显著**；如需追加轮次，另写预注册。
- 附带记录：每腿启动到 ready 的编译耗时（FULL 图模式 + 禁编译缓存，两腿对称）。
