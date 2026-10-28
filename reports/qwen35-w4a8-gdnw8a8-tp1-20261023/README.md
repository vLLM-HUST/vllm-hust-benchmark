# Qwen3.5-35B-A3B W4A8 (专家 int4 + GDN W8A8) · 单卡 SWE prefix reuse 并发曲线 (C1 / C2 / C4 / C8 / C16)

本报告由实测产物直接生成, 数值零推断。每一档是一个独立的 900s 固定窗口观测, 五档在**同一个服务实例**内按 C1→C16 连续完成 (档间不重启服务、不重置 prefix cache),
因此档间唯一变量就是客户端并发 C。

**本页对应 series `swe-w4a8-gdnw8a8-tp1-20261023-mtp2` (TP1, 1×910B3)**。prepared
workload、客户端、协议、硬件型号与软件栈与 W4A8 条目 (GDN 保持 bf16) 一致 —— 两边跑同一份 prepared workload (sha256 相同), 站点上各持自己的
workload.id; 差别只有精度配方 (本产物额外把 90 个 GDN 模块量化成 W8A8) 与随之重新推导的 KV 预算。

## 1. 裁定口径

| 项           | 取值                                                                                                                                                                                                                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 硬件         | Ascend 910B3 × 1 (physical devices [1])                                                                                                                                                                                                                                                         |
| 引擎         | vLLM + vLLM-Ascend `0.28.1.post1.dev143+gf18cf803c / 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`                                                                                                                                                                                               |
| 模型         | Qwen3.5-35B-A3B @ `712cf74392b05026a6db2bf213d343747d1f6d45`                                                                                                                                                                                                                                    |
| 精度         | W4A8 专家层 int4 + GDN linear_attn W8A8 weights · int8 activations · BF16 KV                                                                                                                                                                                                                    |
| 权重         | int4 (MLP 专家层 gate/up/down_proj, ssz per-channel) + int8 (30 层 GDN linear_attn 的 in_proj_qkv / in_proj_z / out_proj 与 10 层 self_attn 的 q/k/v), msModelSlim ascendv1 离线量化, 6 个 quant_model_weights 分片                                                                             |
| 激活         | int8 (动态 per-token, minmax); KV auto (bf16; KV 未量化)                                                                                                                                                                                                                                        |
| 并行         | TP1 · PP1 · DP1 · EP=False                                                                                                                                                                                                                                                                      |
| 并发座位     | max-num-seqs 16                                                                                                                                                                                                                                                                                 |
| KV 预算      | `35956596497` (33.49 GiB, 显式) — 服务端实测 1,430,557 token, 262144 上下文下最大并发 5.46x                                                                                                                                                                                                     |
| 投机解码     | real MTP, num_speculative_tokens=2 (真实接受/拒绝, 无 synthetic 覆盖); 服务端日志实测 Mean acceptance length 2.887–2.919 (每档 n=90 采样)                                                                                                                                                       |
| MOD          | ascend-mtp-contract-2patch, ascend-npugraph-ex-attr-shim                                                                                                                                                                                                                                        |
| Graph        | FULL_DECODE_ONLY (请求 FULL, 平台因 AscendGDNAttentionBackend 仅支持 UNIFORM_BATCH 而降级); max_cudagraph_capture_size=256                                                                                                                                                                      |
| runtime 提交 | vllm `f18cf803c5f6` / vllm-ascend `74f0c0a27237`                                                                                                                                                                                                                                                |
| runtime 版本 | vllm 0.28.1.post1.dev143+gf18cf803c; vllm-ascend 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27; torch 2.13.0+cpu; torch_npu 2.13.0.rc1; Ascend CANN 9.1.0 (逐进程临时环境: /data/jxd/Ascend-9.1.0/{ascend-toolkit,nnal/atb}; 系统自带 /usr/local/Ascend CANN 8.5.0 未改动、未覆盖), ATB CXX ABI=1 |
| 客户端       | swe-prefix-reuse `6861242dbd9f17b707003191e4200b7752911d7c` (0.1.2)                                                                                                                                                                                                                             |
| workload     | `https://github.com/vLLM-HUST/swe-prefix-reuse` prepared/qwen35.json sha256 `ac1fbf9885076c369fd25c5525d1163ff8b7945972cdf4c0c6d2d523ab7c1172`                                                                                                                                                  |
| 窗口         | 每档 60s 协议/缓存资格 + 900s 正式窗口; 总吞吐只计窗口内收到的实际输出 token                                                                                                                                                                                                                    |
| 坐标轴       | x = P90 across fully completed in-window requests of (N-1)/(last-token-time-first-token-time); grouped SSE token times, not inverse P90 TPOT; y = In-window generated tokens / 900 seconds / every allocated chip                                                                               |

## 2. 结果

| C   | run_id         | output tps | tps/卡   | P90 decode tps | TTFT p95 (s) | 窗口内完成 / 启动 | 在途均值 | 满载占比 | 最大 prompt | 边际前缀命中率 | drain (s) |
| --- | -------------- | ---------- | -------- | -------------- | ------------ | ----------------- | -------- | -------- | ----------- | -------------- | --------- |
| C1  | `fe7ffa29a479` | 90.1878    | 90.1878  | 101.2736       | 0.752        | 147 / 148         | 0.9994   | 0.9994   | 52505       | 81.24%         | 14.22     |
| C2  | `a2f70b9e0d33` | 142.9478   | 142.9478 | 87.3835        | 1.179        | 196 / 198         | 1.9991   | 0.9991   | 64999       | 83.91%         | 11.10     |
| C4  | `317b21ccc5ef` | 203.7911   | 203.7911 | 67.8348        | 1.533        | 290 / 294         | 3.9985   | 0.9985   | 67634       | 84.90%         | 10.86     |
| C8  | `ea8b195e292b` | 312.7022   | 312.7022 | 57.0162        | 1.987        | 460 / 468         | 7.9974   | 0.9975   | 80630       | 86.09%         | 50.12     |
| C16 | `675742aea20c` | 404.8133   | 404.8133 | 36.2642        | 2.636        | 608 / 624         | 15.9974  | 0.9975   | 59957       | 82.31%         | 73.09     |

单卡时 `tps/卡` 等于总吞吐 (只有一张卡)。五档单调上升, **C16 是本配置的峰值** (max-num-seqs 16 = C16 供给负载, 即饱和座位数), TTFT p95 由
0.752s 升到 2.636s。

## 3. 每档工作量与资格窗口

| C   | 完成会话 | 覆盖轨迹 | 最大 turn | 60s 资格窗口   | 资格失败请求 | 正式窗口失败请求 | 正式窗口有效 |
| --- | -------- | -------- | --------- | -------------- | ------------ | ---------------- | ------------ |
| C1  | 5        | 6 / 8    | 38        | `a9e82b609489` | 0            | 0                | 是           |
| C2  | 5        | 7 / 8    | 43        | `a370da694594` | 0            | 0                | 是           |
| C4  | 7        | 8 / 8    | 45        | `f5d8bc69bf10` | 0            | 0                | 是           |
| C8  | 9        | 8 / 8    | 63        | `73ed07d318d9` | 0            | 0                | 是           |
| C16 | 11       | 8 / 8    | 41        | `35d60e0930db` | 0            | 0                | 是           |

## 4. 曲线与站点数据

cohort `qwen35-35b-a3b-w4a8-gdnw8a8-tp1-1x910b3-full-decode-only-sweprefix-900s-v1`。cohort / point /
run 的权威数据在本仓库的 `website-publication/data/leaderboard_frontier.json` 与
`website-publication/data/leaderboard_frontier_swe_evidence.json` (packed 源), 网站 `data/` 由 CI 按固定
revision 投影生成; 站点侧的渲染器与 SVG 在 vLLM-HUST/vllm-hust-website 的
`scripts/render_swe_w4a8_gdnw8a8_tp1_curves.py` 与
`assets/frontier-qwen35-w4a8-gdnw8a8-tp1-swe-concurrency.svg`。

## 5. 量化范围与精度代价

| 组                                                                                                     | 层数 × 投影      | 权重/激活   |
| ------------------------------------------------------------------------------------------------------ | ---------------- | ----------- |
| MLP experts (gate/up/down)                                                                             | 40 × 256 experts | int4 / int8 |
| GDN `linear_attn` in_proj_qkv / in_proj_z / out_proj                                                   | 30 × 3           | int8 / int8 |
| `self_attn` q/k/v                                                                                      | 10 × 3           | int8 / int8 |
| 未量化: `self_attn.o_proj`, `in_proj_ba` (a/b 融合, N=16), `shared_expert`, `lm_head`, `visual`, `mtp` | —                | FLOAT       |

产物 quant_model_description 实测: W4A8_DYNAMIC 122880 / W8A8_DYNAMIC 361 / FLOAT 1611, 无 W8A16。权重 22.38
GiB (24,096,662,520 bytes)。

同一 wikitext-2 协议 (298 个 999-token chunk, 并发 8, 同 session 内配 bf16 基线, 逐 chunk 配对 ΔNLL) 下的 PPL:

| checkpoint | 量化范围                                 | PPL        | vs bf16 |
| ---------- | ---------------------------------------- | ---------- | ------- |
| bf16       | –                                        | 7.1694     | –       |
| W8A8       | experts W8A8 + attn q/k/v W8A8           | 7.1750     | +0.08%  |
| 专家 int4  | experts W4A8 + attn q/k/v W8A8, GDN bf16 | 7.8551     | +9.6%   |
| **本产物** | 上者 + GDN 三投影 W8A8                   | **7.9070** | +10.3%  |

即: GDN 的 int8 权重在 PPL 上多付 +0.66%, 换来 90 个模块的权重从 bf16 降到 int8 以及随之可用的 KV 预算。本页只报告该精度-容量-吞吐的实际工作点,
**不构成 int4 或 GDN W8A8 更快的结论**。

## 6. 为什么 C16 是本曲线的峰值

- **KV 容量**: 显式 pin 35956596497 (33.49 GiB) 在服务端记为 `GPU KV cache size: 1,430,557 tokens`, 即 262144
  上下文的 5.46 路等价通道。本 workload 的实测 prompt p50 17865、最大 80630 token, 16 路在途的 KV 占用远低于该预算, 因此 C16 不需要排队等
  KV。
- **前缀复用未被冲掉**: 边际命中率 (窗口内 Δhits/Δqueries) 81.24%/83.91%/84.90%/86.09%/82.31% (C1→C16); C16 仍为
  82.31%, 与 C8 的 86.09% 同一量级。作为交叉校验, 五个 900s 窗口的 Δqueries 合计占末行累计查询的 96.3%, 余下即五个 60s 资格窗的流量。
- **仍然满载**: C16 的在途均值 15.9974 (满载占比 0.9975), 无失败请求, 窗口内完成 608 个请求。
- **对照**: 同 workload 的 W8A8 单卡条目只有 3.07 路并发, C16 时命中率掉到 34.6%, 吞吐在 C8 见顶后回落 32.5%。本曲线没有该拐点, 因为 KV
  预算与权重占用是镜像配置的 (见 caliber_note)。

## 7. 与同 workload 其他 cohort 的对照 (仅作定位, 不作加速声明)

| cohort              | 权重                        | KV pin    | C1    | C2     | C4     | C8     | C16    |
| ------------------- | --------------------------- | --------- | ----- | ------ | ------ | ------ | ------ |
| 已发布 W8A8 TP1     | W8A8 (experts + attn q/k/v) | 18.85 GiB | 88.03 | 148.91 | 198.66 | 304.70 | 205.58 |
| W4A8 (GDN bf16) TP1 | 专家 int4, GDN bf16         | 32.55 GiB | 87.40 | 150.87 | 202.14 | 313.37 | 394.34 |
| **本 cohort**       | 专家 int4 + GDN W8A8        | 33.49 GiB | 90.19 | 142.95 | 203.79 | 312.70 | 404.81 |

读法: C1 上三条曲线相差在 3% 量级, 与该 workload 单次 900s 观测的波动同阶, 不作归因。C2 上本产物比另外两条低 5.3% / 4.0% —— C1/C2 的 KV
占用都远低于预算 (C2 时仅 ~2 路在途), 所以这一档差异来自权重配方 (GDN 三投影的 int8 权重), 与 KV 无关。C16 的差异与 KV 预算同时变化, 不能归给某一项; 本
cohort 只声明自己的五点在同一实例内可比。

## 8. 口径限制 (必须与数值一起引用)

- 客户端与 engine 同机 (127.0.0.1), 客户端调度/JSON/网络开销仍在观测路径内
- 未做预热: 每个 play 使用新 cache_salt 冷启动, 窗口结束排空在途请求
- 本轮为 60s 短测 + 900s 固定窗口两个显式测量, 无自动认证语义
- 首次冷编译: 因关闭 torch AOT autograd 缓存, 每轮启动都要重跑 torch.compile 与 ACL 图捕获, 该时间不计入测量窗口
- CANN 9.1 为逐进程临时环境 (未覆盖系统 8.5.0); 引擎实际加载的 CANN 版本见启动日志的 ASCEND_HOME_PATH
- 样例 workload 仅 8 条轨迹/360 轮, 非统计代表性 SWE 样本; 最长会话 141269 tokens, 未触及 262144 上限
- MTP 特有: 请求的 cudagraph_mode=FULL 被平台降级为 FULL_DECODE_ONLY (AscendGDNAttentionBackend 仅声明
  UNIFORM_BATCH)
- MTP 特有: 每次带 MTP 的运行都会留下一条不可用的 AOT 编译缓存条目, 故每轮实验前需隔离 torch_aot_compile/\*
- MTP 特有: 前缀复用可用, 但受 EAGLE/MTP 尾块丢弃语义影响, 每个请求比无 MTP 时少复用 1 个 block (2048 token)
- 三个本地 engine 改动未进入上游; 第三处 (npugraph_ex 属性 stand-in) 对本产物为惰性: 其生效条件是编译图中存在
  npu.npu_weight_quant_batchmatmul (W8A16) 节点, 而本产物 quant_model_description 中零个 W8A16 条目
- 采集时本臂独占卡 1, 同时卡 0 上另跑一轮 W8A8 单卡实验 (不同芯片/端口/编译缓存)。HBM 不共享, 宿主 192 核 load≈12; 若要做 1-3% 级跨会话对比应复跑核对
- W4A8/GDN 特有: 单卡 KV 预算 33.49 GiB (显式 pin, 与同一 workload 的 W4A8 条目同一条公式) -> 1,430,557 token -> 262144
  上下文下最大并发 5.46x; C16 时边际前缀命中率仍为 82.31% (C8 86.09%), 故吞吐在 C16 继续上升, 没有出现 W8A8 单卡 (3.07x) 的 C16 回落
- W4A8/GDN 特有: GDN 三投影的 int8 权重带来实测精度代价。同一 wikitext-2 协议下 (同 session 配 bf16 基线, 298 chunk, 并发 8)
  PPL: bf16 7.1694 / W8A8 7.1750 / 专家 int4 (GDN bf16) 7.8551 / 本产物 7.9070, 即本产物相对 bf16 +10.3%, 相对同
  experts 配方多付 +0.66%; 本页不因此声称任何速度收益

服务启动命令 (五档同一实例, 来源 server-metadata):

```
ASCEND_RT_VISIBLE_DEVICES=1 TP=1 PORT=8111 MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.95 LABEL=w8a8nr EXTRA_SERVER_ARGS='--kv-cache-memory-bytes 35956596497 --max-num-seqs 16 --max-num-batched-tokens 4096 --speculative-config {"method":"mtp","num_speculative_tokens":2} --compilation-config {"max_cudagraph_capture_size":256,"cudagraph_mode":"FULL"}' ./run_swe_server.sh
```
