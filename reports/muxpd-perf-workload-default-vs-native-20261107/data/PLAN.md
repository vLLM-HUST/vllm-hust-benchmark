# MuxPD 默认模式 vs native：真实文本 workload 验证（#36 后续，2026-11-07 冻结，任何运行之前）

只测默认模式性能；不测 ROW，不设答题质量 / token 一致性门槛，不扩展 TP8 / 其他模型，不重跑 13 个合成负载。

## 1. 从 #36 选出的场景类别（依据 results/perf36-tp{2,4}/{native-s16,mux43-s16}/pass1，分 kind 复核见 tools/bykind36.py）

- A 长文档 prefill 与短对话 decode 混合、中等负载（对应 mixed-r1.0）：Goodput TP2 20.9/21.9% -> 79.1/77.2%（s2/s3），
  TP4 同向；收益来自长 prefill 不再打断 decode（native 违规以 0.5 s 最大间隔为主）。代价：TTFT p50 +30~60%；TP4 s1
  native 本已 98.3%，MuxPD 降到 87.4%（TTFT 越过 1 s）。
- B 同一混合、高负载（对应 mixed-r1.6）：Goodput TP2 7.8/0.5% -> 69.8/64.4%，TP4 5.7/1.4% -> 72.9/62.5%；TPOT p50
  -30~40%，s2 吞吐 +5~6%。代价：ITL p50 约 2 倍。
- C 纯短对话、过载（对应 short-r2）：吞吐 TP2 +5.9/+8.4%，TP4 +9.2/+9.6%，TPOT p50 -8~11%；Goodput 两边都接近 0
  （仅 0 -> 5~8%），属于饱和吞吐场景，不作 Goodput 收益结论。
- 不选：contend（TP2 吞吐 -7%、TPOT +16%，TP4 Goodput 下降，属于代价场景，已在 #36 报告）、burst（收益小、ITL p99 变差）、
  short-r4（与 C 同类且更深饱和）。

## 2. workload 选择（benchmark 仓库 docs/workload-data/DATASET_CHECKLIST.zh-CN.md 及引用文档 / 验证回执）

| 用途 | 来源（固定版本 / SHA256） | 清单状态 | 用法 |
|---|---|---|---|
| 长请求 | LongBench v1 qasper.jsonl（THUDM/LongBench 5e628be，data.zip cb45b11a…，成员 29aa07d2…，200 条）；官方 dataset2prompt / dataset2maxlen（源码 2e00731） | [x] 原 200 题已原生执行（LONGBENCH_V1_QASPER_ORIGINAL） | 官方提示整体作为单条 user 消息，Qwen3.5 聊天模板，enable_thinking=false，官方 128 输出预算 |
| 短请求 | OASST1 validation parquet（fdf72ae，24002597…，4401 行） | [x] 原始会话语料可用（OASST1_ORIGINAL_CONTINUATION） | 完整原始祖先链作为历史，续接下一条 assistant 回复；规则同仓库入口：assistant、非 deleted/synthetic、角色交替、首尾为 prompter |
| 不用 | BurstGPT_3（本机已有，SHA 2299986a… 与回执一致） | 仅元数据 | 只有长度与时间，无正文；其开头 1 小时平均到达仅约 0.02 req/s，远低于 A/B/C 所需负载，不用作到达 |
| 不用 | Dolly | [x] | 输入 p50 55 token、输出参考 p50 44，比 OASST 更短；OASST 多轮历史更贴近对话 decode |
| 不用 | LongBench v2 / Azure 2023 / Mooncake | 容量阻塞或仅元数据 | v2 超 16384 上下文；trace 无正文 |

实测长度（服务模型 tokenizer + 聊天模板，tools/profile_sources.py）：qasper 输入 p10/p50/p90/max = 2798/4853/7628/21843，
200 条中 197 条满足 输入 + 128 <= 16384（max-model-len，两臂一致）；行 15、54、57（18035/20204/21843 token）超容量，排除
且记录，不截断。OASST 合格目标 2704 个，输入 p50 63 / p90 393，参考回复 p50 114 / p90 377 token。

## 3. 派生测试负载（明确为派生，不是原始请求回放）

- 请求正文为原始文本；到达过程与长短比例为派生：Poisson，参数沿用对应合成场景。
  - wl-docchat-r1.0：mixed，1.0 req/s，120 s，长请求占比 0.2（类别 A）
  - wl-docchat-r1.6：mixed，1.6 req/s，120 s，长请求占比 0.2（类别 B）
  - wl-chat-r2：short，2.0 req/s，90 s，仅 OASST（类别 C）
- 输出长度（派生）：每条固定为原参考答案 / 原历史回复的 token 数，夹到 [16, 预算]（qasper 预算 128 为官方值；OASST 预算 2048
  为仓库入口值），ignore_eos 强制输出到该长度，两臂完全相同。参考文本只用于取长度，不进入输入。
- 选样：合格池用 random.Random(37) / Random(38) 打乱一次，三个计划按顺序取互不重叠的片段；不按结果换样本。
- 生成：tools/build_plans.py（运行前校验所有源文件 SHA256），输出 plans/*.json 与 plans/MANIFEST.json（逐条来源行号 /
  ID、长度统计），校验和 plans/SHA256SUMS。

## 4. 服务、负载与指标（与 #36 一致）

- 栈：Qwen3.5-35B-A3B BF16，910B2，Core d0f22d2，Ascend c182a9e，MuxPD main 43fcaf4（干净）。HCCL_DETERMINISTIC 不设置；
  NPU keepalive 在全部臂运行期间停止，结束后恢复。
- 顺序（每次只有一个服务，端口 8300）：TP2 卡 0,1：MuxPD -> native；TP4 卡 0-3：MuxPD -> native。每臂依次回放 3 个计划各一次。
- MuxPD 默认：scripts/serve-qwen35-muxwise.sh --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16
  --compilation-config G16。Native：tools/serve-native.sh --max-num-seqs 16 --max-num-batched-tokens 16384
  --compilation-config G16。G16 = mode 3，FULL_DECODE_ONLY，graphs [1,2,4,8,16]（图模式）。已知差异：MuxPD 不分块 prefill，
  native 分块 prefill 与 async scheduling 为默认值；调度器 / worker 类不同。
- 负载生成：tools/loadgen.py run（未修改；计划 JSON 多出的 src 字段不被读取），流式，temperature 0，ignore_eos。
- 超时：服务健康等待 1800 s；单请求 httpx 超时 1800 s（loadgen 现值）；单计划整体 `timeout 2400`。失败的计划保留为失败；
  服务启动失败只重启一次；不重跑、不换样本、不改参数。
- SLO 与指标：tools/aggregate.py slo() 不变（TTFT <= 1 s，长请求 3 s；TPOT <= 0.1 s；最大 token 间隔 <= 0.5 s）；
  out_tok_s、goodput_frac、TTFT p50/p99、TPOT p50/p99、ITL p50/p99、worst_stall、errors，并按 kind 拆分。
- 结果：results/wl37-tp{2,4}/{mux43,native}/pass1/<plan>.json；单轮，不做显著性结论。
