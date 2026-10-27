# TP4 交互负载 Goodput：六窗口配对确认

2026-10-11 的预登记场景中，相比已测原生配置里主指标最好的 **16K token
budget**，raw P 四层切片插件使 SLO 合格请求从 **117/378 增至 368/378**。
六个测量窗口共 720 s，Goodput 为 **0.162500 → 0.511111 requests/s，
即 3.1453×**；固定窗口输出吞吐为 **103.591667 → 104.086111 tokens/s
（+0.4773%）**。两个卡组交换后的六个窗口全部完成且满足比较条件。

收益是更少的**每请求最坏停顿**：插件在全部六窗口的 request worst-gap
P99 更低，但 **pooled token-gap P99 和 TTFT P99 均更差**。不能写成所有
token 延迟改善，也不据此宣称最大容量或普遍吞吐提升。同一卡组的独立
串行确认也得到 **3.0909× Goodput**，但单个 seed 不能穷尽主机干扰。

## 主指标与全部 SLO

主口径要求完整请求的 scheduled TTFT≤1 s，且**整个响应每个可观测单
token 间隔≤100 ms**；响应在测量到达窗口结束后仍须完成全部 SLO 检查。
Goodput=合格请求数/测量窗口总时长，失败请求保留在分母中。

| 每请求阈值 | 原生合格数 / 378 | 插件合格数 / 378 | 原生→插件 Goodput / requests/s | 倍率 |
| --- | ---: | ---: | --- | ---: |
| gap≤100 ms、TTFT≤1 s（主指标） | 117 | 368 | 0.162500 → 0.511111 | 3.1453× |
| gap≤100 ms、TTFT≤2 s | 117 | 369 | 0.162500 → 0.512500 | 3.1538× |
| gap≤200 ms、TTFT≤1 s | 185 | 377 | 0.256944 → 0.523611 | 2.0378× |
| gap≤200 ms、TTFT≤2 s | 185 | 378 | 0.256944 → 0.525000 | 2.0432× |

scheduled TTFT 从计划到达时刻起算，包含客户端派发迟延；派发迟延上限为
50 ms。原始 SSE/chunk 时间是客户端观测，不是设备 token 产生时间。
pooled gap P99 汇集所有单 token 间隔，而 request worst-gap P99 对每条
请求的最大间隔取分位数。少数长停顿可能被 pooled P99 稀释，两者不能替代。
分位数采用线性插值，**不平均各窗口分位数，也不由分位数重建合并分位数**。
固定窗口 tokens/s 只计窗口内收到、最终通过完整传输校验的输出 tokens，
包括落入该窗口的预热请求输出；它受供给负载限制，不是饱和处理容量。

## 确认计划与逐窗口结果

名义 Poisson 到达率固定为 0.5 requests/s，每 seed 预热 60 s、测量
120 s、固定排空 120 s。合成分布为 80% P128→256 tokens、20%
P12000→8；有限样本不强制精确 80/20。temperature=0、ignore_eos=True，
计划及实际 payload 的 seed 为 1001/1002/1003；只有 priming 使用 42。
三个 seed 分别有 71/57/61 条测量请求，实际到达率分别为 0.5917/0.4750/
0.5083 requests/s。每方向 189 条，两方向共 **378 条测量请求/角色**。

第一方向为原生 A（卡 0–3）/插件 B（4–7），交换方向为原生 B/插件 A。
每方向/角色新启动一次，三个 seed 顺序共享该次启动；自然语言 16 条及
priming 3 条只执行一次，加 98 预热和 189 测量，共 306 次生成。
每 seed 双客户端完整排空后，先记录 CPU-after，再完成独立审计，才进入
下一 seed。每角色共两次启动，**不是六次独立冷启动**。

下表每格为**原生 → 插件**；SLO 四格顺序与上表一致，分母按该 seed。

| 方向 / seed（请求数） | 四格 SLO 合格数 | 窗口 tokens/s | Pooled gap P99 / ms | Request worst-gap P99 / ms | TTFT P99 / ms |
| --- | --- | --- | --- | --- | --- |
| 第一 / 1001（71） | 22/22/33/33 → 66/67/70/71 | 113.592 → 114.142 | 22.529 → 60.589 | 462.924 → 154.470 | 629.263 → 930.397 |
| 第一 / 1002（57） | 19/19/30/30 → 57/57/57/57 | 89.608 → 88.458 | 20.658 → 56.746 | 470.159 → 81.918 | 720.487 → 939.163 |
| 第一 / 1003（61） | 17/17/29/29 → 60/60/61/61 | 107.600 → 109.708 | 22.269 → 56.708 | 465.931 → 95.836 | 595.238 → 897.666 |
| 交换 / 1001（71） | 23/23/37/37 → 67/67/71/71 | 113.642 → 114.175 | 22.379 → 58.361 | 460.856 → 129.104 | 637.380 → 819.559 |
| 交换 / 1002（57） | 19/19/30/30 → 57/57/57/57 | 89.608 → 88.492 | 20.247 → 54.261 | 454.692 → 76.741 | 733.187 → 905.591 |
| 交换 / 1003（61） | 17/17/26/26 → 61/61/61/61 | 107.500 → 109.542 | 21.060 → 55.033 | 453.347 → 91.148 | 591.369 → 873.598 |

吞吐合计使用原生 74586、插件 74942 个窗口内输出 tokens，各除以 720 s；
不是窗口吞吐比的平均。原生 pooled gap P99 为 20.247–22.529 ms，明显
低于插件 54.261–60.589 ms。插件虽有更稳定的请求最坏间隔，仍有 10 条
请求未达到主 SLO；不能用合计 Goodput 隐去逐窗口的代价和失败。

## 保留的 pilot 与原生选参边界

三组 pilot 使用同一 seed=101 计划及 payload：预热 20 s、测量 60 s，
实际 7 预热+38 测量（33 短、5 长），测量到达率 **38/60=0.6333**，
并非名义 0.5。每组另有自然语言 16 条和 priming 3 条，共 64 请求/角色。
每一行是该原生配置与对应插件运行的配对值，不平均三次插件结果。

| 原生预算 | 四格 SLO 合格数（分母 38）：原生→插件 | 窗口 tokens/s | Pooled gap P99 / ms | Request worst-gap P99 / ms | TTFT P99 / ms |
| --- | --- | --- | --- | --- | --- |
| 256 | 0/0/1/1 → 38/38/38/38 | 112.650 → 136.867 | 219.835 → 60.380 | 234.508 → 78.116 | 17787.094 → 953.017 |
| 512 | 1/1/3/4 → 38/38/38/38 | 127.050 → 137.167 | 209.143 → 60.866 | 230.921 → 80.549 | 4995.293 → 925.019 |
| 16384 | 5/5/16/16 → 38/38/38/38 | 137.417 → 136.900 | 193.553 → 57.446 | 468.034 → 81.085 | 565.520 → 921.729 |

原生 16K 主指标 5/38 优于已测的 256/512，故预先选入确认；不只引用
较弱的 256 对照。Pilot 的 7.6× 是探索结果，不替代正式确认的 3.1453×。
这里只调了三个预算及相应图配置：256 的 P capture 上限为 256，其他为
512。不是仅改变预算的单因素实验，也没有穷尽原生优化空间。结论限于
此固定到达率与合成应用场景，没有最大可持续容量或统计显著性证明。

## 配置、启动入口与实现边界

双方为 BF16 Qwen3.5-35B-A3B、TP4+EP4、DP/PP=1、16K context、maxseq8、
内存比例 0.85、纯文本、APC 关闭、Mamba cache mode=none。每组分配 12 个
逻辑 CPU，OMP=1，关闭 Ascend CPU binding 与额外 NUMA binding，隔离
端口和缓存。双方共享主机 CPU、内存与互联；交换卡组并未消除同机干扰。
原生使用异步调度、chunked prefill、`FULL_AND_PIECEWISE`，gears 为
1/2/4/8/128/136/256/512；插件同步调度，完整 16K budget，P/D 独立 batch
和 TP/EP 通信组，D `FULL_DECODE_ONLY` gears 为 1/2/4/8，P 为 raw forward。
插件 policy：`execution=decoupled, stream_mode=overlap, layers_per_slice=4,
overlap_communication=isolated, decode_graph=true`；`adaptive_layers`、
`prefill_first`、`decode_budget_guard`、`compile_full_prefill` 全部 false，
`prefill_layer_token_budget=0`。无活跃 D 时可以一次完成剩余 P。
计时无 trace/profiler/observer，没有 early sampling、short-P PIECEWISE
或 group FX；这些联合配置的收益不能归因于一个开关。

可选入口如下；须先安装插件、准备相同 runtime/model，并按部署选择空闲
设备、CPU affinity、独立缓存和端口。脚本不替用户选择绑核或缓存目录。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 MUXPD_TP_SIZE=4 OMP_NUM_THREADS=1 \
  VLLM_SERVER_PORT=8002 scripts/serve-qwen35-streaming.sh
```

该入口默认 TP4/OMP1，保留显式环境值，末尾 CLI 参数可覆盖预设。
HUST vLLM=`d0f22d2bd`、Ascend=`03766ac`、插件运行时=`5342e8cd9f7407f4b7d43ea34517b59eff2b3253`。
论文边界见[实验说明](goodput-experiments.md#方法与论文的关系)：未实现
CUDA Green Context 核分区及完整预测器；通信隔离不是核分区，客户端结果
不证明设备 kernel 重叠。自然语言 16 条文本精确对照通过，不是任意输入
的 token 数值等价或语义质量保证。旧 [work256 实验](performance-tp4.md)
的负载、指标及运行时不同，不与本轮合并。

## 已完成的串行确认

按 `confirmation-serial-protocol.json`，seed1001
在同一卡组 A 分别新启动原生、插件，各 16+3+31+71=121 请求；71 条测量
请求/角色单列，不混入上述 378 条。双方均完整、独立审计通过且 eligible。

| 串行指标（120 s，71 条测量请求/角色） | 原生 16K | 插件 raw q4 |
| --- | ---: | ---: |
| 四格 SLO 合格数（顺序同上） | 22 / 22 / 33 / 33 | 68 / 68 / 71 / 71 |
| 主 SLO Goodput / requests/s | 0.183333 | 0.566667 |
| 固定窗口 tokens/s | 113.516667 | 114.141667 |
| Pooled gap P99 / ms | 22.920115 | 57.758548 |
| Request worst-gap P99 / ms | 464.372593 | 104.371173 |
| Scheduled TTFT P99 / ms | 622.050727 | 810.355171 |

串行 Goodput 为 **68/22=3.0909×**，窗口吞吐 **+0.55058%**；仍然是
每请求最坏停顿改善，而 pooled gap 与 TTFT 变差。单服务时收益仍存在，
说明收益不依赖两服务同时运行；一次串行 seed 不能排除所有主机状态影响。

## 可追溯证据

本地证据前缀为 `local-results/tp4-goodput/`，最终汇总
`tp4-final-summary.{json,md}` 的状态为 `passed`；独立元数据复核为
`confirm-first-independent-review.json`、`confirm-cross-independent-metadata-review.json`。
串行入口为 `tp4-confirm-serial-{native,plugin}-result.json`。十四份确认
raw 审计全部通过；汇总重新绑定小文件 SHA，但未重放或完整
重哈希大 raw。各结果目录保留原始 SSE、chunk journal、独立审计和 CPU
快照；manifest 绑定服务 birth、卡组、参数、计划及源码。CPU 共享 cgroup
计数无 throttling，不能相加或归为单角色 CPU 利用率，也不排除其他争用。
Pilot 入口为 `tp4-budget256-pilot-result.json`、
`tp4-budget512-pilot-v4-result.json`、`tp4-budget16k-pilot-v4-result.json`。
配对生产者 SHA256：`a0b1dc9e6387e063b8beb6ac1327155ae6596559024fafa415b0ac4fd0f0c2d7`；
串行生产者 SHA256：`d80706a15e7b3f6cbd11a9fce054f07f819154576fff922c2c02462ed6a6cb3f`。
parser=`571eb1ce`、审计器=`da12eb1f`，完整 SHA 在回执。协议文字遗留
seed42，实际执行按冻结计划及 payload。归档入口为
[证据 README (archive member: `evidence/2026-10-11-tp4-goodput/README.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)，发布标签
[`evidence-tp4-goodput-20261011`](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-tp4-goodput-20261011)；原始数据下载及逐文件校验以 benchmark 入口为准。
归档随后对全部 763 个文件重新计算 SHA256，并逐一验证解压后的原始字节；
约 94 MB 压缩包保留约 1.09 GB 证据。本地测试为 657 passed、2 skipped，
Ruff、启动脚本语法和 diff 检查通过。
生成前的 readiness 环境缺项、API helper 宽 affinity 和并发锁拒绝均保留
在 `root-pilot-readiness-environment-diagnosis.json` 及
`tp4-fixed4-b-budget512-pilot-v4-root-lock-deviation.json`；零生成检查失败
不纳入性能表，全部已测结果保留，未观测配置不当成已验证配置。
