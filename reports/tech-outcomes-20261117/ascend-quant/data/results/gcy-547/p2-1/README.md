# P2-1 稀疏激活耦合：top-8 路由对比与分桶误差

单次前向、逐 token 比较 **BF16 / W8A8 / W4A8** 三个臂在同一段 token 序列上的
专家路由（每层 256 选 8），并把逐 token NLL 的差异按"路由翻了几层"分桶。

## 要回答的问题

P1-2 / P1-3 的所有误差都是在**固定路由**下测的：同一个专家、同一批 token。
但真实推理里每层只激活 8/256 个专家，路由是 router 对 hidden state 的**硬判决**——
第 8 名和第 9 名的 logits 只差一点点。hidden state 被量化扰动之后，这个排序可能翻过来，
**换掉一整个专家**，输出是不连续跳变的。这类误差在模块级 rel_fro 里看不见。

**关键事实**：两个量化产物的 `mlp.gate`（router）都是 `FLOAT`
（`quant_model_description.json` 里 41 个 `mlp.gate` 全 FLOAT）。所以这里测的不是
"router 被量化坏了"，而是"**router 保住了，它看到的输入已经被扰动，这个保护够不够**"。

## 协议

| 项 | 取值 |
|---|---|
| 服务 | `run_server.sh`（复用 `run_swe_server_ab.sh` 的环境前置），端口 **8210** |
| 拓扑 | **TP2 + `--enable-expert-parallel`**，卡 **1 / 4** |
| 为什么 TP2 | BF16 权重 66.99 GiB > 单卡 64 GB，TP1 起不来；三个臂统一 TP2 以免拓扑混入对照 |
| 上下文 | `--max-model-len 4096`、`--gpu-memory-utilization 0.9`、**`--no-enable-prefix-caching`** |
| 路由抓取 | `--enable-return-routed-experts`（vllm-ascend capturer，MRV1；MRV2 对本模型不支持） |
| 客户端 | `capture_routing.py`：`/v1/completions`，`prompt=<token ids>`、`max_tokens 1`、`temperature 0`、`prompt_logprobs 0`，**并发必须为 1**（见下） |
| 语料 | `../logs/wikitext2_chunks_1000.json`（298 块 × 1000 token + 53 token 尾块，与 PPL 实验同一批） |

`max_tokens=1` 让每个请求只做一次前向：返回的 `routed_experts` 形状是
`(num_tokens-1, L, topk)`，即**正好覆盖提示词本身**（最后一个被采样的 token 没有再前向）。
因为提示词以 token id 传入、temperature=0、前缀缓存关闭，三个臂的 token 序列逐位置相同，
路由可以逐位置对比。

**为什么不用自回归解码来比**：只要某个位置采样出的 token 不同，后面全部错位，
比出来的"路由差异"就是假的。所以这里只做 teacher-forced 的提示词前向。

### 抓取必须串行（实测出来的坑）

同一个服务、同一批请求重复抓两次（不做任何模型侧变化）：

| 客户端并发 | chunk_000 | chunk_001 | chunk_002 | chunk_003 |
|---|---|---|---|---|
| 4 | 100% | 71.25% | 71.18% | 72.44% |
| 1 | 100% | 100% | 100% | 100% |

并发批处理时，响应里的 `routed_experts` 行与请求 token 的对应关系会错位。所以
`capture_routing.py` 的默认并发是 **1**，全量 298 块的耗时约 2–3 分钟，代价可接受。
**任何并发 >1 抓出来的数据都不能用**（早期 BF16 全量抓取即为 6 并发，已作废）。

## 文件

| 文件 | 作用 |
|---|---|
| `run_server.sh` | 起服务（`LABEL=bf16|w8a8|w4a8`） |
| `capture_routing.py` | 抓路由 + 逐 token NLL，写 `out/routed_<arm>.npz`、`out/nll_<arm>.npz`、`out/meta_<arm>.json` |
| `compare_routing.py` | 汇总，写 `out/routing_coupling.csv`、`out/routing_buckets.csv`、`out/expert_flips.csv`、`out/routing_coupling.json` |

## 跑法

```bash
LABEL=bf16 ./run_server.sh                 # 约 5–8 分钟就绪
python3 capture_routing.py --base-url http://127.0.0.1:8210 --model qwen35-a3b-bf16 \
    --arm bf16 --chunks ../logs/wikitext2_chunks_1000.json --out-dir out
kill -- -$(cat ../logs/swe_server_ab_bf16.pid)

# 换 w8a8 / w4a8 重复上面的三步（换 --model 名字与 --arm）
python3 compare_routing.py                 # 纯 numpy，不需要 NPU
```

## 判据（先写死，免得结果出来再找解释）

| 观测 | 结论 |
|---|---|
| 完全一致率 ≥99.5% 且分桶 ΔNLL 平坦 | 精度损失是平滑的权重误差，与路由无关 |
| 一致率 95–99%，翻转集中在"翻 1–2 层"的 token | 边界抖动，量级有限 |
| 一致率 <95%，或"翻 ≥3 层"桶的 ΔNLL 显著更高 | 离散失稳，比权重截断更值得修 |
| 翻转与 P1-2 的误差集中度相关 | "坏专家被路由放大"；P1-2 已证明无坏层坏专家，预期不相关 |

## 边界

* 语料是 wikitext-2（与 PPL 同源），不是生产流量；只覆盖 40 个主层
  （MTP 层专家在两个产物里都是 FLOAT，且本配置不启用 MTP）。
* TP2 是为了让 BF16 能起来而强制统一的拓扑；**同拓扑内的三臂对照**成立，
  但本节数字不能与 TP1 的性能/精度实验逐位相减。
* 抓路由会多一次 D2H 拷贝，本服务**不用于任何吞吐/时延测量**。
