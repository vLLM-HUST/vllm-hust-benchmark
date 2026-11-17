# P2-2 长上下文 PPL：4K / 32K / 128K

同一段连续文本，按 4K、32K、128K 切成不重叠窗口，teacher forcing 算 PPL。
回答两个问题：量化误差在长上下文下放大多少；这个放大是否随**位置**累积。

## 协议

| 项 | 取值 |
|---|---|
| 服务 | `run_server.sh`，端口 **8220** |
| W8A8 / W4A8 | **单卡 TP1、同一张卡 7、同一 KV 预算 8 GiB** |
| BF16 | 参考臂，**TP2 卡 0/1**（66.99 GiB 单卡放不下；2026-11-02 复跑时卡 1/4 被占用改到 0/1），不与量化臂做单卡相减 |
| 上下文 | `--max-model-len 133120`（131072 提示词 + 1 生成 + 余量） |
| 抓取 | `/v1/completions`，`prompt=<token ids>`，`max_tokens=1`，`temperature=0`，`prompt_logprobs=0`，**串行** |
| 语料 | `../logs/wikitext2_chunks_1000.json`（298 块，297,053 token，按原序拼接） |
| 窗口 | 不重叠：4K × 72、32K × 9、128K × 2 |

**为什么 KV 预算取 8 GiB**：这个模型的 40 层里有 30 层是 linear attention
（`layer_types` 每 4 层才一个 `full_attention`，共 10 层），只有那 10 层占 KV。
每 token 每满注意力层 2（K,V）× 2（kv_heads）× 256（head_dim）× 2 B = 2 KiB，
10 层合计 **20 KiB/token**。131072 token 的一条序列只需 **2.6 GiB**。
8 GiB 给 3 倍余量，同时把剩下的显存留给 `prompt_logprobs` 的 logits
（4096 token 的分块 × 15 万词表 × 2 B ≈ 1.2 GiB/块）。

**为什么关前缀缓存**：命中缓存时 `prompt_logprobs` 可能来自另一条序列的缓存，
而本实验的 `max_tokens=1` 提示词之间大量共用前缀（同一段文本按不同长度切窗），
不关会静默拿到错的数据。

**为什么串行**：P2-1 实测并发批处理会让响应里的逐位置数据错位；NLL 与路由来自
同一个响应体，本轮不冒险。

## 判据（先写死）

| 观测 | 结论 |
|---|---|
| W4A8 相对 BF16 在三个长度的 ΔPPL 都在 2% 内且基本一致 | 误差与长度无关，短上下文 PPL 的 +1.9% 可以外推到长上下文 |
| ΔPPL 随长度单调上升（128K 明显大于 4K） | 存在长度相关放大，需要按长上下文重新验收 |
| 位置分段里 64–128K 段的 ΔNLL 明显大于 0–4K 段（>2×） | 误差随位置累积，不适合长文档场景 |
| W8A8 在各长度几乎不动、W4A8 明显变差 | 差异可归因到 int4 权重截断，而非拓扑/服务 |

## 文件

| 文件 | 作用 |
|---|---|
| `run_server.sh` | 起服务（`LABEL=w8a8|w4a8|bf16`） |
| `score_long.py` | 抓逐位置 NLL，写 `out/nll_<arm>_<L>.npz`、`out/meta_<arm>_<L>.json` |
| `analyze_difficulty.py` | 按 BF16 的 token 难度分箱统计 ΔNLL → `out/longctx_difficulty.csv` |
| `analyze_chunking.py` | 同一模型不同 prefill chunk 大小的逐块对比（诊断用） |
| `compare_long.py` | 汇总 `out/longctx_ppl.csv`、`out/longctx_common.csv`、`out/longctx_position.csv`、`out/longctx_tail.csv`、`out/longctx.json` |

## 跑法

```bash
LABEL=w8a8 ./run_server.sh
python3 score_long.py --base-url http://127.0.0.1:8220 --model qwen35-a3b-w8a8 --arm w8a8
kill -- -$(cat ../logs/swe_server_ab_w8a8.pid)

LABEL=w4a8 ./run_server.sh
python3 score_long.py --base-url http://127.0.0.1:8220 --model qwen35-a3b-w4a8 --arm w4a8
kill -- -$(cat ../logs/swe_server_ab_w4a8.pid)

LABEL=bf16 TP=2 ASCEND_RT_VISIBLE_DEVICES=0,1 ./run_server.sh   # 2026-11-02 复跑
python3 score_long.py --base-url http://127.0.0.1:8220 --model qwen35-a3b-bf16 --arm bf16
kill -- -$(cat ../logs/swe_server_ab_bf16.pid)

python3 compare_long.py        # 纯 numpy
```

## 必须同时看中位数（尾部效应）

逐 token 的 ΔNLL（W4A8 − W8A8）在中位数上≈0，但均值在 32K/128K 是 −0.035 nats。
原因是一小撮 token：约 0.3–1% 的位置上两个臂相差 >2 nats（极端例子一个臂给
p≈1、另一个臂给 p≈1e-7）。这类 token 的数量随**绝对位置**变多，所以均值呈现出
"越深越好"的假象。`longctx_tail.csv` 同时给出中位数、去掉 |Δ|>3 后的截尾均值、
以及 |Δ| 最大 1% 的 token 对总差值的贡献占比。**判断"谁更接近 BF16"只能靠
BF16 锚点，不能靠均值符号。**

## 边界

* 语料只有 wikitext-2 测试集（297K token），128K 窗口只有 **2 个**——token 级
  统计量很大（26 万），但文档级样本极少，不能据此谈"文档间方差"。
* 窗口是不重叠的连续切分，跨文档边界（wikitext 按行拼），与常规 PPL 口径一致。
* 只有 4K/32K/128K 三个点，不覆盖模型上限 262144。
* 只测 PPL，不测检索/长生成——那是 P2-3。
* **prefill 分块固定 4096，且无法改大**：bt>4096 时引擎要申请固定的 7.58 GiB
  工作区，BF16 TP2（剩余 3–7 GiB）和 W8A8 单卡（剩余 0.31 GiB）都 OOM，bt=32768
  要 15.16 GiB。因此"4096 边界与 Δ 转负重合"这一混淆项无法在本机排除，见
  `RESULT.md` 的分块实验一节。
* 抓取会返回 13 万条 `prompt_logprobs`，单请求响应 5–10 MB，本服务不用于吞吐测量。
