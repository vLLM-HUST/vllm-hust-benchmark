# MuxPD prefix 布局：原三个失分所在完整批次的 24 题后验复测

**原 batches 3、6、14 的 24 题完整答案中，prefix 候选为 22/24，与两次历史 native 对照的正确数相同；旧 raw 为
19/24。原三个失分均恢复，候选没有截断，也没有“两次 native 均正确、候选错误”的新增病例。** 这是按旧失分选择的后验诊断，不能替代原 200 题质量门槛。完整 token 序列仅
5/24 与每次 native 相同，答案恢复不等于逐字生成一致。

本候选[已公开发布](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-prefix-quality-losses-20261028)；[匿名公开读回终态](public-readback/posterior24/final-status-v1.json)为
**PASS**：12个附件实际HTTP200下载，归档SHA、45个payload与1份嵌入manifest全部核验，gzip
EOF/CRC通过，未替换本地payload、未解压或执行下载代码。该公开可用性结论不扩大后验子集的质量资格。随后[独立全200复测](FULL200-v1.md)为167/200，但存在1个new
wrong和11个length，原质量门仍FAIL；这两次候选归档相互独立。

## 原输入、预算与评分均保留

复测使用原冻结 MMLU-Pro 200 题计划中的三个完整八题批次，保留原顺序、seed 与 peers：seed 产生恰好一个输出 token
后才加入同批其余七题，整批自然完成后开始下一批。没有单独挑出三题、改变同批上下文或追加请求 warmup。

原 5-shot CoT 输入、token IDs、gold、tokenizer、canonical `extract_answer`、greedy/seed42、**8192 输出 token
上限**均不变。长度截断、缺失、无效答案继续判错；未把 length 重标为 stop。原计划 SHA 为
`dd3e1f305475ec556dc824947f26d5fb5743242e82b5099155c95b8954bb2e65`。包中保留全体 200 题冻结输入以维持计划身份，但本候选只有这
24 题的输出。

运行限定于原 BF16 Qwen3.5-35B-A3B、TP4/EP4、同步 raw-P、isolated communication、decode graph 与共同编译后端配方。设备使用保存的四张
910B2 实体卡 0/2/3/4，进程内可见序号为 0/1/2/3；不把逻辑可见序号等同于历史实体卡身份。插件设备源码为
`6c606f789450a45ef35afee398b0a3e516caa1a1`，旧 f7 中已有的 20 个 runtime 源文件逐字相同；新增外部 prefix 模块和新 producer
的身份单独记录，不伪装成旧 producer。该证据不资格化当前 main、async、group、SP 或安装包。

prefix 在原 LLM 构造与 native warmup 完成后由 RPC 安装。其布局是按真实 admission D 行数，在实际 P collective payload
前加零行；仍调用既有 P communicator 恰好一次，再 copy-back 到原 tensor 并返回同一对象。D0 直接旁路。这个干预没有改变精度、模型行、KV/attention
metadata 或生成预算，也不意味着所有 native 批内 P 顺序或真实 D 值均被重现。

## 三个完整批次的结果

| 角色         | 正确 / 24 | 自然 stop | length 截断 | 相对两 native 共同正确集新增错误 |
| ------------ | --------: | --------: | ----------: | -------------------------------: |
| 历史 Native1 |        22 |        24 |           0 |                                — |
| 历史 Native2 |        22 |        24 |           0 |                                — |
| 历史 raw     |        19 |        23 |           1 |                                3 |
| prefix 候选  |    **22** |    **24** |       **0** |                            **0** |

| 原失分 case    | Gold | 旧 raw 完整结果                   | prefix 完整结果          |
| -------------- | ---- | --------------------------------- | ------------------------ |
| mmlu-pro-03766 | B    | A；stop；6396 tokens              | **B；stop；1053 tokens** |
| mmlu-pro-10672 | A    | G；stop；1046 tokens              | **A；stop；2613 tokens** |
| mmlu-pro-01741 | E    | length；8192 tokens；按原规则判错 | **E；stop；590 tokens**  |

prefix 的完整 token 数组与 Native1、Native2 各只有 **5/24** 完全相同，与旧 raw 为 3/24。独立审计逐题保存完整输出长度、token/text
SHA、首个差异位置、解析答案和终止原因。不能将目标位置 logits 的恢复或这三题答案修复扩大为全部完整输出逐字等同。

候选 `assessment.json` 明确为
`scope=loss-batches`、`full_quality_gate_attempted=false`、`full_frozen_plan_passed=null`。两个旧 native
的全 200 题原件各含 9 个截断；旧 raw 全体有 7 个。因此使用这些旧 controls 时，原 strict gate 的 native
完整性条件依然失败；本报告没有重命名该失败为通过，也没有改变逐题不退化规则。

## 独立评分与运行终态

候选 worker 在 2026-10-28 07:23:02 UTC 返回终态，控制器在 07:23:07 UTC 完成。3/6/14 每批都核验到十个实际 P 层进展与 D token
退役同事务的步骤。每 rank 有 6 个完成 cohort、3 个 treated cohort；真实 admission 模式为 seed D0、peers D1。四 rank 共 972 条
treated collective 记录逐项核对原 shape/实际 payload、BF16、单次原 collective、原 tensor alias 和 isolated P 路由。

成功路径仅做一次空 maintenance turn，未取消 live 请求。scheduler、worker、KV 空闲状态、正常恢复三个 layout hook、shutdown
返回、controller exit0、owned process group 为空、保存的前后四卡 idle 收据均通过；terminal process log 无 Python
traceback 或已知 graph ownership 异常。这里只报告这些具体终态证据，不把 worker 的退出机制描述为“全局无信号 graceful shutdown”。

独立审计没有调用质量 driver、controller、原 scorer 或新 scoring wrapper。它只复用 SHA 固定的原 canonical answer extractor 与
tokenizer，独立读取冻结输入、逐题评分、比较完整输出、检查实际 prompt 指纹/调度/布局/生命周期；最后才与候选 assessment 逐题交叉核对。独立结果同为
`22/22/19/22`。

原 `score_quality.py` 字节保持不变，SHA 为
`157b52ffdbae6d7f0f990bb27379c463df1807daeebf7298fd396968e96b17fd`。新的 `score_layout_quality_v1.py`
是显式 provenance bridge，SHA 为
`249762dd55dbc28e5b3f60b3219afbd02e11cc9fdf654a5fa579a65ca8f1255c`；子集只报告原规则下的诊断比较，不调用全量 `evaluate`
来生成新质量 gate。外部 prefix 模块 SHA 为 `6431c4c0abdd869ef17e7122f681c457ad2b4d5741f86c6a0653e9fca44e02fc`。

## 新候选包与历史对照的边界

新包
[muxpd-prefix-quality-losses-01-20261028-v1.tar.gz](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-prefix-quality-losses-20261028/muxpd-prefix-quality-losses-01-20261028-v1.tar.gz)
为 **636,414 字节**，SHA256：

`eeeb082a111f5236ea620582922bea383d5ef8a06c287b9bbd4978b7d2668615`

它包含 45 个 payload 文件以及一份逐字相同的嵌入 manifest：28 个候选原件（包含冻结生产者和输入）、8 个 CPU/冻结回执、8 个独立审计文件、1 个历史引用索引。排除
compiler-cache、pycache 和工具缓存；不附带历史 native/raw 的完整 `run.json` 或 token 输出原件。候选 assessment
与独立审计中确有历史派生评分/比较行，这些不是历史完整运行的替代品。

[逐成员 manifest](publication/posterior24/muxpd-prefix-quality-losses-01-20261028-v1.manifest.json)
绑定相对路径、原本地来源、字节数和 SHA；[完整本地验证](publication/posterior24/local-verification-v1.json) 逐项读取 tar
内容并重新核对原件，验证精确成员集合、普通文件类型、安全路径、嵌入 manifest 身份和全部原字节。不解压或执行包内代码。全部原始 candidate run/assessment 字节保持不变：

| 原件                    | SHA256                                                             |
| ----------------------- | ------------------------------------------------------------------ |
| 候选 `results/run.json` | `65f245d915a0a4ff7e0b867a49f1e757612053dfd3ebf15d7be938c2a976af3a` |
| 候选 `assessment.json`  | `40804616e0fd02668514933272913a035b3f3a6886329788aa2536b257e12c60` |
| producer freeze         | `07a796c568e8cd69ae7157b3386269b1a624a94b6b8393bba232a5cec68a9345` |
| 独立审计 summary        | `af08f14b697877268d9da1363a4d42447a23b6099e5ec15b3fbbf479d2eb0aa3` |

历史原件继续引用既有公开胶囊；下列 URL 和 SHA 来自固定 benchmark snapshot `b77e05b0d906b0e14b274533edf6750ffacd0170` 的
canonical v3/v5 索引。本次没有再次下载这些旧包，不声称重新验证当前远端可用性。既有
[v3 公开读回](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b77e05b0d906b0e14b274533edf6750ffacd0170/reports/muxpd-followup-20261025/device-evidence-v3/PUBLIC-READBACK-v3.md)与
[v5 公开读回](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b77e05b0d906b0e14b274533edf6750ffacd0170/reports/muxpd-followup-20261025/quality-reference-v5/PUBLIC-READBACK-v5.md)分别记录其历史下载核验。

| 外部历史原件                                                                                                                                                     | 归档 SHA256                                                        | 包内原 run SHA256                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| [Native1（v3）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-native-quality-v2-hccl-true-01.tar.gz) | `5aa0cfebf7ae804100ba6a440512e4daefd823e062efcb7ef66fa660c91e2adf` | `cb7c36fb75a5f586c01d3e4f30a7369b5befa7bbc6021d1fcf0c5cb3a71a6a28` |
| [Native2（v5）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-native-quality-v2-hccl-true-02.tar.gz) | `3556615a84daf4b8f8b7abc762259f43cf845d8932d304f60fad6180d66f5033` | `14454401799a1745ca369891f177624ce1ff4348c2a29c8addb00f8b5d90500e` |
| [Raw（v5）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-raw-quality-v2-hccl-true-01.tar.gz)        | `bbfdef527e51194205b7f7cf1e5d5613f2b63e3c7c2e3e8482e075dc8cfabf8a` | `75aa8495902fd795dbe845da35eb1bf06bb3adea7c81b515e8c3375b07a13f9e` |

[historical-controls-reference-v1.json](publication/posterior24/historical-controls-reference-v1.json)
还列明每个 archive 的大小、完整包内路径、公开 manifest/readback 文档与 SHA，并验证其 run hash 恰好是本候选评分所引用的历史 run。旧目录准备时保留的
`prepared_not_uploaded` 字段不被改写；其后来公开状态由原 readback 文档说明。

公开复核可先只校验包，不接触原本地路径：

```sh
python verify_archive_v1.py \
  --archive muxpd-prefix-quality-losses-01-20261028-v1.tar.gz \
  --manifest muxpd-prefix-quality-losses-01-20261028-v1.manifest.json \
  --output reader-verification.json
```

包内源文件保留原绝对路径，不能假定重定位后 scoring/audit 脚本即可直接执行。完整重评分仍需取得上述外部历史 controls、匹配 tokenizer
和解释器依赖。本地custody与实际匿名公开读回均有独立回执；公开PASS仅证明指定资产字节可读，不构成全量质量、吞吐改善、根因已完全证明或当前插件包资格。原发布暂存稿按原字节保留在publication/posterior24，其当时“未上传”状态由这里所链接的后续读回回执补充，未倒改冻结记录。

本报告对应的小型独立证据见[24题审计](independent-audit/posterior24/README.md)。完整候选raw文件从固定Release归档取得，本目录不复制大型raw。
