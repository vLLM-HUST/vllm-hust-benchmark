# 新 loss 07338 的 batch7 上下文 v1

**结论：第207个输出 token 的分叉附近，没有收据可见的请求结束、调度成员减少或 scheduler 字典顺序变化。** 四份终态运行在目标第190–220个输出 token 的31个事务中，都有原8成员按保存顺序参与 scheduled 与 decode retirement。该结论不等于 runner 实际 tensor 行序已获验证。

目标为原 batch_id=7（零起，第8批），原成员序为07151、07213、07275、07338、07401、07461、07524、07585（均为mmlu-pro）。07151为seed；07338为第4成员、第3个peer。所有 token 序号均从1开始，global_step是各run自己的全局事务编号，不能跨run当作时间戳比较。

| 成员 | role | native1/2长度 | old raw长度 | prefix长度 | prefix/native首分叉 | prefix/raw首分叉 |
|---|---|---:|---:|---:|---:|---:|
| 07151 | seed | 289/289 | 445 | 445 | 56 | 完整相同 |
| 07213 | peer | 302/302 | 237 | 302 | 完整相同 | 51 |
| 07275 | peer | 379/379 | 805 | 447 | 230 | 10 |
| 07338 | peer | 4723/4723 | 5133 | 8192 | 207 | 38 |
| 07401 | peer | 575/575 | 702 | 682 | 273 | 100 |
| 07461 | peer | 468/468 | 481 | 462 | 203 | 57 |
| 07524 | peer | 467/467 | 302 | 417 | 233 | 15 |
| 07585 | peer | 532/532 | 544 | 533 | 401 | 325 |

表中除prefix目标07338为8192 length外，其余输出均stop。first divergence从完整token ID列表逐项计算，没有只比较答案字母或短前缀。seed的prefix与old raw完整445 tokens相同；peer07461在第203个token先于目标分歧，仍未结束。该相邻分叉只作探针选点，不能推断请求间因果。

| run | peers首token时seed累计 | 目标207时seed累计 | 最早结束成员（目标相对token） | seed结束时目标token | 目标207 global_step |
|---|---:|---:|---|---:|---:|
| native1 | 2 | 208 | 07151 (288) | 288 | 28684 |
| native2 | 2 | 208 | 07151 (288) | 288 | 28684 |
| old_raw | 11 | 217 | 07213 (237) | 435 | 31458 |
| prefix | 11 | 217 | 07213 (302) | 435 | 31127 |

seed确实在raw/prefix中比native多领先9个输出token：native的seed始终领先目标1，raw/prefix领先10（seed未结束前）。但prefix最早的peer07213到目标302才结束，seed到435才结束；native最早seed在288，raw最早peer07213在237，均未邻接207。现有证据排除了“seed/peer恰在207附近natural finish导致scheduler成员缩减”这一具体解释。

`target-window190-220.json`保留四run共124对原schedule/retirement JSON及解码后的case ID映射、保存的字典/列表顺序、派生每请求输出计数，包含所需200–210窗口。`summary.json`保留全8成员最终D退役步骤和其后不再出现检查，`members.json`保留长度、结束原因、完整token SHA和首分叉。

对齐方法和证据界限：

- 原receipt producer仅在旧num_output_tokens>0且新计数增加时记入decode_retired_ids。因此其每条D退役对应token2起的输出；这里逐项核实所有相关退役均scheduled_tokens=1、phase=D、没有speculative decoding，并验证每个成员最终输出长度恰等于D退役总次数+1，才将目标第n个token对齐到第n−1次退役。
- natural finish所在事务由最终D退役、run中的stop/length终态和该ID在所有后续scheduler记录缺席联合派生；receipt没有显式finished_req_ids字段。
- scheduled_tokens是scheduler输出字典；phases_before来自scheduler.requests字典；decode_retired_ids也依scheduler对象遍历产生。其保存顺序不是模型runner的物理row index，不能替代input_batch.req_ids。
- 收据没有实际decode input_batch行序、graph bucket、KV/slot映射、logits、层张量或设备时间；不能排除内部行序变化，也不能据此确定第207分叉的数值原因。

建议后续探针保留原完整batch与seed放行规则，在相当于目标190–220的D事务采集实际runner req_ids/row index、graph bucket、positions/KV slots及目标logits，同时保留seed/peer相对输出计数。观察器需先与无观察器输出做bridge。不要通过case特判、fallback、重试或扩充8192预算改变质量结果。

本分析使用四个已终态run、各自scheduler receipts及冻结producer/plan/source manifest；来源路径来自冻结source-manifest，完整字节SHA及大小见`source-identities.json`。未读取main bridge live child、未访问设备、未修改源或旧审计。脚本`derive_context_v1.py`首轮成功，输出采用排他创建，需重跑时请复制脚本至新空目录，避免覆盖本证据。
