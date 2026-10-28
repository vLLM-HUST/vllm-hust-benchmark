# 新增退化 07338：batch7 有界诊断报告

native 与 prefix 在 target `mmlu-pro-07338` 的第 207 个输出 token 分别选择 `318` 与 `198`。各自 top1−top2 为 **0.25
/ 0.125，均非平局**，因此这个窗口的贪心选择变化不能解释为平局时的 argmax 次序。完整答案质量结论没有改变：此前 full200 的 native 两次为 165/200，raw 为
166/200，prefix 为 167/200，但 prefix 有新增错误 07338、11 个 length，原严格门仍 FAIL。

本轮只重现并观察一个后验窗口，没有测量性能，也未证明卷积、规约或其他算子的唯一因果关系。

## 运行范围与真实回执

采用原 batch7 八题与原 prompt，TP4、EP4、BF16、greedy、seed42、原 max_tokens8192。每次以新进程运行，seed07151 先生成 1
token，再加入七个 peer；target 达到 208 token 后由 supervisor 中止、排空和清理。这是有界前缀诊断，不是完成 8192 预算的完整答案评测，也没有重放先前七个
batch 的缓存历史。

四次运行 native/prefix × observed/plain 全部终态通过；每个角色的八题整个保留前缀均复现对应历史 full-run，两组 observed/plain 桥各 8/8
相同。native seed 保留 209 token，prefix seed 保留 218，其余各题 208。前缀通过不能证明内部 KV/GDN 状态相同。

每个 observed 运行捕获各题第 `1,200,201,202,203,204,205,206,207,208` 次决策，共 80 个完整词表向量；两次共 160 个 NPY，每向量长度
248320，以 FP32 序列化，原推理 dtype 仍为 BF16。独立 CPU 审计直接核对文件 SHA、大小、dtype、shape、有限值、采样 token、prompt 和输出历史并重算
top2，未导入 campaign analyzer。[独立审计摘要](independent-audit/summary-v1.json)

四次 scheduler/worker/KV 清理通过；两个 prefix 运行各四个 rank 恢复 3 个 hook。原日志保留 executor grace-period
SIGTERM，不表述为所有 worker 都自然退出。生命周期通过与答案质量通过是独立判断。[数据及生命周期审计](independent-audit/audit-v1.json)

## 决策窗口与解释边界

位置均为 1-based。

| 请求 / 位置       | native top2 token : logit | prefix top2 token : logit | native / prefix margin |
| ----------------- | ------------------------- | ------------------------- | ---------------------- |
| seed07151 / 1     | 9764:31.625，1206:26.75   | 9764:31.625，1206:26.75   | 4.875 / 4.875          |
| peer07461 / 203   | 11421:21.25，10813:21.0   | 10813:21.5，11421:21.375  | 0.25 / 0.125           |
| target07338 / 207 | 318:26.375，198:26.125    | 198:26.375，318:26.25     | 0.25 / 0.125           |

seed 第 1 步整个向量相同。peer 第 203 步、target 第 207 步各自此前输出历史相同，但同批 seed 已在第 **56** 步分叉，peer 又先于 target
分叉；其余五题在保留窗口尚未分叉。八题仍活跃和当前顺序相同，不等于完整同批历史或内部状态相同。

seed 第 56 步不在捕获窗口，不能报告该步 logits。seed 第 200–208 步、target 第 208 步自身历史已不同，不计算或解释这些位置的跨角色向量距离。target207
的自身历史一致也不足以排除更早同批历史、GDN/KV 状态或批处理上下文差异；本轮不作唯一根因判断。[完整 v2 窗口分析](analysis/v2-analysis.json)

## v1 FAIL 保留与 v2 的限定修正

冻结 v1 analyzer 的四次检查均因历史源码回执格式不兼容而失败。旧 quality producer 在模块目录执行 status，记录
`M ../csrc/build_aclnn.sh`；新 producer 在仓库根目录记录 `M csrc/build_aclnn.sh`。旧 diff 保存原始末尾 LF，新 helper 去掉该
LF。[原 v1 失败原件](analysis/original-v1-failure.json)

v2 只替换离线源码格式比较：按各 producer 的 cwd 重现 status 并逐字匹配；实际模块 SHA、固定提交、dirty 文件集合与 SHA、原始 binary diff 和历史
diff SHA 都须匹配。现代 diff 只接受少一个已知末尾 LF。原生命周期、恰好四次 attempt、任何失败 attempt 全局阻断、历史复现与 observed/plain
桥均保留，没有重跑设备或改算子。独立审计另从 Ascend 固定提交 `03766ac696fde5ab1980d80ca0b8543d3580c989` 与单 hunk 补丁重建文件，SHA
`b089d52475c6cb36fffd733bb5be3fbe4158b6845074751268938e98a1d0042c`
同时匹配历史和当前原件。[源码独立证明](independent-audit/source-schema-audit-v1.json)

core 固定为 `d0f22d2bda562156e4dbf433ce645e1769b4f804`，新探针插件为
`6c606f789450a45ef35afee398b0a3e516caa1a1`；历史 f7 到 6c 的原有 20 个 runtime 文件逐字节一致。prefix 外部布局模块仍由冻结
producer 的路径与 SHA 约束。新增诊断与物理运行环境差异须按原件披露，不能据此宣称当前 main 安装包已通过。

v2 分析与独立审计通过。v2 manifest 中 `review-ready-draft`、`analysis_executed:false` 是创建时原始记录，未追改；后续执行结果由
[v2 analysis](analysis/v2-analysis.json) 的原始字节、复制清单与归档 manifest 绑定。

## 归档与历史依赖

[Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-new-loss-batch7-20261028)
的[完整证据包](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-new-loss-batch7-20261028/muxpd-new-loss-batch7-20261028-v1.tar.gz)为
84,817,778 字节，SHA256 `19fb56eaf5201c30e38bcf9ced5736ccae09dfc1d943aa4b18291bb90d0af9bf`。它包含四次终态
run、冻结 producer/输入/CPU 回执、160 个向量、v1 FAIL、v2 和独立审计；共 324 原件与一份内嵌 manifest，排除 cache。本目录只复制小型回执，不复制
NPY、完整控制 run 或 producer
代码。[逐成员清单](publication/archive-manifest-v1.json)、[本地完整核验](publication/local-verification-v1.json)

完整重跑 analyzer 需要历史两次 native、一次 raw、一次 prefix full-run。原外部引用清单将 raw
描述为上下文，容易误读为可省略；[raw 补充说明](publication/README-EXTERNAL-REFERENCES-ADDENDUM-v1.md)明确 raw 的
controller/run/log 仍是 frozen analyzer
的必需哈希输入。[原历史引用](publication/external-references-v1.json)与[补充精确引用](publication/external-references-addendum-v1.json)必须合用。补充单独发布，未重建
v1 tar 或改其原报告。

14 个 Release 附件已由 [API inventory](publication/release-inventory-v1.json)确认 uploaded、大小与 digest
匹配。匿名完整读回结果为
**INCOMPLETE**，见[终态回执](public-readback/final-status-v1.json)与[完整读回记录](public-readback/readback-v1.json)。13个小附件HTTP200/大小/SHA及两份公开checksum清单通过；大归档仅一次GET，180秒后超时，保留26,606,208/84,817,778字节。完整归档SHA、gzip及325成员公开验证均未完成。没有重试或分段下载；API匹配和本地核验不替代这一结果。
