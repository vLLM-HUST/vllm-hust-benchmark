# Full200 prefix 独立审计 v1

**证据审计及 collection/lifecycle PASS；原 full quality gate FAIL。** 这两个结论不可混用。

终态对象仅为 `quality-runs/prefix-quality-full-01`。未访问任何 live main bridge child，未调用 NPU 工具或控制模型进程。本审计没有导入或调用质量 driver、controller、原 score_quality 或包装 scorer；独立实现 JSON 校验、评分循环、逐题转换与质量门。唯一复用的评分逻辑是 SHA 固定的原 canonical `extract_answer`，并用原 tokenizer 对四份 run 的全部 800 条完整 token 序列重新解码，核对保存文本。

| run | 正确/200 | stop | length | 原 all_complete |
|---|---:|---:|---:|---|
| native 1 | 165 | 191 | 9 | false |
| native 2 | 165 | 191 | 9 | false |
| old raw | 166 | 193 | 7 | false |
| prefix | 167 | 189 | 11 | false |

165 个 both-native-correct 案例中，prefix 新丢失 `mmlu-pro-07338`，因此逐题不退化门失败。该题在原 batch 7，第4成员、第3个 peer；native 两次均 H / 4723 tokens / stop，raw 为 H / 5133 / stop，prefix 为8192 / length，原评分 answer=null。它与 native 在第207个输出 token 首次分歧（318→198），与 raw 在第38个首次分歧（7661→26068）；完整批次信息和来源身份见 `new-loss-batch7.json`。

旧 raw 三个 loss（03766、10672、01741）均修复。旧四个 gain 中00882、08107保留，06074、06741回到 length；prefix 相对 native 的第三个 gain 为11363。正确数净增不抵消任何逐题 loss。native 原有9个 length 以及 candidate 的11个 length 均按原规则保留，原8192预算、温度0、seed42、完整题集及评分未改动。

完整 token 序列与 native1/native2/old raw 相同的题数分别为89/89/27（共200）；这不能表述为完整答案恢复 native。所有逐题正确性、截断、token/text SHA和首分叉记录在 `rows.json`；相关旧 loss、gain和新变化保存在 `notable-cases.json`。

独立核验原25批次和200案例原序、每批先seed精确生成1 token再释放7 peers；实际50次admission，各rank 50个closed cohort、25个treated。每批有10个真实P层前进且D token退役的事务。4 rank各2025次treated collective，累计8100；各2025次D0 bypass。实际P prompt/positions SHA与原题匹配，admission D行数为seed D0/peer D1；每次treated调用保留原BF16、单次原P collective及原tensor alias。

终态证据验证正常空 maintenance、无aborted IDs、无未完成scheduler或worker所有权、KV恢复、正常3 hooks恢复、无终态traceback；controller exit0且owned_members_after_cleanup=[]，保存的前后物理卡0/2/3/4均idle。排空和collection成功不用于推断质量成功。

原core/Ascend来源保持，外部prefix诊断模块与冻结producer身份单独验证；plugin为6c，相比旧controls f7已有20个runtime文件字节等价。原producer以及物理卡组合与历史controls存在显式差异，本审计没有改写历史run元数据，也不声称通过原compare_provenance。该结果限于冻结6c外部诊断，不是current main、安装包或性能资格。

`assessment.json`仅在独立评分和生命周期判定完成后作为被审计输出交叉核对，SHA为 `5fa2b1be429abb35daae9494590688b3f8c7978bd31cb9643a0838f921a9ba62`；所有800条评分行、原严格门和配对转换一致。`audit_passed=true`仅表示证据审计成功，`full_frozen_plan_passed=false`保留原质量结论。

`audit_v1.py`使用固定终态路径，向自身所在目录以排他创建方式输出，不覆盖旧审计。首轮运行记录见 `attempt-01.log`。`source-identities.json`绑定输入文件，`SHA256SUMS`冻结本目录成员；复核使用 `sha256sum -c SHA256SUMS`。如需完整重跑，应把脚本复制到新的空目录并使用 `PYTHONDONTWRITEBYTECODE=1 /root/workspace/.venv/bin/python audit_v1.py`，不要覆盖本冻结目录。
