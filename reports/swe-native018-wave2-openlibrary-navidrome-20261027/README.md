# SWE 第二批 Openlibrary 与 Navidrome

新增两次原协议单 Agent 尝试，均使用 vLLM 0.18.0 / vLLM-Ascend 0.18.0。每题 100 步、1200 秒、4096 输出 token，零自动重试；不重跑已完成任务。

| 项目               |        Openlibrary |         Navidrome |
| ------------------ | -----------------: | ----------------: |
| 成功 HTTP / 拒绝   |             37 / 0 |            62 / 0 |
| Agent 秒           | 352.83412449620664 | 533.3633364606649 |
| 提交状态           |          Submitted |         Submitted |
| 补丁字节           |             270398 |             12813 |
| apply / 受影响路径 |              0 / 3 |            0 / 10 |
| F2P                |        0/2，缺失 0 |       0/2，缺失 2 |
| P2P                |        0/1，缺失 0 |            无目标 |
| 官方结果           |              false |             false |

两题均取得完整应用前、应用后及测试前（实际待评分）状态，应用后与测试前身份一致。Openlibrary 原三个目标实际执行，列表断言失败；原镜像 PYTEST_ADDOPTS 中的 reruns=3
产生 9 次 pytest 内部重试，不是重复 Agent 或重复评分。原两项 tracked fixture 字节保持，精确复合准入没有删除或重置原 fixture。

Navidrome 原补丁新增的 imaging.Copy、imaging.DrawAt 调用未定义，并导入缺失的 internal/storage/fs；原日志为编译/setup
失败，目标测试体未执行。原 parser 为空，两项 F2P 缺失，不能表述为两项功能断言失败。评分进程退出 0 不等于构建或测试成功。

累计 26/731 次不同任务尝试：原官方记录 3/26；披露同一 Protonmail 答案的依赖恢复后 4/26；705 项未执行。恢复没有增加 Agent 次数。此前 24 题
catalog、V098 快照及所有旧失败保持不变。

summary-*.json 为 CPU 原派生摘要；review-*.json 为 storage 公开白名单独审投影，原字节不改；custody-\*.json
记录受限原件封存验证，未包含答案、补丁或轨迹正文。源码派生副本仅用于复算摘要。运行 python verify.py 可离线核验公开身份、累计口径和覆盖边界，不执行 Agent 或评分。
