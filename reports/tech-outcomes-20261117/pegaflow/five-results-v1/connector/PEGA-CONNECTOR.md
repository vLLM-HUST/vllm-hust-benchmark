# PegaFlow connector 与额外开销

这是五项报告的独立机制补充，使用真实KV保存/恢复/失败计数，不使用Bid policy字段。

| 阶段              | prefix零复用已核验 |  save success | restore success |               load/save失败 |
| ----------------- | ------------------ | ------------: | --------------: | --------------------------: |
| native-mmlu       | True               | 未验证/不适用 |   未验证/不适用 | 未验证/不适用/未验证/不适用 |
| native-hle        | True               | 未验证/不适用 |   未验证/不适用 | 未验证/不适用/未验证/不适用 |
| native-frontier   | True               | 未验证/不适用 |   未验证/不适用 | 未验证/不适用/未验证/不适用 |
| native-swe        | True               | 未验证/不适用 |   未验证/不适用 | 未验证/不适用/未验证/不适用 |
| native-terminal   | True               | 未验证/不适用 |   未验证/不适用 | 未验证/不适用/未验证/不适用 |
| pegaflow-mmlu     | True               |          12.0 |             0.0 |                     0.0/0.0 |
| pegaflow-hle      | True               |           8.0 |             0.0 |                     0.0/0.0 |
| pegaflow-frontier | True               |           0.0 |             0.0 |                     0.0/0.0 |
| pegaflow-swe      | True               |         402.0 |             0.0 |                     0.0/0.0 |
| pegaflow-terminal | True               |         390.0 |             0.0 |                     0.0/0.0 |

零restore不证明offload缓存收益；save路径被实际调用也不能单独证明加速。两臂保留各自prefix计数与未知cache usage，不从缺失计数推零。

sidecar配置参数为30gb（冻结计划声明30GiB），容量字节与常驻量按实际metric/主机观测分别保留，未知不记零。初始化、模型外sidecar准备、连续常驻期间以及模型/sidecar独立telemetry流见PEGA-CONNECTOR.json。容器/HBM总观测不自动当作MOD独占增量，不能替代有效FLOP或货币费用。

主通用报告使用--project PegaFlow --candidate-arm pegaflow；不得把本文件传给仅识别Bid
solver_policy_custody的--mechanism-evidence。C8可另传--c8-summary，仍与EOS五项分开。
