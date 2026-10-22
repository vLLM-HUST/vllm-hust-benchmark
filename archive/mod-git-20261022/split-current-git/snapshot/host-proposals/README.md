# host-proposals/ — 宿主私有面的提案载体目录（PURPOSE v3 §8 档2）

本目录是**合法的档2 载体**：当某个杠杆只能通过宿主私有面（方法体内部的调用点、
无公开配置开关）表达时，按 `/root/PURPOSE.md`（v3，2026-10-16）§8 的三档归宿，
把 **patch 原件 + 生成脚本 + CPU 测试 + 归属记录** 放在这里，运行树只接收部署副本。

规则（本目录**只冻结自身**）：

1. 每个提案一个子目录（`<id>/`），互不牵连；冻结/废弃只作用于该子目录自身，
   **不构成对 `upstream/`、`vllm-ascend` 或任何上游仓的冻结或追认**。
2. 必须含：`host_orig/`（宿主基线原件 + sha256）、`gen_patch.py`（确定性生成、
   对宿主 sha fail-closed）、`patch.diff`、`tests/`（仓内 CPU 门）、`OWNERSHIP.md`
   （责任人 / 何时可删 / 上游状态 / 后续动作）、`proposal_meta.json`（机器可读）。
3. `upstream_status` 必须如实：未提交就写「未提交（proposal snapshot）」并留
   `issue_or_pr=null`；**不得**表述为「上游已接受/已合入」。
4. `host.version_range` 点钉（`==`），禁止 `>=0` 或区间。
5. 默认关闭：未设 env 时零 import / 零 IO / 零 shape 读取、无设备差异；
   仍存在**既定数量的早退 Python 调用**（当前 W1R = 每次 GMM 两次），文档须据实写明，
   **不得声称宿主行为字节等价**（宿主文件文本必然不同）。
6. 运行树只放部署副本，进 `apply-runtime-patches.sh` 分层链；不得只手改运行树。
7. 证据（fixture / 采集物 / receipt）只进 `/root/包体恢复/records`，不可变、只追加。

## 现有提案

| 子目录 | 能力 | 状态 |
|---|---|---|
| `W1R-gmm-tuning-config/` | GMM `tuning_config` + counts dump（default-off） | P0 全门 PASS；**blocked**：blocker `fixture-unreachable-fullgraph-decode`（active FULL graph 拿不到 M=128 真实 counts）；运行树已回滚、自动补丁链已移除 |
