# NPU Runner Watchdog 运维闭环（Issue #125）

Issue #125 是 poy-180 NPU runner watchdog 的固定告警通道。监控范围 NPU 0–4， 每 30 秒扫描一次。本目录维护这个 watchdog
的**审计闭环**：把每次处置记录成 结构化 JSONL 审计事件、按事件去重告警、并保证宿主机审计与 GitHub 摘要一致。

本仓库不承载 watchdog 守护进程本身（它运行在私有 runner 宿主上），而是提供：

- **事件
  schema**：[schemas/npu_watchdog_event_v1.schema.json](../schemas/npu_watchdog_event_v1.schema.json)
- **核心逻辑**：[src/vllm_hust_benchmark/watchdog_ops.py](../src/vllm_hust_benchmark/watchdog_ops.py) ——
  判定/结果字段、告警去重、摘要渲染、一致性校验、NPU 4 策略。
- **可部署参考守护进程**：[scripts/run_npu_runner_watchdog.py](../scripts/run_npu_runner_watchdog.py)
- **样例事件**：[npu-runner-watchdog-sample-events.jsonl](npu-runner-watchdog-sample-events.jsonl)
- **测试**：[tests/test_watchdog_ops.py](../tests/test_watchdog_ops.py)、
  [tests/test_run_npu_runner_watchdog.py](../tests/test_run_npu_runner_watchdog.py)

## 事件记录（JSONL）

每次处置写入一行 JSONL 审计事件（宿主机 `/var/log/npu-runner-watchdog/events.jsonl`）， 字段与语义见 schema。关键字段：

| 字段                       | 含义                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `determination`            | 归属判定：`runner-job` / `sibling-container` / `unauthorized-container` / `unowned-process` |
| `action`                   | 采取的动作：`none` / `sigterm` / `sigkill`                                                  |
| `result`                   | 退出/清理结果：`no-op` / `exited-before-action` / `terminated` / `killed` / `not-found`     |
| `recovery_status`          | `open`（仍占用）或 `recovered`（已释放）                                                    |
| `dedup_key`                | `npuN/pidP/cmd<sha256前12位>`，同一事件的稳定标识                                           |
| `owner`                    | 运维负责人                                                                                  |
| `alert_suppressed`         | 本次事件是否被去重抑制（不重复告警）                                                        |
| `npu4_unregistered_runner` | 是否命中 NPU 4 无 runner 策略                                                               |
| `cmdline_sha256`           | 命令行 sha256（原始命令行**不**记录，避免泄露密钥）                                         |

## 验收清单（对应 Issue 评论区 2026-08-11 的要求）

- [x] `unauthorized-container` 判定字段与退出/清理结果
  - `determination` / `action` / `result` 全部显式记录。
- [x] 同一事件的告警去重、负责人和恢复状态
  - `dedup_key` 稳定标识；`should_alert` 仅在状态/结果变化或新事件时告警； `owner`、`recovery_status` 随每条记录。
- [x] 宿主机 JSONL 审计与 GitHub 摘要的一致性
  - GitHub 摘要由审计记录渲染（单一事实来源），`verify_summary_consistency` 可反解析已发布摘要校验一致。
- [x] NPU 4 无 runner 时的明确告警策略
  - `npu_is_policy_violation(4)` 恒为真，任何占用一律标记并告警。
- [x] 一次 clean runner 与一次违规进程的验收记录
  - `tests/test_run_npu_runner_watchdog.py::test_run_scan_clean_runner_plus_violating_process` 在
    dry-run 下同时产生 runner-job（`no-op`）与 unauthorized-container（`terminated`）两条记录。

## 验证命令

本地/CI 运行全部测试：

```bash
pytest tests/test_watchdog_ops.py tests/test_run_npu_runner_watchdog.py -v
```

在真实宿主上以 dry-run 跑一次扫描（不杀进程、不发布告警）：

```bash
sudo python3 scripts/run_npu_runner_watchdog.py --dry-run --once
```

用 mock 输入做确定性验收（不需要 NPU 硬件）：构造 `npu-smi` 文本与容器事实， 验证 clean runner + 违规进程两条记录的判定与去重行为：

```bash
python3 scripts/run_npu_runner_watchdog.py --dry-run --once \
  --npu-smi-text "$(cat /tmp/npu_smi.txt)" \
  --facts-file /tmp/facts.json \
  --log-dir /tmp/watchdog-accept
```

样例事件验证（全部符合 schema）：

```bash
python3 - <<'EOF'
import json
from pathlib import Path
from vllm_hust_benchmark.watchdog_ops import validate_event_record
for i, line in enumerate(Path("docs/npu-runner-watchdog-sample-events.jsonl").read_text().splitlines(), 1):
    errors = validate_event_record(json.loads(line))
    assert not errors, (i, errors)
print("all sample events conform")
EOF
```

部署（真实守护进程，systemd 或 cron 常驻，30s 间隔）：

```bash
sudo python3 scripts/run_npu_runner_watchdog.py --interval 30 \
  --log-dir /var/log/npu-runner-watchdog
```

## 归属契约

允许的 NPU 占用：注册 runner 容器 `poy-180-21rc-npu0`~`npu3` 内执行的作业， 以及带 `org.vllm-hust.runner=<runner>`
标签、仅映射对应 `/dev/davinciN` 的兄弟 容器（契约细节见 [RUNNER_DOCKER_OWNERSHIP.md](RUNNER_DOCKER_OWNERSHIP.md)）。
其他占用进程先 SIGTERM，5 秒后仍占用则 SIGKILL。
