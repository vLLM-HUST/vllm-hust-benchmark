# issue #95 Layer 1 判定逻辑 — 验证方案

> 本文档记录 `feat/pr_95` 已交付的 Layer 1（判定逻辑 + CLI）验证方法，供后续 Layer 2/3 接入时回归复用。

## 已交付物

| 交付物 | 文件 | 说明 |
|--------|------|------|
| 判定逻辑 | `src/vllm_hust_benchmark/merge_gate.py` | `evaluate_merge_gate()` 10 步 fail-closed 管线 |
| CLI 子命令 | `src/vllm_hust_benchmark/cli.py` 的 `merge-gate-check` | pass→exit 0 / fail→exit 1 / skip→exit 0 |
| TDD 测试 | `tests/test_merge_gate.py` | 30 个测试覆盖 issue §7.1 全部矩阵 |
| 结构化持久化 | `write_decision_json()` | merge-gate-decision.json |

## 判定管线（10 步，fail-closed）

1. docs-only / test-only / website-only 受控 label → skip
2. base/head CI 状态非 accepted → fail closed（missing/cancelled/skipped/resource_busy 全挡）
3. artifact 读不出来 → fail（missing evidence）
4. data_source 不以 `real-online` 开头 → fail（挡 smoke/replay/derived/screenshot）
5. registry 匹配不到 profile → fail（3B perfgate ≠ 14B evidence）
6. PR 声明的 target_id 不在 registry → fail（registry hash mismatch）
7. active profile 字段校验：gpu_memory_utilization≠0.6 / max_model_len≠32768(文本)/30720(vision) → fail（config_drift 挡 0.9/0.92）
8. base/head spec_id 不一致 → fail（paired 不可比）
9. specialty 缺 spec 或 reason → fail
10. 全过 → pass

## A. 本地自动验证

### A.1 单元测试（TDD）
```bash
.venv/bin/python -m pytest tests/test_merge_gate.py -v
# 期望: 30 passed
```

### A.2 端到端 CLI 验证

#### A.2.1 PASS 场景（合规 14B paired evidence）
```bash
.venv/bin/python -m vllm_hust_benchmark.cli merge-gate-check \
  --base-artifact submissions/codex-latest-main-tf5-random-random-online-1chip-20260726T074647Z/run_leaderboard.json \
  --head-artifact submissions/codex-latest-main-tf5-random-random-online-1chip-20260726T074647Z/run_leaderboard.json \
  --repo vllm-hust --pr-number 193 --head-sha abc1234 --base-sha def5678 \
  --decision-output /tmp/decision.json
# 期望: disposition=pass, exit=0
# decision.json 应含 target_id=official-ascend-jan-2026-v0.18.0, profile_id=core-text-14b
```

#### A.2.2 FAIL 场景（fail closed — 缺 artifact）
```bash
.venv/bin/python -m vllm_hust_benchmark.cli merge-gate-check \
  --base-status missing --head-status missing \
  --repo vllm-hust --pr-number 193 --head-sha abc --base-sha def
# 期望: disposition=fail, exit=1
# 日志应含 "base artifact not accepted (ci_status=missing): paired evidence incomplete, fail closed"
```

#### A.2.3 SKIP 场景（docs-only label）
```bash
.venv/bin/python -m vllm_hust_benchmark.cli merge-gate-check \
  --base-status missing --head-status missing \
  --repo vllm-hust --pr-number 193 --head-sha abc --base-sha def \
  --labels "perf-skip:docs-only"
# 期望: disposition=skip, exit=0
# 日志应含 "PR skipped via controlled label: perf-skip:docs-only"
```

### A.3 配置漂移专项验证
构造 artifact with `gpu_memory_utilization=0.9` 或 `max_model_len=30720`（文本线），期望 disposition=fail 且 reason 含 "config_drift"。
对应 TDD 测试：`TestConfigDrift::test_config_drift_0_9_blocked` 等 5 个。

### A.4 Lint / Format
```bash
ruff check src/vllm_hust_benchmark/merge_gate.py src/vllm_hust_benchmark/cli.py tests/test_merge_gate.py
ruff format --check src/vllm_hust_benchmark/merge_gate.py src/vllm_hust_benchmark/cli.py tests/test_merge_gate.py
# 期望: All checks passed! + 3 files already formatted
```

## B. CI 集成验证（Layer 2，待 #103 就绪后执行）

> #103 自托管 Ascend runner 未就绪，无法端到端跑 paired benchmark。以下是 runner 就绪后的接入步骤：

### B.1 在自托管 runner 上跑 paired benchmark
- base commit（PR fork point）+ head commit（PR 最新 push）各跑一次
- 产出两个 `run_leaderboard.json`

### B.2 GitHub Actions workflow 调用判定工具
```yaml
- name: Merge gate check
  run: |
    .venv/bin/python -m vllm_hust_benchmark.cli merge-gate-check \
      --base-artifact ${{ steps.base.outputs.artifact }} \
      --head-artifact ${{ steps.head.outputs.artifact }} \
      --repo ${{ github.repository }} \
      --pr-number ${{ github.event.pull_request.number }} \
      --head-sha ${{ github.event.pull_request.head.sha }} \
      --base-sha ${{ github.event.pull_request.base.sha }} \
      --decision-output merge-gate-decision.json
  # exit 1 让 GitHub Actions job 失败 → required check = fail → 阻塞合并
```

### B.3 注册 required check
在 GitHub branch protection 把 `merge-gate / performance-evidence` 注册为 required check（job 名对应 workflow 的 job name）。

### B.4 端到端演练（issue §7.2 6 个场景）
在 core 和 Ascend 仓库各用一个真实 PR 演练：
1. 无证据 PR 被挡
2. 配置漂移 artifact 被拒
3. 合规 14B/1chip/0.6/32768 通过
4. 同上三组在 Ascend 仓库
5. docs-only label 跳过
6. CI 日志打印判定详情

## C. 跨仓库验证（Layer 3，待用户确认后执行）

core/Ascend 仓库的 PR 模板 + required check 注册涉及跨仓库改动，按用户要求需先确认。

## 已知阻塞

- **#103 自托管 Ascend runner 未确认**：阻塞 Layer 2 端到端实跑，但不阻塞 Layer 1 判定工具。
- **#105 未合并 main**：`feat/pr_95` 已 fast-forward 引入 #105 代码作为基线，#105 合并后 `feat/pr_95` rebase 会干净落下。

## 验证回归清单（每次 Layer 2/3 改动后重跑）

- [ ] `pytest tests/test_merge_gate.py -v` → 30 passed
- [ ] `pytest tests/` → 全套绿
- [ ] CLI PASS/FAIL/SKIP 三场景 exit code 正确
- [ ] ruff lint + format 绿
