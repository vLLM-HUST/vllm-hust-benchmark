# issue #95 Layer 1 + Layer 2 (mock) + Layer 3 (跨仓库) — 验证方案

> **历史方案，已停止实施。** V4.6 不允许把正式性能验收配置为普通 PR required check；
> `.github/workflows/merge-gate.yml` 已删除。本文只保留设计历史，不能作为当前操作手册。

> 本文档记录 `feat/pr_95` 已交付的三层验证方法：
>
> - Layer 1（判定逻辑 + CLI）
> - Layer 2 mock 模式（CI 接线骨架）
> - Layer 3（跨仓库 PR 模板 + label + repository_dispatch 触发）
>
> 供后续 #103 就绪接入 real 模式时回归复用。

## 已交付物

| 交付物                            | 文件                                                   | 说明                                                                        |
| --------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------- |
| 判定逻辑                          | `src/vllm_hust_benchmark/merge_gate.py`                | `evaluate_merge_gate()` 10 步 fail-closed 管线                              |
| CLI 子命令                        | `src/vllm_hust_benchmark/cli.py` 的 `merge-gate-check` | pass→exit 0 / fail→exit 1 / skip→exit 0                                     |
| TDD 测试                          | `tests/test_merge_gate.py`                             | 30 个测试覆盖 issue §7.1 全部矩阵                                           |
| 结构化持久化                      | `write_decision_json()`                                | merge-gate-decision.json                                                    |
| mock artifact 生成器              | `scripts/generate_mock_merge_gate_artifacts.py`        | 10 场景 mock artifact，脱离 NPU 验证 CI 接线                                |
| CI workflow（mock）               | `.github/workflows/merge-gate.yml`                     | `mock-merge-gate-check` job：生成→判定→校验 exit code                       |
| CI workflow（real 骨架）          | `.github/workflows/merge-gate.yml`                     | `real-merge-gate-check` job：阻塞于 #103，骨架就绪                          |
| 本地模拟脚本                      | `scripts/simulate_merge_gate_workflow.sh`              | 不需 GitHub Actions 跑全套场景                                              |
| mock 生成器 TDD                   | `tests/test_mock_merge_gate_artifacts.py`              | 22 个测试覆盖 10 场景                                                       |
| 跨仓库触发（benchmark）           | `merge-gate.yml` 加 `repository_dispatch`              | 接收 core/Ascend 的 PR 事件触发判定                                         |
| 共享 PR 模板 partial              | `.github/perf-evidence-checklist.md`                   | 增强建议 #8：单点维护避免 core/Ascend 模板漂移                              |
| core PR 模板 + label + workflow   | vllm-hust `feat/pr_95` (d099008a0)                     | PR 模板加 perf-evidence section + 6 个受控 label + repository_dispatch 触发 |
| Ascend PR 模板 + label + workflow | vllm-ascend-hust `feat/pr_95` (513c15ef)               | 同 core                                                                     |

## 判定管线（10 步，fail-closed）

1. docs-only / test-only / website-only 受控 label → skip
1. base/head CI 状态非 accepted → fail closed（missing/cancelled/skipped/resource_busy 全挡）
1. artifact 读不出来 → fail（missing evidence）
1. data_source 不以 `real-online` 开头 → fail（挡 smoke/replay/derived/screenshot）
1. registry 匹配不到 profile → fail（3B perfgate ≠ 14B evidence）
1. PR 声明的 target_id 不在 registry → fail（registry hash mismatch）
1. active profile 字段校验：gpu_memory_utilization≠0.6 / max_model_len≠32768(文本)/30720(vision) →
   fail（config_drift 挡 0.9/0.92）
1. base/head spec_id 不一致 → fail（paired 不可比）
1. specialty 缺 spec 或 reason → fail
1. 全过 → pass

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

构造 artifact with `gpu_memory_utilization=0.9` 或 `max_model_len=30720`（文本线），期望 disposition=fail 且
reason 含 "config_drift"。 对应 TDD 测试：`TestConfigDrift::test_config_drift_0_9_blocked` 等 5 个。

### A.4 Lint / Format

```bash
ruff check src/vllm_hust_benchmark/merge_gate.py src/vllm_hust_benchmark/cli.py tests/test_merge_gate.py
ruff format --check src/vllm_hust_benchmark/merge_gate.py src/vllm_hust_benchmark/cli.py tests/test_merge_gate.py
# 期望: All checks passed! + 3 files already formatted
```

## B. CI 集成验证（Layer 2）

### B.1 mock 模式（已交付，不依赖 #103）

用 `generate_mock_merge_gate_artifacts.py` 产出 10 个场景的 mock artifact，验证 merge-gate-check 的 CI
接线正确性，不依赖真实 NPU。

**workflow**：`.github/workflows/merge-gate.yml`

- 触发：`workflow_dispatch` with `mode=mock` + `scenario=<场景|all>`
- job `mock-merge-gate-check`：生成 mock artifact → 调 merge-gate-check → 校验 expected vs actual
  disposition + exit code → 上传 decision.json

**本地模拟（不需 GitHub Actions）**：

```bash
# 跑全部 10 个场景
bash scripts/simulate_merge_gate_workflow.sh
# 期望: All scenarios passed, exit=0

# 跑单个场景
bash scripts/simulate_merge_gate_workflow.sh pass
```

**mock 场景清单**（全部 TDD 覆盖，22 个测试）：

| 场景                          | 期望 disposition | 验证点                       |
| ----------------------------- | ---------------- | ---------------------------- |
| `pass`                        | pass             | 合规 14B paired evidence     |
| `fail_config_drift`           | fail             | gpu_memory_utilization=0.9   |
| `fail_data_source`            | fail             | data_source=smoke-test       |
| `fail_unpaired_spec`          | fail             | base/head spec_id 不一致     |
| `fail_3b_not_14b`             | fail             | 3B perfgate ≠ 14B            |
| `fail_missing_artifact`       | fail             | head 缺失（CI missing）      |
| `skip_docs_only`              | skip             | docs-only label              |
| `specialty_valid`             | pass             | 2-chip + spec + reason       |
| `specialty_no_reason`         | fail             | specialty 缺 reason          |
| `fail_registry_hash_mismatch` | fail             | 声明 target_id 不在 registry |

**GitHub Actions 上跑 mock 模式**：

1. 进入仓库 Actions 页 → 选 `Merge Gate` workflow → Run workflow
1. `mode=mock`, `scenario=all`, Run
1. 检查 job `mock-merge-gate-check` 全绿 + artifact `merge-gate-mock-*` 含各场景 decision.json

### B.2 real 模式（待 #103 就绪）

real-mode job 骨架已在 workflow 中（`real-merge-gate-check`），阻塞于 #103。 #103 就绪后：

1. 在 step `Block on #103 readiness` 之前填入真实 benchmark 命令
1. checkout PR base + head commit
1. 在自托管 Ascend runner 上跑 paired benchmark
1. 产出 base/head run_leaderboard.json
1. 调用 `merge-gate-check` 判定

### B.3 注册 required check

在 GitHub branch protection 把 `merge-gate / mock-merge-gate-check`（mock 模式） 或
`merge-gate / real-merge-gate-check`（real 模式）注册为 required check。

### B.4 端到端演练（issue §7.2 6 个场景，#103 就绪后）

在 core 和 Ascend 仓库各用一个真实 PR 演练：

1. 无证据 PR 被挡
1. 配置漂移 artifact 被拒
1. 合规 14B/1chip/0.6/32768 通过
1. 同上三组在 Ascend 仓库
1. docs-only label 跳过
1. CI 日志打印判定详情

## C. 跨仓库验证（Layer 3，已交付）

### C.1 已交付物（3 个仓库）

| 仓库                | 分支       | commit    | 文件                                                                                                           |
| ------------------- | ---------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| vllm-hust-benchmark | feat/pr_95 | (本仓库)  | `.github/perf-evidence-checklist.md`（共享 partial）+ `merge-gate.yml` 加 `repository_dispatch`                |
| vllm-hust（core）   | feat/pr_95 | d099008a0 | `.github/PULL_REQUEST_TEMPLATE.md` + `scripts/setup_perf_labels.sh` + `.github/workflows/merge-gate-check.yml` |
| vllm-ascend-hust    | feat/pr_95 | 513c15ef  | 同 core                                                                                                        |

### C.2 跨仓库触发流程

```
core/Ascend PR (opened/synchronize/reopened/labeled)
    │
    ▼
core/Ascend workflow: merge-gate-check.yml
    │  用 BENCHMARK_REPO_TOKEN 向 benchmark 仓库发 repository_dispatch
    │  payload: pr_repo, pr_number, pr_head_sha, pr_base_sha, pr_labels
    ▼
benchmark workflow: merge-gate.yml (repository_dispatch 触发)
    │  mock 模式: 生成 mock artifact + 判定
    │  real 模式: #103 就绪后跑真实 paired benchmark + 判定
    │  输出 merge-gate-decision.json
    ▼
回写 check status 到 core/Ascend PR（待 #103 后接入 GitHub Checks API）
```

### C.3 手动部署步骤（手把手）

#### 步骤 1：创建 PAT 并配置 secret

1. 在 GitHub 创建 PAT（fine-grained），权限：

   - `vLLM-HUST/vllm-hust-benchmark`：Actions (write)
   - `vLLM-HUST/vllm-hust`：Contents (read), Pull requests (write)
   - `vLLM-HUST/vllm-ascend-hust`：Contents (read), Pull requests (write)

1. 在 core 和 Ascend 仓库分别添加 secret：

   ```
   仓库 Settings → Secrets and variables → Actions → New repository secret
   Name: BENCHMARK_REPO_TOKEN
   Value: <上面创建的 PAT>
   ```

#### 步骤 2：创建受控 label

```bash
# 在 core 仓库
cd /path/to/vllm-hust
gh auth login  # 确保已登录
bash scripts/setup_perf_labels.sh

# 在 Ascend 仓库
cd /path/to/vllm-ascend-hust
bash scripts/setup_perf_labels.sh
```

验证：`gh label list | grep perf-skip` 应看到 3 个 perf-skip label + 3 个状态 label。

#### 步骤 3：注册 required check

在 core 和 Ascend 仓库的 branch protection：

```
Settings → Branches → main → Edit → Require status checks to pass before merging
  Required status checks: Merge Gate Check / trigger-benchmark-merge-gate
```

#### 步骤 4：推送分支 + 开 PR

```bash
# core
cd /path/to/vllm-hust && git push -u origin feat/pr_95

# Ascend
cd /path/to/vllm-ascend-hust && git push -u origin feat/pr_95

# benchmark
cd /path/to/vllm-hust-benchmark && git push -u origin feat/pr_95
```

### C.4 端到端演练（issue §7.2，需 #103 就绪后做）

| 演练 | 仓库   | 操作                               | 预期                                              |
| ---- | ------ | ---------------------------------- | ------------------------------------------------- |
| 1    | core   | 开无证据 PR                        | check fail，不能合并                              |
| 2    | core   | 开 PR 带配置漂移 artifact          | check fail                                        |
| 3    | core   | 开 PR 带合规 14B evidence          | check pass，可合并                                |
| 4    | Ascend | 同上三组                           | 同上                                              |
| 5    | 任一   | 开 PR 带 perf-skip:docs-only label | check skip，可合并                                |
| 6    | 任一   | 查看 CI 日志                       | 含 artifact/workload/model/hardware/server params |

### C.5 已知限制

- **check status 回写**：当前 repository_dispatch 触发后，benchmark 仓库的判定结果 还不能自动回写到 core/Ascend 的 PR check
  status。需要接入 GitHub Checks API （`POST /repos/{owner}/{repo}/check-runs`）。这是 Layer 3 的后续增强，不阻塞
  判定逻辑本身。
- **real 模式**：阻塞于 #103（自托管 Ascend runner）。mock 模式可先验证接线。

## 已知阻塞

- **#103 自托管 Ascend runner 未确认**：阻塞 Layer 2 端到端实跑，但不阻塞 Layer 1 判定工具。
- **#105 未合并 main**：`feat/pr_95` 已 fast-forward 引入 #105 代码作为基线，#105 合并后 `feat/pr_95` rebase 会干净落下。

## 验证回归清单（每次 Layer 2/3 改动后重跑）

- [ ] `pytest tests/test_merge_gate.py -v` → 30 passed
- [ ] `pytest tests/test_mock_merge_gate_artifacts.py -v` → 22 passed
- [ ] `pytest tests/` → 全套绿（当前 534 passed）
- [ ] `bash scripts/simulate_merge_gate_workflow.sh` → All scenarios passed
- [ ] CLI PASS/FAIL/SKIP 三场景 exit code 正确
- [ ] ruff lint + format 绿
