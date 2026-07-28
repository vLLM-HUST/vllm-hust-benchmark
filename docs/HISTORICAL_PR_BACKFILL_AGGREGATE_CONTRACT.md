# Historical-PR-Backfill 聚合准入契约

## 背景

`scripts/backfill_single_gpu.py aggregate` 在聚合 `submissions/` 到 `leaderboard-data/snapshots/` 时，会调用
website 侧 `vllm-hust-website/scripts/aggregate_results.py` 的
`plugin_commit_mismatch_rejection_reason` 过滤器。该过滤器原本无差别拒绝所有「同 `metadata.git_commit` 但
`runtime_provenance.plugin.commit` 不同」的 entry，目的是防止同一个 vLLM-HUST engine binary 在 trend chart 上被渲染成多个
x 轴点。

这条规则对主线 backfill 是合理的，但会误伤 **historical-PR-backfill** 数据：PR#66 / PR#70 / PR#77 是三个独立的 PR，每个 PR 用不同的
`vllm-ascend-hust` plugin commit 对同一个 vLLM-HUST engine commit 做对比测试。它们是有意的跨 PR 对比数据，不是 backfill
错误。原规则会把 PR#66 / PR#70 的 entry 全部从 public snapshot 里丢掉（实测 190 个 entry 被静默跳过），导致 compare cards 缺数据。

## 契约

### 1. 数据源标识（benchmark repo 侧）

所有 historical-PR-backfill 提交 MUST 在 `run_leaderboard.json` 的 `metadata.data_source` 字段写入字符串
`"real-online-historical-pr-backfill"`。

- 由 `scripts/backfill_historical_pr_benchmarks.py` 在入库时自动写入
- 防御测试：`tests/test_historical_pr_backfill_data_source.py` 扫描 `submissions/historical-pr-*` 目录，发现任何
  entry 缺失该 marker 立即 fail

### 2. 聚合豁免（website repo 侧）

`vllm-hust-website/scripts/aggregate_results.py::plugin_commit_mismatch_rejection_reason` 在检查 plugin
commit 一致性前，先读取 `metadata.data_source`：

- 若 `data_source == "real-online-historical-pr-backfill"` → 跳过 canonical-plugin-commit 检查，entry 进入
  public snapshot
- 否则 → 维持原规则，同 git_commit 不同 plugin commit 的 entry 被 reject

### 3. 共存规则（benchmark repo 侧）

historical-PR-backfill entry 之间仍受 `_find_superseded_coexistence_conflicts` 约束（见 `spec.md` §
"superseded 不得与新 OK 点共存"），但分组逻辑是：

1. 一级分组：`build_series_signature`（model|hardware|precision|workload|chip_count|config_type|engine|engine_version）
1. 二级分组：`(engine_commit, plugin_commit)` 取自 `metadata.runtime_provenance`

只有同 signature 且同 `(engine_commit, plugin_commit)` 的多条 entry 才需要 `supersedes` 标注。不同 plugin commit 的
PR 对比测试合法共存，不触发冲突。

## 验证

| 场景                                               | 期望行为                           | 测试                                                                          |
| -------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------- |
| PR#66/70/77 不同 plugin commit                     | 共存于 leaderboard + compare cards | `test_aggregate_results_keeps_historical_pr_backfill_despite_plugin_mismatch` |
| 主线 backfill 不同 plugin commit                   | 被 reject（保持 trend chart 诚实） | `test_plugin_commit_mismatch_still_rejects_non_historical_backfill`           |
| historical-pr-backfill entry 缺 data_source marker | 防御测试 fail                      | `test_every_historical_pr_backfill_submission_carries_expected_data_source`   |
| 同 (engine_commit, plugin_commit) 多次运行         | 需 supersedes 标注或归档           | `test_same_code_combo_still_conflicts`                                        |

## 相关文件

- benchmark repo:
  - \[src/vllm_hust_benchmark/integration.py\](file:///Users/paul/company/hust/vllm-hust-benchmark/src/vllm_hust_benchmark/integration.py)
    — `_find_superseded_coexistence_conflicts`、`_extract_code_combo`
  - \[tests/test_historical_pr_backfill_data_source.py\](file:///Users/paul/company/hust/vllm-hust-benchmark/tests/test_historical_pr_backfill_data_source.py)
    — data_source marker 防御测试
  - \[tests/test_integration_aggregation_gate.py\](file:///Users/paul/company/hust/vllm-hust-benchmark/tests/test_integration_aggregation_gate.py)
    — 共存冲突检测测试
- website repo:
  - `scripts/aggregate_results.py` — `plugin_commit_mismatch_rejection_reason` 含
    historical-pr-backfill 豁免
  - `tests/test_aggregate_results.py` — 豁免与保留规则的双重测试

## 故障排查

如果 aggregate 输出 `skipped invalid public entries: N`：

1. 看reject reason 是不是 `plugin commit mismatch`
1. 如果是，看被 reject 的 entry 的 `metadata.data_source`：
   - `real-online-historical-pr-backfill` → website 侧豁免逻辑失效，检查
     `plugin_commit_mismatch_rejection_reason` 是否被改回
   - `vllm-hust-benchmark` 或其他 → 主线 backfill 的 plugin commit 不一致，需要数据修复（统一 plugin commit 或重新入库）
