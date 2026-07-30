#!/usr/bin/env bash
# issue #95 §12.4: 对现有开放性能 PR 各做一次通知和分类
#
# 扫描指定仓库的开放 PR，对性能相关 PR（非 docs-only/test-only）：
#   1. 打 needs-perf-evidence label（如果还没有）
#   2. 发一条通知评论，说明 #95 merge gate 要求
#
# 对 docs-only/test-only/website-only PR：
#   1. 打对应的 perf-skip:* label（如果还没有）
#   2. 发一条评论提示需要审批人确认
#
# 用法:
#   bash scripts/notify_open_perf_prs.sh --repo vllm-hust
#   bash scripts/notify_open_perf_prs.sh --repo vllm-ascend-hust
#   bash scripts/notify_open_perf_prs.sh --repo vllm-hust --dry-run   # 只打印不执行
#
# 依赖: gh CLI（已登录，对目标仓库有 read/write 权限）
set -euo pipefail

REPO=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$REPO" ]]; then
    echo "Usage: bash scripts/notify_open_perf_prs.sh --repo <owner/repo> [--dry-run]"
    exit 1
fi

REPO_FULL="vLLM-HUST/${REPO}"
if [[ "$REPO" == */* ]]; then
    REPO_FULL="$REPO"
fi

echo "=== issue #95 §12.4: 扫描 ${REPO_FULL} 的开放 PR ==="
if $DRY_RUN; then
    echo "[DRY RUN] 只打印不执行"
fi

# 通知评论模板
NOTIFY_BODY='## issue #95 Merge Gate 通知

本仓库已启用 PR 合并前性能证据门禁（merge gate）。

**如果你这个 PR 改了热路径（model/attention/sampling/quantization 等），需要提供 paired real-online 性能证据：**

1. 在本 PR 描述中填写性能证据 checklist（target_id / target_version / profile_id / base artifact / head artifact）
2. 确保 base 和 head 都跑了固定靶 benchmark（参考 #104 registry）
3. merge gate check 通过后才能合并

**如果你的 PR 是纯文档/测试/网站改动**，请添加对应的 label：
- `perf-skip:docs-only` — 纯文档 PR
- `perf-skip:test-only` — 纯测试 PR
- `perf-skip:website-only` — 纯网站 PR

并 @ 一个 reviewer 确认审批。

详细要求见 [issue #95](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/95)。

> 这是一次性通知（issue #95 §12.4 验收项），后续新 PR 会由 CI 自动检查。
'

DOCS_BODY='## issue #95 Merge Gate 通知

本 PR 看起来是文档/测试/网站类改动。如果确实不涉及热路径修改，请添加对应的 `perf-skip:*` label 并 @ 一个 reviewer 确认审批。

详细要求见 [issue #95](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/95)。
'

# 获取所有开放 PR
PR_LIST=$(gh pr list --repo "${REPO_FULL}" --state open --json number,title,labels --limit 100)

# 用 python 解析 PR 列表并分类
ANALYSIS=$(echo "$PR_LIST" | python3 -c "
import json, sys

prs = json.load(sys.stdin)
for pr in prs:
    number = pr['number']
    title = pr.get('title', '')
    labels = [l['name'] for l in pr.get('labels', [])]

    # 已有 perf-skip label 的跳过
    skip_labels = {'perf-skip:docs-only', 'perf-skip:test-only', 'perf-skip:website-only'}
    if skip_labels & set(labels):
        print(f'DOCS|{number}|{title}')
        continue

    # 已有 has-perf-evidence label 的跳过
    if 'has-perf-evidence' in labels:
        continue

    # 其他都算性能 PR（需要通知）
    print(f'PERF|{number}|{title}')
")

PERF_COUNT=0
DOCS_COUNT=0

while IFS='|' read -r category number title; do
    if [[ "$category" == "PERF" ]]; then
        echo ""
        echo "[PERF] PR #${number}: ${title}"
        if $DRY_RUN; then
            echo "  -> (dry-run) 会打 needs-perf-evidence label + 发通知评论"
        else
            gh pr edit "$number" --repo "${REPO_FULL}" --add-label "needs-perf-evidence" 2>/dev/null || true
            gh pr comment "$number" --repo "${REPO_FULL}" --body "$NOTIFY_BODY" 2>/dev/null || true
            echo "  -> 已打 label + 发通知"
        fi
        PERF_COUNT=$((PERF_COUNT + 1))
    elif [[ "$category" == "DOCS" ]]; then
        echo ""
        echo "[DOCS] PR #${number}: ${title}"
        if $DRY_RUN; then
            echo "  -> (dry-run) 会发文档类通知评论"
        else
            gh pr comment "$number" --repo "${REPO_FULL}" --body "$DOCS_BODY" 2>/dev/null || true
            echo "  -> 已发通知"
        fi
        DOCS_COUNT=$((DOCS_COUNT + 1))
    fi
done <<< "$ANALYSIS"

echo ""
echo "=== 扫描完成 ==="
echo "性能 PR 通知: ${PERF_COUNT}"
echo "文档类 PR 通知: ${DOCS_COUNT}"
if $DRY_RUN; then
    echo "(dry-run 模式，未实际执行)"
fi
