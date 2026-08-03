#!/usr/bin/env bash
# issue #95 §12.4: 对现有开放性能 PR 各做一次通知和分类
#
# 扫描指定仓库的开放 PR，对性能相关 PR（非 docs-only/test-only）：
#   1. 打 needs-perf-evidence label（如果还没有）
#   2. 发一条通知评论，说明 #95 merge gate 要求（幂等：通过 label + 评论标识判断）
#
# 对 docs-only/test-only/website-only PR：
#   1. 打对应的 perf-skip:* label（如果还没有）
#   2. 发一条评论提示需要审批人确认（幂等：通过评论标识判断）
#
# 幂等性：
#   - PERF 类：若 PR 已有 needs-perf-evidence label，则跳过评论（label 即通知凭证）
#   - DOCS 类：若 PR 最近评论中已包含 <!-- issue-95-perf-notify-v1 --> 标识，则跳过
#   - 重复运行不会产生重复评论
#
# 错误处理：
#   - 单个 PR 失败不会中断整体流程（记录 WARN 后 continue）
#   - 错误信息输出到 stderr，便于排查
#
# 用法:
#   bash scripts/notify_open_perf_prs.sh --repo vllm-hust
#   bash scripts/notify_open_perf_prs.sh --repo vllm-ascend-hust
#   bash scripts/notify_open_perf_prs.sh --repo vllm-hust --dry-run   # 只打印不执行
#
# 依赖: gh CLI（已登录，对目标仓库有 read/write 权限）
set -uo pipefail

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

# 评论幂等标识（隐藏 HTML 注释，不会渲染显示）
NOTIFY_MARKER='<!-- issue-95-perf-notify-v1 -->'

echo "=== issue #95 §12.4: 扫描 ${REPO_FULL} 的开放 PR ==="
if $DRY_RUN; then
    echo "[DRY RUN] 只打印不执行"
fi

# 通知评论模板（含幂等标识）
NOTIFY_BODY="
## issue #95 Merge Gate 通知

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
"

DOCS_BODY="
## issue #95 Merge Gate 通知

本 PR 看起来是文档/测试/网站类改动。如果确实不涉及热路径修改，请添加对应的 `perf-skip:*` label 并 @ 一个 reviewer 确认审批。

详细要求见 [issue #95](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/95)。
"

# 检查 PR 是否已被通知过（评论中包含幂等标识）
# 参数: $1 = PR number
# 返回: 0 = 已通知, 1 = 未通知
has_been_notified() {
    local pr_number="$1"
    local comments
    comments=$(gh pr view "$pr_number" --repo "${REPO_FULL}" --json comments --jq '.comments[].body' 2>/dev/null) || return 1
    if echo "$comments" | grep -qF "${NOTIFY_MARKER}"; then
        return 0
    fi
    return 1
}

# 检查 PR 是否已有某个 label
# 参数: $1 = PR number, $2 = label name
has_label() {
    local pr_number="$1"
    local label="$2"
    gh pr view "$pr_number" --repo "${REPO_FULL}" --json labels --jq '.labels[].name' 2>/dev/null | grep -qxF "$label"
}

# 获取所有开放 PR
PR_LIST=$(gh pr list --repo "${REPO_FULL}" --state open --json number,title,labels --limit 100) || {
    echo "ERROR: 无法获取 ${REPO_FULL} 的开放 PR 列表" >&2
    exit 1
}

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
") || {
    echo "ERROR: 解析 PR 列表失败" >&2
    exit 1
}

PERF_COUNT=0
DOCS_COUNT=0
PERF_SKIP_COUNT=0
DOCS_SKIP_COUNT=0
ERROR_COUNT=0

while IFS='|' read -r category number title; do
    [[ -z "$category" ]] && continue
    if [[ "$category" == "PERF" ]]; then
        echo ""
        echo "[PERF] PR #${number}: ${title}"
        # 幂等检查：已有 needs-perf-evidence label 则跳过评论
        if ! $DRY_RUN && has_label "$number" "needs-perf-evidence"; then
            echo "  -> 已有 needs-perf-evidence label，跳过通知评论（幂等）"
            PERF_SKIP_COUNT=$((PERF_SKIP_COUNT + 1))
            continue
        fi
        if $DRY_RUN; then
            echo "  -> (dry-run) 会打 needs-perf-evidence label + 发通知评论"
        else
            # 打 label（gh 对已有 label 会自动忽略，不会报错）
            if ! gh pr edit "$number" --repo "${REPO_FULL}" --add-label "needs-perf-evidence" 2>&1 | sed 's/^/    label: /' >&2; then
                echo "  -> WARN: 打 label 失败，继续尝试发评论" >&2
                ERROR_COUNT=$((ERROR_COUNT + 1))
            fi
            # 发评论前再做一次幂等检查（防止并发运行）
            if has_been_notified "$number"; then
                echo "  -> 已存在通知评论，跳过（幂等）"
                PERF_SKIP_COUNT=$((PERF_SKIP_COUNT + 1))
                continue
            fi
            if ! gh pr comment "$number" --repo "${REPO_FULL}" --body "$NOTIFY_BODY" 2>&1 | sed 's/^/    comment: /' >&2; then
                echo "  -> WARN: 发通知评论失败" >&2
                ERROR_COUNT=$((ERROR_COUNT + 1))
            else
                echo "  -> 已打 label + 发通知"
            fi
        fi
        PERF_COUNT=$((PERF_COUNT + 1))
    elif [[ "$category" == "DOCS" ]]; then
        echo ""
        echo "[DOCS] PR #${number}: ${title}"
        # 幂等检查：已有通知评论则跳过
        if ! $DRY_RUN && has_been_notified "$number"; then
            echo "  -> 已存在通知评论，跳过（幂等）"
            DOCS_SKIP_COUNT=$((DOCS_SKIP_COUNT + 1))
            continue
        fi
        if $DRY_RUN; then
            echo "  -> (dry-run) 会发文档类通知评论"
        else
            if ! gh pr comment "$number" --repo "${REPO_FULL}" --body "$DOCS_BODY" 2>&1 | sed 's/^/    comment: /' >&2; then
                echo "  -> WARN: 发文档类通知评论失败" >&2
                ERROR_COUNT=$((ERROR_COUNT + 1))
            else
                echo "  -> 已发通知"
            fi
        fi
        DOCS_COUNT=$((DOCS_COUNT + 1))
    fi
done <<< "$ANALYSIS"

echo ""
echo "=== 扫描完成 ==="
echo "性能 PR 通知: ${PERF_COUNT}（跳过已通知: ${PERF_SKIP_COUNT}）"
echo "文档类 PR 通知: ${DOCS_COUNT}（跳过已通知: ${DOCS_SKIP_COUNT}）"
if [[ "$ERROR_COUNT" -gt 0 ]]; then
    echo "错误数: ${ERROR_COUNT}（详见 stderr）" >&2
fi
if $DRY_RUN; then
    echo "(dry-run 模式，未实际执行)"
fi

# 若有错误，以非零退出码退出（便于 CI 检测）
if [[ "$ERROR_COUNT" -gt 0 ]]; then
    exit 1
fi
