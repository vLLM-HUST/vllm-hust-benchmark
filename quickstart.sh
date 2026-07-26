#!/usr/bin/env bash
# vllm-hust-benchmark: Quick Start
#
# Simplified one-shot installer for the benchmark repo:
#   1. Install dev validation dependencies from requirements-dev.txt
#   2. Install the package itself in editable mode (so pytest can import it)
#   3. Wire git hooks (pre-commit / pre-push / post-commit) into .git/hooks/
#
# Unlike vllm-hust-website's quickstart, this one:
#   - does not run a dev/standard mode switch (no PyPI publishing flow here)
#   - does not clean up pre-existing prefixed packages (benchmark repo does
#     not own a published package namespace)
#   - supports a single skip flag: --skip-hooks

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
TEMPLATE_DIR="$PROJECT_ROOT/hooks"
VALIDATION_REQUIREMENTS_FILE="$PROJECT_ROOT/requirements-dev.txt"

SKIP_HOOKS="false"

show_help() {
    echo "vllm-hust-benchmark Quick Start"
    echo ""
    echo "用法:"
    echo "  ./quickstart.sh                 安装 dev 依赖、可编辑安装本仓包、并安装 git hooks"
    echo "  ./quickstart.sh --skip-hooks   跳过 Git hooks 安装"
    echo "  ./quickstart.sh --help          显示帮助"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-hooks)
                SKIP_HOOKS="true"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 未知参数: $1${NC}"
                echo ""
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

detect_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ 未找到可用 Python 命令（python3/python）${NC}"
        exit 1
    fi
    PIP_CMD=("$PYTHON_CMD" -m pip)
}

run_with_diagnostics() {
    local label="$1"
    shift
    local log_file
    log_file=$(mktemp)

    if "$@" >"$log_file" 2>&1; then
        rm -f "$log_file"
        return 0
    fi

    echo -e "${RED}❌ ${label} 失败${NC}"
    echo -e "${YELLOW}--- 详细错误日志开始 ---${NC}"
    cat "$log_file"
    echo -e "${YELLOW}--- 详细错误日志结束 ---${NC}"
    rm -f "$log_file"
    return 1
}

install_validation_dependencies() {
    if [ ! -f "$VALIDATION_REQUIREMENTS_FILE" ]; then
        echo -e "${YELLOW}⚠️ 未找到 requirements-dev.txt，跳过本地校验依赖安装${NC}"
        return 0
    fi

    echo -e "${BLUE}🧪 安装本地校验依赖（requirements-dev.txt）${NC}"
    run_with_diagnostics "安装本地校验依赖" "${PIP_CMD[@]}" install -r "$VALIDATION_REQUIREMENTS_FILE"
    echo -e "${GREEN}✓ 本地校验依赖安装完成${NC}"
}

install_editable_package() {
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo -e "${BLUE}📦 可编辑安装本仓 Python 包（-e .）${NC}"
        run_with_diagnostics "可编辑安装本仓包" "${PIP_CMD[@]}" install -e .
        echo -e "${GREEN}✓ 可编辑安装完成${NC}"
    else
        echo -e "${YELLOW}⚠️ 未找到 pyproject.toml，跳过可编辑安装${NC}"
    fi
}

install_hooks() {
    if [ ! -d "$HOOKS_DIR" ]; then
        echo -e "${YELLOW}⚠️ .git directory not found, skipping hooks installation${NC}"
        return 0
    fi

    if [ -f "$TEMPLATE_DIR/pre-commit" ]; then
        ln -sf "../../hooks/pre-commit" "$HOOKS_DIR/pre-commit"
        chmod +x "$HOOKS_DIR/pre-commit"
        echo -e "${GREEN}✓ Installed pre-commit hook${NC}"
    else
        echo -e "${YELLOW}⚠️ pre-commit template not found, skipping${NC}"
    fi

    if [ -f "$TEMPLATE_DIR/pre-push" ]; then
        ln -sf "../../hooks/pre-push" "$HOOKS_DIR/pre-push"
        chmod +x "$HOOKS_DIR/pre-push"
        echo -e "${GREEN}✓ Installed pre-push hook${NC}"
    else
        echo -e "${YELLOW}⚠️ pre-push template not found, skipping${NC}"
    fi

    if [ -f "$TEMPLATE_DIR/post-commit" ]; then
        ln -sf "../../hooks/post-commit" "$HOOKS_DIR/post-commit"
        chmod +x "$HOOKS_DIR/post-commit"
        echo -e "${GREEN}✓ Installed post-commit hook${NC}"
    else
        echo -e "${YELLOW}⚠️ post-commit template not found, skipping${NC}"
    fi
}

main() {
    parse_args "$@"
    detect_python

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}vllm-hust-benchmark Quick Start${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📂 Project root: ${NC}${PROJECT_ROOT}"
    echo ""

    echo -e "${YELLOW}${BOLD}Step 1/3: Installing local validation dependencies${NC}"
    install_validation_dependencies
    echo ""

    echo -e "${YELLOW}${BOLD}Step 2/3: Installing editable package${NC}"
    install_editable_package
    echo ""

    echo -e "${YELLOW}${BOLD}Step 3/3: Installing Git hooks${NC}"
    if [ "$SKIP_HOOKS" = "true" ]; then
        echo -e "${YELLOW}⚠️ 已跳过 hooks 安装（--skip-hooks）${NC}"
    else
        install_hooks
    fi
    echo ""

    echo -e "${YELLOW}${BOLD}Next: Local CI-parity validation${NC}"
    echo -e "${BLUE}Run:${NC} ./scripts/validate-local.sh"
    echo ""
    echo -e "${GREEN}${BOLD}✓ Setup Complete${NC}"
}

main "$@"