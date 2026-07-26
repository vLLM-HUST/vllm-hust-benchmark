#!/usr/bin/env bash

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_REQUIREMENTS_FILE="$PROJECT_ROOT/requirements-dev.txt"

RUN_PRE_COMMIT=true
RUN_TESTS=true
RUN_HOOK_TEMPLATE_CHECKS=true

PYTHON_BIN=""
PRE_COMMIT_CMD=()
PYTEST_CMD=()

show_help() {
    cat <<'EOF'
Usage: ./scripts/validate-local.sh [options]

Run the local CI-parity checks for vllm-hust-benchmark.

Options:
  --skip-pre-commit       Skip pre-commit run --all-files
  --skip-tests            Skip python -m pytest -q tests
  --skip-hook-templates   Skip hook-template verification
  --help, -h              Show this help message
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-pre-commit)
                RUN_PRE_COMMIT=false
                ;;
            --skip-tests)
                RUN_TESTS=false
                ;;
            --skip-hook-templates)
                RUN_HOOK_TEMPLATE_CHECKS=false
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Unknown argument: $1${NC}"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

resolve_python311() {
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="python3.11"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        local version
        version=$(python3 - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
        if [[ "$version" == "3.11" ]]; then
            PYTHON_BIN="python3"
            return 0
        fi
    fi

    echo -e "${RED}❌ Python 3.11 is required to match the CI pre-commit environment.${NC}"
    echo -e "${YELLOW}👉 Install python3.11 or use an environment where python3 resolves to 3.11.${NC}"
    exit 1
}

resolve_pre_commit_cmd() {
    if command -v pre-commit >/dev/null 2>&1; then
        PRE_COMMIT_CMD=(pre-commit)
        return 0
    fi

    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("pre_commit") else 1)
PY
    then
        PRE_COMMIT_CMD=("$PYTHON_BIN" -m pre_commit)
        return 0
    fi

    if command -v uv >/dev/null 2>&1; then
        PRE_COMMIT_CMD=(uv run --python 3.11 --with-requirements "$DEV_REQUIREMENTS_FILE" pre-commit)
        return 0
    fi

    echo -e "${RED}❌ pre-commit is not available.${NC}"
    echo -e "${YELLOW}👉 Install validation dependencies with $PYTHON_BIN -m pip install -r requirements-dev.txt, or install uv for the fallback path.${NC}"
    exit 1
}

resolve_pytest_cmd() {
    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("pytest") and importlib.util.find_spec("jsonschema") else 1)
PY
    then
        PYTEST_CMD=("$PYTHON_BIN" -m pytest)
        return 0
    fi

    if command -v uv >/dev/null 2>&1; then
        PYTEST_CMD=(uv run --python 3.11 --with-requirements "$DEV_REQUIREMENTS_FILE" pytest)
        return 0
    fi

    echo -e "${RED}❌ pytest/jsonschema are not available in the current environment.${NC}"
    echo -e "${YELLOW}👉 Install validation dependencies with $PYTHON_BIN -m pip install -r requirements-dev.txt, or install uv for the fallback path.${NC}"

    exit 1
}

run_pre_commit_checks() {
    echo -e "${BLUE}Step 1/3: Run pre-commit --all-files${NC}"
    local current_branch=""
    current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
    if [[ "$current_branch" == "main" ]]; then
        SKIP=no-commit-to-branch "${PRE_COMMIT_CMD[@]}" run --all-files
    else
        "${PRE_COMMIT_CMD[@]}" run --all-files
    fi
    echo -e "${GREEN}✓ pre-commit checks passed${NC}"
}

run_unit_tests() {
    echo -e "${BLUE}Step 2/3: Run python -m pytest -q tests${NC}"
    "${PYTEST_CMD[@]}" -q tests
    echo -e "${GREEN}✓ unit tests passed${NC}"
}

verify_hook_templates() {
    echo -e "${BLUE}Step 3/3: Verify repository hook templates${NC}"
    test -f hooks/pre-commit
    test -f hooks/pre-push
    bash -n hooks/pre-commit
    bash -n hooks/pre-push
    bash -n scripts/validate-local.sh
    grep -q 'ln -sf "../../hooks/pre-commit"' quickstart.sh
    grep -q 'ln -sf "../../hooks/pre-push"' quickstart.sh
    echo -e "${GREEN}✓ hook templates verified${NC}"
}

main() {
    parse_args "$@"
    cd "$PROJECT_ROOT"

    echo -e "${CYAN}Running local CI-parity validation in ${PROJECT_ROOT}${NC}"
    resolve_python311

    if [[ "$RUN_PRE_COMMIT" == true ]]; then
        resolve_pre_commit_cmd
        run_pre_commit_checks
    fi

    if [[ "$RUN_TESTS" == true ]]; then
        resolve_pytest_cmd
        run_unit_tests
    fi

    if [[ "$RUN_HOOK_TEMPLATE_CHECKS" == true ]]; then
        verify_hook_templates
    fi

    echo -e "${GREEN}✅ Local validation completed successfully${NC}"
}

main "$@"
