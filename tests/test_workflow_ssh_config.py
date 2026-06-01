from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATHS = [
    ".github/workflows/ci.yml",
    ".github/workflows/push-to-hf.yml",
    ".github/workflows/run-ascend-context-length-current-vs-official.yml",
    ".github/workflows/run-official-ascend-baselines.yml",
]


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS)
def test_workflows_use_standard_github_ssh_without_overwriting_config(
    workflow_path: str,
) -> None:
    workflow_text = (REPO_ROOT / workflow_path).read_text(encoding="utf-8")

    assert "BENCHMARK_CHECKOUT_USE_SSH_443" not in workflow_text
    assert "Configure GitHub SSH over 443" not in workflow_text
    assert "ssh.github.com" not in workflow_text
    assert 'git config --global url."ssh://git@ssh.github.com:443/".insteadOf https://github.com/' not in workflow_text

    assert "BENCHMARK_CHECKOUT_USE_SSH" in workflow_text
    assert "Configure GitHub SSH" in workflow_text
    assert 'if [[ ! -d "$ssh_dir" ]]; then' in workflow_text
    assert 'if [[ ! -f "$config_file" ]]; then' in workflow_text
    assert 'Host github.com' in workflow_text
    assert 'HostName github.com' in workflow_text
    assert 'IdentityFile ~/.ssh/github_actions' in workflow_text
    assert "printf '\\n%s\\n' \"$config_block\" >> \"$config_file\"" in workflow_text


def test_context_sweep_workflow_sets_soc_version_before_plugin_install() -> None:
    workflow_text = (
        REPO_ROOT / ".github/workflows/run-ascend-context-length-current-vs-official.yml"
    ).read_text(encoding="utf-8")

    soc_export = 'export SOC_VERSION="${SOC_VERSION:-ascend910b3}"'
    install_command = (
        'bash "${VLLM_ASCEND_HUST_REPO}/scripts/install_local_ascend_plugin.sh" '
        '"$VLLM_ASCEND_HUST_REPO"'
    )

    assert soc_export in workflow_text
    assert install_command in workflow_text
    assert workflow_text.index(soc_export) < workflow_text.index(install_command)


def test_context_sweep_workflow_repairs_current_runtime_before_plugin_install() -> None:
    workflow_text = (
        REPO_ROOT / ".github/workflows/run-ascend-context-length-current-vs-official.yml"
    ).read_text(encoding="utf-8")

    runtime_check = 'hust_ascend_manager_run runtime check \\'
    runtime_repair = 'hust_ascend_manager_run runtime repair \\'
    direct_vllm_install = (
        '"${CURRENT_RUNTIME_PYTHON}" -m pip install -e "$GITHUB_WORKSPACE/vllm-hust"'
    )
    install_command = (
        'bash "${VLLM_ASCEND_HUST_REPO}/scripts/install_local_ascend_plugin.sh" '
        '"$VLLM_ASCEND_HUST_REPO"'
    )

    assert runtime_check in workflow_text
    assert runtime_repair in workflow_text
    assert '--repo "$GITHUB_WORKSPACE/vllm-hust" \\' in workflow_text
    assert '--python "$CURRENT_RUNTIME_PYTHON"' in workflow_text
    assert '--skip-build-deps' in workflow_text
    assert '--require-npu' in workflow_text
    assert direct_vllm_install not in workflow_text
    assert workflow_text.index(runtime_repair) < workflow_text.index(install_command)