from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATHS = [
    ".github/workflows/ci.yml",
    ".github/workflows/push-to-hf.yml",
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