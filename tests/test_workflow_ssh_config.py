from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW_PATHS = [
    ".github/workflows/run-official-ascend-baselines.yml",
]


def test_pull_request_ci_uses_public_checkout_without_ssh_secret() -> None:
    workflow_text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow_text
    assert "VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" not in workflow_text
    assert "BENCHMARK_CHECKOUT_USE_SSH" not in workflow_text
    assert "ssh-key:" not in workflow_text
    assert "ssh-strict:" not in workflow_text
    assert "Require GitHub SSH checkout key" not in workflow_text
    assert "Configure GitHub SSH" not in workflow_text

    checkout_step = "      - name: Checkout"
    # ci.yml may now declare multiple jobs (tests, lint), each starting with
    # its own Checkout step. Every Checkout must be a *public* checkout,
    # i.e. followed by ``persist-credentials: false`` — one per usage.
    checkout_count = workflow_text.count("uses: actions/checkout@v4")
    assert checkout_count >= 1
    assert workflow_text.count("persist-credentials: false") == checkout_count
    assert "          persist-credentials: false" in workflow_text
    assert workflow_text.index(checkout_step) < workflow_text.index(
        "      - name: Setup Python"
    )
    steps_text = workflow_text.split("    steps:\n", maxsplit=1)[1]
    first_step = next(line for line in steps_text.splitlines() if line.strip())
    assert first_step == checkout_step


def test_hf_upload_uses_public_checkout_without_ssh_secret() -> None:
    workflow_text = (REPO_ROOT / ".github/workflows/push-to-hf.yml").read_text(
        encoding="utf-8"
    )

    assert workflow_text.count("uses: actions/checkout@v4") == 3
    assert workflow_text.count("persist-credentials: false") == 3
    assert "VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" not in workflow_text
    assert "BENCHMARK_CHECKOUT_USE_SSH" not in workflow_text
    assert "Require GitHub SSH checkout key" not in workflow_text
    assert "Configure GitHub SSH" not in workflow_text
    assert "ssh-key:" not in workflow_text


@pytest.mark.parametrize("workflow_path", PUBLISH_WORKFLOW_PATHS)
def test_workflows_use_standard_github_ssh_without_overwriting_config(
    workflow_path: str,
) -> None:
    workflow_text = (REPO_ROOT / workflow_path).read_text(encoding="utf-8")

    assert "BENCHMARK_CHECKOUT_USE_SSH_443" not in workflow_text
    assert "Configure GitHub SSH over 443" not in workflow_text
    assert "ssh.github.com" not in workflow_text
    assert (
        'git config --global url."ssh://git@ssh.github.com:443/".insteadOf https://github.com/'
        not in workflow_text
    )

    assert "BENCHMARK_CHECKOUT_USE_SSH" in workflow_text
    assert "Configure GitHub SSH" in workflow_text
    assert 'if [[ ! -d "$ssh_dir" ]]; then' in workflow_text
    assert 'if [[ ! -f "$config_file" ]]; then' in workflow_text
    assert "Host github.com" in workflow_text
    assert "HostName github.com" in workflow_text
    assert "IdentityFile ~/.ssh/github_actions" in workflow_text
    assert 'printf \'\\n%s\\n\' "$config_block" >> "$config_file"' in workflow_text
