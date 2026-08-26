from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
