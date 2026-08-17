from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_never_executes_on_self_hosted_runners() -> None:
    for workflow in sorted((REPO_ROOT / ".github/workflows").glob("*.y*ml")):
        text = workflow.read_text(encoding="utf-8")
        assert "self-hosted" not in text, workflow


def test_real_hardware_workflows_submit_to_112() -> None:
    for name in ("merge-gate.yml", "run-official-ascend-baselines.yml"):
        text = (REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        assert "evaluation-request.yml@main" in text
        assert (
            "repeat_count: 3" in text
            or "repeat_count: ${{ inputs.repeat_count }}" in text
        )
