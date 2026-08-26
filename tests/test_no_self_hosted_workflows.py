from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_never_executes_on_self_hosted_runners() -> None:
    for workflow in sorted((REPO_ROOT / ".github/workflows").glob("*.y*ml")):
        text = workflow.read_text(encoding="utf-8")
        assert "self-hosted" not in text, workflow


def test_legacy_benchmark_workflows_are_removed() -> None:
    for name in (
        "merge-gate.yml",
        "run-official-ascend-baselines.yml",
        "push-to-hf.yml",
        "notify-website-leaderboard.yml",
    ):
        assert not (REPO_ROOT / ".github/workflows" / name).exists(), name
