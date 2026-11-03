"""Sparse smoke checkout keeps software/metadata and does not weaken test scope."""

import os
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def smoke_job():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/workload-data-smoke.yml").read_text("utf-8")
    )
    return workflow["jobs"]["cpu-correctness"]


def test_sparse_smoke_keeps_timeout_head_binding_and_strict_checks():
    job = smoke_job()
    assert job["timeout-minutes"] == 10
    checkout = job["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["sparse-checkout-cone-mode"] is False
    assert checkout["with"]["ref"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )
    assert not any(step.get("continue-on-error") for step in job["steps"])
    assert any(
        step.get("name") == "Verify exact current-source wheel payload"
        for step in job["steps"]
    )
    assert any(
        step.get("name")
        == "Verify installed wheel payload outside checkout, no datasets"
        for step in job["steps"]
    )


def test_real_git_sparse_patterns_preserve_inputs_and_complete_tracked_index(tmp_path):
    repo = tmp_path / "fixture-repository"
    repo.mkdir()
    keep = [
        "README.md",
        "pyproject.toml",
        "workload-preparation/src/example/__init__.py",
        "workload-preparation/data/catalogs/example.json",
        "tests/test_example.py",
        "scripts/example.py",
        "configs/workload-native/example.json",
        "schemas/example.json",
        "docs/workload-data/validation/example.json",
        "docs/reports/nested-software-reference.json",
        "src/example/__init__.py",
    ]
    omit = ["reports/old.txt", "archive/old.txt", "submissions/old.txt"]
    for name in keep + omit:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    hooks = tmp_path / "empty-hooks"
    hooks.mkdir()
    git = [
        "git",
        "-c",
        "core.autocrlf=false",
        "-c",
        f"core.hooksPath={hooks.as_posix()}",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=Smoke fixture",
        "-c",
        "user.email=smoke@example.invalid",
    ]
    subprocess.run(
        git + ["init", "-q", "--initial-branch=codex/smoke-fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(git + ["add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        git + ["commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    patterns = smoke_job()["steps"][0]["with"]["sparse-checkout"]
    subprocess.run(
        git + ["sparse-checkout", "set", "--no-cone", "--stdin"],
        cwd=repo,
        input=patterns,
        text=True,
        check=True,
        capture_output=True,
    )
    assert all((repo / name).read_text("utf-8") == "fixture\n" for name in keep)
    assert all(not (repo / name).exists() for name in omit)
    tracked = subprocess.check_output(git + ["ls-files"], cwd=repo, text=True)
    assert set(tracked.splitlines()) == set(keep + omit)


def test_actual_smoke_checkout_does_not_materialize_result_capsules():
    # Full CI intentionally retains these directories; only this smoke is sparse.
    if os.environ.get("GITHUB_WORKFLOW") == "Workload data CPU correctness smoke":
        assert all(
            not (ROOT / name).exists() for name in ("reports", "archive", "submissions")
        )
    for name in ("workload-preparation", "tests", "scripts", "configs", "docs"):
        assert (ROOT / name).is_dir()
