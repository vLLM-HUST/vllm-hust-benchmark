import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = (
    REPO_ROOT
    / ".github"
    / "workflows"
    / "scripts"
    / "sync_official_baseline_snapshots_to_github.sh"
)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def _git(repo_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(repo_dir), *args], check=check)


def _write_submission(source_repo: Path, run_id: str, model_name: str) -> None:
    submission_dir = source_repo / "submissions" / run_id
    submission_dir.mkdir(parents=True)
    (submission_dir / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps({"model": {"name": model_name}}) + "\n",
        encoding="utf-8",
    )


def _create_target_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote_repo = tmp_path / "benchmark-remote.git"
    _run(["git", "init", "--bare", "--initial-branch=main", str(remote_repo)])

    seed_repo = tmp_path / "benchmark-seed"
    _run(["git", "init", "--initial-branch=main", str(seed_repo)])
    _git(seed_repo, "config", "user.name", "Test User")
    _git(seed_repo, "config", "user.email", "test@example.com")
    _git(seed_repo, "remote", "add", "origin", str(remote_repo))

    (seed_repo / "README.md").write_text("seed\n", encoding="utf-8")
    snapshot_dir = seed_repo / "leaderboard-data" / "snapshots"
    snapshot_dir.mkdir(parents=True)
    for file_name in (
        "leaderboard_single.json",
        "leaderboard_multi.json",
        "leaderboard_compare.json",
        "last_updated.json",
    ):
        (snapshot_dir / file_name).write_text("{}\n", encoding="utf-8")

    _git(seed_repo, "add", ".")
    _git(seed_repo, "commit", "-m", "seed")
    _git(seed_repo, "push", "origin", "main")

    target_repo = tmp_path / "benchmark-target"
    _run(["git", "clone", str(remote_repo), str(target_repo)])
    _git(target_repo, "checkout", "main")
    return remote_repo, target_repo


def _create_dummy_website_repo(tmp_path: Path) -> Path:
    website_repo = tmp_path / "vllm-hust-website"
    scripts_dir = website_repo / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "aggregate_results.py").write_text(
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--source-dir', type=Path, required=True)\n"
        "parser.add_argument('--output-dir', type=Path, required=True)\n"
        "parser.add_argument('--replace-all', action='store_true')\n"
        "parser.add_argument('--schema', type=Path)\n"
        "args = parser.parse_args()\n"
        "args.output_dir.mkdir(parents=True, exist_ok=True)\n"
        "payload = {\n"
        "    'runs': sorted(path.parent.name for path in args.source_dir.glob('*/run_leaderboard.json'))\n"
        "}\n"
        "for file_name in ('leaderboard_single.json', 'leaderboard_multi.json', 'leaderboard_compare.json', 'last_updated.json'):\n"
        "    (args.output_dir / file_name).write_text(json.dumps(payload) + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    return website_repo


def test_sync_official_baseline_snapshots_to_github_pushes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    _write_submission(
        source_repo,
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "Qwen2.5-14B-Instruct",
    )
    _write_submission(
        source_repo,
        "official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3",
        "Qwen2.5-14B-Instruct",
    )

    remote_repo, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    local_snapshot_output_dir = tmp_path / "published-snapshots"

    env = {
        **dict(subprocess.os.environ),
        "ALLOW_LOCAL_GIT_RESET": "1",
        "SOURCE_BENCHMARK_REPO_DIR": str(source_repo),
        "TARGET_BENCHMARK_REPO_DIR": str(target_repo),
        "WEBSITE_REPO_DIR": str(website_repo),
        "PYTHON_BIN": sys.executable,
        "LOCAL_SNAPSHOT_OUTPUT_DIR": str(local_snapshot_output_dir),
        "SNAPSHOT_COMMIT_MESSAGE": "test: publish official baseline snapshots",
    }

    first_run = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert first_run.returncode == 0, first_run.stderr
    assert "Pushed official baseline publication" in first_run.stdout

    pushed_head = _git(target_repo, "rev-parse", "HEAD").stdout.strip()
    assert pushed_head == _run(
        ["git", f"--git-dir={remote_repo}", "rev-parse", "refs/heads/main"]
    ).stdout.strip()

    for run_id in (
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3",
    ):
        assert (target_repo / "submissions" / run_id / "leaderboard_manifest.json").is_file()
        assert (target_repo / "submissions" / run_id / "run_leaderboard.json").is_file()

    snapshot_payload = json.loads(
        (target_repo / "leaderboard-data" / "snapshots" / "leaderboard_single.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot_payload["runs"] == [
        "official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3",
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
    ]
    assert (local_snapshot_output_dir / "leaderboard_single.json").is_file()

    second_run = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert second_run.returncode == 0, second_run.stderr
    assert "already up to date" in second_run.stdout
    assert pushed_head == _git(target_repo, "rev-parse", "HEAD").stdout.strip()


def test_sync_official_baseline_snapshots_to_github_can_skip_empty_source(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "benchmark-source"
    (source_repo / "submissions").mkdir(parents=True)
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)

    env = {
        **dict(subprocess.os.environ),
        "ALLOW_LOCAL_GIT_RESET": "1",
        "ALLOW_EMPTY_SNAPSHOT_SOURCE": "1",
        "SOURCE_BENCHMARK_REPO_DIR": str(source_repo),
        "TARGET_BENCHMARK_REPO_DIR": str(target_repo),
        "WEBSITE_REPO_DIR": str(website_repo),
        "PYTHON_BIN": sys.executable,
        "SNAPSHOT_SOURCE_PATTERN": "official-ascend-*",
    }

    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "skipping publication sync" in completed.stdout.lower()