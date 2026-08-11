import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = (
    REPO_ROOT
    / ".github"
    / "workflows"
    / "scripts"
    / "sync_official_baseline_snapshots_to_github.sh"
)


def _script_env(**overrides: str) -> dict[str, str]:
    env = dict(subprocess.os.environ)
    env.pop("GITHUB_ACTIONS", None)
    env.pop("GITHUB_ENV", None)
    env.pop("GITHUB_OUTPUT", None)
    env.update(overrides)
    return env


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


def _git(
    repo_dir: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(repo_dir), *args], check=check)


def _write_submission(source_repo: Path, run_id: str, model_name: str) -> None:
    submission_dir = source_repo / "submissions" / run_id
    submission_dir.mkdir(parents=True)
    (submission_dir / "leaderboard_manifest.json").write_text(
        json.dumps({"entries": [{"leaderboard_artifact": "run_leaderboard.json"}]})
        + "\n",
        encoding="utf-8",
    )
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps({"model": {"name": model_name}}) + "\n",
        encoding="utf-8",
    )
    (submission_dir / "env-manifest.json").write_text(
        json.dumps(
            {
                "os": "test",
                "python_version": "test",
                "collected_at": "2026-08-07T00:00:00Z",
                "frozen_inputs_required": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (submission_dir / "pip-packages.json").write_text("[]\n", encoding="utf-8")
    checksum_files = (
        "leaderboard_manifest.json",
        "run_leaderboard.json",
        "env-manifest.json",
        "pip-packages.json",
    )
    (submission_dir / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((submission_dir / name).read_bytes()).hexdigest()}  ./{name}\n"
            for name in checksum_files
        ),
        encoding="utf-8",
    )
    (submission_dir / "STATUS").write_text("OK\n", encoding="utf-8")


def _create_artifact_validator(tmp_path: Path) -> Path:
    validator = tmp_path / "validate-artifact.sh"
    validator.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'test "$(tr -d \'[:space:]\' < "$1/STATUS")" = OK\n'
        '(cd "$1" && sha256sum -c checksums.sha256 >/dev/null)\n',
        encoding="utf-8",
    )
    validator.chmod(0o755)
    return validator


def _create_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$*" == *"validate_public_leaderboard_snapshots.py"* ]]; then
  exit "${FAKE_PUBLIC_VALIDATOR_EXIT:-0}"
fi
if [[ "$1" == "-" && "$#" == "2" ]]; then
  cat >/dev/null
  exit "${FAKE_TREND_VALIDATOR_EXIT:-0}"
fi
if [[ "$1" != "-m" || "$2" != "vllm_hust_benchmark.cli" || "$3" != "publish-website" ]]; then
  echo "unexpected fake python invocation: $*" >&2
  exit 2
fi
shift 3
source_dir=""
output_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --source-dir) source_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$output_dir"
runs_json=$(find "$source_dir" -mindepth 2 -maxdepth 2 -name run_leaderboard.json -print | sed 's|/run_leaderboard.json$||' | xargs -n1 basename | sort | jq -R -s 'split("\\n") | map(select(length > 0))')
printf '{"runs":%s}\n' "$runs_json" > "$output_dir/leaderboard_single.json"
printf '[]\n' > "$output_dir/leaderboard_multi.json"
printf '{}\n' > "$output_dir/leaderboard_compare.json"
printf '{}\n' > "$output_dir/last_updated.json"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def _create_failing_verify_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  if [[ "$argument" == cat-file ]]; then exit 1; fi\n'
        "done\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


def _create_mismatch_verify_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-bin" / "git"
    fake_git.parent.mkdir(exist_ok=True)
    fake_git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  if [[ "$argument" == *":${FAKE_GIT_MISMATCH_PATH:-__never__}" ]]; then\n'
        "    printf 'tampered\\n'\n"
        "    exit 0\n"
        "  fi\n"
        "done\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


def _create_fetch_exhaustion_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-bin" / "git"
    fake_git.parent.mkdir(exist_ok=True)
    fake_git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  if [[ "$argument" == fetch && -f "${FETCH_FAILURE_MARKER:?}" ]]; then\n'
        "    exit 1\n"
        "  fi\n"
        "done\n"
        'if [[ "$1" == -C ]]; then command="$3"; else command="$1"; fi\n'
        '"$REAL_GIT" "$@"\n'
        "status=$?\n"
        'if [[ "$command" == push && $status -eq 0 ]]; then touch "$FETCH_FAILURE_MARKER"; fi\n'
        "exit $status\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


def _create_push_exhaustion_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-bin" / "git"
    fake_git.parent.mkdir(exist_ok=True)
    fake_git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'for argument in "$@"; do\n'
        '  if [[ "$argument" == push ]]; then exit 1; fi\n'
        "done\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


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
    github_env = tmp_path / "github.env"

    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        LOCAL_SNAPSHOT_OUTPUT_DIR=str(local_snapshot_output_dir),
        SNAPSHOT_COMMIT_MESSAGE="test: publish official baseline snapshots",
        GITHUB_ENV=str(github_env),
    )

    first_run = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert first_run.returncode == 0, first_run.stderr
    assert "Pushed official baseline publication" in first_run.stdout

    pushed_head = _git(target_repo, "rev-parse", "HEAD").stdout.strip()
    assert (
        pushed_head
        == _run(
            ["git", f"--git-dir={remote_repo}", "rev-parse", "refs/heads/main"]
        ).stdout.strip()
    )

    for run_id in (
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3",
    ):
        assert (
            target_repo / "submissions" / run_id / "leaderboard_manifest.json"
        ).is_file()
        assert (target_repo / "submissions" / run_id / "run_leaderboard.json").is_file()

    snapshot_payload = json.loads(
        (
            target_repo / "leaderboard-data" / "snapshots" / "leaderboard_single.json"
        ).read_text(encoding="utf-8")
    )
    assert snapshot_payload["runs"] == [
        "official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3",
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
    ]
    assert (local_snapshot_output_dir / "leaderboard_single.json").is_file()
    github_values = dict(
        line.split("=", maxsplit=1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    assert github_values["GITHUB_SNAPSHOT_SYNC_STATUS"] == "pushed"
    assert github_values["GITHUB_SNAPSHOT_SYNC_VERIFICATION"] == "verified"
    assert (
        github_values["GITHUB_SNAPSHOT_SYNC_REPO"]
        == "vLLM-HUST" + "/vllm-hust-benchmark"
    )
    assert github_values["GITHUB_SNAPSHOT_SYNC_BRANCH"] == "main"
    assert github_values["GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH"] == (
        "leaderboard-data" + "/snapshots"
    )
    assert github_values["GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATHS"] == ",".join(
        f"submissions/{run_id}" for run_id in snapshot_payload["runs"]
    )

    second_run = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert second_run.returncode == 0, second_run.stderr
    assert "already up to date" in second_run.stdout
    assert pushed_head == _git(target_repo, "rev-parse", "HEAD").stdout.strip()


def test_sync_verification_rejects_remote_content_mismatch(tmp_path: Path) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    run_id = "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3"
    _write_submission(source_repo, run_id, "Qwen2.5-14B-Instruct")
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    fake_git = _create_mismatch_verify_git(tmp_path)
    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        PATH=f"{fake_git.parent}:{os.environ['PATH']}",
        REAL_GIT=shutil.which("git") or "git",
        FAKE_GIT_MISMATCH_PATH=f"submissions/{run_id}/run_leaderboard.json",
    )
    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert completed.returncode != 0
    assert "content mismatch" in completed.stderr


@pytest.mark.parametrize(
    ("failure_env", "expected_error"),
    (
        ("FAKE_PUBLIC_VALIDATOR_EXIT", "public snapshot validation"),
        ("FAKE_TREND_VALIDATOR_EXIT", "trend validation"),
    ),
)
def test_sync_rejects_snapshot_validation_before_git_write(
    tmp_path: Path,
    failure_env: str,
    expected_error: str,
) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    run_id = "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3"
    _write_submission(source_repo, run_id, "Qwen2.5-14B-Instruct")
    remote_repo, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    original_head = _git(target_repo, "rev-parse", "HEAD").stdout.strip()
    github_env = tmp_path / "github.env"
    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        GITHUB_ENV=str(github_env),
        **{failure_env: "2"},
    )

    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert _git(target_repo, "rev-parse", "HEAD").stdout.strip() == original_head
    assert (
        _run(
            ["git", f"--git-dir={remote_repo}", "rev-parse", "refs/heads/main"]
        ).stdout.strip()
        == original_head
    )
    assert not (target_repo / "submissions" / run_id).exists()
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=rejected" in github_env.read_text(
        encoding="utf-8"
    )


def test_sync_verification_rejects_remote_missing_file(tmp_path: Path) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    _write_submission(
        source_repo,
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "Qwen2.5-14B-Instruct",
    )
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    fake_git = _create_failing_verify_git(tmp_path)
    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        PATH=f"{fake_git.parent}:{os.environ['PATH']}",
        REAL_GIT=shutil.which("git") or "git",
        SNAPSHOT_MAX_FETCH_ATTEMPTS="1",
    )
    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert completed.returncode != 0
    assert "verification failed" in completed.stderr


def test_sync_reports_pushed_but_failed_when_verification_fetch_is_exhausted(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    _write_submission(
        source_repo,
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "Qwen2.5-14B-Instruct",
    )
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    fake_git = _create_fetch_exhaustion_git(tmp_path)
    github_env = tmp_path / "github.env"
    marker = tmp_path / "fetch-failure.marker"
    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        PATH=f"{fake_git.parent}:{os.environ['PATH']}",
        REAL_GIT=shutil.which("git") or "git",
        FETCH_FAILURE_MARKER=str(marker),
        GITHUB_ENV=str(github_env),
        SNAPSHOT_MAX_FETCH_ATTEMPTS="2",
        SNAPSHOT_FETCH_RETRY_SECONDS="0",
        SNAPSHOT_MAX_PUSH_ATTEMPTS="1",
    )
    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert completed.returncode != 0
    github_values = github_env.read_text(encoding="utf-8")
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed" in github_values
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=failed" in github_values


def test_sync_reports_failed_when_unchanged_verification_fails(tmp_path: Path) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    run_id = "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3"
    _write_submission(source_repo, run_id, "Qwen2.5-14B-Instruct")
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    first_env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
    )
    first_run = _run(["bash", str(SYNC_SCRIPT)], env=first_env, check=False)
    assert first_run.returncode == 0, first_run.stderr

    github_env = tmp_path / "github.env"
    fake_git = _create_failing_verify_git(tmp_path)
    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        PATH=f"{fake_git.parent}:{os.environ['PATH']}",
        REAL_GIT=shutil.which("git") or "git",
        SNAPSHOT_MAX_FETCH_ATTEMPTS="1",
        GITHUB_ENV=str(github_env),
    )
    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)

    assert completed.returncode != 0
    github_values = dict(
        line.split("=", maxsplit=1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    assert github_values["GITHUB_SNAPSHOT_SYNC_STATUS"] == "unchanged"
    assert github_values["GITHUB_SNAPSHOT_SYNC_VERIFICATION"] == "failed"


def test_sync_reports_failed_when_push_retries_are_exhausted(tmp_path: Path) -> None:
    source_repo = tmp_path / "benchmark-source"
    source_repo.mkdir()
    _write_submission(
        source_repo,
        "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "Qwen2.5-14B-Instruct",
    )
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)
    fake_git = _create_push_exhaustion_git(tmp_path)
    github_env = tmp_path / "github.env"
    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=_create_fake_python(tmp_path),
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        PATH=f"{fake_git.parent}:{os.environ['PATH']}",
        REAL_GIT=shutil.which("git") or "git",
        SNAPSHOT_MAX_PUSH_ATTEMPTS="2",
        SNAPSHOT_PUSH_RETRY_SECONDS="0",
        GITHUB_ENV=str(github_env),
    )
    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)

    assert completed.returncode != 0
    github_values = dict(
        line.split("=", maxsplit=1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    assert github_values["GITHUB_SNAPSHOT_SYNC_STATUS"] == "failed"


def test_sync_official_baseline_snapshots_to_github_can_skip_empty_source(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "benchmark-source"
    (source_repo / "submissions").mkdir(parents=True)
    _, target_repo = _create_target_repo(tmp_path)
    website_repo = _create_dummy_website_repo(tmp_path)

    env = _script_env(
        ALLOW_LOCAL_GIT_RESET="1",
        ALLOW_EMPTY_SNAPSHOT_SOURCE="1",
        SOURCE_BENCHMARK_REPO_DIR=str(source_repo),
        TARGET_BENCHMARK_REPO_DIR=str(target_repo),
        WEBSITE_REPO_DIR=str(website_repo),
        PYTHON_BIN=sys.executable,
        ARTIFACT_VALIDATOR=str(_create_artifact_validator(tmp_path)),
        SNAPSHOT_SOURCE_PATTERN="official-ascend-*",
    )

    completed = _run(["bash", str(SYNC_SCRIPT)], env=env, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "skipping publication sync" in completed.stdout.lower()
