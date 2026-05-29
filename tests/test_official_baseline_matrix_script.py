import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline-matrix.sh"


def _write_spec(spec_file: Path, spec_id: str) -> None:
    spec_file.write_text(
        json.dumps(
            {
                "id": spec_id,
                "scenario": "random-online",
            }
        ),
        encoding="utf-8",
    )


def _write_prepare_stub(script_path: Path) -> None:
    script_path.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_runner_stub(script_path: Path, *, call_log: Path, fail_repeat_names: tuple[str, ...]) -> None:
    python_bin = sys.executable
    script_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "spec_file=$1\n"
        "repeat_name=$(basename \"$RESULT_DIR\")\n"
        f"printf '%s\\n' \"$repeat_name\" >> {call_log!s}\n"
        "case \"$repeat_name\" in\n"
        + "".join(
            f"  {repeat_name}) echo 'ValueError: Initial test run failed - simulated transient engine crash' >&2; exit 1 ;;&\n"
            for repeat_name in fail_repeat_names
        )
        + "  repeat-01) ttft_ms=110 ;;&\n"
        "  repeat-02) ttft_ms=100 ;;&\n"
        "  repeat-03) ttft_ms=120 ;;&\n"
        "  repeat-04) ttft_ms=105 ;;&\n"
        "  *) ttft_ms=130 ;;&\n"
        "esac\n"
        "mkdir -p \"$RESULT_DIR/submission\"\n"
        f"{python_bin} - <<'PY' \"$spec_file\" \"$RESULT_DIR\" \"$ttft_ms\"\n"
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "result_dir = Path(sys.argv[2])\n"
        "ttft_ms = float(sys.argv[3])\n"
        "submission_dir = result_dir / 'submission'\n"
        "payload = {\n"
        "    'metadata': {'submitter': 'official-ascend-baseline'},\n"
        "    'same_spec': {'spec_id': spec['id']},\n"
        "    'metrics': {'ttft_ms': ttft_ms, 'throughput_tps': 200.0, 'error_rate': 0.0},\n"
        "}\n"
        "(submission_dir / 'run_leaderboard.json').write_text(json.dumps(payload), encoding='utf-8')\n"
        "(submission_dir / 'leaderboard_manifest.json').write_text(json.dumps({'entries': [{'leaderboard_artifact': 'run_leaderboard.json'}]}), encoding='utf-8')\n"
        "PY\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _write_publish_stub(script_path: Path, *, publish_log: Path) -> None:
    script_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"${{SNAPSHOT_SOURCE_PATTERN:-}}\" >> {publish_log!s}\n"
        "mkdir -p \"$TARGET_BENCHMARK_REPO_DIR/submissions\"\n"
        "shopt -s nullglob\n"
        "matches=(\"$SOURCE_BENCHMARK_REPO_DIR\"/submissions/$SNAPSHOT_SOURCE_PATTERN)\n"
        "for source_dir in \"${matches[@]}\"; do\n"
        "  target_dir=\"$TARGET_BENCHMARK_REPO_DIR/submissions/$(basename \"$source_dir\")\"\n"
        "  rm -rf \"$target_dir\"\n"
        "  cp -a \"$source_dir\" \"$target_dir\"\n"
        "done\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _run_matrix(spec_file: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged_env = {**os.environ, **env}
    return subprocess.run(
        ["bash", str(MATRIX_SCRIPT), str(spec_file)],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_matrix_script_accepts_partial_successful_repeats(tmp_path: Path) -> None:
    spec_id = "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3"
    spec_file = tmp_path / "spec.json"
    prepare_stub = tmp_path / "prepare.sh"
    runner_stub = tmp_path / "runner.sh"
    runner_call_log = tmp_path / "runner-calls.log"
    summary_file = tmp_path / "summary.md"
    canonical_root = tmp_path / "submissions-local"
    result_root = tmp_path / "results"

    _write_spec(spec_file, spec_id)
    _write_prepare_stub(prepare_stub)
    _write_runner_stub(
        runner_stub,
        call_log=runner_call_log,
        fail_repeat_names=("repeat-03",),
    )

    completed = _run_matrix(
        spec_file,
        {
            "GOAL_BASELINE_ENV_PREFIX": "/tmp/fake-official-env",
            "PREPARE_SCRIPT": str(prepare_stub),
            "SINGLE_RUNNER": str(runner_stub),
            "PREPARE_OFFICIAL_ENV": "0",
            "REPEAT_COUNT": "3",
            "MIN_SUCCESSFUL_REPEATS": "2",
            "MAX_REPEAT_ATTEMPTS": "3",
            "CANONICAL_SUBMISSIONS_ROOT": str(canonical_root),
            "MATRIX_RESULT_ROOT": str(result_root),
            "MATRIX_SUMMARY_FILE": str(summary_file),
            "PYTHON_BIN": sys.executable,
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert (canonical_root / spec_id / "run_leaderboard.json").is_file()
    assert runner_call_log.read_text(encoding="utf-8").strip().splitlines() == [
        "repeat-01",
        "repeat-02",
        "repeat-03",
    ]
    summary_text = summary_file.read_text(encoding="utf-8")
    assert "Proceeding with degraded sample count" in summary_text
    assert "Failed specs: 0" in summary_text


def test_matrix_script_uses_published_canonical_root_for_resume(tmp_path: Path) -> None:
    spec_id = "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3"
    spec_file = tmp_path / "spec.json"
    prepare_stub = tmp_path / "prepare.sh"
    runner_stub = tmp_path / "runner.sh"
    publish_stub = tmp_path / "publish.sh"
    runner_call_log = tmp_path / "runner-calls.log"
    publish_log = tmp_path / "publish.log"
    summary_file = tmp_path / "summary.md"
    local_canonical_root = tmp_path / "submissions-local"
    remote_repo_dir = tmp_path / "benchmark-main"
    remote_submissions_root = remote_repo_dir / "submissions"
    result_root = tmp_path / "results"
    website_repo_dir = tmp_path / "website"

    _write_spec(spec_file, spec_id)
    _write_prepare_stub(prepare_stub)
    _write_runner_stub(runner_stub, call_log=runner_call_log, fail_repeat_names=())
    _write_publish_stub(publish_stub, publish_log=publish_log)
    (website_repo_dir / "scripts").mkdir(parents=True)
    (website_repo_dir / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")
    remote_repo_dir.mkdir(parents=True)

    env = {
        "GOAL_BASELINE_ENV_PREFIX": "/tmp/fake-official-env",
        "PREPARE_SCRIPT": str(prepare_stub),
        "SINGLE_RUNNER": str(runner_stub),
        "PREPARE_OFFICIAL_ENV": "0",
        "REPEAT_COUNT": "1",
        "MIN_SUCCESSFUL_REPEATS": "1",
        "MAX_REPEAT_ATTEMPTS": "1",
        "CANONICAL_SUBMISSIONS_ROOT": str(local_canonical_root),
        "EXISTING_CANONICAL_SUBMISSIONS_ROOT": str(remote_submissions_root),
        "MATRIX_RESULT_ROOT": str(result_root),
        "MATRIX_SUMMARY_FILE": str(summary_file),
        "PYTHON_BIN": sys.executable,
        "PUBLISH_RESULTS": "1",
        "PUBLICATION_SYNC_HELPER": str(publish_stub),
        "TARGET_BENCHMARK_REPO_DIR": str(remote_repo_dir),
        "WEBSITE_REPO_DIR": str(website_repo_dir),
        "SNAPSHOT_TARGET_BRANCH": "main",
    }

    first_run = _run_matrix(spec_file, env)
    assert first_run.returncode == 0, first_run.stderr
    assert (remote_submissions_root / spec_id / "run_leaderboard.json").is_file()
    assert publish_log.read_text(encoding="utf-8").strip().splitlines() == [spec_id]

    second_run = _run_matrix(spec_file, env)
    assert second_run.returncode == 0, second_run.stderr
    assert runner_call_log.read_text(encoding="utf-8").strip().splitlines() == [
        "repeat-01",
    ]
    summary_text = summary_file.read_text(encoding="utf-8")
    assert f"Skip existing canonical: {spec_id}" in summary_text