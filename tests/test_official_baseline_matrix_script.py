import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tests._bash_utils import bash_executable

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline-matrix.sh"
ONE_CLICK_SCRIPT = REPO_ROOT / "scripts" / "run-official-v0180-baselines.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-official-ascend-baselines.yml"


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
    bash_bin = bash_executable()
    script_path.write_text(f"#!{bash_bin}\nset -euo pipefail\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_runner_stub(
    script_path: Path, *, call_log: Path, fail_repeat_names: tuple[str, ...]
) -> None:
    python_bin = sys.executable
    bash_bin = bash_executable()
    script_path.write_text(
        f"#!{bash_bin}\n"
        "set -euo pipefail\n"
        "spec_file=$1\n"
        'repeat_name=$(basename "$RESULT_DIR")\n'
        f"printf '%s\\n' \"$repeat_name\" >> {call_log!s}\n"
        'case "$repeat_name" in\n'
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
        'mkdir -p "$RESULT_DIR/submission"\n'
        f'{python_bin} - <<\'PY\' "$spec_file" "$RESULT_DIR" "$ttft_ms"\n'
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "result_dir = Path(sys.argv[2])\n"
        "ttft_ms = float(sys.argv[3])\n"
        "submission_dir = result_dir / 'submission'\n"
        "payload = {\n"
        "    'metadata': {'submitter': 'official-ascend-baseline'},\n"
        "    'model': {'canonical_id': 'hf:Qwen/Qwen2.5-14B-Instruct', 'repo_id': 'Qwen/Qwen2.5-14B-Instruct', 'short_name': 'Qwen2.5-14B-Instruct', 'display_name': 'Qwen2.5-14B-Instruct', 'name': 'Qwen/Qwen2.5-14B-Instruct'},\n"
        "    'same_spec': {'spec_id': spec['id']},\n"
        "    'metrics': {'ttft_ms': ttft_ms, 'throughput_tps': 200.0, 'error_rate': 0.0},\n"
        "}\n"
        "(submission_dir / 'run_leaderboard.json').write_text(json.dumps(payload), encoding='utf-8')\n"
        "(submission_dir / 'leaderboard_manifest.json').write_text(json.dumps({'entries': [{'leaderboard_artifact': 'run_leaderboard.json'}]}), encoding='utf-8')\n"
        "(submission_dir / 'env-manifest.json').write_text(json.dumps({'os': 'test', 'python_version': 'test', 'collected_at': '2026-08-07T00:00:00Z', 'frozen_inputs_required': False}), encoding='utf-8')\n"
        "(submission_dir / 'pip-packages.json').write_text('[]\\n', encoding='utf-8')\n"
        "files = ['leaderboard_manifest.json', 'run_leaderboard.json', 'env-manifest.json', 'pip-packages.json']\n"
        "checksums = ''.join(f'{__import__(\"hashlib\").sha256((submission_dir / name).read_bytes()).hexdigest()}  ./{name}\\n' for name in files)\n"
        "(submission_dir / 'checksums.sha256').write_text(checksums, encoding='utf-8')\n"
        "(submission_dir / 'STATUS').write_text('OK\\n', encoding='utf-8')\n"
        "PY\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _write_publish_stub(
    script_path: Path, *, publish_log: Path, source_submissions_root: Path
) -> None:
    # Publish the canonical output created by this test, not a same-named
    # repository fixture that may be archived independently.
    bash_bin = bash_executable()
    script_path.write_text(
        f"#!{bash_bin}\n"
        "set -euo pipefail\n"
        f"source_submissions_root={shlex.quote(str(source_submissions_root))}\n"
        f"printf '%s\\n' \"${{SNAPSHOT_SOURCE_PATTERN:-}}\" >> {publish_log!s}\n"
        'mkdir -p "$TARGET_BENCHMARK_REPO_DIR/submissions"\n'
        "shopt -s nullglob\n"
        'matches=("$source_submissions_root"/$SNAPSHOT_SOURCE_PATTERN)\n'
        'for source_dir in "${matches[@]}"; do\n'
        '  target_dir="$TARGET_BENCHMARK_REPO_DIR/submissions/$(basename "$source_dir")"\n'
        '  rm -rf "$target_dir"\n'
        '  cp -a "$source_dir" "$target_dir"\n'
        "done\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _run_matrix(
    spec_file: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    merged_env = {**os.environ, **env}
    # Forward the modern bash to the matrix script so its child ``$SINGLE_RUNNER``
    # invocation also uses bash 4+ (macOS system bash is 3.2 and chokes on
    # ``;;&`` fall-through and ``mapfile``).
    merged_env.setdefault("MATRIX_RUNNER_BASH", bash_executable())
    return subprocess.run(
        [bash_executable(), str(MATRIX_SCRIPT), str(spec_file)],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_one_click_official_v0180_wrapper_has_help_and_valid_syntax() -> None:
    syntax = subprocess.run(
        [bash_executable(), "-n", str(ONE_CLICK_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        [bash_executable(), str(ONE_CLICK_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "run-official-ascend-goal-baseline-matrix.sh" in help_result.stdout
    assert "SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD" in help_result.stdout


def test_matrix_script_accepts_partial_successful_repeats(tmp_path: Path) -> None:
    spec_id = "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
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
    for file_name in (
        "leaderboard_manifest.json",
        "run_leaderboard.json",
        "env-manifest.json",
        "pip-packages.json",
        "checksums.sha256",
        "STATUS",
    ):
        assert (canonical_root / spec_id / file_name).is_file()
    assert runner_call_log.read_text(encoding="utf-8").strip().splitlines() == [
        "repeat-01",
        "repeat-02",
        "repeat-03",
    ]
    summary_text = summary_file.read_text(encoding="utf-8")
    assert "Proceeding with degraded sample count" in summary_text
    assert "Failed specs: 0" in summary_text


def test_matrix_accepts_mixed_official_source_tuples_as_independent_specs(
    tmp_path: Path,
) -> None:
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    first = specs_dir / "first.json"
    second = specs_dir / "second.json"
    first.write_text(
        json.dumps(
            {
                "id": "first",
                "scenario": "random-online",
                "baseline_target": {
                    "vllm_ref": "v0.18.0",
                    "vllm_ascend_ref": "v0.18.0",
                },
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "id": "second",
                "scenario": "random-online",
                "baseline_target": {"vllm_ref": "main", "vllm_ascend_ref": "main"},
            }
        ),
        encoding="utf-8",
    )
    prepare_stub = tmp_path / "prepare.sh"
    runner_stub = tmp_path / "runner.sh"
    prepare_log = tmp_path / "prepare.log"
    prepare_stub.write_text(
        f"#!{bash_executable()}\n"
        f'printf \'%s\\t%s\\n\' "$OFFICIAL_VLLM_REF" "$OFFICIAL_VLLM_WORKTREE" >> {prepare_log}\n',
        encoding="utf-8",
    )
    prepare_stub.chmod(0o755)
    _write_runner_stub(
        runner_stub, call_log=tmp_path / "calls.log", fail_repeat_names=()
    )

    completed = _run_matrix(
        specs_dir,
        {
            "GOAL_BASELINE_ENV_PREFIX": "/tmp/fake-official-env",
            "PREPARE_SCRIPT": str(prepare_stub),
            "SINGLE_RUNNER": str(runner_stub),
            "PREPARE_OFFICIAL_ENV": "1",
            "REPEAT_COUNT": "1",
            "MIN_SUCCESSFUL_REPEATS": "1",
            "MAX_REPEAT_ATTEMPTS": "1",
            "CANONICAL_SUBMISSIONS_ROOT": str(tmp_path / "submissions"),
            "MATRIX_RESULT_ROOT": str(tmp_path / "results"),
            "PYTHON_BIN": sys.executable,
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "mixed source tuples" not in completed.stderr
    assert "prepared and executed independently" in (
        tmp_path / "results" / "summary.md"
    ).read_text(encoding="utf-8")
    prepare_lines = prepare_log.read_text(encoding="utf-8").splitlines()
    assert prepare_lines[0].startswith("v0.18.0\t")
    assert prepare_lines[1].startswith("main\t")
    assert prepare_lines[0].split("\t", 1)[1] != prepare_lines[1].split("\t", 1)[1]
    assert all("/results/" not in line for line in prepare_lines)


def test_matrix_script_uses_published_canonical_root_for_resume(tmp_path: Path) -> None:
    spec_id = "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
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
    _write_publish_stub(
        publish_stub,
        publish_log=publish_log,
        source_submissions_root=local_canonical_root,
    )
    (website_repo_dir / "scripts").mkdir(parents=True)
    (website_repo_dir / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )
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


def test_matrix_script_requires_complete_evidence_before_promotion(
    tmp_path: Path,
) -> None:
    spec_id = "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
    spec_file = tmp_path / "spec.json"
    prepare_stub = tmp_path / "prepare.sh"
    runner_stub = tmp_path / "runner.sh"
    runner_call_log = tmp_path / "runner-calls.log"
    canonical_root = tmp_path / "submissions-local"
    result_root = tmp_path / "results"

    _write_spec(spec_file, spec_id)
    _write_prepare_stub(prepare_stub)
    _write_runner_stub(runner_stub, call_log=runner_call_log, fail_repeat_names=())
    runner_text = runner_stub.read_text(encoding="utf-8")
    runner_stub.write_text(
        runner_text + 'rm -f "$RESULT_DIR/submission/STATUS"\n',
        encoding="utf-8",
    )

    completed = _run_matrix(
        spec_file,
        {
            "GOAL_BASELINE_ENV_PREFIX": "/tmp/fake-official-env",
            "PREPARE_SCRIPT": str(prepare_stub),
            "SINGLE_RUNNER": str(runner_stub),
            "PREPARE_OFFICIAL_ENV": "0",
            "REPEAT_COUNT": "1",
            "MIN_SUCCESSFUL_REPEATS": "1",
            "MAX_REPEAT_ATTEMPTS": "1",
            "CANONICAL_SUBMISSIONS_ROOT": str(canonical_root),
            "MATRIX_RESULT_ROOT": str(result_root),
            "PYTHON_BIN": sys.executable,
        },
    )

    assert completed.returncode != 0
    assert not (canonical_root / spec_id).exists()


def test_official_runner_finalizes_and_validates_exported_artifact() -> None:
    runner_text = (
        REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline.sh"
    ).read_text(encoding="utf-8")

    export_position = runner_text.index(
        'run_in_official_runtime "$REPO_ROOT/src:$OFFICIAL_RUNTIME_PYTHONPATH"'
    )
    collect_position = runner_text.index(
        'bash "$COLLECT_ARTIFACT_SCRIPT" "$ARTIFACT_DIR"'
    )
    validate_position = runner_text.index(
        'bash "$VALIDATE_ARTIFACT_SCRIPT" "$ARTIFACT_DIR"'
    )

    assert export_position < collect_position < validate_position
    assert 'CURRENT_VLLM_HUST_REPO="$OFFICIAL_VLLM_WORKTREE"' in runner_text
    assert (
        'CURRENT_VLLM_ASCEND_HUST_REPO="$OFFICIAL_VLLM_ASCEND_WORKTREE"' in runner_text
    )
    assert (
        'capture_official_runtime_provenance "$RUNTIME_PROVENANCE_FILE"' in runner_text
    )
    assert (
        'capture_official_runtime_provenance "$RUNTIME_PROVENANCE_AFTER_FILE"'
        in runner_text
    )
    assert (
        "OFFICIAL_ENGINE_RUNTIME_ROOT=${OFFICIAL_ENGINE_RUNTIME_ROOT:-}" in runner_text
    )
    assert '--runtime-image-id "$OFFICIAL_RUNTIME_IMAGE_ID"' in runner_text
    assert '--engine-image-commit "$OFFICIAL_ENGINE_IMAGE_COMMIT"' in runner_text
    assert '--plugin-image-commit "$OFFICIAL_PLUGIN_IMAGE_COMMIT"' in runner_text
    assert 'metadata["official_runtime_provenance"] = runtime' in runner_text
    assert 'manifest["official_runtime_provenance"] = runtime' in runner_text
    assert (
        "recorded engine_version does not match imported runtime package" in runner_text
    )


def test_official_workflow_uses_pinned_v018_runtime_on_112() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    assert "self-hosted" not in workflow_text
    assert "evaluation-request.yml@main" in workflow_text
    assert "bcf2be96120005e9aea171927f85055a6a5c0cf6" in workflow_text
    assert "e18643f8a4d5bd9990727654318ad069ea0b56e2" in workflow_text
    assert "target_registry_version: 1.3.5" in workflow_text
    assert "publish_results" not in workflow_text


def test_official_runner_rejects_stale_source_worktree(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(source_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(source_repo), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source_repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    (source_repo / "pyproject.toml").write_text(
        "[project]\nname='test'\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(source_repo), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(source_repo), "commit", "-m", "first"], check=True)
    stale_worktree = tmp_path / "stale-worktree"
    subprocess.run(
        [
            "git",
            "-C",
            str(source_repo),
            "worktree",
            "add",
            "--detach",
            str(stale_worktree),
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (source_repo / "second.txt").write_text("second\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source_repo), "add", "second.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(source_repo), "commit", "-m", "second"], check=True
    )

    runner_path = shlex.quote(
        str(REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline.sh")
    )
    command = (
        "source <(awk 'BEGIN{capture=0} "
        "/^ensure_worktree\\(\\) \\{/ {capture=1} "
        "/^json2args\\(\\) \\{/ {exit} capture {print}' "
        f"{runner_path}) && ensure_worktree "
        f"{shlex.quote(str(source_repo))} {shlex.quote(str(stale_worktree))} HEAD"
    )
    completed = subprocess.run(
        [bash_executable(), "-lc", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "official source worktree ref mismatch" in completed.stderr
