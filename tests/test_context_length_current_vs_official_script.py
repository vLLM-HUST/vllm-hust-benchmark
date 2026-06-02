import json
import os
import re
import subprocess
import sys
from pathlib import Path

from vllm_hust_benchmark.same_spec import build_same_spec_payload, load_benchmark_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run-ascend-context-length-current-vs-official.sh"
CURRENT_RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run-current-ascend-same-spec.sh"
OFFICIAL_RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run-official-ascend-goal-baseline.sh"


def _write_spec(
    spec_file: Path,
    *,
    spec_id: str = "official-ascend-jan-2026-v0.11.0-random-online-ctx2k-qwen25-14b-910b3",
    input_len: int = 2048,
    output_len: int = 256,
) -> None:
    spec_file.write_text(
        json.dumps(
            {
                "id": spec_id,
                "label": spec_id,
                "scenario": "random-online",
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "model_parameters": "14B",
                "model_precision": "FP16",
                "hardware_vendor": "Huawei",
                "hardware_chip_model": "910B3",
                "chip_count": 1,
                "node_count": 1,
                "server_parameters": {
                    "tensor_parallel_size": 1,
                    "max_model_len": input_len + output_len + 256,
                    "max_num_seqs": 1,
                    "enforce_eager": "",
                    "trust_remote_code": "",
                    "disable_log_stats": "",
                    "disable_log_requests": "",
                    "host": "0.0.0.0",
                    "port": 8000,
                },
                "client_parameters": {
                    "backend": "vllm",
                    "endpoint": "/v1/completions",
                    "dataset_name": "random",
                    "num_prompts": 20,
                    "input_len": input_len,
                    "output_len": output_len,
                    "request_rate": 1,
                    "max_concurrency": 1,
                    "host": "127.0.0.1",
                    "port": 8000,
                },
                "export": {
                    "engine": "vllm",
                    "engine_version": "0.11.0",
                    "submitter": "official-ascend-baseline",
                    "baseline_engine": "vllm",
                    "github_repository": "vllm-project/vllm-ascend",
                    "github_ref": "v0.11.0",
                    "git_commit": "2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367",
                    "data_source": "reference-vllm-ascend-benchmark",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_stub(script_path: Path, exit_code: int) -> None:
    script_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _write_prepare_stub(script_path: Path) -> None:
    script_path.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_result_runner_stub(
    script_path: Path,
    *,
    call_log: Path,
    submitter: str,
    succeed_spec_ids: tuple[str, ...],
) -> None:
    python_bin = sys.executable
    success_pattern = "|".join(re.escape(spec_id) for spec_id in succeed_spec_ids)
    success_check = (
        f"if [[ ! \"$spec_id\" =~ ^({success_pattern})$ ]]; then\n"
        if success_pattern
        else "if true; then\n"
    )
    script_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "spec_file=$1\n"
        f"real_python={python_bin!r}\n"
        "spec_id=$($real_python - <<'PY' \"$spec_file\"\n"
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['id'])\n"
        "PY\n"
        ")\n"
        f"printf '%s\\t%s\\n' \"$(basename \"$0\")\" \"$spec_id\" >> {call_log!s}\n"
        f"{success_check}"
        "  echo \"simulated failure for $spec_id\" >&2\n"
        "  exit 1\n"
        "fi\n"
        "mkdir -p \"$RESULT_DIR/submission\"\n"
        "printf '{}' > \"$RESULT_DIR/raw_benchmark_result.json\"\n"
        f"$real_python - <<'PY' \"$RESULT_DIR\" \"$spec_id\" {submitter!r}\n"
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "result_dir = Path(sys.argv[1])\n"
        "spec_id = sys.argv[2]\n"
        "submitter = sys.argv[3]\n"
        "submission_dir = result_dir / 'submission'\n"
        "submission_dir.mkdir(parents=True, exist_ok=True)\n"
        "payload = {\n"
        "    'metadata': {'submitter': submitter},\n"
        "    'same_spec': {'spec_id': spec_id, 'resolved_spec_hash': 'stub-hash'},\n"
        "    'model': {'name': 'Qwen/Qwen2.5-14B-Instruct'},\n"
        "}\n"
        "(result_dir / 'resolved_same_spec.json').write_text(json.dumps({'spec_id': spec_id, 'resolved_spec_hash': 'stub-hash'}), encoding='utf-8')\n"
        "(submission_dir / 'run_leaderboard.json').write_text(json.dumps(payload), encoding='utf-8')\n"
        "(submission_dir / 'leaderboard_manifest.json').write_text(json.dumps({'entries': [{'leaderboard_artifact': 'run_leaderboard.json'}]}), encoding='utf-8')\n"
        "PY\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _write_publish_wrapper(script_path: Path, *, publish_log: Path) -> None:
    python_bin = sys.executable
    script_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"real_python={python_bin!r}\n"
        "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"vllm_hust_benchmark.cli\" && \"${3:-}\" == \"publish-website\" ]]; then\n"
        "  output_dir=\"\"\n"
        "  source_dir=\"\"\n"
        "  shift 3\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    case \"$1\" in\n"
        "      --output-dir)\n"
        "        output_dir=$2\n"
        "        shift 2\n"
        "        ;;\n"
        "      --source-dir)\n"
        "        source_dir=$2\n"
        "        shift 2\n"
        "        ;;\n"
        "      --execute)\n"
        "        shift\n"
        "        ;;\n"
        "      *)\n"
        "        shift\n"
        "        ;;\n"
        "    esac\n"
        "  done\n"
        f"  printf '%s\\n' \"$source_dir\" >> {publish_log!s}\n"
        "  mkdir -p \"$output_dir\"\n"
        "  printf '{\"entries\": []}\\n' > \"$output_dir/leaderboard_single.json\"\n"
        "  printf '{\"entries\": []}\\n' > \"$output_dir/leaderboard_multi.json\"\n"
        "  printf '{\"group_count\": 1}\\n' > \"$output_dir/leaderboard_compare.json\"\n"
        "  printf '{\"updated_at\": \"2026-06-02T00:00:00Z\"}\\n' > \"$output_dir/last_updated.json\"\n"
        "  exit 0\n"
        "fi\n"
        "exec \"$real_python\" \"$@\"\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _write_checkpoint_result(
    checkpoint_root: Path,
    *,
    side: str,
    spec_file: Path,
    submitter: str,
    current_engine_commit: str = "",
    current_plugin_commit: str = "",
    github_repository: str = "",
    github_ref: str = "",
) -> None:
    same_spec_payload = build_same_spec_payload(
        load_benchmark_spec(spec_file),
        spec_source=spec_file,
    )
    result_dir = (
        checkpoint_root
        / ".benchmarks"
        / "context-length-sweep-previous"
        / side
        / spec_file.stem
    )
    submission_dir = result_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    artifact_payload = {
        "metadata": {
            "submitter": submitter,
            "github_repository": github_repository,
            "github_ref": github_ref,
            "runtime_provenance": {
                "engine": {"commit": current_engine_commit},
                "plugin": {"commit": current_plugin_commit},
            },
        },
        "same_spec": {
            "spec_id": same_spec_payload["spec_id"],
            "resolved_spec_hash": same_spec_payload["resolved_spec_hash"],
        },
        "model": {"name": same_spec_payload["model"]},
    }

    (result_dir / "raw_benchmark_result.json").write_text("{}", encoding="utf-8")
    (result_dir / "resolved_same_spec.json").write_text(
        json.dumps(same_spec_payload),
        encoding="utf-8",
    )
    (submission_dir / "leaderboard_manifest.json").write_text(
        json.dumps({"entries": [{"leaderboard_artifact": "run_leaderboard.json"}]}),
        encoding="utf-8",
    )
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps(artifact_payload),
        encoding="utf-8",
    )


def test_context_sweep_skips_aggregation_when_no_manifests_exist(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    prepare_stub = tmp_path / "prepare.sh"
    official_stub = tmp_path / "official.sh"
    current_stub = tmp_path / "current.sh"
    result_root = tmp_path / "results"
    summary_file = tmp_path / "summary.md"
    website_output_dir = tmp_path / "website"
    current_vllm_repo = tmp_path / "vllm-hust"
    current_plugin_repo = tmp_path / "vllm-ascend-hust"

    _write_spec(spec_file)
    _write_stub(prepare_stub, 0)
    _write_stub(official_stub, 1)
    _write_stub(current_stub, 1)
    current_vllm_repo.mkdir()
    current_plugin_repo.mkdir()

    completed = subprocess.run(
        ["bash", str(SWEEP_SCRIPT), str(spec_file)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CURRENT_ENV_PREFIX": str(tmp_path / "current-env"),
            "CURRENT_RUNTIME_PYTHON": sys.executable,
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(current_plugin_repo),
            "CURRENT_VLLM_HUST_REPO": str(current_vllm_repo),
            "GOAL_BASELINE_ENV_PREFIX": str(tmp_path / "official-env"),
            "HOST_PYTHON_BIN": sys.executable,
            "MATRIX_RESULT_ROOT": str(result_root),
            "MATRIX_SUMMARY_FILE": str(summary_file),
            "OFFICIAL_RUNNER": str(official_stub),
            "CURRENT_RUNNER": str(current_stub),
            "PREPARE_OFFICIAL_ENV": "0",
            "PREPARE_SCRIPT": str(prepare_stub),
            "PUBLISH_WEBSITE": "1",
            "WEBSITE_OUTPUT_DIR": str(website_output_dir),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "No leaderboard_manifest.json found under" not in completed.stdout
    assert "No leaderboard_manifest.json found under" not in completed.stderr

    summary_text = summary_file.read_text(encoding="utf-8")
    assert "Website aggregation skipped" in summary_text
    assert "- Failed runs: 2" in summary_text


def test_context_sweep_continues_after_single_spec_failure_and_publishes_available_results(
    tmp_path: Path,
) -> None:
    spec_dir = tmp_path / "specs"
    prepare_stub = tmp_path / "prepare.sh"
    official_stub = tmp_path / "official.sh"
    current_stub = tmp_path / "current.sh"
    host_python_wrapper = tmp_path / "host-python-wrapper.sh"
    runner_call_log = tmp_path / "runner-calls.log"
    publish_log = tmp_path / "publish.log"
    result_root = tmp_path / "results"
    summary_file = tmp_path / "summary.md"
    website_output_dir = tmp_path / "website"
    current_vllm_repo = tmp_path / "vllm-hust"
    current_plugin_repo = tmp_path / "vllm-ascend-hust"
    fail_spec = spec_dir / "ctx2k-fail.json"
    success_spec = spec_dir / "ctx4k-success.json"

    spec_dir.mkdir()
    _write_spec(fail_spec, spec_id="spec-fail", input_len=2048, output_len=256)
    _write_spec(success_spec, spec_id="spec-success", input_len=4096, output_len=256)
    _write_prepare_stub(prepare_stub)
    _write_result_runner_stub(
        official_stub,
        call_log=runner_call_log,
        submitter="official-ascend-baseline",
        succeed_spec_ids=("spec-success",),
    )
    _write_result_runner_stub(
        current_stub,
        call_log=runner_call_log,
        submitter="same-spec-current",
        succeed_spec_ids=("spec-success",),
    )
    _write_publish_wrapper(host_python_wrapper, publish_log=publish_log)
    current_vllm_repo.mkdir()
    current_plugin_repo.mkdir()

    completed = subprocess.run(
        ["bash", str(SWEEP_SCRIPT), str(spec_dir)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CURRENT_ENV_PREFIX": str(tmp_path / "current-env"),
            "CURRENT_RUNTIME_PYTHON": sys.executable,
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(current_plugin_repo),
            "CURRENT_VLLM_HUST_REPO": str(current_vllm_repo),
            "GOAL_BASELINE_ENV_PREFIX": str(tmp_path / "official-env"),
            "HOST_PYTHON_BIN": str(host_python_wrapper),
            "MATRIX_RESULT_ROOT": str(result_root),
            "MATRIX_SUMMARY_FILE": str(summary_file),
            "OFFICIAL_RUNNER": str(official_stub),
            "CURRENT_RUNNER": str(current_stub),
            "PREPARE_OFFICIAL_ENV": "0",
            "PREPARE_SCRIPT": str(prepare_stub),
            "PUBLISH_WEBSITE": "1",
            "WEBSITE_OUTPUT_DIR": str(website_output_dir),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "spec-fail" in runner_call_log.read_text(encoding="utf-8")
    assert "spec-success" in runner_call_log.read_text(encoding="utf-8")
    assert publish_log.read_text(encoding="utf-8").strip() == str(result_root)
    assert (website_output_dir / "leaderboard_compare.json").is_file()

    summary_text = summary_file.read_text(encoding="utf-8")
    assert "Website aggregation skipped" not in summary_text
    assert "- Failed runs: 2" in summary_text
    assert "- Executed runs: 2" in summary_text
    assert "- Successful runs: 2" in summary_text


def test_context_sweep_reuses_checkpointed_results_without_rerun(tmp_path: Path) -> None:
    spec_file = tmp_path / "ctx2k-success.json"
    prepare_stub = tmp_path / "prepare.sh"
    official_stub = tmp_path / "official.sh"
    current_stub = tmp_path / "current.sh"
    result_root = tmp_path / "results"
    checkpoint_root = tmp_path / "checkpoint"
    summary_file = tmp_path / "summary.md"
    runner_call_log = tmp_path / "runner-calls.log"
    current_vllm_repo = tmp_path / "vllm-hust"
    current_plugin_repo = tmp_path / "vllm-ascend-hust"

    _write_spec(spec_file, spec_id="spec-resume", input_len=2048, output_len=256)
    _write_prepare_stub(prepare_stub)
    _write_result_runner_stub(
        official_stub,
        call_log=runner_call_log,
        submitter="official-ascend-baseline",
        succeed_spec_ids=(),
    )
    _write_result_runner_stub(
        current_stub,
        call_log=runner_call_log,
        submitter="same-spec-current",
        succeed_spec_ids=(),
    )
    current_vllm_repo.mkdir()
    current_plugin_repo.mkdir()

    _write_checkpoint_result(
        checkpoint_root,
        side="official",
        spec_file=spec_file,
        submitter="official-ascend-baseline",
        current_plugin_commit="official-plugin-commit",
        github_repository="vllm-project/vllm-ascend",
        github_ref="v0.11.0",
    )
    _write_checkpoint_result(
        checkpoint_root,
        side="current",
        spec_file=spec_file,
        submitter="same-spec-current",
        current_engine_commit="current-engine-commit",
        current_plugin_commit="current-plugin-commit",
    )

    completed = subprocess.run(
        ["bash", str(SWEEP_SCRIPT), str(spec_file)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CURRENT_ENV_PREFIX": str(tmp_path / "current-env"),
            "CURRENT_RUNTIME_PYTHON": sys.executable,
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(current_plugin_repo),
            "CURRENT_VLLM_HUST_REPO": str(current_vllm_repo),
            "GOAL_BASELINE_ENV_PREFIX": str(tmp_path / "official-env"),
            "HOST_PYTHON_BIN": sys.executable,
            "MATRIX_RESULT_ROOT": str(result_root),
            "MATRIX_SUMMARY_FILE": str(summary_file),
            "OFFICIAL_RUNNER": str(official_stub),
            "CURRENT_RUNNER": str(current_stub),
            "PREPARE_OFFICIAL_ENV": "0",
            "PREPARE_SCRIPT": str(prepare_stub),
            "PUBLISH_WEBSITE": "0",
            "RESUME_CHECKPOINT_ROOT": str(checkpoint_root),
            "CURRENT_GIT_COMMIT": "current-engine-commit",
            "CURRENT_PLUGIN_GIT_COMMIT": "current-plugin-commit",
            "OFFICIAL_PLUGIN_GIT_COMMIT": "official-plugin-commit",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert not runner_call_log.exists()

    summary_text = summary_file.read_text(encoding="utf-8")
    assert "- Executed runs: 0" in summary_text
    assert "- Resumed runs: 2" in summary_text
    assert "- Failed runs: 0" in summary_text
    assert "restored from checkpoint" in summary_text


def test_current_same_spec_runner_limits_plugins_without_service_profiling() -> None:
    script_text = CURRENT_RUNNER_SCRIPT.read_text(encoding="utf-8")

    assert 'CURRENT_VLLM_PLUGINS=${CURRENT_VLLM_PLUGINS:-"ascend,ascend_kv_connector,ascend_model_loader"}' in script_text
    assert 'export VLLM_PLUGINS="$CURRENT_VLLM_PLUGINS"' in script_text
    assert 'ascend_service_profiling' not in script_text


def test_current_same_spec_runner_reuses_selected_ascend_device() -> None:
    script_text = CURRENT_RUNNER_SCRIPT.read_text(encoding="utf-8")

    assert 'CURRENT_DEVICE_PREFERENCE_FILE=${CURRENT_DEVICE_PREFERENCE_FILE:-${GOAL_BASELINE_DEVICE_PREFERENCE_FILE:-}}' in script_text
    assert '[same-spec-current] reusing Ascend device from preference file:' in script_text
    assert 'export ASCEND_VISIBLE_DEVICES="$visible_devices"' in script_text
    assert 'export ASCEND_RT_VISIBLE_DEVICES="$rt_visible_devices"' in script_text


def test_official_runner_force_eager_messages_use_stderr() -> None:
    script_text = OFFICIAL_RUNNER_SCRIPT.read_text(encoding="utf-8")

    assert 'forcing --enforce-eager for ${BENCHMARK_TYPE} benchmark" >&2' in script_text
    assert 'forcing --enforce-eager for serve benchmark server" >&2' in script_text


def test_context_sweep_runner_shares_device_preference_and_long_max_len() -> None:
    script_text = SWEEP_SCRIPT.read_text(encoding="utf-8")

    assert 'ASCEND_DEVICE_PREFERENCE_FILE=${ASCEND_DEVICE_PREFERENCE_FILE:-"$MATRIX_RESULT_ROOT/.runtime-state/preferred-ascend-device.txt"}' in script_text
    assert 'CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN=${CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN:-1}' in script_text
    assert 'GOAL_BASELINE_DEVICE_PREFERENCE_FILE="$ASCEND_DEVICE_PREFERENCE_FILE"' in script_text
    assert 'CURRENT_DEVICE_PREFERENCE_FILE="$ASCEND_DEVICE_PREFERENCE_FILE"' in script_text
    assert 'VLLM_ALLOW_LONG_MAX_MODEL_LEN="$CONTEXT_SWEEP_ALLOW_LONG_MAX_MODEL_LEN"' in script_text