import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run-ascend-context-length-current-vs-official.sh"


def _write_spec(spec_file: Path) -> None:
    spec_file.write_text(
        json.dumps(
            {
                "id": "official-ascend-jan-2026-v0.11.0-random-online-ctx2k-qwen25-14b-910b3",
                "client_parameters": {
                    "input_len": 2048,
                    "output_len": 256,
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