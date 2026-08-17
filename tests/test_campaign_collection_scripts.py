import json
import os
import subprocess
import sys
from pathlib import Path

from _bash_utils import bash_executable

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_RUNNER = REPO_ROOT / "scripts/run-campaign-repetitions.sh"
COLLECTOR = REPO_ROOT / "scripts/collect-run-artifact.sh"
VALIDATOR = REPO_ROOT / "scripts/validate-run-artifact.sh"


def _write_spec(path: Path, *, chip_count: int = 2) -> None:
    path.write_text(
        json.dumps(
            {
                "scenario": "random-online",
                "chip_count": chip_count,
            }
        ),
        encoding="utf-8",
    )


def _write_fake_runner(
    path: Path, *, fail_index: int | None = None, mode: int = 0o755
) -> None:
    failure = (
        f'if [[ "$3" == "{fail_index}" ]]; then exit 19; fi'
        if fail_index is not None
        else ""
    )
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail
printf '%s\\t%s\\t%s\\t%s\\n' "$1" "$2" "$3" "$CAMPAIGN_REPEAT_INDEX" >> "$CALL_LOG"
{failure}
""",
        encoding="utf-8",
    )
    path.chmod(mode)


def _run_campaign(
    tmp_path: Path,
    *,
    fail_index: int | None = None,
    runner_mode: int = 0o755,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    spec = tmp_path / "spec.json"
    fake_runner = tmp_path / "fake-single-runner.sh"
    call_log = tmp_path / "calls.tsv"
    summary = tmp_path / "summary.json"
    _write_spec(spec)
    _write_fake_runner(fake_runner, fail_index=fail_index, mode=runner_mode)
    env = {
        **os.environ,
        "SINGLE_REPETITION_RUNNER": str(fake_runner),
        "CALL_LOG": str(call_log),
        "CAMPAIGN_SUMMARY_FILE": str(summary),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            bash_executable(),
            str(CAMPAIGN_RUNNER),
            str(spec),
            "--campaign-prefix",
            "issue-136-test",
            "--repetitions",
            "3",
            "--cooldown",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_campaign_runner_completes_three_successful_process_repetitions(
    tmp_path: Path,
) -> None:
    result = _run_campaign(tmp_path)

    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls.tsv").read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[2:] for line in calls] == [
        ["1", "0"],
        ["2", "1"],
        ["3", "2"],
    ]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "independent-service-campaign-summary/v1"
    assert summary["requested_repetitions"] == 3
    assert summary["successful_repetitions"] == 3
    assert [run["repeat_index"] for run in summary["runs"]] == [0, 1, 2]


def test_campaign_runner_preserves_failure_after_later_success(tmp_path: Path) -> None:
    result = _run_campaign(tmp_path, fail_index=2)

    assert result.returncode == 19
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["successful_repetitions"] == 2
    assert [run["status"] for run in summary["runs"]] == [
        "ok",
        "failed",
        "ok",
    ]


def test_campaign_runner_accepts_readable_non_executable_runner(tmp_path: Path) -> None:
    result = _run_campaign(tmp_path, runner_mode=0o644)

    assert result.returncode == 0, result.stderr
    assert len((tmp_path / "calls.tsv").read_text(encoding="utf-8").splitlines()) == 3


def test_formal_campaign_rejects_same_server_measurement_repetitions(
    tmp_path: Path,
) -> None:
    frozen = "a" * 40
    result = _run_campaign(
        tmp_path,
        extra_env={
            "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
            "CAMPAIGN_ID": "issue-136/v1",
            "CAMPAIGN_COVERAGE_CLASS": "full-matrix",
            "CAMPAIGN_POINT_ROLE": "checkpoint",
            "CAMPAIGN_LOAD_PROFILE": "fixed-1-rps",
            "CURRENT_GIT_COMMIT": frozen,
            "CURRENT_PLUGIN_GIT_COMMIT": frozen,
            "CURRENT_IMAGE_ID": "b" * 64,
            "CURRENT_MODEL_REVISION": "c" * 40,
            "CURRENT_CANN_VERSION": "9.0.0",
            "CURRENT_TORCH_NPU_VERSION": "2.10.0",
            "CURRENT_TOPOLOGY": "single-node-hccs",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            "ASCEND_VISIBLE_DEVICES": "0,1",
            "CURRENT_VLLM_HUST_REPO": str(tmp_path / "core"),
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(tmp_path / "backend"),
            "PERFGATE_MEASURED_RUNS": "3",
        },
    )

    assert result.returncode == 2
    assert "require PERFGATE_WARMUP_RUNS=0 and PERFGATE_MEASURED_RUNS=1" in (
        result.stderr
    )


def _init_git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _formal_campaign_env(tmp_path: Path) -> dict[str, str]:
    core = tmp_path / "core"
    backend = tmp_path / "backend"
    core_commit = _init_git_repo(core)
    backend_commit = _init_git_repo(backend)
    return {
        "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
        "CAMPAIGN_ID": "issue-136/v1",
        "CAMPAIGN_COVERAGE_CLASS": "full-matrix",
        "CAMPAIGN_POINT_ROLE": "checkpoint",
        "CAMPAIGN_LOAD_PROFILE": "fixed-1-rps",
        "CURRENT_GIT_COMMIT": core_commit,
        "CURRENT_PLUGIN_GIT_COMMIT": backend_commit,
        "CURRENT_IMAGE_ID": "b" * 64,
        "CURRENT_MODEL_REVISION": "c" * 40,
        "CURRENT_CANN_VERSION": "9.0.0",
        "CURRENT_TORCH_NPU_VERSION": "2.10.0",
        "CURRENT_TOPOLOGY": "single-node-hccs",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "ASCEND_VISIBLE_DEVICES": "0,1",
        "CURRENT_VLLM_HUST_REPO": str(core),
        "CURRENT_VLLM_ASCEND_HUST_REPO": str(backend),
        "PERFGATE_WARMUP_RUNS": "0",
        "PERFGATE_MEASURED_RUNS": "1",
    }


def _write_port_busy_after_first_run_tools(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "ss").write_text(
        """#!/bin/bash
if [[ "${PORT_BUSY_FROM_START:-0}" == "1" || -s "$CALL_LOG" ]]; then
  echo "LISTEN 0 4096 0.0.0.0:8001 0.0.0.0:*"
fi
""",
        encoding="utf-8",
    )
    (fake_bin / "sleep").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "ss").chmod(0o755)
    (fake_bin / "sleep").chmod(0o755)
    return fake_bin


def test_formal_campaign_aborts_when_previous_service_keeps_port(
    tmp_path: Path,
) -> None:
    fake_bin = _write_port_busy_after_first_run_tools(tmp_path)
    result = _run_campaign(
        tmp_path,
        extra_env={
            **_formal_campaign_env(tmp_path),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "MAX_PORT_WAIT_SECONDS": "5",
        },
    )

    assert result.returncode == 2
    calls = (tmp_path / "calls.tsv").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "refusing to start repetition 2/3 in formal mode" in result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "aborted"
    assert summary["attempted_repetitions"] == 1
    assert summary["abort_exit_code"] == 2
    assert "still has listeners" in summary["abort_reason"]


def test_formal_campaign_rejects_port_listener_before_first_run(
    tmp_path: Path,
) -> None:
    fake_bin = _write_port_busy_after_first_run_tools(tmp_path)
    result = _run_campaign(
        tmp_path,
        extra_env={
            **_formal_campaign_env(tmp_path),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "MAX_PORT_WAIT_SECONDS": "5",
            "PORT_BUSY_FROM_START": "1",
        },
    )

    assert result.returncode == 2
    assert not (tmp_path / "calls.tsv").exists()
    assert "refusing to start repetition 1/3 in formal mode" in result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "aborted"
    assert summary["attempted_repetitions"] == 0


def test_diagnostic_campaign_warns_and_continues_when_port_stays_busy(
    tmp_path: Path,
) -> None:
    fake_bin = _write_port_busy_after_first_run_tools(tmp_path)
    result = _run_campaign(
        tmp_path,
        extra_env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "MAX_PORT_WAIT_SECONDS": "5",
        },
    )

    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls.tsv").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert result.stderr.count("proceeding in diagnostic mode") == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["attempted_repetitions"] == 3


def test_formal_campaign_accepts_matching_frozen_repositories(tmp_path: Path) -> None:
    core = tmp_path / "core"
    backend = tmp_path / "backend"
    core_commit = _init_git_repo(core)
    backend_commit = _init_git_repo(backend)
    result = _run_campaign(
        tmp_path,
        extra_env={
            "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
            "CAMPAIGN_ID": "issue-136/v1",
            "CAMPAIGN_COVERAGE_CLASS": "full-matrix",
            "CAMPAIGN_POINT_ROLE": "checkpoint",
            "CAMPAIGN_LOAD_PROFILE": "fixed-1-rps",
            "CURRENT_GIT_COMMIT": core_commit,
            "CURRENT_PLUGIN_GIT_COMMIT": backend_commit,
            "CURRENT_IMAGE_ID": "b" * 64,
            "CURRENT_MODEL_REVISION": "c" * 40,
            "CURRENT_CANN_VERSION": "9.0.0",
            "CURRENT_TORCH_NPU_VERSION": "2.10.0",
            "CURRENT_TOPOLOGY": "single-node-hccs",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            "ASCEND_VISIBLE_DEVICES": "0,1",
            "CURRENT_VLLM_HUST_REPO": str(core),
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(backend),
            "PERFGATE_WARMUP_RUNS": "0",
            "PERFGATE_MEASURED_RUNS": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign_id"] == "issue-136/v1"
    assert summary["coverage_class"] == "full-matrix"


def test_strict_campaign_accepts_explicit_experimental_series(tmp_path: Path) -> None:
    core = tmp_path / "core"
    backend = tmp_path / "backend"
    core_commit = _init_git_repo(core)
    backend_commit = _init_git_repo(backend)
    result = _run_campaign(
        tmp_path,
        extra_env={
            "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
            "CAMPAIGN_ID": "issue-136-integration/v1",
            "CAMPAIGN_COVERAGE_CLASS": "experimental",
            "CAMPAIGN_POINT_ROLE": "",
            "CAMPAIGN_COMPARISON_ID": "",
            "CAMPAIGN_LOAD_PROFILE": "fixed-1-rps",
            "CURRENT_GIT_COMMIT": core_commit,
            "CURRENT_PLUGIN_GIT_COMMIT": backend_commit,
            "CURRENT_IMAGE_ID": "b" * 64,
            "CURRENT_MODEL_REVISION": "c" * 40,
            "CURRENT_CANN_VERSION": "9.0.0",
            "CURRENT_TORCH_NPU_VERSION": "2.10.0",
            "CURRENT_TOPOLOGY": "single-node-hccs",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            "ASCEND_VISIBLE_DEVICES": "0,1",
            "CURRENT_VLLM_HUST_REPO": str(core),
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(backend),
            "PERFGATE_WARMUP_RUNS": "0",
            "PERFGATE_MEASURED_RUNS": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage_class"] == "experimental"
    assert summary["comparison_id"] == ""
    assert summary["point_role"] == ""


def test_formal_campaign_rejects_dirty_frozen_repository(tmp_path: Path) -> None:
    core = tmp_path / "core"
    backend = tmp_path / "backend"
    core_commit = _init_git_repo(core)
    backend_commit = _init_git_repo(backend)
    (backend / "untracked.txt").write_text("not frozen\n", encoding="utf-8")

    result = _run_campaign(
        tmp_path,
        extra_env={
            "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
            "CAMPAIGN_ID": "issue-136/v1",
            "CAMPAIGN_COVERAGE_CLASS": "full-matrix",
            "CAMPAIGN_POINT_ROLE": "checkpoint",
            "CAMPAIGN_LOAD_PROFILE": "fixed-1-rps",
            "CURRENT_GIT_COMMIT": core_commit,
            "CURRENT_PLUGIN_GIT_COMMIT": backend_commit,
            "CURRENT_IMAGE_ID": "b" * 64,
            "CURRENT_MODEL_REVISION": "c" * 40,
            "CURRENT_CANN_VERSION": "9.0.0",
            "CURRENT_TORCH_NPU_VERSION": "2.10.0",
            "CURRENT_TOPOLOGY": "single-node-hccs",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            "ASCEND_VISIBLE_DEVICES": "0,1",
            "CURRENT_VLLM_HUST_REPO": str(core),
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(backend),
            "PERFGATE_WARMUP_RUNS": "0",
            "PERFGATE_MEASURED_RUNS": "1",
        },
    )

    assert result.returncode == 2
    assert "formal campaigns require a clean frozen source tree" in result.stderr


def test_collector_records_declared_and_observed_worktree_commits(
    tmp_path: Path,
) -> None:
    core = tmp_path / "core"
    backend = tmp_path / "backend"
    core_commit = _init_git_repo(core)
    backend_commit = _init_git_repo(backend)
    core_worktree = tmp_path / "core-worktree"
    backend_worktree = tmp_path / "backend-worktree"
    subprocess.run(
        ["git", "-C", str(core), "worktree", "add", "-q", str(core_worktree)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(backend), "worktree", "add", "-q", str(backend_worktree)],
        check=True,
    )
    assert (core_worktree / ".git").is_file()

    artifact = tmp_path / "submissions" / "run"
    artifact.mkdir(parents=True)
    env = {
        **os.environ,
        "CURRENT_RUNTIME_PYTHON": sys.executable,
        "CURRENT_VLLM_HUST_REPO": str(core_worktree),
        "CURRENT_VLLM_ASCEND_HUST_REPO": str(backend_worktree),
        "CURRENT_GIT_COMMIT": core_commit,
        "CURRENT_PLUGIN_GIT_COMMIT": backend_commit,
        "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
        "CAMPAIGN_ID": "issue-136/v1",
        "CAMPAIGN_COVERAGE_CLASS": "full-matrix",
        "CAMPAIGN_POINT_ROLE": "checkpoint",
        "CAMPAIGN_LOAD_PROFILE": "fixed-1-rps",
        "CAMPAIGN_REPEAT_INDEX": "1",
        "CAMPAIGN_REPETITIONS": "3",
        "CURRENT_IMAGE_ID": "d" * 64,
        "CURRENT_MODEL_REVISION": "e" * 40,
        "CURRENT_CANN_VERSION": "9.0.0",
        "CURRENT_TORCH_NPU_VERSION": "2.10.0",
        "CURRENT_TOPOLOGY": "single-node-hccs",
    }
    result = subprocess.run(
        [bash_executable(), str(COLLECTOR), str(artifact)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((artifact / "env-manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "run-env-manifest/v2"
    assert manifest["git_info"]["vllm_hust"] == {
        "declared": core_commit,
        "observed": core_commit,
    }
    assert manifest["git_info"]["vllm_ascend_hust"] == {
        "declared": backend_commit,
        "observed": backend_commit,
    }
    assert manifest["campaign"]["repeat_index"] == 1
    assert manifest["frozen_inputs"]["image_id"] == "d" * 64


def test_validator_rejects_missing_environment_manifest_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "STATUS").write_text("OK\n", encoding="utf-8")
    (artifact / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (artifact / "leaderboard_manifest.json").write_text(
        json.dumps({"entries": [{"leaderboard_artifact": "run_leaderboard.json"}]})
        + "\n",
        encoding="utf-8",
    )
    (artifact / "env-manifest.json").write_text("{}\n", encoding="utf-8")
    checksums = subprocess.run(
        [
            "sha256sum",
            "run_leaderboard.json",
            "leaderboard_manifest.json",
            "env-manifest.json",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=artifact,
    ).stdout
    (artifact / "checksums.sha256").write_text(checksums, encoding="utf-8")

    result = subprocess.run(
        [bash_executable(), str(VALIDATOR), str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode > 0
    assert "missing env-manifest fields: os, python_version, collected_at" in (
        result.stderr
    )
    assert "validation error(s)" in result.stderr


def test_validator_reports_frozen_input_parser_failure_and_continues(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "STATUS").write_text("OK\n", encoding="utf-8")
    (artifact / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (artifact / "leaderboard_manifest.json").write_text(
        json.dumps({"entries": [{"leaderboard_artifact": "run_leaderboard.json"}]})
        + "\n",
        encoding="utf-8",
    )
    (artifact / "env-manifest.json").write_text(
        json.dumps(
            {
                "os": "test",
                "python_version": "test",
                "collected_at": "2026-08-03T00:00:00Z",
                "frozen_inputs_required": True,
                "frozen_inputs": {
                    "image_id": "a" * 64,
                    "model_revision": "b" * 40,
                    "topology": "single-node-hccs",
                    "cann": {"declared": "9.0.0", "detected": "9.0.0"},
                    "torch_npu_version": {
                        "declared": "2.10.0",
                        "detected": "2.10.0",
                    },
                },
                "git_info": {
                    "vllm_hust": {"declared": "c" * 40, "observed": "c" * 40},
                    "vllm_ascend_hust": {
                        "declared": "d" * 40,
                        "observed": "d" * 40,
                    },
                },
                "campaign": {
                    "campaign_id": "issue-136/v1",
                    "coverage_class": "full-matrix",
                    "point_role": "checkpoint",
                    "load_profile": "fixed-1-rps",
                    "repetitions": "three",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = subprocess.run(
        [
            "sha256sum",
            "run_leaderboard.json",
            "leaderboard_manifest.json",
            "env-manifest.json",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=artifact,
    ).stdout
    (artifact / "checksums.sha256").write_text(checksums, encoding="utf-8")

    result = subprocess.run(
        [bash_executable(), str(VALIDATOR), str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode > 0
    assert "parse-error" in result.stderr
    assert "checksums.sha256 all pass" in result.stdout
    assert "validation error(s)" in result.stderr


def test_validator_rejects_mismatched_official_runtime_provenance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    runtime = {
        "schema_version": "official-runtime-provenance/v1",
        "python_executable": "/runtime/bin/python",
        "python_version": "3.12.0",
        "sources": {
            role: {
                "module": module,
                "module_path": f"/prepared/{module}/__init__.py",
                "module_version": "1.2.3",
                "distribution": distribution,
                "distribution_version": "1.2.3",
                "prepared_worktree": f"/prepared/{module}",
                "prepared_commit": commit * 40,
                "source_version": "v1.2.3",
                "extension_policy": "none-discovered",
                "source_patch_sha256": "d" * 64,
                "source_tree_sha256": "e" * 64,
                "source_status": "clean",
                "extensions": [],
            }
            for role, module, distribution, commit in (
                ("engine", "vllm", "vllm", "a"),
                ("plugin", "vllm_ascend", "vllm-ascend", "b"),
            )
        },
    }
    (artifact / "STATUS").write_text("OK\n", encoding="utf-8")
    (artifact / "run_leaderboard.json").write_text(
        json.dumps({"metadata": {"official_runtime_provenance": runtime}}) + "\n",
        encoding="utf-8",
    )
    stale_runtime = json.loads(json.dumps(runtime))
    stale_runtime["sources"]["engine"]["prepared_commit"] = "c" * 40
    (artifact / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "entries": [{"leaderboard_artifact": "run_leaderboard.json"}],
                "official_runtime_provenance": stale_runtime,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "env-manifest.json").write_text(
        json.dumps(
            {
                "os": "test",
                "python_version": "test",
                "collected_at": "2026-08-13T00:00:00Z",
                "frozen_inputs_required": False,
                "official_runtime_provenance": runtime,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = subprocess.run(
        [
            "sha256sum",
            "run_leaderboard.json",
            "leaderboard_manifest.json",
            "env-manifest.json",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=artifact,
    ).stdout
    (artifact / "checksums.sha256").write_text(checksums, encoding="utf-8")

    result = subprocess.run(
        [bash_executable(), str(VALIDATOR), str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode > 0
    assert "official_runtime_provenance differs across artifact and manifests" in (
        result.stderr
    )
    assert "official runtime provenance is complete and consistent" in result.stderr


def test_validator_resolves_repo_before_changing_to_artifact_dir(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [bash_executable(), "scripts/validate-run-artifact.sh", str(artifact)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode > 0
    assert "cd: scripts: No such file or directory" not in result.stderr
    assert "artifact contract normalization passes" in result.stderr
