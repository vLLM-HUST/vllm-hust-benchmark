import builtins
from pathlib import Path
import types

import pytest

from vllm_hust_benchmark import integration
from vllm_hust_benchmark.integration import (
    RepoLayout,
    _build_effective_env,
    aggregate_to_website,
    build_benchmark_script_command,
    build_performance_suite_command,
    build_vllm_bench_command,
    resolve_repo_layout,
    sync_submission_to_huggingface,
    validate_repo_layout,
)


def test_resolve_repo_layout_defaults_to_repo_sibling_workspace(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_HUST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("VLLM_HUST_REPO", raising=False)
    monkeypatch.delenv("VLLM_HUST_WEBSITE_REPO", raising=False)

    layout = resolve_repo_layout()

    expected_workspace_root = Path(integration.__file__).resolve().parents[3]
    assert layout.workspace_root == expected_workspace_root
    assert layout.vllm_hust_repo == (expected_workspace_root / "vllm-hust").resolve()


def test_resolve_repo_layout_from_workspace_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_HUST_WORKSPACE_ROOT", str(tmp_path))

    layout = resolve_repo_layout()

    assert layout.vllm_hust_repo == (tmp_path / "vllm-hust").resolve()
    assert layout.website_repo == (tmp_path / "vllm-hust-website").resolve()


def test_validate_repo_layout_requires_expected_repos(tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )

    with pytest.raises(ValueError, match="vllm-hust repository not found"):
        validate_repo_layout(layout)


def test_build_benchmark_script_command(tmp_path: Path) -> None:
    repo = tmp_path / "vllm-hust"
    script = repo / "benchmarks" / "benchmark_serving.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=repo,
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_benchmark_script_command(layout, "benchmark_serving.py", ["--help"])

    assert command[1] == str(script)
    assert command[-1] == "--help"


def test_build_vllm_bench_command_prefers_console_script(monkeypatch) -> None:
    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: "/tmp/bin/vllm-hust" if name == "vllm-hust" else None,
    )

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command == [
        "/tmp/bin/vllm-hust",
        "bench",
        "serve",
        "--model",
        "foo/bar",
    ]


def test_build_vllm_bench_command_falls_back_to_vllm_console_script(monkeypatch) -> None:
    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: "/tmp/bin/vllm" if name == "vllm" else None,
    )

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command == ["/tmp/bin/vllm", "bench", "serve", "--model", "foo/bar"]


def test_build_vllm_bench_command_falls_back_to_python_module(monkeypatch) -> None:
    monkeypatch.setattr(integration.shutil, "which", lambda name: None)

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command[:4] == [integration.sys.executable, "-m", "vllm.entrypoints.cli.main", "bench"]
    assert command[-3:] == ["serve", "--model", "foo/bar"]


def test_build_effective_env_stringifies_values_and_prepends_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    effective_env = _build_effective_env(tmp_path, {"DRY_RUN": 1, "FLAG": True})

    assert effective_env["DRY_RUN"] == "1"
    assert effective_env["FLAG"] == "True"
    assert effective_env["PYTHONPATH"].split(":")[0] == str(tmp_path)
    assert "/existing/path" in effective_env["PYTHONPATH"].split(":")


def test_discover_vllm_flags_falls_back_for_bench_serve(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "No module named vllm"

        return Result()

    monkeypatch.setattr(integration.subprocess, "run", fake_run)
    integration.discover_vllm_flags.cache_clear()

    flags = integration.discover_vllm_flags("bench", "serve")

    assert "dataset_name" in flags
    assert "endpoint" in flags
    assert "num_prompts" in flags
    integration.discover_vllm_flags.cache_clear()


def test_build_performance_suite_command(tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_performance_suite_command(layout)

    assert command[0] == "bash"
    assert command[1].endswith("run-performance-benchmarks.sh")


def test_aggregate_to_website_without_execute_prints_command(
    capsys, tmp_path: Path
) -> None:
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )

    exit_code = aggregate_to_website(
        layout=layout,
        source_dir=tmp_path / "exports",
        execute=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aggregate_results.py" in captured.out
    assert "--output-dir" in captured.out

def test_split_vllm_serve_scenario_parameters_uses_cli_help(monkeypatch) -> None:
    monkeypatch.setattr(
        integration,
        "discover_vllm_flags",
        lambda *parts: frozenset(
            {
                "backend",
                "endpoint",
                "dataset_name",
                "num_prompts",
                "input_len",
                "output_len",
                "num_warmups",
                "random_batch_size",
            }
        ),
    )

    bench_parameters, serve_parameters = integration.split_vllm_serve_scenario_parameters(
        {
            "backend": "vllm",
            "dataset_name": "random",
            "input_len": 8,
            "output_len": 8,
            "num_warmups": 1,
            "random_batch_size": 1,
            "gpu_memory_utilization": 0.6,
            "enforce_eager": True,
        }
    )

    assert bench_parameters == {
        "backend": "vllm",
        "dataset_name": "random",
        "input_len": 8,
        "output_len": 8,
        "num_warmups": 1,
        "random_batch_size": 1,
    }
    assert serve_parameters == {
        "gpu_memory_utilization": 0.6,
        "enforce_eager": True,
    }


def test_run_local_serve_benchmark_starts_server_then_client(monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )

    popen_calls: list[list[str]] = []
    run_calls: list[list[str]] = []
    wait_calls: list[str] = []

    class DummyProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(command, cwd, env):
        popen_calls.append(command)
        return DummyProcess(command)

    def fake_run(command, cwd, check, env):
        run_calls.append(command)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(integration.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(integration.subprocess, "run", fake_run)
    monkeypatch.setattr(
        integration,
        "_wait_for_local_server_ready",
        lambda base_url, timeout_seconds: wait_calls.append(base_url),
    )

    exit_code = integration.run_local_serve_benchmark(
        layout=layout,
        model="foo/bar",
        bench_parameters={"backend": "vllm", "dataset_name": "random", "input_len": 8},
        serve_parameters={"gpu_memory_utilization": 0.6, "enforce_eager": True},
    )

    assert exit_code == 0
    assert popen_calls == [
        [
            *integration.build_vllm_command(["serve", "foo/bar", "--gpu-memory-utilization", "0.6", "--enforce-eager"]),
        ]
    ]
    assert run_calls == [
        [
            *integration.build_vllm_command(
                [
                    "bench",
                    "serve",
                    "--model",
                    "foo/bar",
                    "--backend",
                    "vllm",
                    "--dataset-name",
                    "random",
                    "--input-len",
                    "8",
                ]
            ),
        ]
    ]
    assert wait_calls == ["http://127.0.0.1:8000"]


def test_sync_submission_to_huggingface_merges_existing_submission_and_uploads(
    monkeypatch, tmp_path: Path
) -> None:
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )

    submission_dir = tmp_path / "submission-a"
    submission_dir.mkdir()
    (submission_dir / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (submission_dir / "leaderboard_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    downloaded_submission = tmp_path / "downloaded-existing.json"
    downloaded_submission.write_text("{}\n", encoding="utf-8")

    aggregate_calls: list[tuple[Path, Path]] = []
    merged_markers: dict[str, bool] = {}
    uploaded_paths: list[str] = []

    def fake_aggregate_to_website(*, layout, source_dir, output_dir, execute):
        aggregate_calls.append((source_dir, output_dir))
        merged_markers["existing"] = source_dir.joinpath(
            "existing-run", "run_leaderboard.json"
        ).exists()
        merged_markers["current"] = source_dir.joinpath(
            "submission-a", "run_leaderboard.json"
        ).exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "leaderboard_single.json",
            "leaderboard_multi.json",
            "leaderboard_compare.json",
            "last_updated.json",
        ):
            (output_dir / file_name).write_text("{}\n", encoding="utf-8")
        return 0

    class FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            uploaded_paths.append(path_in_repo)
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id, repo_type, revision):
            return ["submissions-auto/existing-run/run_leaderboard.json"]

        def repo_info(self, repo_id, repo_type):
            return {"repo_id": repo_id, "repo_type": repo_type}

        def create_repo(self, repo_id, repo_type, private, exist_ok):
            return None

        def create_commit(
            self,
            repo_id,
            repo_type,
            operations,
            commit_message,
            revision=None,
        ):
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "branch": revision,
                "commit_message": commit_message,
                "count": len(operations),
            }

    def fake_hf_hub_download(repo_id, repo_type, filename, revision, token):
        return str(downloaded_submission)

    monkeypatch.setattr(integration, "aggregate_to_website", fake_aggregate_to_website)
    fake_hf_module = types.SimpleNamespace(
        CommitOperationAdd=FakeCommitOperationAdd,
        HfApi=FakeHfApi,
        hf_hub_download=fake_hf_hub_download,
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "huggingface_hub":
            return fake_hf_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    aggregate_output_dir = tmp_path / "aggregated"
    exit_code = sync_submission_to_huggingface(
        layout=layout,
        submission_dirs=submission_dir,
        aggregate_output_dir=aggregate_output_dir,
        repo_id="owner/repo",
        submissions_prefix="submissions-auto",
    )

    assert exit_code == 0
    assert aggregate_calls
    assert merged_markers == {"existing": True, "current": True}
    assert "leaderboard_single.json" in uploaded_paths
    assert "submissions-auto/submission-a/run_leaderboard.json" in uploaded_paths


def test_sync_submission_to_huggingface_merges_multiple_submissions_and_uploads(
    monkeypatch, tmp_path: Path
) -> None:
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )

    submission_a = tmp_path / "submission-a"
    submission_a.mkdir()
    (submission_a / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (submission_a / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")

    submission_b = tmp_path / "submission-b"
    submission_b.mkdir()
    (submission_b / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (submission_b / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")

    downloaded_submission = tmp_path / "downloaded-existing.json"
    downloaded_submission.write_text("{}\n", encoding="utf-8")

    merged_markers: dict[str, bool] = {}
    uploaded_paths: list[str] = []

    def fake_aggregate_to_website(*, layout, source_dir, output_dir, execute):
        merged_markers["existing"] = source_dir.joinpath(
            "existing-run", "run_leaderboard.json"
        ).exists()
        merged_markers["submission_a"] = source_dir.joinpath(
            "submission-a", "run_leaderboard.json"
        ).exists()
        merged_markers["submission_b"] = source_dir.joinpath(
            "submission-b", "run_leaderboard.json"
        ).exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "leaderboard_single.json",
            "leaderboard_multi.json",
            "leaderboard_compare.json",
            "last_updated.json",
        ):
            (output_dir / file_name).write_text("{}\n", encoding="utf-8")
        return 0

    class FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            uploaded_paths.append(path_in_repo)
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id, repo_type, revision):
            return ["submissions-auto/existing-run/run_leaderboard.json"]

        def repo_info(self, repo_id, repo_type):
            return {"repo_id": repo_id, "repo_type": repo_type}

        def create_repo(self, repo_id, repo_type, private, exist_ok):
            return None

        def create_commit(
            self,
            repo_id,
            repo_type,
            operations,
            commit_message,
            revision=None,
        ):
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "branch": revision,
                "commit_message": commit_message,
                "count": len(operations),
            }

    def fake_hf_hub_download(repo_id, repo_type, filename, revision, token):
        return str(downloaded_submission)

    monkeypatch.setattr(integration, "aggregate_to_website", fake_aggregate_to_website)
    fake_hf_module = types.SimpleNamespace(
        CommitOperationAdd=FakeCommitOperationAdd,
        HfApi=FakeHfApi,
        hf_hub_download=fake_hf_hub_download,
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "huggingface_hub":
            return fake_hf_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    aggregate_output_dir = tmp_path / "aggregated"
    exit_code = sync_submission_to_huggingface(
        layout=layout,
        submission_dirs=[submission_a, submission_b],
        aggregate_output_dir=aggregate_output_dir,
        repo_id="owner/repo",
        submissions_prefix="submissions-auto",
    )

    assert exit_code == 0
    assert merged_markers == {
        "existing": True,
        "submission_a": True,
        "submission_b": True,
    }
    assert "submissions-auto/submission-a/run_leaderboard.json" in uploaded_paths
    assert "submissions-auto/submission-b/run_leaderboard.json" in uploaded_paths


def test_sync_submission_to_huggingface_requires_submission_dir(
    tmp_path: Path,
) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )

    exit_code = sync_submission_to_huggingface(
        layout=layout,
        submission_dirs=tmp_path / "missing",
        aggregate_output_dir=tmp_path / "aggregated",
        repo_id="owner/repo",
    )

    assert exit_code == 2
