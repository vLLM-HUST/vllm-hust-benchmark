import builtins
import json
from pathlib import Path
import types

import pytest

from vllm_hust_benchmark import integration
from vllm_hust_benchmark.integration import (
    RepoLayout,
    _build_effective_env,
    aggregate_to_website,
    build_ascend_benchmark_ci_command,
    build_benchmark_script_command,
    build_performance_suite_command,
    build_vllm_bench_command,
    resolve_repo_layout,
    sync_submission_to_huggingface,
    upload_to_huggingface,
    validate_aggregated_leaderboard_outputs,
    validate_repo_layout,
)


def _minimal_manifest(artifact_name: str = "run_leaderboard.json") -> str:
    return json.dumps(
        {
            "schema_version": "leaderboard-export-manifest/v2",
            "generated_at": "2026-05-26T00:00:00Z",
            "entries": [{"leaderboard_artifact": artifact_name}],
        }
    )


def _minimal_artifact(model_name: str = "Qwen/Qwen2.5-14B-Instruct") -> str:
    return json.dumps({"model": {"name": model_name}})


def test_resolve_repo_layout_defaults_to_repo_sibling_workspace(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_HUST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("VLLM_HUST_REPO", raising=False)
    monkeypatch.delenv("VLLM_HUST_WEBSITE_REPO", raising=False)

    layout = resolve_repo_layout()

    expected_workspace_root = Path(integration.__file__).resolve().parents[3]
    assert layout.workspace_root == expected_workspace_root
    assert layout.vllm_hust_repo == (expected_workspace_root / "vllm-hust").resolve()
    assert (
        layout.reference_vllm_repo
        == (expected_workspace_root / "reference-repos" / "vllm").resolve()
    )


def test_resolve_repo_layout_from_workspace_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_HUST_WORKSPACE_ROOT", str(tmp_path))

    layout = resolve_repo_layout()

    assert layout.vllm_hust_repo == (tmp_path / "vllm-hust").resolve()
    assert (
        layout.reference_vllm_repo == (tmp_path / "reference-repos" / "vllm").resolve()
    )
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
    (repo / "pyproject.toml").write_text(
        "[project]\nname='vllm-hust'\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=repo,
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_benchmark_script_command(layout, "benchmark_serving.py", ["--help"])

    assert command[0] == integration.sys.executable
    assert command[1] == str(script)
    assert command[-1] == "--help"


def test_build_benchmark_script_command_uses_bash_for_shell_scripts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "vllm-hust"
    script = repo / "benchmarks" / "run_structured_output_benchmark.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    script.chmod(0o755)
    (repo / "pyproject.toml").write_text(
        "[project]\nname='vllm-hust'\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=repo,
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_benchmark_script_command(
        layout,
        "run_structured_output_benchmark.sh",
        ["--dry-run"],
    )

    assert command == ["bash", str(script), "--dry-run"]


def test_build_vllm_bench_command_prefers_console_script(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "vllm-hust"
    executable.write_text(
        f"#!{integration.sys.executable}\nprint('ok')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: str(executable) if name == "vllm-hust" else None,
    )

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command == [
        str(executable),
        "bench",
        "serve",
        "--model",
        "foo/bar",
    ]


def test_build_vllm_bench_command_falls_back_to_vllm_console_script(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "vllm"
    executable.write_text(
        f"#!{integration.sys.executable}\nprint('ok')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: str(executable) if name == "vllm" else None,
    )

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command == [str(executable), "bench", "serve", "--model", "foo/bar"]


def test_build_vllm_bench_command_falls_back_to_python_module(monkeypatch) -> None:
    monkeypatch.setattr(integration.shutil, "which", lambda name: None)

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command[:4] == [
        integration.sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
    ]
    assert command[-3:] == ["serve", "--model", "foo/bar"]


def test_build_vllm_bench_command_skips_broken_console_script(
    monkeypatch,
    tmp_path: Path,
) -> None:
    broken_vllm_hust = tmp_path / "vllm-hust"
    broken_vllm_hust.write_text(
        "#!/missing/python\nprint('broken')\n",
        encoding="utf-8",
    )
    broken_vllm_hust.chmod(0o755)
    fallback_vllm = tmp_path / "vllm"
    fallback_vllm.write_text(
        f"#!{integration.sys.executable}\nprint('ok')\n",
        encoding="utf-8",
    )
    fallback_vllm.chmod(0o755)

    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: (
            str(broken_vllm_hust)
            if name == "vllm-hust"
            else str(fallback_vllm)
            if name == "vllm"
            else None
        ),
    )

    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])

    assert command == [str(fallback_vllm), "bench", "serve", "--model", "foo/bar"]


def test_build_vllm_bench_command_falls_back_when_console_script_shebang_is_stale(
    monkeypatch,
    tmp_path: Path,
) -> None:
    broken_vllm = tmp_path / "vllm"
    broken_vllm.write_text(
        "#!/missing/python\nprint('broken')\n",
        encoding="utf-8",
    )
    broken_vllm.chmod(0o755)

    monkeypatch.setattr(
        integration.shutil,
        "which",
        lambda name: str(broken_vllm) if name == "vllm" else None,
    )

    command = build_vllm_bench_command(
        ["serve", "--model", "foo/bar"],
        runtime_engine="vllm",
    )

    assert command[:4] == [
        integration.sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
    ]
    assert command[-3:] == ["serve", "--model", "foo/bar"]


def test_build_effective_env_stringifies_values_and_prepends_cwd(
    monkeypatch, tmp_path: Path
) -> None:
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


def test_discover_vllm_flags_falls_back_to_benchmark_repo_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    benchmark_repo = tmp_path / "vllm-hust-benchmark"
    benchmark_repo.mkdir()
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=benchmark_repo,
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]

        class Result:
            returncode = 0
            stdout = "  --dataset-name DATASET_NAME\n  --num-prompts NUM_PROMPTS\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(integration, "resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(integration.shutil, "which", lambda name: None)
    monkeypatch.setattr(integration.subprocess, "run", fake_run)
    integration.discover_vllm_flags.cache_clear()

    flags = integration.discover_vllm_flags("bench", "serve")

    assert flags == frozenset({"dataset_name", "num_prompts"})
    assert captured["cwd"] == benchmark_repo
    assert str(benchmark_repo) in str(captured["env"]["PYTHONPATH"])
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


def test_build_ascend_benchmark_ci_command(tmp_path: Path) -> None:
    repo = tmp_path / "vllm-hust"
    script = repo / ".github" / "workflows" / "scripts" / "run_ascend_benchmark_ci.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=repo,
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_ascend_benchmark_ci_command(layout)

    assert command == ["bash", str(script)]


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


def test_validate_aggregated_leaderboard_outputs_rejects_single_engine_distribution(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps(
            [
                {"engine": "vllm-hust", "config_type": "single_gpu"},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_multi.json").write_text(
        json.dumps(
            [
                {
                    "engine": "vllm-hust",
                    "config_type": "multi_gpu",
                    "cluster": {"node_count": 1},
                },
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_compare.json").write_text(
        json.dumps(
            {"groups": [{"scope_key": "dummy"}], "goal_progress": {"pairs": []}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only vllm-hust entries"):
        validate_aggregated_leaderboard_outputs(data_dir)


def test_validate_aggregated_leaderboard_outputs_rejects_incomplete_compare_snapshot(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    single_entry = {
        "engine": "vllm-hust",
        "config_type": "single_gpu",
        "model": {"name": "Qwen2.5-7B-Instruct"},
        "hardware": {"chip_model": "910B3"},
        "workload": {"name": "sharegpt-online"},
        "constraints": {
            "accountable_scope": {
                "representative_business_scenario": "online-chat",
                "baseline_engine": "vllm",
            }
        },
    }
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps([single_entry, {**single_entry, "engine": "vllm"}]),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_multi.json").write_text("[]", encoding="utf-8")
    (data_dir / "leaderboard_compare.json").write_text(
        json.dumps(
            {
                "groups": [],
                "goal_progress": {"pairs": []},
                "hard_constraints": {
                    "scopes": [
                        {
                            "scope_key": "vllm-hust|Qwen2.5-7B-Instruct|910B3|sharegpt-online|single_gpu|online-chat|vllm"
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"neither groups nor goal_progress\.pairs"):
        validate_aggregated_leaderboard_outputs(data_dir)


def test_validate_aggregated_leaderboard_outputs_rejects_missing_scope_keys(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    single_entry = {
        "engine": "vllm-hust",
        "config_type": "single_gpu",
        "model": {"name": "Qwen2.5-7B-Instruct"},
        "hardware": {"chip_model": "910B3"},
        "workload": {"name": "sharegpt-online"},
        "constraints": {
            "accountable_scope": {
                "representative_business_scenario": "online-chat",
                "baseline_engine": "vllm",
            }
        },
    }
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps([single_entry, {**single_entry, "engine": "vllm"}]),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_multi.json").write_text("[]", encoding="utf-8")
    (data_dir / "leaderboard_compare.json").write_text(
        json.dumps(
            {
                "groups": [{"scope_key": "present-group"}],
                "goal_progress": {"pairs": []},
                "hard_constraints": {"scopes": [{"scope_key": "missing|scope|key"}]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scope keys are missing"):
        validate_aggregated_leaderboard_outputs(data_dir)


def test_validate_aggregated_leaderboard_outputs_rejects_missing_baseline_rows(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    current_entry = {
        "engine": "vllm-hust",
        "config_type": "single_gpu",
        "model": {"name": "Qwen2.5-7B-Instruct"},
        "hardware": {"chip_model": "910B3"},
        "workload": {"name": "sharegpt-online"},
        "constraints": {
            "accountable_scope": {
                "representative_business_scenario": "online-chat",
                "baseline_engine": "vllm",
            }
        },
    }
    unrelated_baseline_entry = {
        **current_entry,
        "engine": "vllm",
        "workload": {"name": "random-online"},
    }
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps([current_entry, unrelated_baseline_entry]),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_multi.json").write_text("[]", encoding="utf-8")
    (data_dir / "leaderboard_compare.json").write_text(
        json.dumps(
            {
                "groups": [{"scope_key": "present-group"}],
                "goal_progress": {"pairs": []},
                "hard_constraints": {
                    "scopes": [
                        {
                            "scope_key": "vllm-hust|Qwen2.5-7B-Instruct|910B3|sharegpt-online|single_gpu|online-chat|vllm",
                            "scope": {
                                "model": "Qwen2.5-7B-Instruct",
                                "hardware": "910B3",
                                "workload": "sharegpt-online",
                                "config_type": "single_gpu",
                                "accountable_scope": {
                                    "baseline_engine": "vllm",
                                },
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing matching baseline rows"):
        validate_aggregated_leaderboard_outputs(data_dir)


def test_validate_aggregated_leaderboard_outputs_allows_pending_baseline_rows(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    current_entry = {
        "engine": "vllm-hust",
        "config_type": "single_gpu",
        "model": {"name": "Qwen2.5-7B-Instruct"},
        "hardware": {"chip_model": "910B3"},
        "workload": {"name": "sharegpt-online"},
        "constraints": {
            "accountable_scope": {
                "representative_business_scenario": "online-chat",
                "baseline_engine": "",
                "declared_baseline_engine": "vllm",
                "baseline_status": "pending-baseline",
            }
        },
    }
    unrelated_baseline_entry = {
        **current_entry,
        "engine": "vllm",
        "workload": {"name": "random-online"},
        "constraints": {"accountable_scope": {}},
    }
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps([current_entry, unrelated_baseline_entry]),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_multi.json").write_text("[]", encoding="utf-8")
    (data_dir / "leaderboard_compare.json").write_text(
        json.dumps(
            {
                "groups": [{"scope_key": "present-group"}],
                "goal_progress": {"pairs": []},
                "hard_constraints": {
                    "scopes": [
                        {
                            "scope_key": "vllm-hust|Qwen2.5-7B-Instruct|910B3|sharegpt-online|single_gpu|online-chat|vllm",
                            "scope": {
                                "model": "Qwen2.5-7B-Instruct",
                                "hardware": "910B3",
                                "workload": "sharegpt-online",
                                "config_type": "single_gpu",
                                "accountable_scope": {
                                    "baseline_engine": "",
                                    "declared_baseline_engine": "vllm",
                                    "baseline_status": "pending-baseline",
                                },
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    validate_aggregated_leaderboard_outputs(data_dir)


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

    bench_parameters, serve_parameters = (
        integration.split_vllm_serve_scenario_parameters(
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


def test_run_local_serve_benchmark_starts_server_then_client(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "vllm-hust"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\nname='vllm-hust'\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=repo,
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
            *integration.build_vllm_command(
                [
                    "serve",
                    "foo/bar",
                    "--gpu-memory-utilization",
                    "0.6",
                    "--enforce-eager",
                ]
            ),
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
    (submission_dir / "run_leaderboard.json").write_text(
        _minimal_artifact() + "\n", encoding="utf-8"
    )
    (submission_dir / "leaderboard_manifest.json").write_text(
        _minimal_manifest() + "\n", encoding="utf-8"
    )

    downloaded_submission = tmp_path / "downloaded-existing.json"
    downloaded_submission.write_text(_minimal_artifact() + "\n", encoding="utf-8")

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
    (submission_a / "run_leaderboard.json").write_text(
        _minimal_artifact() + "\n", encoding="utf-8"
    )
    (submission_a / "leaderboard_manifest.json").write_text(
        _minimal_manifest() + "\n", encoding="utf-8"
    )

    submission_b = tmp_path / "submission-b"
    submission_b.mkdir()
    (submission_b / "run_leaderboard.json").write_text(
        _minimal_artifact("Qwen/Qwen2.5-7B-Instruct") + "\n", encoding="utf-8"
    )
    (submission_b / "leaderboard_manifest.json").write_text(
        _minimal_manifest() + "\n", encoding="utf-8"
    )

    downloaded_submission = tmp_path / "downloaded-existing.json"
    downloaded_submission.write_text(_minimal_artifact() + "\n", encoding="utf-8")

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


def test_sync_submission_to_huggingface_rejects_invalid_aggregated_snapshots(
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
    (submission_dir / "run_leaderboard.json").write_text(
        _minimal_artifact() + "\n", encoding="utf-8"
    )
    (submission_dir / "leaderboard_manifest.json").write_text(
        _minimal_manifest() + "\n", encoding="utf-8"
    )

    downloaded_submission = tmp_path / "downloaded-existing.json"
    downloaded_submission.write_text(_minimal_artifact() + "\n", encoding="utf-8")
    uploaded_paths: list[str] = []

    def fake_aggregate_to_website(*, layout, source_dir, output_dir, execute):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "leaderboard_single.json").write_text(
            json.dumps([{"engine": "vllm-hust", "config_type": "single_gpu"}]),
            encoding="utf-8",
        )
        (output_dir / "leaderboard_multi.json").write_text(
            json.dumps(
                [
                    {
                        "engine": "vllm-hust",
                        "config_type": "multi_gpu",
                        "cluster": {"node_count": 1},
                    }
                ]
            ),
            encoding="utf-8",
        )
        (output_dir / "leaderboard_compare.json").write_text(
            json.dumps(
                {
                    "groups": [],
                    "goal_progress": {"pairs": []},
                    "hard_constraints": {"scopes": []},
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "last_updated.json").write_text("{}\n", encoding="utf-8")
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

    assert exit_code == 2
    assert uploaded_paths == []


def test_sync_submission_to_huggingface_normalizes_unsupported_historical_baselines(
    monkeypatch, tmp_path: Path
) -> None:
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    benchmark_repo = tmp_path / "vllm-hust-benchmark"
    official_specs_dir = benchmark_repo / "docs" / "official-baselines"
    official_specs_dir.mkdir(parents=True)
    (official_specs_dir / "official-random-online.json").write_text(
        json.dumps(
            {
                "scenario": "random-online",
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "hardware_chip_model": "910B3",
                "chip_count": 1,
                "node_count": 1,
                "export": {"baseline_engine": "vllm"},
            }
        ),
        encoding="utf-8",
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=benchmark_repo,
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )

    submission_dir = tmp_path / "submission-a"
    submission_dir.mkdir()
    current_payload = {
        "model": {"name": "Qwen/Qwen2.5-14B-Instruct"},
        "hardware": {"chip_model": "910B3"},
        "workload": {"name": "random-online"},
        "config_type": "single_gpu",
        "constraints": {
            "accountable_scope": {
                "baseline_engine": "vllm",
                "declared_baseline_engine": "vllm",
                "baseline_status": "pending-baseline",
            }
        },
    }
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps(current_payload), encoding="utf-8"
    )
    (submission_dir / "leaderboard_manifest.json").write_text(
        _minimal_manifest() + "\n", encoding="utf-8"
    )

    downloaded_run = tmp_path / "downloaded-existing-run.json"
    downloaded_manifest = tmp_path / "downloaded-existing-manifest.json"
    downloaded_run.write_text(
        json.dumps(
            {
                "model": {"name": "Qwen2.5-7B-Instruct"},
                "hardware": {"chip_model": "910B3"},
                "workload": {"name": "sharegpt-online"},
                "config_type": "single_gpu",
                "constraints": {
                    "accountable_scope": {
                        "baseline_engine": "vllm",
                        "declared_baseline_engine": "vllm",
                        "baseline_status": "pending-baseline",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    downloaded_manifest.write_text(_minimal_manifest() + "\n", encoding="utf-8")

    seen_baselines: dict[str, dict[str, str]] = {}

    def fake_aggregate_to_website(*, layout, source_dir, output_dir, execute):
        historical = json.loads(
            source_dir.joinpath("existing-run", "run_leaderboard.json").read_text(
                encoding="utf-8"
            )
        )
        current = json.loads(
            source_dir.joinpath("submission-a", "run_leaderboard.json").read_text(
                encoding="utf-8"
            )
        )
        historical_accountable = historical["constraints"]["accountable_scope"]
        current_accountable = current["constraints"]["accountable_scope"]
        seen_baselines["historical"] = {
            "baseline_engine": historical_accountable["baseline_engine"],
            "declared_baseline_engine": historical_accountable[
                "declared_baseline_engine"
            ],
            "baseline_status": historical_accountable["baseline_status"],
        }
        seen_baselines["current"] = {
            "baseline_engine": current_accountable["baseline_engine"],
            "declared_baseline_engine": current_accountable[
                "declared_baseline_engine"
            ],
            "baseline_status": current_accountable["baseline_status"],
        }
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
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id, repo_type, revision):
            return [
                "submissions-auto/existing-run/leaderboard_manifest.json",
                "submissions-auto/existing-run/run_leaderboard.json",
            ]

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
        if filename.endswith("leaderboard_manifest.json"):
            return str(downloaded_manifest)
        return str(downloaded_run)

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
    monkeypatch.setattr(
        integration,
        "validate_aggregated_leaderboard_outputs",
        lambda _data_dir: None,
    )

    aggregate_output_dir = tmp_path / "aggregated"
    exit_code = sync_submission_to_huggingface(
        layout=layout,
        submission_dirs=submission_dir,
        aggregate_output_dir=aggregate_output_dir,
        repo_id="owner/repo",
        submissions_prefix="submissions-auto",
    )

    assert exit_code == 0
    assert seen_baselines == {
        "historical": {
            "baseline_engine": "",
            "declared_baseline_engine": "vllm",
            "baseline_status": "pending-baseline",
        },
        "current": {
            "baseline_engine": "vllm",
            "declared_baseline_engine": "vllm",
            "baseline_status": "official-covered",
        },
    }


def test_sync_submission_to_huggingface_existing_only_backfills_historical_artifacts(
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

    downloaded_submission = tmp_path / "downloaded-existing.json"
    downloaded_submission.write_text(
        json.dumps({"model": {"name": "Qwen2.5-7B-Instruct"}}),
        encoding="utf-8",
    )

    uploaded_paths: list[str] = []
    seen_model: dict[str, str] = {}

    def fake_aggregate_to_website(*, layout, source_dir, output_dir, execute):
        historical = json.loads(
            source_dir.joinpath("existing-run", "run_leaderboard.json").read_text(
                encoding="utf-8"
            )
        )
        seen_model.update(historical["model"])
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
    monkeypatch.setattr(
        integration,
        "validate_aggregated_leaderboard_outputs",
        lambda _data_dir: None,
    )
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
        submission_dirs=None,
        aggregate_output_dir=aggregate_output_dir,
        repo_id="owner/repo",
        submissions_prefix="submissions-auto",
        allow_existing_only=True,
    )

    assert exit_code == 0
    assert seen_model["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert seen_model["repo_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert seen_model["canonical_id"] == "hf:Qwen/Qwen2.5-7B-Instruct"
    assert "submissions-auto/existing-run/run_leaderboard.json" in uploaded_paths
    assert "submissions-auto/existing-run/leaderboard_manifest.json" in uploaded_paths


def test_upload_to_huggingface_rejects_invalid_aggregated_snapshots(
    monkeypatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "aggregated"
    data_dir.mkdir()
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps([{"engine": "vllm-hust", "config_type": "single_gpu"}]),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_multi.json").write_text(
        json.dumps([]),
        encoding="utf-8",
    )
    (data_dir / "leaderboard_compare.json").write_text(
        json.dumps(
            {
                "groups": [],
                "goal_progress": {"pairs": []},
                "hard_constraints": {"scopes": []},
            }
        ),
        encoding="utf-8",
    )

    called = {"upload": False}

    def fake_upload_leaderboard_to_hf(**kwargs):
        called["upload"] = True

    monkeypatch.setattr(
        "vllm_hust_benchmark.hf_publisher.upload_leaderboard_to_hf",
        fake_upload_leaderboard_to_hf,
    )

    exit_code = upload_to_huggingface(
        data_dir=data_dir,
        repo_id="owner/repo",
    )

    assert exit_code == 2
    assert called["upload"] is False


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
