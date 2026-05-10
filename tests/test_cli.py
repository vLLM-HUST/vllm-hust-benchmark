import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import vllm_hust_benchmark.cli as cli_module
from vllm_hust_benchmark.cli import _parse_set_arguments
from vllm_hust_benchmark.cli import main
from vllm_hust_benchmark.integration import RepoLayout
from vllm_hust_benchmark.integration import DEFAULT_RUNTIME_ENGINE
from vllm_hust_benchmark.integration import build_vllm_bench_command


def test_build_command_prints_upstream_equivalent(capsys) -> None:
    exit_code = main(
        [
            "build-command",
            "sharegpt-online",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "vllm bench serve" in captured.out
    assert "--dataset-name sharegpt" in captured.out


def test_run_without_execute_only_prints(capsys) -> None:
    exit_code = main(
        [
            "run",
            "random-latency",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--set",
            "input_len=2048",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "vllm bench latency" in captured.out
    assert "--input-len 2048" in captured.out


def test_run_without_execute_prints_env_prefix(capsys) -> None:
    exit_code = main(
        [
            "run",
            "random-latency",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--env",
            "HF_HOME=/tmp/hf-home",
            "--set",
            "input_len=64",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "HF_HOME=/tmp/hf-home vllm bench latency" in captured.out


def test_run_both_without_execute_prints_both_runtimes(capsys) -> None:
    exit_code = main(
        [
            "run-both",
            "random-latency",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--set",
            "input_len=2048",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "runtime: vllm-hust" in captured.out
    assert "runtime: vllm" in captured.out
    assert captured.out.count("vllm bench latency") == 2


def test_run_both_execute_runs_runtimes_in_order(monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
        reference_vllm_repo=tmp_path / "reference-repos" / "vllm",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.validate_runtime_repo",
        lambda _layout, runtime, require_benchmarks=False: (
            layout.reference_vllm_repo if runtime == "vllm" else layout.vllm_hust_repo
        ),
    )

    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run_external_command(command, *, cwd, execute, env=None):
        calls.append((command, cwd, execute))
        return 0

    monkeypatch.setattr("vllm_hust_benchmark.cli.run_external_command", fake_run_external_command)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.build_vllm_bench_command",
        lambda bench_args, runtime_engine=DEFAULT_RUNTIME_ENGINE: [runtime_engine, *bench_args],
    )

    exit_code = main(
        [
            "run-both",
            "random-latency",
            "--model",
            "foo/bar",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert [cwd for _, cwd, _ in calls] == [
        layout.vllm_hust_repo,
        layout.reference_vllm_repo,
    ]
    assert [command[0] for command, _, _ in calls] == ["vllm-hust", "vllm"]
    assert all(execute is True for _, _, execute in calls)
    assert all("--model" in command and "foo/bar" in command for command, _, _ in calls)


def test_run_both_forwards_env_to_nested_runs(monkeypatch) -> None:
    captured_argv: list[list[str]] = []
    real_main = cli_module.main

    def fake_main(argv=None):
        if argv and argv[0] == "run":
            captured_argv.append(list(argv))
            return 0
        return real_main(argv)

    monkeypatch.setattr(cli_module, "main", fake_main)

    exit_code = fake_main(
        [
            "run-both",
            "random-latency",
            "--model",
            "foo/bar",
            "--env",
            "HF_HOME=/tmp/hf-home",
        ]
    )

    assert exit_code == 0
    assert len(captured_argv) == 2
    assert all("--env" in argv and "HF_HOME=/tmp/hf-home" in argv for argv in captured_argv)


def test_parse_set_arguments_normalizes_hyphenated_keys() -> None:
    parsed = _parse_set_arguments([
        "input-len=8",
        "output_len=4",
        "enforce-eager=true",
        "no-enable-prefix-caching=true",
        "gpu-memory-utilization=0.6",
    ])

    assert parsed == {
        "input_len": 8,
        "output_len": 4,
        "enforce_eager": True,
        "no_enable_prefix_caching": True,
        "gpu_memory_utilization": 0.6,
    }


def test_run_without_execute_accepts_hyphenated_overrides(capsys) -> None:
    exit_code = main(
        [
            "run",
            "random-latency",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--set",
            "input-len=8",
            "--set",
            "output-len=4",
            "--set",
            "enforce-eager=true",
            "--set",
            "no-enable-prefix-caching=true",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.count("--input-len") == 1
    assert captured.out.count("--output-len") == 1
    assert "--input-len 8" in captured.out
    assert "--output-len 4" in captured.out
    assert "--enforce-eager" in captured.out
    assert "--no-enable-prefix-caching" in captured.out


def test_show_repos_prints_resolved_layout(capsys, monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
        reference_vllm_repo=tmp_path / "reference-repos" / "vllm",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["show-repos"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(layout.vllm_hust_repo) in captured.out
    assert str(layout.reference_vllm_repo) in captured.out
    assert str(layout.website_repo) in captured.out


def test_list_tests_prints_upstream_tests(capsys, monkeypatch, tmp_path: Path) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests_dir / "serving-tests.json").write_text(
        '[{"test_name":"serving_llama8B_tp1_sharegpt","qps_list":["inf"],"server_parameters":{"model":"foo/bar"},"client_parameters":{"model":"foo/bar"}}]\n',
        encoding="utf-8",
    )
    (tests_dir / "latency-tests.json").write_text("[]\n", encoding="utf-8")
    (tests_dir / "throughput-tests.json").write_text("[]\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["list-tests", "--benchmark-type", "serve"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "serving_llama8B_tp1_sharegpt" in captured.out


def test_show_test_prints_wrapped_commands(capsys, monkeypatch, tmp_path: Path) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests_dir / "serving-tests.json").write_text(
        '[{"test_name":"serving_llama8B_tp1_sharegpt","qps_list":["inf"],"server_parameters":{"model":"foo/bar","tensor_parallel_size":1},"client_parameters":{"model":"foo/bar","backend":"vllm"}}]\n',
        encoding="utf-8",
    )
    (tests_dir / "latency-tests.json").write_text("[]\n", encoding="utf-8")
    (tests_dir / "throughput-tests.json").write_text("[]\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["show-test", "serving_llama8B_tp1_sharegpt"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "server_command: vllm serve foo/bar --tensor-parallel-size 1" in captured.out
    assert "client_command: vllm bench serve --request-rate inf --model foo/bar --backend vllm" in captured.out


def test_run_test_without_execute_prints_suite_wrapper(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests_dir / "serving-tests.json").write_text("[]\n", encoding="utf-8")
    (tests_dir / "latency-tests.json").write_text(
        '[{"test_name":"latency_llama8B_tp1","parameters":{"model":"foo/bar"}}]\n',
        encoding="utf-8",
    )
    (tests_dir / "throughput-tests.json").write_text("[]\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["run-test", "latency_llama8B_tp1", "--env", "DRY_RUN=1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "TEST_SELECTOR='^latency_llama8B_tp1$' DRY_RUN=1 bash" in captured.out


def test_run_ascend_ci_without_execute_prints_wrapped_command(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    vllm_repo.mkdir()
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    ci_dir = vllm_repo / ".github" / "workflows" / "scripts"
    ci_dir.mkdir(parents=True)
    (ci_dir / "run_ascend_benchmark_ci.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main([
        "run-ascend-ci",
        "--scenario",
        "random-online",
        "--model",
        "foo/bar",
        "--num-prompts",
        "4",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "WORKSPACE_ROOT=" in captured.out
    assert "BENCH_SCENARIO=random-online" in captured.out
    assert "MODEL_NAME=foo/bar" in captured.out
    assert "BENCH_NUM_PROMPTS=4" in captured.out
    assert "run_ascend_benchmark_ci.sh" in captured.out


def test_run_ascend_ci_execute_forwards_environment(monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr("vllm_hust_benchmark.cli.validate_repo_layout", lambda _layout: None)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.build_ascend_benchmark_ci_command",
        lambda _layout: ["bash", "run_ascend_benchmark_ci.sh"],
    )

    captured: dict[str, object] = {}

    def fake_run_external_command(command, *, cwd, execute, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["execute"] = execute
        captured["env"] = env
        return 0

    monkeypatch.setattr("vllm_hust_benchmark.cli.run_external_command", fake_run_external_command)

    exit_code = main(
        [
            "run-ascend-ci",
            "--execute",
            "--env",
            "EXTRA_FLAG=1",
            "--scenario",
            "sharegpt-online",
            "--dataset-path",
            "/tmp/sharegpt.json",
            "--constraints-file",
            "/tmp/constraints.json",
            "--publish-to-hf",
            "--hf-repo-id",
            "owner/repo",
            "--max-concurrency",
            "2",
        ]
    )

    assert exit_code == 0
    assert captured["command"] == ["bash", "run_ascend_benchmark_ci.sh"]
    assert captured["cwd"] == layout.vllm_hust_repo
    assert captured["execute"] is True
    assert captured["env"]["EXTRA_FLAG"] == 1
    assert captured["env"]["BENCH_SCENARIO"] == "sharegpt-online"
    assert captured["env"]["BENCH_DATASET_PATH"] == "/tmp/sharegpt.json"
    assert captured["env"]["BENCH_CONSTRAINTS_FILE"] == "/tmp/constraints.json"
    assert captured["env"]["PUBLISH_TO_HF"] == "1"
    assert captured["env"]["HF_REPO_ID"] == "owner/repo"
    assert captured["env"]["BENCH_MAX_CONCURRENCY"] == 2


def test_bench_without_execute_prints_wrapped_command(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.validate_runtime_repo",
        lambda _layout, _runtime, require_benchmarks=False: layout.vllm_hust_repo,
    )

    exit_code = main(["bench", "--", "serve", "--model", "foo/bar"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "bench serve --model foo/bar" in captured.out
    assert any(
        prefix in captured.out
        for prefix in ("vllm-hust", "vllm ", "-m vllm.entrypoints.cli.main")
    )


def test_bench_execute_forwards_env(monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.validate_runtime_repo",
        lambda _layout, _runtime, require_benchmarks=False: layout.vllm_hust_repo,
    )

    captured: dict[str, object] = {}

    def fake_run_external_command(command, *, cwd, execute, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["execute"] = execute
        captured["env"] = env
        return 0

    monkeypatch.setattr("vllm_hust_benchmark.cli.run_external_command", fake_run_external_command)

    exit_code = main(
        [
            "bench",
            "--execute",
            "--env",
            "HF_HOME=/tmp/hf-home",
            "throughput",
            "--model",
            "foo/bar",
        ]
    )

    assert exit_code == 0
    assert captured["cwd"] == layout.vllm_hust_repo
    assert captured["execute"] is True
    assert captured["env"] == {"HF_HOME": "/tmp/hf-home"}


def test_run_script_execute_forwards_env(monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.validate_runtime_repo",
        lambda _layout, _runtime, require_benchmarks=False: layout.vllm_hust_repo,
    )
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.build_benchmark_script_command",
        lambda _layout, _name, _args, runtime_engine=DEFAULT_RUNTIME_ENGINE: ["bash", "benchmark.sh"],
    )

    captured: dict[str, object] = {}

    def fake_run_external_command(command, *, cwd, execute, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["execute"] = execute
        captured["env"] = env
        return 0

    monkeypatch.setattr("vllm_hust_benchmark.cli.run_external_command", fake_run_external_command)

    exit_code = main(
        [
            "run-script",
            "--execute",
            "--env",
            "FOO=bar",
            "run_structured_output_benchmark.sh",
        ]
    )

    assert exit_code == 0
    assert captured["command"] == ["bash", "benchmark.sh"]
    assert captured["env"] == {"FOO": "bar"}


def test_run_execute_serve_forwards_env_to_local_runner(monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr("vllm_hust_benchmark.cli.validate_runtime_repo", lambda _layout, _runtime: layout.vllm_hust_repo)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.split_vllm_serve_scenario_parameters",
        lambda *_args, **_kwargs: (
            {"backend": "vllm", "dataset_name": "random", "input_len": 8},
            {"gpu_memory_utilization": 0.4},
        ),
    )

    captured: dict[str, object] = {}

    def fake_run_local_serve_benchmark(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("vllm_hust_benchmark.cli.run_local_serve_benchmark", fake_run_local_serve_benchmark)

    exit_code = main(
        [
            "run",
            "random-online",
            "--model",
            "foo/bar",
            "--execute",
            "--env",
            "HF_HOME=/tmp/hf-home",
        ]
    )

    assert exit_code == 0
    assert captured["env"] == {"HF_HOME": "/tmp/hf-home"}


def test_bench_runtime_vllm_uses_reference_repo(capsys, monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
        reference_vllm_repo=tmp_path / "reference-repos" / "vllm",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.validate_runtime_repo",
        lambda _layout, runtime, require_benchmarks=False: (
            layout.reference_vllm_repo if runtime == "vllm" else layout.vllm_hust_repo
        ),
    )

    captured: dict[str, object] = {}

    def fake_run_external_command(command, *, cwd, execute, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["execute"] = execute
        return 0

    monkeypatch.setattr("vllm_hust_benchmark.cli.run_external_command", fake_run_external_command)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.build_vllm_bench_command",
        lambda bench_args, runtime_engine=DEFAULT_RUNTIME_ENGINE: [runtime_engine, *bench_args],
    )

    exit_code = main(["bench", "--runtime", "vllm", "--", "serve", "--model", "foo/bar"])

    assert exit_code == 0
    assert captured["cwd"] == layout.reference_vllm_repo
    assert captured["command"] == ["vllm", "serve", "--model", "foo/bar"]


def test_build_command_runtime_vllm_passes_runtime_to_split(capsys, monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
        reference_vllm_repo=tmp_path / "reference-repos" / "vllm",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.validate_runtime_repo",
        lambda _layout, runtime, require_benchmarks=False: (
            layout.reference_vllm_repo if runtime == "vllm" else layout.vllm_hust_repo
        ),
    )

    captured: dict[str, object] = {}

    def fake_split(merged, *, runtime_engine=DEFAULT_RUNTIME_ENGINE, runtime_repo=None):
        captured["runtime_engine"] = runtime_engine
        captured["runtime_repo"] = runtime_repo
        return ({"backend": "vllm", "dataset_name": "random", "input_len": 8}, {"enforce_eager": True})

    monkeypatch.setattr("vllm_hust_benchmark.cli.split_vllm_serve_scenario_parameters", fake_split)

    exit_code = main([
        "build-command",
        "random-online",
        "--runtime",
        "vllm",
        "--model",
        "foo/bar",
        "--set",
        "input-len=8",
    ])

    assert exit_code == 0
    assert captured["runtime_engine"] == "vllm"
    assert captured["runtime_repo"] == str(layout.reference_vllm_repo)


def test_publish_website_without_execute_prints_aggregate_command(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr("vllm_hust_benchmark.cli.validate_repo_layout", lambda _layout: None)

    source_dir = tmp_path / "exports"
    exit_code = main(["publish-website", "--source-dir", str(source_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aggregate_results.py" in captured.out
    assert f"--source-dir {source_dir}" in captured.out


def test_sync_submission_to_hf_accepts_multiple_submission_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr(cli_module, "resolve_repo_layout", lambda: layout)
    monkeypatch.setattr(cli_module, "validate_repo_layout", lambda _layout: None)

    captured: dict[str, object] = {}

    def fake_sync_submission_to_huggingface(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        cli_module,
        "sync_submission_to_huggingface",
        fake_sync_submission_to_huggingface,
    )

    first = tmp_path / "submission-a"
    second = tmp_path / "submission-b"

    exit_code = main(
        [
            "sync-submission-to-hf",
            "--submission-dir",
            str(first),
            "--submission-dir",
            str(second),
            "--aggregate-output-dir",
            str(tmp_path / "aggregated"),
            "--repo-id",
            "owner/repo",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert captured["submission_dirs"] == [first.resolve(), second.resolve()]


def test_resolve_github_metadata_falls_back_to_local_git_config(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_load_github_event_payload", lambda: {})
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    values = {
        "remote.origin.url": "git@github.com:vLLM-HUST/vllm-hust.git",
        "remote.upstream.url": None,
        "github.user": None,
        "user.name": "hust-benchmark-bot",
    }
    monkeypatch.setattr(cli_module, "_run_git_config", lambda key: values.get(key))

    metadata = cli_module._resolve_github_metadata(Namespace(git_commit="abc123def"))

    assert metadata["github_repository"] == "vLLM-HUST/vllm-hust"
    assert metadata["github_user"] == "hust-benchmark-bot"
    assert (
        metadata["github_commit_url"]
        == "https://github.com/vLLM-HUST/vllm-hust/commit/abc123def"
    )


def test_resolve_github_metadata_prefers_ci_environment(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_load_github_event_payload", lambda: {})
    monkeypatch.setenv("GITHUB_REPOSITORY", "vLLM-HUST/vllm-ascend-hust")
    monkeypatch.setenv("GITHUB_ACTOR", "ci-runner")
    monkeypatch.setenv("GITHUB_SHA", "def456")
    monkeypatch.setattr(cli_module, "_run_git_config", lambda _key: "ignored")

    metadata = cli_module._resolve_github_metadata(Namespace())

    assert metadata["github_repository"] == "vLLM-HUST/vllm-ascend-hust"
    assert metadata["github_user"] == "ci-runner"
    assert (
        metadata["github_commit_url"]
        == "https://github.com/vLLM-HUST/vllm-ascend-hust/commit/def456"
    )


def test_export_leaderboard_artifact(tmp_path) -> None:
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        """
{
    "metrics": {
        "ttft_ms": 42.0,
        "throughput_tps": 321.0,
        "peak_mem_mb": 10240,
        "error_rate": 0.0
    },
    "constraints_metrics": {
        "single_chip_effective_utilization_pct": 92.0,
        "typical_throughput_ratio_vs_baseline": 2.2,
        "typical_ttft_reduction_pct_vs_baseline": 23.0,
        "typical_tpot_reduction_pct_vs_baseline": 25.0,
        "long_context_length": 32768,
        "long_context_throughput_stable": true,
        "long_context_ttft_p95_ms": 80.0,
        "long_context_ttft_p99_ms": 95.0,
        "long_context_tpot_p95_ms": 9.0,
        "long_context_tpot_p99_ms": 10.0,
        "long_context_ttft_p95_stable": true,
        "long_context_ttft_p99_stable": true,
        "long_context_tpot_p95_stable": true,
        "long_context_tpot_p99_stable": true,
        "unit_token_cost_reduction_pct": 35.0,
        "multi_tenant_high_utilization": true
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export"
    exit_code = main(
        [
            "export-leaderboard-artifact",
            "sharegpt-online",
            "--metrics-file",
            str(metrics_file),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "smoke-run-1",
            "--engine",
            "vllm-hust",
            "--engine-version",
            "0.7.3",
            "--model-name",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--hardware-chip-model",
            "Ascend-910B",
            "--submitter",
            "ci",
            "--git-commit",
            "abc123def456",
            "--github-user",
            "octocat",
            "--github-repository",
            "vLLM-HUST/vllm-hust",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "run_leaderboard.json").is_file()
    assert (output_dir / "leaderboard_manifest.json").is_file()
    artifact = json.loads((output_dir / "run_leaderboard.json").read_text(encoding="utf-8"))
    assert artifact["metadata"]["git_commit"] == "abc123def456"
    assert artifact["metadata"]["github_user"] == "octocat"
    assert artifact["metadata"]["github_commit_url"] == "https://github.com/vLLM-HUST/vllm-hust/commit/abc123def456"


def test_export_leaderboard_artifact_from_raw_benchmark_result(tmp_path) -> None:
    benchmark_result = tmp_path / "serve_result.json"
    benchmark_result.write_text(
        """
{
    "completed": 10,
    "failed": 1,
    "request_throughput": 5.5,
    "output_throughput": 321.0,
    "mean_ttft_ms": 42.0,
    "errors": [null, null, null, null, null, null, null, null, null, null, "boom"]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    constraints_file = tmp_path / "constraints.json"
    constraints_file.write_text(
        """
{
    "single_chip_effective_utilization_pct": 92.0,
    "typical_throughput_ratio_vs_baseline": 2.2,
    "typical_ttft_reduction_pct_vs_baseline": 23.0,
    "typical_tpot_reduction_pct_vs_baseline": 25.0,
    "long_context_length": 32768,
    "long_context_throughput_stable": true,
    "long_context_ttft_p95_ms": 80.0,
    "long_context_ttft_p99_ms": 95.0,
    "long_context_tpot_p95_ms": 9.0,
    "long_context_tpot_p99_ms": 10.0,
    "long_context_ttft_p95_stable": true,
    "long_context_ttft_p99_stable": true,
    "long_context_tpot_p95_stable": true,
    "long_context_tpot_p99_stable": true,
    "unit_token_cost_reduction_pct": 35.0,
    "multi_tenant_high_utilization": true
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export_raw"
    exit_code = main(
        [
            "export-leaderboard-artifact",
            "sharegpt-online",
            "--benchmark-result-file",
            str(benchmark_result),
            "--constraints-file",
            str(constraints_file),
            "--peak-mem-mb",
            "10240",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "smoke-run-2",
            "--engine",
            "vllm-hust",
            "--engine-version",
            "0.7.3",
            "--model-name",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--hardware-chip-model",
            "Ascend-910B",
            "--submitter",
            "ci",
            "--github-user",
            "benchmark-bot",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "run_leaderboard.json").is_file()
    assert (output_dir / "leaderboard_manifest.json").is_file()
    artifact = json.loads((output_dir / "run_leaderboard.json").read_text(encoding="utf-8"))
    assert artifact["metadata"]["github_user"] == "benchmark-bot"


def test_export_leaderboard_artifact_embeds_same_spec_payload(tmp_path: Path) -> None:
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        """
{
    "metrics": {
        "ttft_ms": 150.0,
        "throughput_tps": 98.5,
        "peak_mem_mb": 2048,
        "error_rate": 0.0
    },
    "constraints_metrics": {
        "single_chip_effective_utilization_pct": 91.0,
        "typical_throughput_ratio_vs_baseline": 2.1,
        "typical_ttft_reduction_pct_vs_baseline": 22.0,
        "typical_tpot_reduction_pct_vs_baseline": 21.0,
        "long_context_length": 32768,
        "long_context_throughput_stable": true,
        "long_context_ttft_p95_ms": 100.0,
        "long_context_ttft_p99_ms": 120.0,
        "long_context_tpot_p95_ms": 15.0,
        "long_context_tpot_p99_ms": 18.0,
        "long_context_ttft_p95_stable": true,
        "long_context_ttft_p99_stable": true,
        "long_context_tpot_p95_stable": true,
        "long_context_tpot_p99_stable": true,
        "unit_token_cost_reduction_pct": 30.0,
        "multi_tenant_high_utilization": true
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    same_spec_file = tmp_path / "same_spec.json"
    same_spec_file.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-same-spec/v1",
                "spec_id": "spec-1",
                "resolved_spec_hash": "abc123",
                "resolved_server_parameters": {"dtype": "float16"},
                "resolved_client_parameters": {"dataset_name": "random"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export_same_spec"
    exit_code = main(
        [
            "export-leaderboard-artifact",
            "sharegpt-online",
            "--metrics-file",
            str(metrics_file),
            "--same-spec-file",
            str(same_spec_file),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "same-spec-run-1",
            "--engine",
            "vllm-hust",
            "--engine-version",
            "0.7.3",
            "--model-name",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--hardware-chip-model",
            "Ascend-910B",
            "--submitter",
            "ci",
            "--runtime-python",
            "/root/miniconda3/envs/vllm-hust-dev/bin/python",
            "--engine-source-repository",
            "vLLM-HUST/vllm-hust",
            "--engine-source-ref",
            "main",
            "--engine-source-commit",
            "abc123def456",
            "--plugin-source-engine",
            "vllm-ascend-hust",
            "--plugin-source-repository",
            "vLLM-HUST/vllm-ascend-hust",
            "--plugin-source-ref",
            "ws/fix/same_spec_current_artifact",
            "--plugin-source-commit",
            "fedcba654321",
        ]
    )

    assert exit_code == 0
    artifact = json.loads((output_dir / "run_leaderboard.json").read_text(encoding="utf-8"))
    assert artifact["same_spec"]["spec_id"] == "spec-1"
    assert artifact["same_spec"]["resolved_spec_hash"] == "abc123"
    assert artifact["metadata"]["runtime_provenance"]["python"] == (
        "/root/miniconda3/envs/vllm-hust-dev/bin/python"
    )
    assert artifact["metadata"]["runtime_provenance"]["engine"]["repository"] == (
        "vLLM-HUST/vllm-hust"
    )
    assert artifact["metadata"]["runtime_provenance"]["plugin"]["repository"] == (
        "vLLM-HUST/vllm-ascend-hust"
    )


def test_export_leaderboard_artifact_rejects_zero_long_context_length(
    tmp_path: Path, capsys
) -> None:
    benchmark_result = tmp_path / "serve_result.json"
    benchmark_result.write_text(
        """
{
    "completed": 2,
    "failed": 0,
    "output_throughput": 10.0,
    "mean_ttft_ms": 42.0
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    constraints_file = tmp_path / "constraints.json"
    constraints_file.write_text(
        """
{
    "single_chip_effective_utilization_pct": null,
    "typical_throughput_ratio_vs_baseline": null,
    "typical_ttft_reduction_pct_vs_baseline": null,
    "typical_tpot_reduction_pct_vs_baseline": null,
    "long_context_length": 0,
    "long_context_throughput_stable": null,
    "long_context_ttft_p95_ms": null,
    "long_context_ttft_p99_ms": null,
    "long_context_tpot_p95_ms": null,
    "long_context_tpot_p99_ms": null,
    "long_context_ttft_p95_stable": null,
    "long_context_ttft_p99_stable": null,
    "long_context_tpot_p95_stable": null,
    "long_context_tpot_p99_stable": null,
    "unit_token_cost_reduction_pct": null,
    "multi_tenant_high_utilization": null
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "export-leaderboard-artifact",
            "random-online",
            "--benchmark-result-file",
            str(benchmark_result),
            "--constraints-file",
            str(constraints_file),
            "--output-dir",
            str(tmp_path / "export_invalid"),
            "--run-id",
            "bad-run-1",
            "--engine",
            "vllm-ascend-hust",
            "--engine-version",
            "28568c22",
            "--model-name",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--hardware-chip-model",
            "910B3",
            "--submitter",
            "ci",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "constraints_metrics.long_context_length must be null or >= 1" in captured.err



def test_build_vllm_bench_command_prefers_console_script(tmp_path: Path):
    executable = tmp_path / "vllm"
    executable.write_text(
        "#!/usr/bin/env python3\nprint('ok')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    with patch("vllm_hust_benchmark.integration.shutil.which", return_value=str(executable)):
        command = build_vllm_bench_command(["latency", "--model", "tiny-model"])

    assert command == [str(executable), "bench", "latency", "--model", "tiny-model"]


def test_build_vllm_bench_command_falls_back_to_python_module():
    with patch("vllm_hust_benchmark.integration.shutil.which", return_value=None), patch(
        "vllm_hust_benchmark.integration.sys.executable", "/usr/bin/python3"
    ):
        command = build_vllm_bench_command(["latency", "--model", "tiny-model"])

    assert command == [
        "/usr/bin/python3",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        "tiny-model",
    ]


def test_build_command_prints_server_and_client_commands_for_local_serve(
    capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        "vllm_hust_benchmark.cli.split_vllm_serve_scenario_parameters",
        lambda merged, runtime_engine=DEFAULT_RUNTIME_ENGINE, runtime_repo=None: (
            {
                key: merged[key]
                for key in (
                    "backend",
                    "endpoint",
                    "dataset_name",
                    "num_prompts",
                    "input_len",
                    "output_len",
                    "num_warmups",
                    "random_batch_size",
                )
                if key in merged
            },
            {
                key: value
                for key, value in merged.items()
                if key
                not in {
                    "backend",
                    "endpoint",
                    "dataset_name",
                    "num_prompts",
                    "input_len",
                    "output_len",
                    "num_warmups",
                    "random_batch_size",
                }
            },
        ),
    )

    exit_code = main(
        [
            "build-command",
            "random-online",
            "--model",
            "foo/bar",
            "--set",
            "input-len=8",
            "--set",
            "output-len=8",
            "--set",
            "batch-size=1",
            "--set",
            "num-iters-warmup=1",
            "--set",
            "gpu-memory-utilization=0.6",
            "--set",
            "enforce-eager=true",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "server_command:" in captured.out
    assert "client_command:" in captured.out
    assert "--random-batch-size 1" in captured.out
    assert "--num-warmups 1" in captured.out
    assert "--gpu-memory-utilization 0.6" in captured.out
    assert "--enforce-eager" in captured.out
    assert "--batch-size" not in captured.out
    assert "--num-iters-warmup" not in captured.out
